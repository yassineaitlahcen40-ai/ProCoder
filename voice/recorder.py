"""Explicit, bounded microphone capture with temporary WAV cleanup."""

import select
import sys
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from voice.base import AudioRecording
from voice.errors import MicrophoneUnavailableError, VoiceRecordingError


class _AudioChunk(Protocol):
    def tobytes(self) -> bytes: ...


class _AudioInputStream(Protocol):
    def start(self) -> None: ...

    def read(self, frames: int) -> tuple[bytes | _AudioChunk, bool]: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class MicrophoneRecorder:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        chunk_duration_seconds: float = 0.1,
        stream_factory: Callable[..., _AudioInputStream] | None = None,
        input_available: Callable[[], bool] | None = None,
        stop_requested: Callable[[], bool] | None = None,
        on_started: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_duration_seconds = chunk_duration_seconds
        self._stream_factory = stream_factory or self._default_stream_factory
        self._input_available = input_available or self._default_input_available
        self._stop_requested = stop_requested or _enter_was_pressed
        self._on_started = on_started
        self._clock = clock

    def record(self, max_duration_seconds: int) -> AudioRecording:
        if max_duration_seconds < 1:
            raise ValueError("Maximum recording duration must be positive.")
        try:
            available = self._input_available()
        except Exception as exc:
            raise MicrophoneUnavailableError() from exc
        if not available:
            raise MicrophoneUnavailableError()
        try:
            stream = self._stream_factory(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
            )
        except Exception as exc:
            raise VoiceRecordingError() from exc

        if self._on_started is None:
            print(
                "Recording started. Press Enter to stop; recording stops "
                f"automatically after {max_duration_seconds} seconds."
            )
        audio = bytearray()
        started = self._clock()
        deadline = started + max_duration_seconds
        try:
            stream.start()
            if self._on_started is not None:
                self._on_started()
            while True:
                remaining = deadline - self._clock()
                if remaining <= 0 or self._stop_requested():
                    break
                frames = max(
                    1,
                    min(
                        round(self._sample_rate * self._chunk_duration_seconds),
                        int(remaining * self._sample_rate),
                    ),
                )
                chunk, _ = stream.read(frames)
                audio.extend(chunk if isinstance(chunk, bytes) else chunk.tobytes())
        except Exception as exc:
            raise VoiceRecordingError() from exc
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                if audio:
                    raise VoiceRecordingError() from exc

        if not audio:
            raise VoiceRecordingError()

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            with wave.open(str(temporary_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self._sample_rate)
                wav_file.writeframes(audio)
        except Exception as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise VoiceRecordingError() from exc

        return AudioRecording(
            path=temporary_path,
            duration_seconds=len(audio) / (self._sample_rate * 2),
            sample_rate=self._sample_rate,
        )

    @staticmethod
    def _default_input_available() -> bool:
        import sounddevice

        sounddevice.query_devices(kind="input")
        return True

    @staticmethod
    def _default_stream_factory(**kwargs: object) -> _AudioInputStream:
        import sounddevice

        return sounddevice.InputStream(**kwargs)


def _enter_was_pressed() -> bool:
    if sys.platform == "win32":
        import msvcrt

        if msvcrt.kbhit():
            return msvcrt.getwch() in ("\r", "\n")
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(ready and sys.stdin.readline())
