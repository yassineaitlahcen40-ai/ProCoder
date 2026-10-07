"""Provider-independent types for audio capture and transcription."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from voice.errors import VoiceRecordingError


@dataclass(frozen=True)
class AudioRecording:
    path: Path
    duration_seconds: float
    sample_rate: int

    def cleanup(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError as exc:
            raise VoiceRecordingError() from exc


class Recorder(Protocol):
    def record(self, max_duration_seconds: int) -> AudioRecording: ...


class TranscriptionProvider(Protocol):
    def transcribe(self, recording: AudioRecording) -> str: ...
