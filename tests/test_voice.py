import io
import json
import logging
import os
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from types import ModuleType
from pathlib import Path
from unittest.mock import Mock, patch

from core.config import Settings
from main import main
from voice.base import AudioRecording
from voice.errors import (
    InvalidTranscriptError,
    MicrophoneUnavailableError,
    VoiceTranscriptionError,
)
from voice.providers.faster_whisper import FasterWhisperProvider
from voice.providers import transcribe_worker
from voice.recorder import MicrophoneRecorder
from voice.service import VoiceInputService, validate_transcript


class FakeStream:
    def __init__(self, **_kwargs) -> None:
        self.started = False
        self.stopped = False
        self.closed = False
        self.frames_read: list[int] = []

    def start(self) -> None:
        self.started = True

    def read(self, frames: int) -> tuple[bytes, bool]:
        self.frames_read.append(frames)
        return b"\x00\x00" * frames, False

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class VoiceRecordingTests(unittest.TestCase):
    def test_voice_settings_have_safe_bounded_defaults(self) -> None:
        settings = Settings.from_environment({})
        self.assertEqual(settings.voice_max_duration_seconds, 30)
        self.assertEqual(settings.voice_model, "base")
        self.assertEqual(settings.voice_transcription_timeout, 300)
        with self.assertRaisesRegex(ValueError, "VOICE_MAX_DURATION_SECONDS"):
            Settings.from_environment({"VOICE_MAX_DURATION_SECONDS": "301"})

    def test_recorder_stops_on_explicit_stop_and_writes_temporary_wav(self) -> None:
        stream = FakeStream()
        stop_checks = iter((False, True))
        recorder = MicrophoneRecorder(
            sample_rate=10,
            stream_factory=lambda **kwargs: stream,
            input_available=lambda: True,
            stop_requested=lambda: next(stop_checks),
        )

        recording = recorder.record(10)

        try:
            self.assertTrue(recording.path.is_file())
            self.assertGreater(recording.duration_seconds, 0)
            self.assertTrue(stream.started)
            self.assertTrue(stream.stopped)
            self.assertTrue(stream.closed)
        finally:
            recording.cleanup()
        self.assertFalse(recording.path.exists())

    def test_recorder_stops_at_configured_duration(self) -> None:
        stream = FakeStream()
        now = [0.0]
        recorder = MicrophoneRecorder(
            sample_rate=10,
            chunk_duration_seconds=0.1,
            stream_factory=lambda **kwargs: stream,
            input_available=lambda: True,
            stop_requested=lambda: False,
            clock=lambda: now[0],
        )

        def read_with_elapsed_time(frames: int) -> tuple[bytes, bool]:
            stream.frames_read.append(frames)
            now[0] += 0.25
            return b"\x00\x00" * frames, False

        stream.read = read_with_elapsed_time
        recording = recorder.record(1)
        try:
            self.assertLessEqual(recording.duration_seconds, 1)
            self.assertLessEqual(sum(stream.frames_read), 10)
        finally:
            recording.cleanup()

    def test_missing_microphone_is_reported(self) -> None:
        recorder = MicrophoneRecorder(input_available=lambda: False)
        with self.assertRaises(MicrophoneUnavailableError):
            recorder.record(5)

    def test_transcript_validation_trims_and_rejects_unsafe_shapes(self) -> None:
        self.assertEqual(validate_transcript("  Make a prime checker. \n"), "Make a prime checker.")
        for invalid in (" \n\t ", "a" * 4_001, "request\x00"):
            with self.subTest(invalid=invalid[:20]), self.assertRaises(
                InvalidTranscriptError
            ):
                validate_transcript(invalid)

    def test_service_cleans_audio_after_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "capture.wav"
            audio_path.write_bytes(b"audio")
            recording = AudioRecording(audio_path, 1, 16_000)
            recorder = Mock()
            recorder.record.return_value = recording
            provider = Mock()
            provider.transcribe.return_value = "  Create a calculator. "

            transcript = VoiceInputService(recorder, provider).transcribe_request(20)

            self.assertEqual(transcript, "Create a calculator.")
            provider.transcribe.assert_called_once_with(recording)
            self.assertFalse(audio_path.exists())

    def test_service_cleans_audio_when_provider_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "capture.wav"
            audio_path.write_bytes(b"audio")
            recorder = Mock()
            recorder.record.return_value = AudioRecording(audio_path, 1, 16_000)
            provider = Mock()
            provider.transcribe.side_effect = VoiceTranscriptionError()

            with self.assertRaises(VoiceTranscriptionError):
                VoiceInputService(recorder, provider).transcribe_request(20)
            self.assertFalse(audio_path.exists())

    def test_local_worker_receives_no_provider_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "capture.wav"
            audio_path.write_bytes(b"wav")
            process = Mock(
                returncode=0,
                stdout='{"ok":true,"text":"spoken request"}',
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "GEMINI_API_KEY": "do-not-forward-gemini",
                        "NVIDIA_API_KEY": "do-not-forward-nvidia",
                        "OPENAI_API_KEY": "do-not-forward-openai",
                    },
                ),
                patch(
                    "voice.providers.faster_whisper.subprocess.run",
                    return_value=process,
                ) as run_worker,
            ):
                provider = FasterWhisperProvider(timeout_seconds=17)
                transcript = provider.transcribe(AudioRecording(audio_path, 1, 16_000))

            self.assertEqual(transcript, "spoken request")
            worker_environment = run_worker.call_args.kwargs["env"]
            self.assertNotIn("GEMINI_API_KEY", worker_environment)
            self.assertNotIn("NVIDIA_API_KEY", worker_environment)
            self.assertNotIn("OPENAI_API_KEY", worker_environment)
            self.assertEqual(run_worker.call_args.kwargs["timeout"], 17)
            self.assertEqual(
                run_worker.call_args.args[0][1:3],
                ["-m", "voice.providers.transcribe_worker"],
            )
            self.assertEqual(
                run_worker.call_args.kwargs["cwd"],
                Path(__file__).resolve().parents[1],
            )

    def test_real_worker_module_imports_without_model_download(self) -> None:
        result = FasterWhisperProvider(timeout_seconds=30).check_worker_import()
        self.assertTrue(result["ok"])
        self.assertEqual(result["safe_reason"],
                         "Voice worker module imported successfully.")
        self.assertIsNone(result["model_cached"])
        self.assertFalse(result["model_download_attempted"])

    def test_worker_validates_wav_before_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid_wav = Path(directory) / "invalid.wav"
            invalid_wav.write_bytes(b"not-a-wave")
            output = io.StringIO()
            fake_module = ModuleType("faster_whisper")
            with (
                patch.object(transcribe_worker, "_model_is_cached", return_value=True),
                patch.dict(
                    sys.modules,
                    {"faster_whisper": fake_module},
                ),
                patch.object(sys, "argv", ["worker", "base", str(invalid_wav)]),
                redirect_stdout(output),
            ):
                result = transcribe_worker.main()
            payload = json.loads(output.getvalue())
        self.assertEqual(result, 1)
        self.assertEqual(payload["failure_phase"], "audio_validation")
        self.assertFalse(payload["audio_valid"])
        self.assertNotIn("WhisperModel", fake_module.__dict__)

    def test_worker_reports_model_loading_exception_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "valid.wav"
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes(b"\x00\x00" * 16)
            fake_module = ModuleType("faster_whisper")

            def fail_model_load(*_args, **_kwargs):
                try:
                    raise OSError("failed at C:\\private\\user\\cache\\model")
                except OSError as cause:
                    raise RuntimeError("model initialization failed") from cause

            fake_module.WhisperModel = fail_model_load
            output = io.StringIO()
            with (
                patch.object(transcribe_worker, "_model_is_cached", return_value=False),
                patch.dict(sys.modules, {"faster_whisper": fake_module}),
                patch.object(sys, "argv", ["worker", "base", str(audio_path)]),
                redirect_stdout(output),
            ):
                result = transcribe_worker.main()
            payload = json.loads(output.getvalue())
        self.assertEqual(result, 1)
        self.assertEqual(payload["failure_phase"], "model_loading")
        self.assertEqual(payload["underlying_exception_type"], "RuntimeError")
        self.assertEqual(payload["cause_type"], "OSError")
        self.assertTrue(payload["audio_valid"])
        self.assertTrue(payload["model_download_attempted"])
        self.assertNotIn(directory, payload["safe_reason"])
        self.assertNotIn("private", payload["safe_reason"])

    def test_setup_worker_loads_model_without_audio_or_transcription(self) -> None:
        fake_module = ModuleType("faster_whisper")
        loaded: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs) -> None:
                loaded.append((args, kwargs))

            def transcribe(self, *_args, **_kwargs):
                raise AssertionError("setup must not transcribe audio")

        fake_module.WhisperModel = FakeWhisperModel
        output = io.StringIO()
        with (
            patch.object(transcribe_worker, "_model_is_cached", return_value=False),
            patch.dict(sys.modules, {"faster_whisper": fake_module}),
            patch.object(sys, "argv", ["worker", "--setup-model", "base"]),
            redirect_stdout(output),
        ):
            result = transcribe_worker.main()
        payload = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["failure_phase"], None)
        self.assertTrue(payload["model_download_attempted"])
        self.assertEqual(loaded[0][0], ("base",))
        self.assertEqual(loaded[0][1]["device"], "cpu")
        self.assertEqual(loaded[0][1]["compute_type"], "int8")
        self.assertFalse(loaded[0][1]["local_files_only"])

    def test_provider_preserves_only_safe_worker_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "capture.wav"
            audio_path.write_bytes(b"audio")
            worker_result = {
                "ok": False,
                "failure_phase": "model_loading",
                "model": "base",
                "device": "cpu",
                "compute_type": "int8",
                "model_cached": False,
                "model_download_attempted": True,
                "underlying_exception_type": "RuntimeError",
                "cause_type": "OSError",
                "safe_reason": "safe reason",
                "audio_valid": True,
                "private_path": "must not propagate",
            }
            with patch(
                "voice.providers.faster_whisper.subprocess.run",
                return_value=Mock(
                    returncode=1,
                    stdout=json.dumps(worker_result),
                    stderr="never log stderr",
                ),
            ):
                with self.assertRaises(VoiceTranscriptionError) as raised:
                    FasterWhisperProvider().transcribe(
                        AudioRecording(audio_path, 1, 16_000)
                    )
        self.assertEqual(
            raised.exception.diagnostics["underlying_exception_type"], "RuntimeError"
        )
        self.assertNotIn("private_path", raised.exception.diagnostics)
        self.assertNotIn("stderr", repr(raised.exception.diagnostics))

    def test_voice_failure_log_contains_only_safe_diagnostic_fields(self) -> None:
        settings = Settings.from_environment({})
        logger = Mock(spec=logging.Logger)
        output = io.StringIO()
        errors = io.StringIO()
        diagnostic_error = VoiceTranscriptionError(
            {
                "failure_phase": "model_loading",
                "underlying_exception_type": "RuntimeError",
                "cause_type": "OSError",
                "safe_reason": "connection refused <url>",
                "model": "base",
                "device": "cpu",
                "compute_type": "int8",
                "model_cached": False,
                "model_download_attempted": True,
                "audio_valid": True,
                "audio_size_bytes": 32044,
            }
        )
        service = Mock()
        service.return_value.transcribe_request.side_effect = diagnostic_error
        with (
            patch("main.Settings.from_environment", return_value=settings),
            patch("main.configure_logging", return_value=logger),
            patch("main.VoiceInputService", service),
            patch("main.MicrophoneRecorder"),
            patch("main.FasterWhisperProvider"),
            patch("builtins.input", return_value=""),
            patch("sys.stdout", output),
            patch("sys.stderr", errors),
        ):
            self.assertEqual(main(["voice"]), 1)
        event_data = logger.warning.call_args.kwargs["extra"]["event_data"]
        self.assertEqual(event_data["underlying_exception_type"], "RuntimeError")
        self.assertEqual(event_data["cause_type"], "OSError")
        self.assertEqual(event_data["failure_phase"], "model_loading")
        self.assertEqual(event_data["model_download_attempted"], True)
        self.assertNotIn("API_KEY", repr(event_data))

    def test_model_check_is_cached_only_and_does_not_run_workflow(self) -> None:
        settings = Settings.from_environment({})
        logger = Mock(spec=logging.Logger)
        payload = {
            "ok": False,
            "model": "base",
            "device": "cpu",
            "compute_type": "int8",
            "model_cached": False,
            "model_download_attempted": False,
            "failure_phase": "model_cache_check",
            "safe_reason": "Model is not available in the local cache.",
        }
        output = io.StringIO()
        errors = io.StringIO()
        with (
            patch("main.Settings.from_environment", return_value=settings),
            patch("main.configure_logging", return_value=logger),
            patch(
                "main.FasterWhisperProvider"
            ) as provider_factory,
            patch("main.run_generate_command") as run_workflow,
            patch("sys.stdout", output),
            patch("sys.stderr", errors),
        ):
            provider_factory.return_value.check_model.return_value = payload
            result = main(["voice-check"])
        self.assertEqual(result, 1)
        self.assertNotIn("VOICE MODEL READY", output.getvalue())
        run_workflow.assert_not_called()
        self.assertFalse(payload["model_download_attempted"])

    def test_voice_setup_only_calls_local_model_setup(self) -> None:
        settings = Settings.from_environment({})
        logger = Mock(spec=logging.Logger)
        output = io.StringIO()
        errors = io.StringIO()
        provider_factory = Mock()
        provider_factory.return_value.setup_model.return_value = {
            "ok": True,
            "model": "base",
            "device": "cpu",
            "compute_type": "int8",
            "model_cached": False,
            "model_download_attempted": True,
            "failure_phase": None,
            "safe_reason": "Model loaded successfully on CPU with int8.",
        }
        with (
            patch("main.Settings.from_environment", return_value=settings),
            patch("main.configure_logging", return_value=logger),
            patch("main.FasterWhisperProvider", provider_factory),
            patch("main.MicrophoneRecorder") as recorder,
            patch("main.GeminiPromptAgent") as gemini,
            patch("main.OpenAICodexCodingAgent") as codex,
            patch("main.NvidiaNimReviewAgent") as nvidia,
            patch("main.DockerSandboxRunner") as docker,
            patch("main.run_generate_command") as workflow,
            patch("sys.stdout", output),
            patch("sys.stderr", errors),
        ):
            result = main(["voice-setup"])

        self.assertEqual(result, 0)
        self.assertIn("Model files may be downloaded", output.getvalue())
        self.assertIn("VOICE MODEL READY", output.getvalue())
        self.assertEqual(errors.getvalue(), "")
        provider_factory.return_value.setup_model.assert_called_once_with()
        recorder.assert_not_called()
        gemini.assert_not_called()
        codex.assert_not_called()
        nvidia.assert_not_called()
        docker.assert_not_called()
        workflow.assert_not_called()

    def test_voice_setup_reports_safe_failure_diagnostics(self) -> None:
        settings = Settings.from_environment({})
        logger = Mock(spec=logging.Logger)
        output = io.StringIO()
        errors = io.StringIO()
        provider = Mock()
        provider.return_value.setup_model.return_value = {
            "ok": False,
            "model": "base",
            "device": "cpu",
            "compute_type": "int8",
            "failure_phase": "model_loading",
            "underlying_exception_type": "RuntimeError",
            "cause_type": "OSError",
            "safe_reason": "download failed <url>",
        }
        with (
            patch("main.Settings.from_environment", return_value=settings),
            patch("main.configure_logging", return_value=logger),
            patch("main.FasterWhisperProvider", provider),
            patch("sys.stdout", output),
            patch("sys.stderr", errors),
        ):
            result = main(["voice-setup"])
        self.assertEqual(result, 1)
        for expected in (
            "RuntimeError",
            "OSError",
            "download failed <url>",
            "model_loading",
            "base",
            "cpu",
            "int8",
        ):
            self.assertIn(expected, errors.getvalue())


class VoiceCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings.from_environment({})
        self.logger = Mock(spec=logging.Logger)
        self.output = io.StringIO()
        self.errors = io.StringIO()

    def _run_cli_with_transcript(self, confirmation: str) -> tuple[int, Mock]:
        service = Mock()
        service.return_value.transcribe_request.return_value = (
            "Create a Python prime checker."
        )
        with (
            patch("main.Settings.from_environment", return_value=self.settings),
            patch("main.configure_logging", return_value=self.logger),
            patch("main.VoiceInputService", service),
            patch("main.MicrophoneRecorder"),
            patch("main.FasterWhisperProvider"),
            patch("main.run_generate_command", return_value=0) as run_workflow,
        ):
            result = main(["voice"])
            if confirmation in {"y", "yes"}:
                run_workflow.assert_called_once()
                self.assertEqual(
                    run_workflow.call_args.args[:2],
                    ("Create a Python prime checker.", self.settings),
                )
                self.assertTrue(run_workflow.call_args.kwargs["execute_tests"])
            else:
                run_workflow.assert_not_called()
        return result, run_workflow

    def test_confirmed_transcript_uses_normal_complete_workflow(self) -> None:
        with (
            patch("sys.stdout", self.output),
            patch("sys.stderr", self.errors),
            patch("builtins.input", side_effect=("", "yes")),
        ):
            result, _workflow = self._run_cli_with_transcript("yes")
        self.assertEqual(result, 0)
        self.assertIn("Create a Python prime checker.", self.output.getvalue())
        self.assertEqual(self.errors.getvalue(), "")

    def test_rejected_transcript_never_invokes_coding_workflow(self) -> None:
        with (
            patch("sys.stdout", self.output),
            patch("sys.stderr", self.errors),
            patch("builtins.input", side_effect=("", "no")),
        ):
            result, workflow = self._run_cli_with_transcript("no")
        self.assertEqual(result, 0)
        workflow.assert_not_called()
        self.assertIn("nothing was run", self.output.getvalue())

    def test_cancelled_before_recording_does_not_create_voice_service(self) -> None:
        with (
            patch("main.Settings.from_environment", return_value=self.settings),
            patch("main.configure_logging", return_value=self.logger),
            patch("main.VoiceInputService") as service,
            patch("builtins.input", side_effect=KeyboardInterrupt),
            patch("sys.stdout", self.output),
        ):
            self.assertEqual(main(["voice"]), 0)
        service.assert_not_called()

    def test_provider_failure_does_not_invoke_coding_workflow(self) -> None:
        service = Mock()
        service.return_value.transcribe_request.side_effect = VoiceTranscriptionError()
        with (
            patch("main.Settings.from_environment", return_value=self.settings),
            patch("main.configure_logging", return_value=self.logger),
            patch("main.VoiceInputService", service),
            patch("main.MicrophoneRecorder"),
            patch("main.FasterWhisperProvider"),
            patch("main.run_generate_command") as workflow,
            patch("builtins.input", return_value=""),
            patch("sys.stdout", self.output),
            patch("sys.stderr", self.errors),
        ):
            self.assertEqual(main(["voice"]), 1)
        workflow.assert_not_called()
        self.assertNotIn("API_KEY", self.output.getvalue() + self.errors.getvalue())


if __name__ == "__main__":
    unittest.main()
