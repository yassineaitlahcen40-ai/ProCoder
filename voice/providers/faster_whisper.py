"""Credential-isolated local Faster-Whisper subprocess adapter."""

import json
import os
import subprocess
import sys
from pathlib import Path

from voice.base import AudioRecording
from voice.errors import VoiceTranscriptionError


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WORKER_MODULE = "voice.providers.transcribe_worker"
_WORKER_ENV_ALLOWLIST = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "HOME",
)
_MAX_AUDIO_BYTES = 10 * 1024 * 1024


def _safe_model_name(model: str) -> str:
    if model.startswith(("/", "\\")) or (
        len(model) >= 3 and model[1:3] == ":\\"
    ):
        return "<local-model-path>"
    return model[:120]


class FasterWhisperProvider:
    def __init__(
        self,
        model: str = "base",
        timeout_seconds: int = 300,
        python_executable: str = sys.executable,
    ) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._python_executable = python_executable

    def transcribe(self, recording: AudioRecording) -> str:
        try:
            audio_path = recording.path.resolve(strict=True)
            if (
                not audio_path.is_file()
                or audio_path.suffix.lower() != ".wav"
                or audio_path.stat().st_size > _MAX_AUDIO_BYTES
            ):
                raise VoiceTranscriptionError()
            completed = subprocess.run(
                [
                    self._python_executable,
                    "-m",
                    _WORKER_MODULE,
                    self._model,
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self._timeout_seconds,
                check=False,
                cwd=_PROJECT_ROOT,
                env={
                    key: value
                    for key in _WORKER_ENV_ALLOWLIST
                    if (value := os.environ.get(key)) is not None
                },
            )
            payload = json.loads(completed.stdout)
            if not isinstance(payload, dict):
                raise VoiceTranscriptionError(
                    self._base_diagnostics("worker_protocol", "Invalid worker response.")
                )
            if completed.returncode != 0 or payload.get("ok") is not True:
                raise VoiceTranscriptionError(
                    self._diagnostics_from_payload(payload)
                )
            transcript = payload.get("text")
            if not isinstance(transcript, str):
                raise VoiceTranscriptionError(
                    self._base_diagnostics("worker_protocol", "Missing transcript.")
                )
            return transcript
        except VoiceTranscriptionError:
            raise
        except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
            diagnostics = self._base_diagnostics(
                "worker_start",
                "The local transcription worker could not complete.",
            )
            diagnostics["underlying_exception_type"] = type(exc).__name__
            diagnostics["cause_type"] = (
                type(exc.__cause__).__name__ if exc.__cause__ is not None else None
            )
            raise VoiceTranscriptionError(diagnostics) from exc

    def check_model(self) -> dict[str, object]:
        """Load only already-cached model files; this never downloads a model."""
        return self._run_model_command("--check-model")

    def setup_model(self) -> dict[str, object]:
        """Download if needed and load via Faster-Whisper's normal mechanism."""
        return self._run_model_command("--setup-model")

    def check_worker_import(self) -> dict[str, object]:
        """Start the worker module without touching model files or audio."""
        return self._run_model_command("--import-check")

    def _run_model_command(self, operation: str) -> dict[str, object]:
        try:
            completed = subprocess.run(
                [
                    self._python_executable,
                    "-m",
                    _WORKER_MODULE,
                    operation,
                    self._model,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self._timeout_seconds,
                check=False,
                cwd=_PROJECT_ROOT,
                env={
                    key: value
                    for key in _WORKER_ENV_ALLOWLIST
                    if (value := os.environ.get(key)) is not None
                },
            )
            payload = json.loads(completed.stdout)
            if isinstance(payload, dict):
                return payload
        except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
            diagnostics = self._base_diagnostics(
                "worker_start", "The local model check could not complete."
            )
            diagnostics["underlying_exception_type"] = type(exc).__name__
            diagnostics["cause_type"] = (
                type(exc.__cause__).__name__ if exc.__cause__ is not None else None
            )
            return {"ok": False, **diagnostics}
        return {
            "ok": False,
            **self._base_diagnostics("worker_protocol", "Invalid worker response."),
        }

    def _base_diagnostics(self, phase: str, reason: str) -> dict[str, object]:
        return {
            "failure_phase": phase,
            "model": _safe_model_name(self._model),
            "device": "cpu",
            "compute_type": "int8",
            "model_cached": None,
            "model_download_attempted": False,
            "safe_reason": reason[:240],
            "underlying_exception_type": None,
            "cause_type": None,
            "audio_valid": None,
        }

    def _diagnostics_from_payload(
        self, payload: dict[str, object]
    ) -> dict[str, object]:
        allowed_fields = {
            "failure_phase",
            "model",
            "device",
            "compute_type",
            "model_cached",
            "model_download_attempted",
            "safe_reason",
            "underlying_exception_type",
            "cause_type",
            "audio_valid",
            "audio_size_bytes",
        }
        diagnostics = self._base_diagnostics(
            str(payload.get("failure_phase", "worker")),
            str(payload.get("safe_reason", "Local transcription failed.")),
        )
        for key in allowed_fields:
            if key in payload:
                diagnostics[key] = payload[key]
        diagnostics["model"] = _safe_model_name(self._model)
        diagnostics["device"] = "cpu"
        diagnostics["compute_type"] = "int8"
        if isinstance(diagnostics["safe_reason"], str):
            diagnostics["safe_reason"] = diagnostics["safe_reason"][:240]
        return diagnostics
