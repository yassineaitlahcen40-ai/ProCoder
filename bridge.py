"""Versioned JSON Lines interface for the VS Code extension."""

import json
import logging
import re
import sys
import threading
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import TextIO

from core.config import PROJECT_ROOT, Settings
from core.logging import configure_logging
from main import run_generate_command
from voice.errors import VoiceError, VoiceRecordingError
from voice.providers.faster_whisper import FasterWhisperProvider
from voice.recorder import MicrophoneRecorder
from voice.service import validate_transcript
from workspace.generated_project import GeneratedProjectWriter


PROTOCOL_VERSION = 1
_MAX_LINE_BYTES = 65_536
_MAX_REQUEST_CHARACTERS = 12_000
_RUN_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "workflow_started": ("run_id",),
    "planning_started": ("run_id",),
    "planning_completed": ("run_id", "provider", "model"),
    "generation_started": ("run_id",),
    "code_generation_completed": (
        "run_id",
        "files_created",
        "files_modified",
        "success",
    ),
    "testing_started": ("run_id", "attempt"),
    "testing_completed": (
        "run_id",
        "attempt",
        "passed",
        "exit_code",
        "timed_out",
        "infrastructure_error",
        "tests_passed",
        "tests_failed",
        "tests_skipped",
        "output_truncated",
    ),
    "review_started": ("run_id", "attempt"),
    "review_completed": (
        "run_id",
        "attempt",
        "verdict",
        "specification_satisfied",
        "tests_passed",
        "repair_required",
        "confidence",
    ),
    "repair_attempt_started": ("run_id", "attempt", "max_attempts"),
    "repair_attempt_completed": (
        "run_id",
        "attempt",
        "patch_applied",
        "files_created",
        "files_modified",
        "files_deleted",
        "termination_reason",
    ),
    "repair_workflow_completed": (
        "run_id",
        "success",
        "repair_attempts",
        "termination_reason",
        "final_verdict",
    ),
    "repair_workflow_terminated": ("run_id", "termination_reason", "error_code"),
    "code_generation_failed": ("run_id", "phase", "error_code"),
}


_EVENT_NAMES = {
    "code_generation_completed": "generation_completed",
    "repair_attempt_started": "repair_started",
    "repair_attempt_completed": "repair_completed",
    "repair_workflow_terminated": "workflow_stage_failed",
    "code_generation_failed": "workflow_stage_failed",
}

_FAILURE_MESSAGES = {
    "CodexCliNotFoundError": "Codex CLI is unavailable.",
    "PromptAgentError": "Gemini planning failed.",
    "CodingAgentError": "Code generation failed.",
    "ReviewAgentError": "NVIDIA review failed.",
    "DockerSandboxError": "Docker testing failed.",
    "VoiceRecordingError": "Microphone recording failed.",
    "MicrophoneUnavailableError": "A microphone is unavailable.",
    "VoiceTranscriptionError": "Local voice transcription failed.",
}


def _safe_failure_message(error_code: object, default: str) -> str:
    code = str(error_code)
    exact = _FAILURE_MESSAGES.get(code)
    if exact is not None:
        return exact
    normalized = code.casefold()
    if normalized.startswith(("gemini", "promptagent")):
        return "Gemini planning failed."
    if normalized.startswith(("nvidia", "reviewagent")):
        return "NVIDIA review failed."
    if normalized.startswith(("codex", "codingagent")):
        return "Code generation failed."
    if normalized.startswith(("docker", "sandbox")):
        return "Docker testing failed."
    return default


class _ProtocolLogHandler(logging.Handler):
    def __init__(self, server: "BridgeServer") -> None:
        super().__init__(logging.INFO)
        self._server = server

    def emit(self, record: logging.LogRecord) -> None:
        event_data = getattr(record, "event_data", None)
        if not isinstance(event_data, Mapping):
            return
        event = event_data.get("event")
        if not isinstance(event, str) or event not in _EVENT_FIELDS:
            return
        fields = {
            key: event_data[key]
            for key in _EVENT_FIELDS[event]
            if key in event_data
        }
        self._server._observe_backend_event(event, fields)
        if event == "code_generation_failed":
            self._server.emit(
                "workflow_stage_failed",
                **fields,
                message=_safe_failure_message(
                    fields.get("error_code"),
                    "The workflow could not continue.",
                ),
            )
            return
        if event == "repair_workflow_terminated":
            reason = fields.get("termination_reason")
            message = {
                "SANDBOX_ERROR": "Docker testing failed.",
                "PROVIDER_ERROR": _safe_failure_message(
                    fields.get("error_code"),
                    "An AI provider could not complete its workflow stage.",
                ),
                "INVALID_REPAIR": "A proposed repair failed workspace validation.",
                "PERSISTENCE_ERROR": "ProCoder could not save run results.",
            }.get(str(reason), "The workflow reached a terminal failure.")
            self._server.emit("workflow_stage_failed", **fields, message=message)
            return
        self._server.emit(_EVENT_NAMES.get(event, event), **fields)


class BridgeServer:
    def __init__(
        self,
        settings: Settings,
        logger: logging.Logger,
        *,
        project_root: Path = PROJECT_ROOT,
        input_stream: TextIO = sys.stdin,
        output_stream: TextIO = sys.stdout,
    ) -> None:
        self._settings = settings
        self._logger = logger
        self._project_root = project_root.resolve()
        self._input = input_stream
        self._output = output_stream
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._active_operation: str | None = None
        self._voice_stop = threading.Event()
        self._operation_finished = threading.Event()
        self._operation_finished.set()
        self._voice_cancelled = False
        self._pending_transcript: str | None = None
        self._run_state: dict[str, object] = {}
        self._shutdown = False
        self._log_handler = _ProtocolLogHandler(self)
        self._logger.addHandler(self._log_handler)

    def serve(self) -> int:
        self.emit(
            "ready",
            capabilities={
                "text_requests": True,
                "voice_recording": True,
                "transcript_confirmation": True,
                "voice_cancellation": True,
                "workflow_cancellation": False,
                "generated_file_opening": True,
            },
        )
        try:
            for line in self._input:
                if self._shutdown:
                    break
                if len(line.encode("utf-8")) > _MAX_LINE_BYTES:
                    self.emit("protocol_error", message="The message is too large.")
                    continue
                try:
                    message = json.loads(line)
                except (json.JSONDecodeError, UnicodeError, RecursionError):
                    self.emit("protocol_error", message="Invalid JSON message.")
                    continue
                if not isinstance(message, dict):
                    self.emit("protocol_error", message="Expected a JSON object.")
                    continue
                self._dispatch(message)
            self._operation_finished.wait()
        finally:
            self._logger.removeHandler(self._log_handler)
            self._log_handler.close()
        return 0

    def emit(self, event: str, **fields: object) -> None:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "type": event,
            **fields,
        }
        serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        with self._write_lock:
            self._output.write(serialized + "\n")
            self._output.flush()

    def _observe_backend_event(
        self, event: str, fields: dict[str, object]
    ) -> None:
        if event == "workflow_started" and isinstance(fields.get("run_id"), str):
            self._run_state["run_id"] = fields["run_id"]
        elif event == "testing_completed":
            self._run_state["docker"] = fields
        elif event == "review_completed":
            self._run_state["nvidia"] = fields
        elif event == "code_generation_completed":
            self._run_state["files_created"] = fields.get("files_created", [])
            self._run_state["files_modified"] = fields.get("files_modified", [])
        elif event == "repair_attempt_completed":
            history = self._run_state.setdefault("repair_history", [])
            if isinstance(history, list):
                history.append(fields)
        elif event == "repair_workflow_completed":
            self._run_state["final"] = {
                "success": fields.get("success", False),
                "repair_attempts": fields.get("repair_attempts", 0),
                "termination_reason": fields.get("termination_reason"),
                "final_verdict": fields.get("final_verdict"),
            }
        elif event == "repair_workflow_terminated":
            self._run_state["termination_reason"] = fields.get(
                "termination_reason"
            )

    def _dispatch(self, message: dict[str, object]) -> None:
        version = message.get("protocol_version")
        if type(version) is not int or version != PROTOCOL_VERSION:
            self.emit(
                "protocol_error",
                message=f"Unsupported protocol version; expected {PROTOCOL_VERSION}.",
            )
            return
        command = message.get("type")
        if command == "run":
            request = message.get("request")
            if not isinstance(request, str) or not request.strip():
                self.emit("protocol_error", message="Request text is required.")
                return
            if len(request) > _MAX_REQUEST_CHARACTERS:
                self.emit("protocol_error", message="Request text is too long.")
                return
            self._start_operation("workflow", lambda: self._run_workflow(request))
            return
        if command == "voice_start":
            self._start_operation("voice", self._run_voice)
            return
        if command == "voice_stop":
            with self._state_lock:
                is_recording = self._active_operation == "voice"
            if not is_recording:
                self.emit("protocol_error", message="Voice recording is not active.")
                return
            self._voice_stop.set()
            self.emit("voice_stop_requested")
            return
        if command == "transcript_decision":
            self._decide_transcript(message)
            return
        if command == "cancel":
            self._cancel_active_operation()
            return
        if command == "shutdown":
            with self._state_lock:
                active = self._active_operation
            if active is not None:
                self.emit(
                    "cancel_unavailable",
                    message="The active operation must finish before shutdown.",
                )
                return
            self._shutdown = True
            return
        self.emit("protocol_error", message="Unsupported command.")

    def _start_operation(self, operation: str, target: Callable[[], None]) -> None:
        with self._state_lock:
            if self._active_operation is not None:
                self.emit("protocol_error", message="Another operation is active.")
                return
            self._active_operation = operation
            self._operation_finished.clear()
            self._pending_transcript = None
            if operation == "workflow":
                self._run_state = {}
            if operation == "voice":
                self._voice_stop.clear()
                self._voice_cancelled = False
        if operation == "voice":
            self.emit("voice_starting")
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

    def _finish_operation(self, operation: str) -> None:
        with self._state_lock:
            if self._active_operation == operation:
                self._active_operation = None
                self._operation_finished.set()

    def _run_workflow(self, request: str) -> None:
        self._run_state = {}
        stdout = StringIO()
        stderr = StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = run_generate_command(
                    request,
                    self._settings,
                    self._logger,
                    stdout=stdout,
                    stderr=stderr,
                    project_root=self._project_root,
                    execute_tests=True,
                )
            state = dict(self._run_state)
            run_id = state.get("run_id")
            if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
                self.emit(
                    "workflow_failed",
                    message="The workflow did not produce a valid run result.",
                    error_code="WorkflowResultUnavailable",
                )
                return
            files_available = True
            try:
                files = sorted(GeneratedProjectWriter(self._project_root).read_files(run_id))
            except Exception as exc:
                files_available = False
                self._logger.error(
                    "bridge_generated_files_unavailable",
                    extra={
                        "event_data": {
                            "event": "bridge_generated_files_unavailable",
                            "run_id": run_id,
                            "error_code": type(exc).__name__,
                        }
                    },
                )
                self.emit(
                    "generated_files_unavailable",
                    message="Generated files could not be safely listed.",
                )
                files = []
            final = state.get("final", {})
            if not isinstance(final, dict):
                final = {}
            termination_reason = final.get(
                "termination_reason", state.get("termination_reason")
            )
            succeeded = bool(final.get("success", False)) and exit_code == 0
            self.emit(
                "workflow_completed",
                run_id=run_id,
                success=succeeded,
                termination_reason=termination_reason,
                repair_attempts=final.get("repair_attempts", 0),
                docker=state.get("docker"),
                nvidia=state.get("nvidia"),
                files=files,
                files_available=files_available,
                files_created=state.get("files_created", []),
                files_modified=state.get("files_modified", []),
                repair_history=state.get("repair_history", []),
                exit_code=exit_code,
            )
        except Exception as exc:
            self._logger.error(
                "bridge_workflow_failed",
                extra={
                    "event_data": {
                        "event": "bridge_workflow_failed",
                        "error_code": type(exc).__name__,
                    }
                },
            )
            self.emit(
                "workflow_failed",
                message="The backend could not complete the workflow.",
                error_code="BackendWorkflowError",
            )
        finally:
            self._finish_operation("workflow")

    def _run_voice(self) -> None:
        recording = None
        cleanup_attempted = False
        try:
            recorder = MicrophoneRecorder(
                stop_requested=self._voice_stop.is_set,
                on_started=lambda: self.emit("voice_recording_started"),
            )
            recording = recorder.record(self._settings.voice_max_duration_seconds)
            with self._state_lock:
                cancelled = self._voice_cancelled
            if cancelled:
                self.emit("voice_recording_cancelled")
                return
            self.emit("voice_transcription_started")
            transcript = FasterWhisperProvider(
                self._settings.voice_model,
                self._settings.voice_transcription_timeout,
            ).transcribe(recording)
            transcript = validate_transcript(transcript)
            cleanup_attempted = True
            try:
                recording.cleanup()
            except OSError as exc:
                raise VoiceRecordingError() from exc
            recording = None
            with self._state_lock:
                cancelled = self._voice_cancelled
                if not cancelled:
                    self._pending_transcript = transcript
                    if self._active_operation == "voice":
                        self._active_operation = None
                        self._operation_finished.set()
            if cancelled:
                self.emit("voice_recording_cancelled")
                return
            self.emit("transcript_ready", transcript=transcript)
        except VoiceError as exc:
            with self._state_lock:
                cancelled = self._voice_cancelled
            if cancelled:
                self.emit("voice_recording_cancelled")
                return
            self.emit(
                "voice_failed",
                error_code=type(exc).__name__,
                message=_FAILURE_MESSAGES.get(
                    type(exc).__name__, "Local voice input failed."
                ),
            )
        except (OSError, ValueError, TypeError) as exc:
            with self._state_lock:
                cancelled = self._voice_cancelled
            if cancelled:
                self.emit("voice_recording_cancelled")
                return
            self._logger.error(
                "bridge_voice_failed",
                extra={
                    "event_data": {
                        "event": "bridge_voice_failed",
                        "error_code": type(exc).__name__,
                    }
                },
            )
            self.emit(
                "voice_failed",
                error_code="VoiceInputError",
                message="Local voice input failed.",
            )
        except Exception as exc:
            self._logger.error(
                "bridge_voice_failed",
                extra={
                    "event_data": {
                        "event": "bridge_voice_failed",
                        "error_code": type(exc).__name__,
                    }
                },
            )
            self.emit(
                "voice_failed",
                error_code="VoiceInputError",
                message="Local voice input failed.",
            )
        finally:
            if recording is not None and not cleanup_attempted:
                try:
                    recording.cleanup()
                except OSError:
                    self.emit(
                        "voice_failed",
                        error_code="VoiceRecordingError",
                        message="Temporary voice audio could not be removed.",
                    )
            self._finish_operation("voice")

    def _decide_transcript(self, message: dict[str, object]) -> None:
        with self._state_lock:
            transcript = self._pending_transcript
            active = self._active_operation
        if transcript is None or active is not None:
            self.emit("protocol_error", message="There is no transcript awaiting approval.")
            return
        approved = message.get("approved")
        if not isinstance(approved, bool):
            self.emit("protocol_error", message="Transcript approval must be explicit.")
            return
        if not approved:
            with self._state_lock:
                self._pending_transcript = None
            self.emit("transcript_rejected")
            return
        candidate = message.get("transcript", transcript)
        if not isinstance(candidate, str):
            self.emit("protocol_error", message="Transcript text is invalid.")
            return
        try:
            approved_text = validate_transcript(candidate)
        except VoiceError:
            self.emit("protocol_error", message="Transcript text is invalid.")
            return
        with self._state_lock:
            self._pending_transcript = None
        self.emit("transcript_approved")
        self._start_operation("workflow", lambda: self._run_workflow(approved_text))

    def _cancel_active_operation(self) -> None:
        with self._state_lock:
            active = self._active_operation
            if active == "voice":
                self._voice_cancelled = True
                self._voice_stop.set()
        if active == "voice":
            self.emit("voice_cancel_requested")
        elif active == "workflow":
            self.emit(
                "cancel_unavailable",
                message="Workflow cancellation is unavailable while agents or Docker are running.",
            )
        else:
            self.emit("cancel_unavailable", message="There is no cancellable operation.")


def main() -> int:
    try:
        settings = Settings.from_environment()
    except ValueError:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "type": "workflow_failed",
            "message": "Backend configuration is invalid.",
            "error_code": "ConfigurationError",
        }
        sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
        return 2
    logger = configure_logging(PROJECT_ROOT / "logs" / "procoder.jsonl", stream=False)
    return BridgeServer(settings, logger).serve()


if __name__ == "__main__":
    raise SystemExit(main())
