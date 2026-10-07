"""Subprocess entry point for local speech recognition."""

import json
import re
import sys
import wave
from pathlib import Path


_DEVICE = "cpu"
_COMPUTE_TYPE = "int8"
_MODEL_CACHE: bool | None = None


def _safe_reason(message: str) -> str:
    value = str(message)
    value = re.sub(r"https?://\S+", "<url>", value, flags=re.IGNORECASE)
    value = re.sub(r"[A-Za-z]:\\(?:[^\s\"']+\\?)*", "<path>", value)
    value = re.sub(r"(?<!\w)/(?:[^/\s]+/)*[^/\s]*", "<path>", value)
    value = re.sub(
        r"(?i)\b(token|api[_ -]?key|authorization)\s*[:=]\s*\S+",
        r"\1=<redacted>",
        value,
    )
    value = re.sub(r"(?i)\bBearer\s+\S+", "Bearer <redacted>", value)
    value = re.sub(r"\bhf_[A-Za-z0-9_-]{8,}\b", "<redacted-token>", value)
    return value[:240]


def _safe_model_name(model_name: str) -> str:
    if re.match(r"^(?:[A-Za-z]:\\|/|\\\\)", model_name):
        return "<local-model-path>"
    return _safe_reason(model_name)[:120]


def _exception_details(exc: BaseException) -> tuple[str, str | None, str]:
    cause = exc.__cause__ or exc.__context__
    return (
        type(exc).__name__,
        type(cause).__name__ if cause is not None else None,
        _safe_reason(str(exc) or type(exc).__name__),
    )


def _model_is_cached(model_name: str) -> bool:
    from faster_whisper.utils import download_model

    try:
        download_model(model_name, local_files_only=True)
        return True
    except Exception:
        return False


def _emit(payload: dict[str, object], exit_code: int) -> int:
    print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    return exit_code


def _diagnostic_base(model_name: str) -> dict[str, object]:
    return {
        "ok": False,
        "model": _safe_model_name(model_name),
        "device": _DEVICE,
        "compute_type": _COMPUTE_TYPE,
        "model_cached": _MODEL_CACHE,
        "model_download_attempted": False,
        "audio_valid": None,
        "audio_size_bytes": None,
        "underlying_exception_type": None,
        "cause_type": None,
        "safe_reason": "",
    }


def main() -> int:
    global _MODEL_CACHE
    check_only = len(sys.argv) == 3 and sys.argv[1] == "--check-model"
    setup_only = len(sys.argv) == 3 and sys.argv[1] == "--setup-model"
    import_check = len(sys.argv) == 3 and sys.argv[1] == "--import-check"
    if import_check:
        return _emit(
            {
                "ok": True,
                "failure_phase": None,
                "model": _safe_model_name(sys.argv[2]),
                "device": _DEVICE,
                "compute_type": _COMPUTE_TYPE,
                "model_cached": None,
                "model_download_attempted": False,
                "safe_reason": "Voice worker module imported successfully.",
            },
            0,
        )
    if check_only or setup_only:
        model_name, audio_path = sys.argv[2], None
    elif len(sys.argv) == 3:
        model_name, audio_path = sys.argv[1:3]
    else:
        return _emit(
            {
                "ok": False,
                "failure_phase": "worker_arguments",
                "safe_reason": "Invalid local worker arguments.",
            },
            2,
        )

    payload = _diagnostic_base(model_name)
    try:
        _MODEL_CACHE = _model_is_cached(model_name)
    except Exception as exc:
        exception_type, cause_type, reason = _exception_details(exc)
        payload.update(
            {
                "failure_phase": "model_loading",
                "underlying_exception_type": exception_type,
                "cause_type": cause_type,
                "safe_reason": reason,
            }
        )
        return _emit(payload, 1)
    payload["model_cached"] = _MODEL_CACHE

    if check_only and not _MODEL_CACHE:
        payload.update(
            {
                "failure_phase": "model_cache_check",
                "safe_reason": "Model is not available in the local cache; no download was attempted.",
            }
        )
        return _emit(payload, 1)

    if not check_only and not setup_only:
        assert audio_path is not None
        try:
            file_path = Path(audio_path)
            payload["audio_size_bytes"] = file_path.stat().st_size
            with wave.open(str(file_path), "rb") as audio:
                frame_count = audio.getnframes()
                channels = audio.getnchannels()
                sample_width = audio.getsampwidth()
                sample_rate = audio.getframerate()
                payload["audio_valid"] = (
                    frame_count > 0
                    and channels > 0
                    and sample_width > 0
                    and sample_rate > 0
                )
            if payload["audio_valid"] is not True:
                payload.update(
                    {
                        "failure_phase": "audio_validation",
                        "safe_reason": "Temporary WAV file contained no valid audio frames.",
                    }
                )
                return _emit(payload, 1)
        except Exception as exc:
            exception_type, cause_type, reason = _exception_details(exc)
            payload.update(
                {
                    "failure_phase": "audio_validation",
                    "audio_valid": False,
                    "underlying_exception_type": exception_type,
                    "cause_type": cause_type,
                    "safe_reason": reason,
                }
            )
            return _emit(payload, 1)

    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            model_name,
            device=_DEVICE,
            compute_type=_COMPUTE_TYPE,
            local_files_only=check_only,
        )
    except Exception as exc:
        exception_type, cause_type, reason = _exception_details(exc)
        payload.update(
            {
                "failure_phase": "model_loading",
                "model_download_attempted": (
                    not check_only and not _MODEL_CACHE
                ),
                "underlying_exception_type": exception_type,
                "cause_type": cause_type,
                "safe_reason": reason,
            }
        )
        return _emit(payload, 1)

    if check_only or setup_only:
        payload.update(
            {
                "ok": True,
                "failure_phase": None,
                "model_cached": True,
                "model_download_attempted": setup_only and not _MODEL_CACHE,
                "safe_reason": (
                    "Model loaded successfully on CPU with int8."
                    if setup_only
                    else "Cached model loaded successfully on CPU with int8."
                ),
            }
        )
        return _emit(payload, 0)

    assert audio_path is not None
    try:
        segments, _ = model.transcribe(audio_path, beam_size=5)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:
        exception_type, cause_type, reason = _exception_details(exc)
        payload.update(
            {
                "failure_phase": "transcription",
                "underlying_exception_type": exception_type,
                "cause_type": cause_type,
                "safe_reason": reason,
            }
        )
        return _emit(payload, 1)

    payload.update(
        {
            "ok": True,
            "failure_phase": None,
            "model_download_attempted": not _MODEL_CACHE,
            "audio_valid": True,
            "text": transcript,
            "safe_reason": "",
        }
    )
    return _emit(payload, 0)


if __name__ == "__main__":
    raise SystemExit(main())
