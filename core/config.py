"""Validated environment-backed application settings."""

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DOCKER_MEMORY_PATTERN = re.compile(r"^[1-9]\d*(?:b|k|kb|m|mb|g|gb)?$", re.IGNORECASE)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-3.1-flash-lite"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    nvidia_api_key: str | None
    nvidia_model: str = DEFAULT_NVIDIA_MODEL
    nvidia_base_url: str = DEFAULT_NVIDIA_BASE_URL
    nvidia_timeout: int = 60
    nvidia_max_retries: int = 2
    gemini_api_key: str | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_fallback_model: str | None = DEFAULT_GEMINI_FALLBACK_MODEL
    gemini_timeout: int = 30
    gemini_max_retries: int = 2
    codex_cli: str = "codex"
    codex_model: str | None = None
    codex_timeout: int = 600
    max_repair_attempts: int = 3
    sandbox_timeout_seconds: int = 30
    sandbox_memory_limit: str = "256m"
    sandbox_cpu_limit: float = 0.5
    sandbox_pids_limit: int = 64
    sandbox_max_output_bytes: int = 1_048_576
    sandbox_docker_image: str = "procoder-sandbox:local"
    voice_max_duration_seconds: int = 30
    voice_model: str = "base"
    voice_transcription_timeout: int = 300

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        if environ is None:
            load_dotenv(PROJECT_ROOT / ".env", override=False)
            values: Mapping[str, str] = os.environ
        else:
            values = environ

        settings = cls(
            openai_api_key=_optional_secret(values.get("OPENAI_API_KEY")),
            nvidia_api_key=_optional_secret(values.get("NVIDIA_API_KEY")),
            nvidia_model=values.get(
                "NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL
            ).strip(),
            nvidia_base_url=values.get(
                "NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL
            ).strip().rstrip("/"),
            nvidia_timeout=_integer(values, "NVIDIA_TIMEOUT", 60, minimum=1),
            nvidia_max_retries=_integer(
                values, "NVIDIA_MAX_RETRIES", 2, minimum=0, maximum=5
            ),
            gemini_api_key=_optional_secret(values.get("GEMINI_API_KEY")),
            gemini_model=values.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip(),
            gemini_fallback_model=_optional_model(
                values.get("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL)
            ),
            gemini_timeout=_integer(values, "GEMINI_TIMEOUT", 30, minimum=1),
            gemini_max_retries=_integer(
                values, "GEMINI_MAX_RETRIES", 2, minimum=0, maximum=5
            ),
            codex_cli=values.get("CODEX_CLI", "codex").strip(),
            codex_model=values.get("CODEX_MODEL", "").strip() or None,
            codex_timeout=_integer(values, "CODEX_TIMEOUT", 600, minimum=1),
            max_repair_attempts=_integer(values, "MAX_REPAIR_ATTEMPTS", 3, minimum=0),
            sandbox_timeout_seconds=_integer(
                values, "SANDBOX_TIMEOUT_SECONDS", 30, minimum=1
            ),
            sandbox_memory_limit=values.get("SANDBOX_MEMORY_LIMIT", "256m").strip(),
            sandbox_cpu_limit=_positive_float(values, "SANDBOX_CPU_LIMIT", 0.5),
            sandbox_pids_limit=_integer(
                values, "SANDBOX_PIDS_LIMIT", 64, minimum=1, maximum=4096
            ),
            sandbox_max_output_bytes=_integer(
                values, "SANDBOX_MAX_OUTPUT_BYTES", 1_048_576, minimum=1
            ),
            sandbox_docker_image=values.get(
                "SANDBOX_DOCKER_IMAGE", "procoder-sandbox:local"
            ).strip(),
            voice_max_duration_seconds=_integer(
                values, "VOICE_MAX_DURATION_SECONDS", 30, minimum=1, maximum=300
            ),
            voice_model=values.get("VOICE_MODEL", "base").strip(),
            voice_transcription_timeout=_integer(
                values, "VOICE_TRANSCRIPTION_TIMEOUT", 300, minimum=1, maximum=1800
            ),
        )
        if not _DOCKER_MEMORY_PATTERN.fullmatch(settings.sandbox_memory_limit):
            raise ValueError(
                "SANDBOX_MEMORY_LIMIT must be a positive Docker memory value, "
                "such as 256m or 1g."
            )
        if not settings.sandbox_docker_image:
            raise ValueError("SANDBOX_DOCKER_IMAGE must not be empty.")
        if not settings.gemini_model:
            raise ValueError("GEMINI_MODEL must not be empty.")
        if not settings.nvidia_model:
            raise ValueError("NVIDIA_MODEL must not be empty.")
        if not settings.voice_model:
            raise ValueError("VOICE_MODEL must not be empty.")
        if not settings.nvidia_base_url.startswith("https://"):
            raise ValueError("NVIDIA_BASE_URL must use HTTPS.")
        if not settings.codex_cli:
            raise ValueError("CODEX_CLI must not be empty.")
        if settings.gemini_fallback_model == settings.gemini_model:
            raise ValueError("GEMINI_FALLBACK_MODEL must differ from GEMINI_MODEL.")
        return settings


def _integer(
    values: Mapping[str, str],
    name: str,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw_value = values.get(name)
    try:
        value = default if raw_value is None else int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return value


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw_value = values.get(name)
    try:
        value = default if raw_value is None else float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _optional_secret(value: str | None) -> str | None:
    cleaned = value.strip() if value else ""
    return cleaned or None


def _optional_model(value: str | None) -> str | None:
    cleaned = value.strip() if value else ""
    return cleaned or None
