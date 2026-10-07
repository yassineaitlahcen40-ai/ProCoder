"""Safe persistence for per-run specifications and Docker test evidence."""

import json
import re
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.models import (
    FinalRunResult,
    GeneratedCode,
    RepairAttemptRecord,
    Specification,
    TestResult,
)


_RUN_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_MAX_METADATA_BYTES = 2_500_000
_SECRET_VALUE_PATTERNS = (
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?key|token|secret|password|authorization)"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{8,}"
    ),
    re.compile(r"(?i)\bAIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ),
)


class RunMetadataError(RuntimeError):
    """Safe persistence/lookup failure shown without underlying file details."""

    safe_message = (
        "Run metadata could not be safely saved or loaded. "
        "Check the workspace/test_results directory and run ID."
    )


class RunMetadataStore:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.resolve()
        self._workspace = self._project_root / "workspace"
        self._metadata_root = self._workspace / "test_results"
        if _is_link(self._workspace) or _is_link(self._metadata_root):
            raise RunMetadataError()
        if not self._metadata_root.resolve().is_relative_to(self._project_root):
            raise RunMetadataError()

    def save_specification(self, run_id: str, specification: Specification) -> None:
        record = self._read_record(run_id, missing_ok=True)
        record["specification"] = _specification_to_dict(specification)
        self._write_record(run_id, record)

    def save_generated(self, run_id: str, generated: GeneratedCode) -> None:
        record = self._read_record(run_id, missing_ok=True)
        record["generated"] = _generated_to_dict(generated)
        self._write_record(run_id, record)

    def save_test_result(self, run_id: str, result: TestResult) -> None:
        record = self._read_record(run_id, missing_ok=True)
        record["test_result"] = _test_result_to_dict(result)
        self._write_record(run_id, record)

    def save_repair_history(
        self, run_id: str, history: tuple[RepairAttemptRecord, ...]
    ) -> None:
        record = self._read_record(run_id)
        record["repair_history"] = _safe_json_value(history)
        self._write_record(run_id, record)

    def save_final_result(self, result: FinalRunResult) -> None:
        record = self._read_record(result.run_id)
        record["final_result"] = _safe_json_value(result)
        self._write_record(result.run_id, record)

    def load_specification(self, run_id: str) -> Specification:
        record = self._read_record(run_id)
        try:
            data = record["specification"]
            if not isinstance(data, dict):
                raise ValueError
            return Specification(
                request=_required_string(data, "request"),
                summary=_required_string(data, "summary"),
                requirements=_string_tuple(data, "requirements"),
                acceptance_criteria=_string_tuple(data, "acceptance_criteria"),
                provider=_optional_string(data.get("provider")),
                model=_optional_string(data.get("model")),
                tokens_used=_optional_int(data.get("tokens_used")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunMetadataError() from exc

    def load_generated_metadata(self, run_id: str) -> GeneratedCode:
        record = self._read_record(run_id)
        try:
            data = record["generated"]
            if not isinstance(data, dict):
                raise ValueError
            return GeneratedCode(
                files={},
                provider=_optional_string(data.get("provider")),
                model=_optional_string(data.get("model")),
                tokens_used=_optional_int(data.get("tokens_used")),
                language=_optional_string(data.get("language")),
                summary=_optional_string(data.get("summary")),
                files_created=_string_tuple(data, "files_created"),
                files_modified=_string_tuple(data, "files_modified"),
                duration_seconds=_optional_float(data.get("duration_seconds")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunMetadataError() from exc

    def load_test_result(self, run_id: str) -> TestResult:
        record = self._read_record(run_id)
        try:
            data = record["test_result"]
            if not isinstance(data, dict):
                raise ValueError
            return TestResult(
                passed=_required_bool(data, "passed"),
                exit_code=_optional_int(data.get("exit_code")),
                stdout=_required_string(data, "stdout"),
                stderr=_required_string(data, "stderr"),
                duration_seconds=_required_float(data, "duration_seconds"),
                timed_out=_required_bool(data, "timed_out"),
                output_truncated=_required_bool(data, "output_truncated"),
                command=_string_tuple(data, "command"),
                tests_passed=_optional_int(data.get("tests_passed")),
                tests_failed=_optional_int(data.get("tests_failed")),
                tests_skipped=_optional_int(data.get("tests_skipped")),
                runtime=_optional_string(data.get("runtime")),
                container_image=_optional_string(data.get("container_image")),
                infrastructure_error=(
                    _required_bool(data, "infrastructure_error")
                    if "infrastructure_error" in data
                    else False
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunMetadataError() from exc

    def _read_record(self, run_id: str, *, missing_ok: bool = False) -> dict[str, Any]:
        self._validate_run_id(run_id)
        self._validate_root(create=False)
        metadata_path = self._metadata_path(run_id)
        if not metadata_path.exists():
            if missing_ok:
                return {"run_id": run_id}
            raise RunMetadataError()
        try:
            if _is_link(metadata_path) or not metadata_path.is_file():
                raise RunMetadataError()
            metadata_stat = metadata_path.stat()
            if (
                metadata_stat.st_nlink > 1
                or metadata_stat.st_size > _MAX_METADATA_BYTES
            ):
                raise RunMetadataError()
            parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RunMetadataError() from exc
        if (
            not isinstance(parsed, dict)
            or parsed.get("run_id") != run_id
            or parsed.get("version") != 1
        ):
            raise RunMetadataError()
        return parsed

    def _write_record(self, run_id: str, record: dict[str, Any]) -> None:
        self._validate_run_id(run_id)
        self._validate_root(create=True)
        metadata_path = self._metadata_path(run_id)
        if _is_link(metadata_path):
            raise RunMetadataError()
        record.update({"run_id": run_id, "version": 1})
        try:
            encoded = json.dumps(record, ensure_ascii=True, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
                raise RunMetadataError()
            temporary_path = self._metadata_root / f".{run_id}.{uuid.uuid4().hex}.tmp"
            with temporary_path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(encoded)
            if _is_link(metadata_path):
                raise RunMetadataError()
            temporary_path.replace(metadata_path)
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            raise RunMetadataError() from exc
        finally:
            if "temporary_path" in locals():
                temporary_path.unlink(missing_ok=True)

    def _validate_root(self, *, create: bool) -> None:
        if _is_link(self._workspace) or _is_link(self._metadata_root):
            raise RunMetadataError()
        try:
            if create:
                self._metadata_root.mkdir(parents=True, exist_ok=True)
            if (
                not self._metadata_root.resolve().is_relative_to(self._project_root)
                or _is_link(self._workspace)
                or _is_link(self._metadata_root)
            ):
                raise RunMetadataError()
        except OSError as exc:
            raise RunMetadataError() from exc

    def _metadata_path(self, run_id: str) -> Path:
        return self._metadata_root / f"{run_id}.json"

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise RunMetadataError()


def _specification_to_dict(specification: Specification) -> dict[str, Any]:
    return {
        "request": _redact(specification.request),
        "summary": _redact(specification.summary),
        "requirements": [_redact(item) for item in specification.requirements],
        "acceptance_criteria": [
            _redact(item) for item in specification.acceptance_criteria
        ],
        "provider": specification.provider,
        "model": specification.model,
        "tokens_used": specification.tokens_used,
    }


def _generated_to_dict(generated: GeneratedCode) -> dict[str, Any]:
    return {
        "provider": generated.provider,
        "model": generated.model,
        "tokens_used": generated.tokens_used,
        "language": generated.language,
        "summary": _redact(generated.summary) if generated.summary else None,
        "files_created": list(generated.files_created),
        "files_modified": list(generated.files_modified),
        "duration_seconds": generated.duration_seconds,
    }


def _test_result_to_dict(result: TestResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "exit_code": result.exit_code,
        "stdout": _redact(result.stdout),
        "stderr": _redact(result.stderr),
        "duration_seconds": result.duration_seconds,
        "timed_out": result.timed_out,
        "output_truncated": result.output_truncated,
        "command": list(result.command),
        "tests_passed": result.tests_passed,
        "tests_failed": result.tests_failed,
        "tests_skipped": result.tests_skipped,
        "runtime": result.runtime,
        "container_image": result.container_image,
        "infrastructure_error": result.infrastructure_error,
    }


def _redact(value: str) -> str:
    for pattern in _SECRET_VALUE_PATTERNS:
        value = pattern.sub("[REDACTED]", value)
    return value


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, TestResult):
        return _test_result_to_dict(value)
    if isinstance(value, BaseModel):
        return _safe_json_value(value.model_dump(mode="json"))
    if is_dataclass(value):
        return _safe_json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, dict):
        return {key: _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json_value(item) for item in value]
    return value


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise ValueError
    return value


def _string_tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError
    return tuple(value)


def _optional_string(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError


def _required_bool(data: dict[str, Any], key: str) -> bool:
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError
    return value


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    raise ValueError


def _required_float(data: dict[str, Any], key: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    return float(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    return float(value)


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()
