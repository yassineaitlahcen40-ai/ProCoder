"""Official Codex CLI adapter for structured, non-executed code generation."""

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from agents.coding_agent.base import CodingAgent
from agents.coding_agent.errors import (
    CodexCliNotFoundError,
    CodexExecutionError,
    CodexResponseError,
)
from agents.coding_agent.instructions import (
    CODING_RESULT_SCHEMA,
    REPAIR_RESULT_SCHEMA,
    build_coding_prompt,
    build_repair_prompt,
)
from core.config import Settings
from core.models import (
    CodeRepair,
    GeneratedCode,
    RepairFile,
    RepairInstructions,
    Specification,
)


_ENVIRONMENT_ALLOWLIST = {
    "APPDATA",
    "CODEX_HOME",
    "COMSPEC",
    "HOMEDRIVE",
    "HOMEPATH",
    "HOME",
    "LOCALAPPDATA",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERDOMAIN",
    "USERNAME",
    "USERPROFILE",
    "WINDIR",
    "XDG_CONFIG_HOME",
}


class OpenAICodexCodingAgent(CodingAgent):
    def __init__(self, settings: Settings, workspace: Path) -> None:
        self._settings = settings
        self._workspace = workspace.resolve()
        if not self._workspace.is_dir():
            raise CodexExecutionError()

    def generate(self, specification: Specification) -> GeneratedCode:
        prompt = build_coding_prompt(specification)
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="procoder-codex-") as temp_dir:
                output_path = Path(temp_dir) / "coding-result.json"
                events_path = Path(temp_dir) / "codex-events.jsonl"
                schema_path = Path(temp_dir) / "coding-result-schema.json"
                schema_path.write_text(
                    json.dumps(CODING_RESULT_SCHEMA, separators=(",", ":")),
                    encoding="utf-8",
                )
                executable = resolve_codex_executable(self._settings.codex_cli)
                if executable is None:
                    raise CodexCliNotFoundError()
                command = [
                    executable,
                    "exec",
                    "--json",
                    "--disable",
                    "shell_tool",
                    "--sandbox",
                    "read-only",
                    "--cd",
                    str(self._workspace),
                    "--output-last-message",
                    str(output_path),
                    "--output-schema",
                    str(schema_path),
                    "--ignore-user-config",
                    "--skip-git-repo-check",
                    "--ephemeral",
                ]
                if self._settings.codex_model:
                    command.extend(["--model", self._settings.codex_model])
                command.append("-")
                with events_path.open("wb") as events_file:
                    completed = subprocess.run(
                        command,
                        input=prompt,
                        text=True,
                        cwd=self._workspace,
                        env=self._codex_environment(),
                        timeout=self._settings.codex_timeout,
                        stdout=events_file,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        shell=False,
                    )
                if completed.returncode != 0:
                    raise CodexExecutionError()
                try:
                    if output_path.stat().st_size > 5_000_000:
                        raise CodexResponseError()
                    raw_response = output_path.read_text(encoding="utf-8")
                    tokens_used = _read_reported_token_usage(events_path)
                except (OSError, UnicodeError) as exc:
                    raise CodexResponseError() from exc
        except FileNotFoundError as exc:
            raise CodexCliNotFoundError() from exc
        except subprocess.TimeoutExpired as exc:
            raise CodexExecutionError() from exc
        except OSError as exc:
            raise CodexExecutionError() from exc

        generated = _parse_coding_result(
            raw_response,
            model=self._settings.codex_model,
            tokens_used=tokens_used,
            duration_seconds=time.perf_counter() - started,
        )
        return generated

    def repair(
        self,
        specification: Specification,
        current_code: GeneratedCode,
        instructions: RepairInstructions,
    ) -> CodeRepair:
        try:
            prompt = build_repair_prompt(
                specification, dict(current_code.files), instructions
            )
        except (TypeError, ValueError) as exc:
            raise CodexResponseError() from exc
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="procoder-codex-repair-") as temp_dir:
                output_path = Path(temp_dir) / "repair-result.json"
                events_path = Path(temp_dir) / "codex-events.jsonl"
                schema_path = Path(temp_dir) / "repair-result-schema.json"
                schema_path.write_text(
                    json.dumps(REPAIR_RESULT_SCHEMA, separators=(",", ":")),
                    encoding="utf-8",
                )
                executable = resolve_codex_executable(self._settings.codex_cli)
                if executable is None:
                    raise CodexCliNotFoundError()
                command = [
                    executable,
                    "exec",
                    "--json",
                    "--disable",
                    "shell_tool",
                    "--sandbox",
                    "read-only",
                    "--cd",
                    str(self._workspace),
                    "--output-last-message",
                    str(output_path),
                    "--output-schema",
                    str(schema_path),
                    "--ignore-user-config",
                    "--skip-git-repo-check",
                    "--ephemeral",
                ]
                if self._settings.codex_model:
                    command.extend(["--model", self._settings.codex_model])
                command.append("-")
                with events_path.open("wb") as events_file:
                    completed = subprocess.run(
                        command,
                        input=prompt,
                        text=True,
                        cwd=self._workspace,
                        env=self._codex_environment(),
                        timeout=self._settings.codex_timeout,
                        stdout=events_file,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        shell=False,
                    )
                if completed.returncode != 0:
                    raise CodexExecutionError()
                try:
                    if output_path.stat().st_size > 5_000_000:
                        raise CodexResponseError()
                    raw_response = output_path.read_text(encoding="utf-8")
                    tokens_used = _read_reported_token_usage(events_path)
                except (OSError, UnicodeError) as exc:
                    raise CodexResponseError() from exc
        except (CodexCliNotFoundError, CodexExecutionError, CodexResponseError):
            raise
        except FileNotFoundError as exc:
            raise CodexCliNotFoundError() from exc
        except subprocess.TimeoutExpired as exc:
            raise CodexExecutionError() from exc
        except OSError as exc:
            raise CodexExecutionError() from exc

        return _parse_repair_result(
            raw_response,
            model=self._settings.codex_model,
            tokens_used=tokens_used,
            duration_seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _codex_environment() -> dict[str, str]:
        return {
            key: value
            for key, value in os.environ.items()
            if key.upper() in _ENVIRONMENT_ALLOWLIST
        }


def resolve_codex_executable(configured: str = "codex") -> str | None:
    executable = shutil.which(configured)
    if executable is not None:
        return executable
    configured_path = Path(configured).expanduser()
    if configured_path.is_file():
        return str(configured_path.resolve())
    if configured.casefold() != "codex":
        return None

    machine = platform.machine().casefold()
    if machine in {"amd64", "x86_64"}:
        architecture = "windows-x86_64"
    elif machine in {"arm64", "aarch64"}:
        architecture = "windows-aarch64"
    else:
        return None

    extension_roots = (
        Path.home() / ".vscode" / "extensions",
        Path.home() / ".vscode-insiders" / "extensions",
    )
    candidates = [
        candidate
        for extension_root in extension_roots
        for candidate in extension_root.glob(
            f"openai.chatgpt-*/bin/{architecture}/codex.exe"
        )
        if candidate.is_file()
    ]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime).resolve())


def _parse_coding_result(
    raw_response: str,
    *,
    model: str | None,
    tokens_used: int | None,
    duration_seconds: float,
) -> GeneratedCode:
    try:
        response: Any = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as exc:
        raise CodexResponseError() from exc
    if (
        not isinstance(response, dict)
        or set(response) != {"language", "summary", "files"}
        or not isinstance(response["language"], str)
        or not response["language"].strip()
        or not isinstance(response["summary"], str)
        or not response["summary"].strip()
        or not isinstance(response["files"], list)
        or not response["files"]
        or len(response["files"]) > 100
    ):
        raise CodexResponseError()

    files: dict[str, str] = {}
    total_bytes = 0
    for entry in response["files"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "content"}
            or not isinstance(entry["path"], str)
            or not entry["path"].strip()
            or not isinstance(entry["content"], str)
            or entry["path"] in files
        ):
            raise CodexResponseError()
        content_bytes = entry["content"].encode("utf-8")
        total_bytes += len(content_bytes)
        if len(content_bytes) > 1_000_000 or total_bytes > 5_000_000:
            raise CodexResponseError()
        files[entry["path"]] = entry["content"]

    return GeneratedCode(
        files=files,
        provider="openai-codex-cli",
        model=model,
        tokens_used=tokens_used,
        language=response["language"].strip(),
        summary=response["summary"].strip(),
        duration_seconds=duration_seconds,
    )


def _parse_repair_result(
    raw_response: str,
    *,
    model: str | None,
    tokens_used: int | None,
    duration_seconds: float,
) -> CodeRepair:
    try:
        response: Any = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as exc:
        raise CodexResponseError() from exc
    expected_keys = {
        "files_created",
        "files_modified",
        "files_deleted",
        "summary",
    }
    if (
        not isinstance(response, dict)
        or set(response) != expected_keys
        or not isinstance(response["summary"], str)
        or not response["summary"].strip()
        or len(response["summary"]) > 2000
    ):
        raise CodexResponseError()
    created = _parse_repair_files(response["files_created"])
    modified = _parse_repair_files(response["files_modified"])
    deleted = response["files_deleted"]
    if (
        not isinstance(deleted, list)
        or len(deleted) > 100
        or any(not isinstance(path, str) or not path.strip() for path in deleted)
        or len({path.casefold() for path in deleted}) != len(deleted)
        or len({item.path.casefold() for item in (*created, *modified)})
        != len(created) + len(modified)
        or {item.path.casefold() for item in (*created, *modified)}
        & {path.casefold() for path in deleted}
    ):
        raise CodexResponseError()
    if not created and not modified and not deleted:
        raise CodexResponseError()
    total_bytes = sum(len(item.content.encode("utf-8")) for item in (*created, *modified))
    if (
        len(created) + len(modified) + len(deleted) > 100
        or any(len(item.content.encode("utf-8")) > 1_000_000 for item in (*created, *modified))
        or total_bytes > 5_000_000
    ):
        raise CodexResponseError()
    return CodeRepair(
        files_created=created,
        files_modified=modified,
        files_deleted=tuple(deleted),
        summary=response["summary"].strip(),
        provider="openai-codex-cli",
        model=model,
        tokens_used=tokens_used,
        duration_seconds=duration_seconds,
    )


def _parse_repair_files(value: Any) -> tuple[RepairFile, ...]:
    if not isinstance(value, list) or len(value) > 100:
        raise CodexResponseError()
    files: list[RepairFile] = []
    seen: set[str] = set()
    for entry in value:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "content"}
            or not isinstance(entry["path"], str)
            or not entry["path"].strip()
            or not isinstance(entry["content"], str)
            or entry["path"].casefold() in seen
        ):
            raise CodexResponseError()
        seen.add(entry["path"].casefold())
        files.append(RepairFile(entry["path"], entry["content"]))
    return tuple(files)


def _read_reported_token_usage(events_path: Path) -> int | None:
    if events_path.stat().st_size > 10_000_000:
        raise CodexExecutionError()
    total = 0
    found_usage = False
    with events_path.open("r", encoding="utf-8") as events_file:
        for line in events_file:
            try:
                event: Any = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "turn.completed":
                continue
            usage = event.get("usage")
            if not isinstance(usage, dict):
                continue
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if (
                isinstance(input_tokens, bool)
                or not isinstance(input_tokens, int)
                or input_tokens < 0
                or isinstance(output_tokens, bool)
                or not isinstance(output_tokens, int)
                or output_tokens < 0
            ):
                continue
            total += input_tokens + output_tokens
            found_usage = True
    return total if found_usage else None
