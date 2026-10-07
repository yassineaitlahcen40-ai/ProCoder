"""Secure writer for isolated, per-run generated project directories."""

import re
import os
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

from agents.coding_agent.errors import CodingAgentError
from core.models import CodeRepair, GeneratedCode


class GeneratedProjectError(CodingAgentError):
    safe_message = "Generated files could not be safely written to the project workspace."


_MAX_FILES = 100
_MAX_FILE_BYTES = 1_000_000
_MAX_TOTAL_BYTES = 5_000_000
_RUN_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_DOCKER_FILES = {
    ".dockerignore",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
}
_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
_SENSITIVE_FILES = {
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    ".aws",
    ".azure",
    ".gcloud",
    ".ssh",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
    "secrets.toml",
}
_SENSITIVE_NAME_MARKERS = (
    "secret",
    "credential",
    "token",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "id_rsa",
    "id_ed25519",
)
_SENSITIVE_EXTENSIONS = {".key", ".pem"}


class GeneratedProjectWriter:
    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.resolve()
        workspace_path = self._project_root / "workspace"
        generated_root = workspace_path / "generated_code"
        if _is_link(workspace_path) or _is_link(generated_root):
            raise GeneratedProjectError()
        self._generated_root = generated_root.resolve()
        if not self._generated_root.is_relative_to(self._project_root):
            raise GeneratedProjectError()

    def create_run_directory(self, run_id: str) -> Path:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise GeneratedProjectError()
        self._validate_generated_root()
        try:
            self._generated_root.mkdir(parents=True, exist_ok=True)
            if _is_link(self._generated_root):
                raise GeneratedProjectError()
            run_directory = self._generated_root / run_id
            run_directory.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise GeneratedProjectError() from exc
        except OSError as exc:
            raise GeneratedProjectError() from exc
        return run_directory.resolve()

    def write(self, run_id: str, generated: GeneratedCode) -> GeneratedCode:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise GeneratedProjectError()
        self._validate_generated_root()
        run_directory = self._generated_root / run_id
        if not run_directory.is_dir() or _is_link(run_directory):
            raise GeneratedProjectError()
        resolved_run_directory = run_directory.resolve()
        if not resolved_run_directory.is_relative_to(self._generated_root):
            raise GeneratedProjectError()
        if not generated.files or len(generated.files) > _MAX_FILES:
            raise GeneratedProjectError()

        checked_files: list[tuple[str, Path, str]] = []
        seen_paths: set[str] = set()
        total_bytes = 0
        for raw_path, content in generated.files.items():
            normalized, destination = self._resolve_file(
                resolved_run_directory, raw_path
            )
            normalized_key = normalized.casefold()
            if normalized_key in seen_paths or not isinstance(content, str):
                raise GeneratedProjectError()
            seen_paths.add(normalized_key)
            try:
                content_bytes = content.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise GeneratedProjectError() from exc
            total_bytes += len(content_bytes)
            if len(content_bytes) > _MAX_FILE_BYTES or total_bytes > _MAX_TOTAL_BYTES:
                raise GeneratedProjectError()
            checked_files.append((normalized, destination, content))

        created: list[str] = []
        modified: list[str] = []
        for normalized, destination, content in checked_files:
            self._ensure_no_symlink(resolved_run_directory, destination)
            existed = destination.exists()
            if existed and not destination.is_file():
                raise GeneratedProjectError()
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._ensure_no_symlink(resolved_run_directory, destination)
                destination.write_text(content, encoding="utf-8", newline="")
            except OSError as exc:
                raise GeneratedProjectError() from exc
            (modified if existed else created).append(normalized)

        return GeneratedCode(
            files=dict(generated.files),
            provider=generated.provider,
            model=generated.model,
            tokens_used=generated.tokens_used,
            language=generated.language,
            summary=generated.summary,
            files_created=tuple(created),
            files_modified=tuple(modified),
            duration_seconds=generated.duration_seconds,
        )

    def read_files(self, run_id: str) -> dict[str, str]:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise GeneratedProjectError()
        self._validate_generated_root()
        run_directory = self._generated_root / run_id
        if not run_directory.is_dir() or _is_link(run_directory):
            raise GeneratedProjectError()
        resolved_run_directory = run_directory.resolve()
        if not resolved_run_directory.is_relative_to(self._generated_root):
            raise GeneratedProjectError()

        files: dict[str, str] = {}
        total_bytes = 0
        try:
            for path in resolved_run_directory.rglob("*"):
                if _is_link(path):
                    raise GeneratedProjectError()
                if path.is_dir():
                    continue
                if not path.is_file() or path.stat().st_nlink > 1:
                    raise GeneratedProjectError()
                normalized, destination = self._resolve_file(
                    resolved_run_directory,
                    path.relative_to(resolved_run_directory).as_posix(),
                )
                self._ensure_no_symlink(resolved_run_directory, destination)
                if len(files) >= _MAX_FILES:
                    raise GeneratedProjectError()
                content = destination.read_bytes()
                total_bytes += len(content)
                if len(content) > _MAX_FILE_BYTES or total_bytes > _MAX_TOTAL_BYTES:
                    raise GeneratedProjectError()
                files[normalized] = content.decode("utf-8")
        except (OSError, UnicodeError, ValueError) as exc:
            raise GeneratedProjectError() from exc
        if not files:
            raise GeneratedProjectError()
        return files

    def apply_repair(self, run_id: str, repair: CodeRepair) -> CodeRepair:
        if not _RUN_ID_PATTERN.fullmatch(run_id):
            raise GeneratedProjectError()
        self._validate_generated_root()
        run_directory = self._generated_root / run_id
        if not run_directory.is_dir() or _is_link(run_directory):
            raise GeneratedProjectError()
        resolved_run_directory = run_directory.resolve()
        if not resolved_run_directory.is_relative_to(self._generated_root):
            raise GeneratedProjectError()
        if not repair.summary.strip():
            raise GeneratedProjectError()

        entries = (
            *((item.path, item.content, "create") for item in repair.files_created),
            *((item.path, item.content, "modify") for item in repair.files_modified),
            *((path, None, "delete") for path in repair.files_deleted),
        )
        if not entries or len(entries) > _MAX_FILES:
            raise GeneratedProjectError()
        checked: list[tuple[str, Path, str | None, str]] = []
        seen_paths: set[str] = set()
        total_bytes = 0
        for raw_path, content, operation in entries:
            normalized, destination = self._resolve_file(
                resolved_run_directory, raw_path
            )
            key = normalized.casefold()
            if key in seen_paths:
                raise GeneratedProjectError()
            seen_paths.add(key)
            self._ensure_no_symlink(resolved_run_directory, destination)
            exists = destination.exists()
            if operation == "create" and exists:
                raise GeneratedProjectError()
            if operation in {"modify", "delete"} and not exists:
                raise GeneratedProjectError()
            if exists:
                try:
                    file_stat = destination.stat(follow_symlinks=False)
                except OSError as exc:
                    raise GeneratedProjectError() from exc
                if not destination.is_file() or file_stat.st_nlink > 1:
                    raise GeneratedProjectError()
            if operation != "delete":
                if not isinstance(content, str):
                    raise GeneratedProjectError()
                try:
                    encoded_size = len(content.encode("utf-8"))
                except UnicodeEncodeError as exc:
                    raise GeneratedProjectError() from exc
                total_bytes += encoded_size
                if (
                    encoded_size > _MAX_FILE_BYTES
                    or total_bytes > _MAX_TOTAL_BYTES
                ):
                    raise GeneratedProjectError()
            checked.append((normalized, destination, content, operation))

        originals: dict[Path, bytes | None] = {}
        staged: list[tuple[Path, Path]] = []
        changed: list[Path] = []
        try:
            for _, destination, content, operation in checked:
                if operation == "delete":
                    originals[destination] = destination.read_bytes()
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._ensure_no_symlink(resolved_run_directory, destination)
                originals[destination] = (
                    destination.read_bytes() if destination.exists() else None
                )
                temporary = destination.with_name(
                    f".{destination.name}.{uuid.uuid4().hex}.procoder-tmp"
                )
                with temporary.open("x", encoding="utf-8", newline="") as stream:
                    stream.write(content or "")
                staged.append((temporary, destination))
            for temporary, destination in staged:
                self._ensure_no_symlink(resolved_run_directory, destination)
                os.replace(temporary, destination)
                changed.append(destination)
            for _, destination, _, operation in checked:
                if operation == "delete":
                    destination.unlink()
                    changed.append(destination)
        except OSError as exc:
            for destination in reversed(changed):
                original = originals[destination]
                if original is None:
                    destination.unlink(missing_ok=True)
                else:
                    destination.write_bytes(original)
            raise GeneratedProjectError() from exc
        finally:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)

        return repair

    def _validate_generated_root(self) -> None:
        workspace_path = self._project_root / "workspace"
        generated_root = workspace_path / "generated_code"
        if (
            _is_link(workspace_path)
            or _is_link(generated_root)
            or generated_root.resolve() != self._generated_root
            or not self._generated_root.is_relative_to(self._project_root)
        ):
            raise GeneratedProjectError()

    def _resolve_file(self, run_directory: Path, raw_path: str) -> tuple[str, Path]:
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
            raise GeneratedProjectError()
        posix_path = PurePosixPath(raw_path)
        windows_path = PureWindowsPath(raw_path)
        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or any(":" in part for part in posix_path.parts)
            or any(part.endswith((".", " ")) for part in posix_path.parts)
            or any(
                part.split(".", maxsplit=1)[0].casefold() in _WINDOWS_RESERVED_NAMES
                for part in posix_path.parts
            )
            or any(part in {"", ".", ".."} for part in posix_path.parts)
            or any(
                any(character in part for character in '<>"|?*')
                for part in posix_path.parts
            )
        ):
            raise GeneratedProjectError()
        parts = posix_path.parts
        if any(part.casefold() == ".git" for part in parts):
            raise GeneratedProjectError()
        if any(part.casefold().startswith(".env") for part in parts):
            raise GeneratedProjectError()
        if any(part.casefold() in _SENSITIVE_FILES for part in parts):
            raise GeneratedProjectError()
        if any(
            any(marker in part.casefold() for marker in _SENSITIVE_NAME_MARKERS)
            or PurePosixPath(part).suffix.casefold() in _SENSITIVE_EXTENSIONS
            for part in parts
        ):
            raise GeneratedProjectError()
        if any(_is_docker_configuration(part) for part in parts):
            raise GeneratedProjectError()
        destination = run_directory.joinpath(*parts)
        if not destination.resolve().is_relative_to(run_directory):
            raise GeneratedProjectError()
        return posix_path.as_posix(), destination

    @staticmethod
    def _ensure_no_symlink(run_directory: Path, destination: Path) -> None:
        try:
            relative = destination.relative_to(run_directory)
        except ValueError as exc:
            raise GeneratedProjectError() from exc
        cursor = run_directory
        if _is_link(cursor):
            raise GeneratedProjectError()
        for part in relative.parts:
            cursor = cursor / part
            if _is_link(cursor):
                raise GeneratedProjectError()


def _is_docker_configuration(part: str) -> bool:
    normalized = part.casefold()
    return (
        normalized in _DOCKER_FILES
        or normalized.startswith("dockerfile.")
        or normalized.startswith("docker-compose.")
        or normalized.startswith("compose.")
    )


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()
