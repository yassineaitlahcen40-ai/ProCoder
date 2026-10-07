"""Restricted Docker execution for the fixed smoke test and Python test projects."""

import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from core.config import Settings
from core.models import TestResult
from sandbox.base import SandboxRunner


_RUN_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_UNITTEST_FILE_PATTERN = re.compile(r"^test.*\.py$")
_MAX_PROJECT_FILES = 100
_MAX_PROJECT_BYTES = 5_000_000
_SENSITIVE_PROJECT_NAMES = {
    ".git",
    ".aws",
    ".azure",
    ".gcloud",
    ".ssh",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "secrets.json",
    "secrets.toml",
    "id_ed25519",
    "id_rsa",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
}


class DockerUnavailableError(RuntimeError):
    """Raised when Docker cannot be invoked."""


@dataclass
class _OutputCapture:
    limit: int
    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)
    size: int = 0
    truncated: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def consume(self, destination: bytearray, chunk: bytes) -> None:
        with self.lock:
            remaining = self.limit - self.size
            accepted = min(max(remaining, 0), len(chunk))
            destination.extend(chunk[:accepted])
            self.size += accepted
            if accepted < len(chunk):
                self.truncated = True


class DockerSandboxRunner(SandboxRunner):
    def __init__(
        self,
        settings: Settings,
        *,
        docker_executable: str = "docker",
        project_root: Path | None = None,
    ):
        self._settings = settings
        self._docker_executable = docker_executable
        self._project_root = (
            project_root.resolve()
            if project_root is not None
            else Path(__file__).resolve().parent.parent
        )

    def run_project(
        self, project_directory: Path, *, language: str | None = None
    ) -> TestResult:
        started = time.perf_counter()
        try:
            project = self._validate_project_directory(project_directory)
            detected_language, command = self._detect_test_command(
                project, requested_language=language
            )
        except (OSError, ValueError) as exc:
            return self._failure_result(
                f"Project rejected: {exc}",
                duration_seconds=time.perf_counter() - started,
                infrastructure_error=True,
            )
        if command is None:
            return self._failure_result(
                f"No supported {detected_language} test files were found.",
                duration_seconds=time.perf_counter() - started,
                tests_failed=0,
                tests_passed=0,
            )

        container_name = f"procoder-test-{uuid.uuid4().hex}"
        docker_command = (
            self._docker_executable,
            "run",
            "--rm",
            "--pull=never",
            f"--name={container_name}",
            "--network=none",
            f"--memory={self._settings.sandbox_memory_limit}",
            f"--cpus={self._settings.sandbox_cpu_limit:g}",
            f"--pids-limit={self._settings.sandbox_pids_limit}",
            "--read-only",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=32m",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--user=65532:65532",
            "--env=HOME=/tmp",
            "--env=TMPDIR=/tmp",
            "--mount",
            f"type=bind,source={project},target=/workspace,readonly",
            "--workdir=/workspace",
            "--entrypoint=python",
            self._settings.sandbox_docker_image,
            "-I",
            "-B",
            "-u",
            *command[1:],
        )

        try:
            return self._execute_project_command(
                docker_command,
                container_name=container_name,
                test_command=command,
                started=started,
            )
        except FileNotFoundError:
            return self._failure_result(
                "Docker CLI was not found. Install Docker Desktop and ensure "
                "docker.exe is available on PATH.",
                duration_seconds=time.perf_counter() - started,
                command=command,
                runtime="docker",
                infrastructure_error=True,
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            return self._failure_result(
                f"Docker execution failed: {exc}",
                duration_seconds=time.perf_counter() - started,
                command=command,
                runtime="docker",
                infrastructure_error=True,
            )

    def _validate_project_directory(self, project_directory: Path) -> Path:
        generated_root = self._project_root / "workspace" / "generated_code"
        if any(_is_link_or_junction(path) for path in (
            self._project_root / "workspace",
            generated_root,
        )):
            raise ValueError("generated workspace must not contain links.")
        if not _RUN_ID_PATTERN.fullmatch(project_directory.name):
            raise ValueError("run ID is invalid.")
        expected = generated_root / project_directory.name
        if project_directory.resolve(strict=True) != expected.resolve(strict=True):
            raise ValueError("project must be inside its assigned generated run directory.")
        if not expected.is_dir() or _is_link_or_junction(expected):
            raise ValueError("generated project directory is missing or unsafe.")

        project_size = 0
        entry_count = 0
        for entry in expected.rglob("*"):
            if _is_link_or_junction(entry):
                raise ValueError("project symlinks and junctions are not allowed.")
            if not entry.is_file() and not entry.is_dir():
                raise ValueError("project must contain only regular files and directories.")
            if entry.is_file() and entry.stat(follow_symlinks=False).st_nlink > 1:
                raise ValueError("project hard links are not allowed.")
            entry_count += 1
            if entry.is_file():
                project_size += entry.stat(follow_symlinks=False).st_size
            if entry_count > _MAX_PROJECT_FILES or project_size > _MAX_PROJECT_BYTES:
                raise ValueError("project exceeds the sandbox file-count or size limit.")
            name = entry.name.casefold()
            if _is_sensitive_project_name(name):
                raise ValueError("project contains a protected credential or config file.")
        return expected.resolve(strict=True)

    def _detect_test_command(
        self, project: Path, *, requested_language: str | None
    ) -> tuple[str, tuple[str, ...] | None]:
        python_files = tuple(project.rglob("*.py"))
        known_files = tuple(
            path for path in project.rglob("*") if path.is_file()
        )
        normalized_language = requested_language.casefold() if requested_language else ""
        if normalized_language and not (
            normalized_language == "py" or normalized_language.startswith("python")
        ):
            return requested_language, None
        if not python_files and known_files:
            extension = known_files[0].suffix.lstrip(".") or "unknown"
            return extension, None
        if not python_files:
            return "Python", None
        has_test_files = any(
            _UNITTEST_FILE_PATTERN.fullmatch(path.name)
            for path in python_files
        )
        if not has_test_files:
            return "Python", None
        return "Python", (
            "python",
            "-I",
            "-B",
            "-u",
            "-m",
            "unittest",
            "discover",
            "-v",
        )

    def _execute_project_command(
        self,
        docker_command: tuple[str, ...],
        *,
        container_name: str,
        test_command: tuple[str, ...],
        started: float,
    ) -> TestResult:
        process = subprocess.Popen(
            docker_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_docker_cli_environment(),
        )
        capture = _OutputCapture(self._settings.sandbox_max_output_bytes)
        if process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            return self._failure_result(
                "Docker output streams could not be created.",
                duration_seconds=time.perf_counter() - started,
                command=test_command,
                runtime="docker",
                infrastructure_error=True,
            )

        readers = [
            threading.Thread(
                target=self._drain,
                args=(process.stdout, capture.stdout, capture),
                daemon=True,
            ),
            threading.Thread(
                target=self._drain,
                args=(process.stderr, capture.stderr, capture),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        deadline = started + self._settings.sandbox_timeout_seconds
        timed_out = False
        while process.poll() is None:
            if capture.truncated:
                process.terminate()
                break
            if time.perf_counter() >= deadline:
                timed_out = True
                process.terminate()
                break
            time.sleep(0.02)

        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        for reader in readers:
            reader.join(timeout=2)
            if reader.is_alive():
                process.kill()
                self._remove_container(container_name)
                raise RuntimeError("Docker output reader did not stop after process exit.")
        process.stdout.close()
        process.stderr.close()
        cleanup_error: str | None = None
        if timed_out or capture.truncated:
            try:
                self._remove_container(container_name)
            except RuntimeError as exc:
                cleanup_error = str(exc)

        stdout = capture.stdout.decode("utf-8", errors="replace")
        stderr = capture.stderr.decode("utf-8", errors="replace")
        if cleanup_error:
            stderr = f"{stderr}\n{cleanup_error}".strip()
        exit_code = process.returncode
        tests_run, tests_failed, tests_skipped = _parse_unittest_summary(stdout, stderr)
        tests_passed = (
            max(tests_run - tests_failed - tests_skipped, 0)
            if (
                tests_run is not None
                and tests_failed is not None
                and tests_skipped is not None
            )
            else None
        )
        passed = (
            not timed_out
            and not capture.truncated
            and cleanup_error is None
            and exit_code == 0
            and tests_run is not None
            and tests_run > 0
            and tests_failed == 0
        )
        return TestResult(
            passed=passed,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.perf_counter() - started,
            timed_out=timed_out,
            output_truncated=capture.truncated,
            command=test_command,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            runtime="docker",
            container_image=self._settings.sandbox_docker_image,
            infrastructure_error=exit_code in {125, 126, 127},
        )

    def _failure_result(
        self,
        message: str,
        *,
        duration_seconds: float,
        command: tuple[str, ...] = (),
        runtime: str | None = None,
        tests_passed: int | None = None,
        tests_failed: int | None = None,
        infrastructure_error: bool = False,
    ) -> TestResult:
        return TestResult(
            passed=False,
            exit_code=None,
            stdout="",
            stderr=message,
            duration_seconds=duration_seconds,
            command=command,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            runtime=runtime,
            container_image=self._settings.sandbox_docker_image,
            infrastructure_error=infrastructure_error,
        )

    def run_smoke_test(self) -> TestResult:
        container_name = f"procoder-smoke-{uuid.uuid4().hex}"
        command = (
            self._docker_executable,
            "run",
            "--rm",
            "--pull=never",
            f"--name={container_name}",
            "--network=none",
            f"--memory={self._settings.sandbox_memory_limit}",
            f"--cpus={self._settings.sandbox_cpu_limit:g}",
            f"--pids-limit={self._settings.sandbox_pids_limit}",
            "--read-only",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--user=65532:65532",
            "--entrypoint=python",
            self._settings.sandbox_docker_image,
            "-c",
            "print('ProCoder sandbox smoke test passed')",
        )
        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise DockerUnavailableError(
                "Docker CLI was not found. Install Docker Desktop and ensure "
                "docker.exe is available on PATH."
            ) from exc

        capture = _OutputCapture(self._settings.sandbox_max_output_bytes)
        readers = [
            threading.Thread(
                target=self._drain,
                args=(process.stdout, capture.stdout, capture),
                daemon=True,
            ),
            threading.Thread(
                target=self._drain,
                args=(process.stderr, capture.stderr, capture),
                daemon=True,
            ),
        ]
        if process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            raise RuntimeError("Docker process output pipes were not created.")
        for reader in readers:
            reader.start()

        deadline = started + self._settings.sandbox_timeout_seconds
        timed_out = False
        while process.poll() is None:
            if capture.truncated:
                process.terminate()
                break
            if time.perf_counter() >= deadline:
                timed_out = True
                process.terminate()
                break
            time.sleep(0.02)

        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        for reader in readers:
            reader.join(timeout=2)
            if reader.is_alive():
                raise RuntimeError("Docker output reader did not stop after process exit.")
        process.stdout.close()
        process.stderr.close()

        if timed_out or capture.truncated:
            self._remove_container(container_name)

        stdout = capture.stdout.decode("utf-8", errors="replace")
        stderr = capture.stderr.decode("utf-8", errors="replace")
        exit_code = process.returncode
        passed = (
            not timed_out
            and not capture.truncated
            and exit_code == 0
            and stdout.strip() == "ProCoder sandbox smoke test passed"
        )
        return TestResult(
            passed=passed,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=time.perf_counter() - started,
            timed_out=timed_out,
            output_truncated=capture.truncated,
            command=command,
        )

    @staticmethod
    def _drain(
        stream: BinaryIO, destination: bytearray, capture: _OutputCapture
    ) -> None:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            capture.consume(destination, chunk)

    def _remove_container(self, container_name: str) -> None:
        try:
            cleanup = subprocess.run(
                (self._docker_executable, "rm", "--force", container_name),
                capture_output=True,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out while removing sandbox container {container_name}."
            ) from exc
        if cleanup.returncode != 0 and b"No such container" not in cleanup.stderr:
            detail = cleanup.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Could not remove sandbox container {container_name}: {detail}"
            )


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _docker_cli_environment() -> dict[str, str]:
    allowed = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
    }
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed
    }


def _is_sensitive_project_name(name: str) -> bool:
    if name.startswith(".env") or name in _SENSITIVE_PROJECT_NAMES:
        return True
    if name in {".gitconfig", ".git-credentials"}:
        return True
    sensitive_markers = (
        "credential",
        "secret",
        "token",
        "api_key",
        "apikey",
        "access_key",
        "private_key",
        "id_rsa",
        "id_ed25519",
    )
    return (
        any(marker in name for marker in sensitive_markers)
        or name.endswith((".pem", ".key"))
    )


def _parse_unittest_summary(
    stdout: str, stderr: str
) -> tuple[int | None, int | None, int | None]:
    output = f"{stdout}\n{stderr}"
    match = re.search(r"Ran\s+(\d+)\s+tests?\s+in\s+[\d.]+s", output)
    if not match:
        return None, None, None
    tests_run = int(match.group(1))
    failed = 0
    skipped = 0
    failure_match = re.search(r"FAILED\s*\(([^)]*)\)", output)
    if failure_match:
        for name, count in re.findall(r"(failures|errors|skipped)=(\d+)", failure_match.group(1)):
            if name == "skipped":
                skipped += int(count)
            else:
                failed += int(count)
    else:
        skipped_match = re.search(r"skipped=(\d+)", output)
        if skipped_match:
            skipped = int(skipped_match.group(1))
    return tests_run, failed, skipped
