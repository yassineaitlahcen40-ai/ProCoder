import io
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from core.config import Settings
from sandbox.runner import DockerSandboxRunner, _OutputCapture


class _FakeProcess:
    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        *,
        exit_code: int = 0,
        poll_values: tuple[int | None, ...] = (),
    ) -> None:
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = exit_code
        self._poll_values = list(poll_values)
        self.wait_timeout: float | None = None

    def poll(self) -> int | None:
        return self._poll_values.pop(0) if self._poll_values else self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeout = timeout
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class SandboxSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project_root = Path(self.temporary_directory.name)
        self.run_id = str(uuid4())
        self.project = (
            self.project_root / "workspace" / "generated_code" / self.run_id
        )
        self.project.mkdir(parents=True)
        (self.project / "test_example.py").write_text(
            "import unittest\nclass ExampleTests(unittest.TestCase):\n"
            "    def test_ok(self): self.assertTrue(True)\n",
            encoding="utf-8",
        )
        self.settings = Settings.from_environment({})
        self.runner = DockerSandboxRunner(
            self.settings, project_root=self.project_root
        )

    def _mock_docker(self, process: _FakeProcess) -> Mock:
        return patch("sandbox.runner.subprocess.Popen", return_value=process).start()

    def test_successful_unittest_run_uses_hardened_container(self) -> None:
        process = _FakeProcess(
            stderr=b".\n----------------------------------------------------------------------\n"
            b"Ran 1 test in 0.001s\n\nOK\n"
        )
        popen = self._mock_docker(process)
        self.addCleanup(patch.stopall)

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "test-gemini-marker",
                "OPENAI_API_KEY": "test-openai-marker",
                "NVIDIA_API_KEY": "test-nvidia-marker",
                "GIT_ASKPASS": "test-git-marker",
            },
        ):
            result = self.runner.run_project(self.project)

        self.assertTrue(result.passed)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.tests_passed, 1)
        self.assertEqual(result.tests_failed, 0)
        self.assertEqual(
            result.command,
            ("python", "-I", "-B", "-u", "-m", "unittest", "discover", "-v"),
        )
        self.assertEqual(result.runtime, "docker")
        self.assertEqual(result.container_image, self.settings.sandbox_docker_image)
        command = popen.call_args.args[0]
        for restriction in (
            "--network=none",
            "--memory=256m",
            "--cpus=0.5",
            f"--pids-limit={self.settings.sandbox_pids_limit}",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--user=65532:65532",
            "--env=HOME=/tmp",
            "--env=TMPDIR=/tmp",
        ):
            self.assertIn(restriction, command)
        mounts = [
            command[index + 1]
            for index, argument in enumerate(command[:-1])
            if argument == "--mount"
        ]
        self.assertEqual(
            mounts,
            [
                f"type=bind,source={self.project.resolve()},target=/workspace,readonly"
            ],
        )
        self.assertNotIn("--env-file", command)
        self.assertNotIn("/var/run/docker.sock", command)
        self.assertNotIn("--privileged", command)
        self.assertNotIn("--network=host", command)
        self.assertFalse(any("API_KEY" in value for value in command))
        self.assertFalse(any("GIT" in value for value in command))
        docker_environment = popen.call_args.kwargs["env"]
        self.assertNotIn("GEMINI_API_KEY", docker_environment)
        self.assertNotIn("OPENAI_API_KEY", docker_environment)
        self.assertNotIn("NVIDIA_API_KEY", docker_environment)
        self.assertNotIn("GIT_ASKPASS", docker_environment)
        self.assertIn("-B", command)
        self.assertIn("-u", command)

    def test_failing_assertion_is_a_valid_test_result(self) -> None:
        process = _FakeProcess(
            stderr=b"FAIL: test_bad\nRan 1 test in 0.001s\n"
            b"FAILED (failures=1)\n",
            exit_code=1,
        )
        self._mock_docker(process)
        self.addCleanup(patch.stopall)

        result = self.runner.run_project(self.project)

        self.assertFalse(result.passed)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.tests_passed, 0)
        self.assertEqual(result.tests_failed, 1)
        self.assertIn("FAIL: test_bad", result.stderr)

    def test_syntax_error_and_runtime_exception_are_captured(self) -> None:
        for error_text in (
            "SyntaxError: invalid syntax",
            "RuntimeError: generated test failed",
        ):
            with self.subTest(error_text=error_text):
                self._mock_docker(
                    _FakeProcess(
                        stderr=(
                            f"ERROR: test_module\n{error_text}\n"
                            "Ran 1 test in 0.001s\nFAILED (errors=1)\n"
                        ).encode(),
                        exit_code=1,
                    )
                )
                result = self.runner.run_project(self.project)
                self.assertFalse(result.passed)
                self.assertEqual(result.tests_failed, 1)
                self.assertIn(error_text, result.stderr)
                patch.stopall()

    def test_timeout_terminates_and_removes_container(self) -> None:
        process = _FakeProcess(poll_values=(None,))
        settings = replace(self.settings, sandbox_timeout_seconds=1)
        runner = DockerSandboxRunner(settings, project_root=self.project_root)
        popen = self._mock_docker(process)
        run = patch("sandbox.runner.subprocess.run", return_value=Mock(
            returncode=0, stderr=b""
        )).start()
        perf_counter = patch(
            "sandbox.runner.time.perf_counter", side_effect=(10.0, 12.0, 12.1)
        ).start()
        patch("sandbox.runner.time.sleep").start()
        self.addCleanup(patch.stopall)

        result = runner.run_project(self.project)

        self.assertFalse(result.passed)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.exit_code, -15)
        self.assertEqual(process.wait_timeout, 2)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        self.assertEqual(run.call_args.args[0][:3], ("docker", "rm", "--force"))
        self.assertEqual(perf_counter.call_count, 3)
        self.assertEqual(popen.call_count, 1)

    def test_missing_test_files_and_unsupported_language_do_not_start_docker(self) -> None:
        (self.project / "test_example.py").unlink()
        (self.project / "main.py").write_text("print('not run')", encoding="utf-8")
        with patch("sandbox.runner.subprocess.Popen") as popen:
            missing = self.runner.run_project(self.project)
            self.assertFalse(missing.passed)
            self.assertIn("test files", missing.stderr)
            popen.assert_not_called()

            unsupported = self.runner.run_project(self.project, language="JavaScript")
            self.assertFalse(unsupported.passed)
            self.assertIn("JavaScript", unsupported.stderr)
            popen.assert_not_called()

    def test_rejects_paths_outside_generated_workspace(self) -> None:
        outside = self.project_root / "outside"
        outside.mkdir()

        result = self.runner.run_project(outside)

        self.assertFalse(result.passed)
        self.assertIn("run ID is invalid", result.stderr)

    def test_rejects_dotenv_and_project_symlinks(self) -> None:
        secret_file = self.project / ".env"
        secret_file.write_text("DO_NOT_MOUNT", encoding="utf-8")

        result = self.runner.run_project(self.project)

        self.assertFalse(result.passed)
        self.assertIn("protected credential", result.stderr)
        secret_file.unlink()
        (self.project / "api_key.txt").write_text(
            "must not be mounted", encoding="utf-8"
        )
        result = self.runner.run_project(self.project)
        self.assertFalse(result.passed)
        self.assertIn("protected credential", result.stderr)
        (self.project / "api_key.txt").unlink()
        link = self.project / "linked.txt"
        link.write_text("not actually linked", encoding="utf-8")
        with patch(
            "sandbox.runner._is_link_or_junction",
            side_effect=lambda path: path == link,
        ):
            result = self.runner.run_project(self.project)
        self.assertFalse(result.passed)
        self.assertIn("symlinks", result.stderr)

    def test_output_capture_enforces_a_combined_byte_limit(self) -> None:
        capture = _OutputCapture(limit=5)

        capture.consume(capture.stdout, b"1234")
        capture.consume(capture.stderr, b"567")

        self.assertEqual(capture.stdout, b"1234")
        self.assertEqual(capture.stderr, b"5")
        self.assertTrue(capture.truncated)
        self.assertEqual(capture.size, 5)

    def test_container_start_failure_is_returned_not_raised(self) -> None:
        with patch(
            "sandbox.runner.subprocess.Popen", side_effect=FileNotFoundError
        ):
            result = self.runner.run_project(self.project)

        self.assertFalse(result.passed)
        self.assertIsNone(result.exit_code)
        self.assertIn("Docker CLI was not found", result.stderr)

    def test_docker_engine_failure_is_a_structured_result(self) -> None:
        process = _FakeProcess(
            stderr=b"image is not available locally", exit_code=125
        )
        self._mock_docker(process)
        self.addCleanup(patch.stopall)

        result = self.runner.run_project(self.project)

        self.assertFalse(result.passed)
        self.assertEqual(result.exit_code, 125)
        self.assertIsNone(result.tests_passed)
        self.assertIsNone(result.tests_failed)
        self.assertIn("image is not available locally", result.stderr)
