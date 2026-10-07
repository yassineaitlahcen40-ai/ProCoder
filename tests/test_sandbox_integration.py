import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from core.config import Settings
from sandbox.runner import DockerSandboxRunner


def _sandbox_ready() -> bool:
    try:
        daemon = subprocess.run(
            ("docker", "info"),
            capture_output=True,
            timeout=5,
            check=False,
        )
        image = subprocess.run(
            ("docker", "image", "inspect", "procoder-sandbox:local"),
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return daemon.returncode == 0 and image.returncode == 0


@unittest.skipUnless(
    _sandbox_ready(),
    "Docker daemon and procoder-sandbox:local image are required.",
)
class DockerSandboxIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.project_root = Path(self.temporary_directory.name)
        self.run_directory = (
            self.project_root
            / "workspace"
            / "generated_code"
            / str(uuid4())
        )
        self.run_directory.mkdir(parents=True)

    def _run_project(
        self, *, timeout: int = 10, image: str = "procoder-sandbox:local"
    ):
        settings = replace(
            Settings.from_environment({}),
            sandbox_timeout_seconds=timeout,
            sandbox_docker_image=image,
        )
        return DockerSandboxRunner(
            settings, project_root=self.project_root
        ).run_project(self.run_directory)

    def test_generated_unittest_passes_in_container(self) -> None:
        (self.run_directory / "prime.py").write_text(
            "def is_prime(number):\n"
            "    if number < 2: return False\n"
            "    return all(number % divisor for divisor in range(2, number))\n",
            encoding="utf-8",
        )
        (self.run_directory / "test_prime.py").write_text(
            "import unittest\n"
            "from prime import is_prime\n"
            "class PrimeTests(unittest.TestCase):\n"
            "    def test_known_true(self): self.assertTrue(is_prime(2))\n"
            "    def test_known_false(self): self.assertFalse(is_prime(1))\n",
            encoding="utf-8",
        )

        result = self._run_project()

        self.assertTrue(result.passed, result.stderr)
        self.assertEqual(result.tests_passed, 2)
        self.assertEqual(result.tests_failed, 0)

    def test_failing_assertion_is_reported_from_container(self) -> None:
        (self.run_directory / "test_failure.py").write_text(
            "import unittest\n"
            "class FailureTests(unittest.TestCase):\n"
            "    def test_expected_failure(self): self.fail('expected')\n",
            encoding="utf-8",
        )

        result = self._run_project()

        self.assertFalse(result.passed)
        self.assertEqual(result.tests_failed, 1)
        self.assertIn("expected", result.stderr)

    def test_syntax_error_is_reported_from_container(self) -> None:
        (self.run_directory / "test_syntax.py").write_text(
            "def broken(:\n    pass\n", encoding="utf-8"
        )

        result = self._run_project()

        self.assertFalse(result.passed)
        self.assertGreaterEqual(result.tests_failed or 0, 1)
        self.assertIn("SyntaxError", result.stderr)

    def test_runtime_exception_is_reported_from_container(self) -> None:
        (self.run_directory / "test_runtime.py").write_text(
            "import unittest\n"
            "class RuntimeTests(unittest.TestCase):\n"
            "    def test_exception(self): raise RuntimeError('generated failure')\n",
            encoding="utf-8",
        )

        result = self._run_project()

        self.assertFalse(result.passed)
        self.assertEqual(result.tests_failed, 1)
        self.assertIn("generated failure", result.stderr)

    def test_infinite_loop_is_terminated_by_timeout(self) -> None:
        (self.run_directory / "test_loop.py").write_text(
            "import unittest\n"
            "class LoopTests(unittest.TestCase):\n"
            "    def test_loop(self):\n"
            "        while True: pass\n",
            encoding="utf-8",
        )

        result = self._run_project(timeout=2)

        self.assertFalse(result.passed)
        self.assertTrue(result.timed_out)

    def test_network_access_is_disabled(self) -> None:
        (self.run_directory / "test_network.py").write_text(
            "import socket\n"
            "import unittest\n"
            "class NetworkTests(unittest.TestCase):\n"
            "    def test_external_network_is_unavailable(self):\n"
            "        with self.assertRaises(OSError):\n"
            "            socket.create_connection(('192.0.2.1', 80), timeout=0.5)\n",
            encoding="utf-8",
        )

        result = self._run_project(timeout=5)

        self.assertTrue(result.passed, result.stderr)

    def test_project_escape_cannot_read_host_sibling(self) -> None:
        marker = "PROC0DER_HOST_SENTINEL_NOT_MOUNTED"
        (self.project_root / "workspace" / "sentinel.txt").write_text(
            marker, encoding="utf-8"
        )
        (self.run_directory / "test_escape.py").write_text(
            "import pathlib\n"
            "import unittest\n"
            "class EscapeTests(unittest.TestCase):\n"
            "    def test_sibling_is_not_mounted(self):\n"
            "        candidate = pathlib.Path('/workspace/../sentinel.txt')\n"
            "        self.assertFalse(candidate.exists())\n",
            encoding="utf-8",
        )

        result = self._run_project()

        self.assertTrue(result.passed, result.stderr)
        self.assertNotIn(marker, result.stdout + result.stderr)

    def test_docker_container_start_failure_is_structured(self) -> None:
        (self.run_directory / "test_example.py").write_text(
            "import unittest\nclass Example(unittest.TestCase):\n"
            "    def test_ok(self): self.assertTrue(True)\n",
            encoding="utf-8",
        )

        result = self._run_project(image="procoder-image-that-does-not-exist:local")

        self.assertFalse(result.passed)
        self.assertIsNotNone(result.exit_code)
        self.assertFalse(result.timed_out)
