import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import BinaryIO, cast
from unittest.mock import patch

from agents.coding_agent.errors import (
    CodexCliNotFoundError,
    CodexExecutionError,
    CodexResponseError,
)
from agents.coding_agent.providers.openai_codex import (
    OpenAICodexCodingAgent,
    _parse_coding_result,
    _read_reported_token_usage,
    resolve_codex_executable,
)
from core.config import Settings
from core.models import (
    CodeRepair,
    GeneratedCode,
    RepairFile,
    RepairInstructions,
    ReviewResult,
    ReviewVerdict,
    Specification,
    TestResult,
)
from workspace.generated_project import GeneratedProjectError, GeneratedProjectWriter


class CodexCodingAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specification = Specification(
            request="Create a Python prime checker.",
            summary="Determine whether an integer is prime.",
            requirements=("Accept an integer.", "Return a boolean result."),
            acceptance_criteria=("2 is prime.", "4 is not prime."),
            provider="google-gemini",
            model="gemini-test-model",
        )
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.captured_schema: dict[str, object] = {}
        self.codex_executable = self.workspace / "codex.exe"
        self.codex_executable.write_text("offline test stub", encoding="utf-8")
        self.settings = Settings.from_environment(
            {
                "CODEX_CLI": str(self.codex_executable),
                "CODEX_MODEL": "configured-test-model",
                "CODEX_TIMEOUT": "12",
            }
        )
        self.agent = OpenAICodexCodingAgent(self.settings, self.workspace)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_cli_result(
        self, command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess:
        cast(BinaryIO, kwargs["stdout"]).write(
            b'{"type":"turn.completed","usage":{"input_tokens":21,"output_tokens":10}}\n'
        )
        schema_path = Path(command[command.index("--output-schema") + 1])
        self.captured_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "language": "Python",
                    "summary": "Adds a primality function and unit tests.",
                    "files": [
                        {"path": "prime.py", "content": "def is_prime(n): return n > 1"},
                        {"path": "test_prime.py", "content": "def test_prime(): pass"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    def test_returns_validated_multi_file_coding_result(self) -> None:
        with patch(
            "agents.coding_agent.providers.openai_codex.subprocess.run",
            side_effect=self._write_cli_result,
        ) as run:
            result = self.agent.generate(self.specification)

        self.assertEqual(tuple(result.files), ("prime.py", "test_prime.py"))
        self.assertEqual(result.language, "Python")
        self.assertEqual(result.summary, "Adds a primality function and unit tests.")
        self.assertEqual(result.provider, "openai-codex-cli")
        self.assertEqual(result.model, "configured-test-model")
        self.assertEqual(result.tokens_used, 31)
        self.assertGreaterEqual(result.duration_seconds or -1, 0)
        call = run.call_args
        command = call.args[0]
        self.assertIn("--sandbox", command)
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertIn("--disable", command)
        self.assertEqual(command[command.index("--disable") + 1], "shell_tool")
        self.assertIn("--output-schema", command)
        self.assertEqual(
            self.captured_schema["required"],
            ["language", "summary", "files"],
        )
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--skip-git-repo-check", command)
        self.assertIn("--ephemeral", command)
        self.assertFalse(call.kwargs["shell"])
        prompt = call.kwargs["input"]
        self.assertIn("Do not run, execute, build, test, or install anything.", prompt)
        self.assertIn("validated specification", prompt)
        self.assertIn("2 is prime.", prompt)
        self.assertNotIn("OPENAI_API_KEY", call.kwargs["env"])
        self.assertNotIn("GEMINI_API_KEY", call.kwargs["env"])

    def test_rejects_malformed_provider_output(self) -> None:
        with patch(
            "agents.coding_agent.providers.openai_codex.subprocess.run",
            side_effect=lambda command, **_: self._write_text_response(
                "{not JSON", command
            ),
        ):
            with self.assertRaises(CodexResponseError):
                self.agent.generate(self.specification)

    def test_provider_failure_is_mapped_to_safe_error(self) -> None:
        with patch(
            "agents.coding_agent.providers.openai_codex.subprocess.run",
            return_value=subprocess.CompletedProcess(["codex"], 1),
        ):
            with self.assertRaises(CodexExecutionError) as context:
                self.agent.generate(self.specification)

        self.assertNotIn("secret", context.exception.safe_message)

    def test_missing_cli_has_safe_error(self) -> None:
        with (
            patch(
                "agents.coding_agent.providers.openai_codex.resolve_codex_executable",
                return_value=None,
            ),
            patch(
                "agents.coding_agent.providers.openai_codex.subprocess.run"
            ) as run,
        ):
            with self.assertRaises(CodexCliNotFoundError):
                self.agent.generate(self.specification)
        run.assert_not_called()

    def test_result_parser_requires_exact_structured_shape(self) -> None:
        for response in (
            "",
            "```json\n{}\n```",
            json.dumps({"language": "Python", "files": []}),
            json.dumps(
                {
                    "language": "Python",
                    "summary": "Complete.",
                    "files": [{"path": "main.py", "content": "pass", "extra": True}],
                }
            ),
        ):
            with self.subTest(response=response):
                with self.assertRaises(CodexResponseError):
                    _parse_coding_result(
                        response,
                        model=None,
                        tokens_used=None,
                        duration_seconds=0,
                    )

    def test_reads_token_usage_only_from_completed_cli_turn_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            event_path = Path(temporary_directory) / "events.jsonl"
            event_path.write_text(
                "\n".join(
                    (
                        '{"type":"item.completed","item":{"type":"agent_message","text":"ignored"}}',
                        '{"type":"turn.completed","usage":{"input_tokens":12,"output_tokens":5}}',
                        '{"type":"turn.completed","usage":{"input_tokens":3,"output_tokens":2}}',
                    )
                ),
                encoding="utf-8",
            )

            self.assertEqual(_read_reported_token_usage(event_path), 22)

    def test_codex_subprocess_does_not_inherit_api_credentials(self) -> None:
        with patch.dict(
                    os.environ,
                    {
                        "OPENAI_API_KEY": "must-not-forward",
                        "GEMINI_API_KEY": "must-not-forward",
                        "NVIDIA_API_KEY": "must-not-forward",
                        "PATH": "safe-path",
                        "CODEX_HOME": "local-codex-auth",
                    },
                    clear=True,
        ):
                    environment = OpenAICodexCodingAgent._codex_environment()

        self.assertEqual(environment["PATH"], "safe-path")
        self.assertEqual(environment["CODEX_HOME"], "local-codex-auth")
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertNotIn("GEMINI_API_KEY", environment)
        self.assertNotIn("NVIDIA_API_KEY", environment)

    def test_discovers_codex_cli_bundled_with_vscode_extension(self) -> None:
        extension_binary = (
            self.workspace
            / ".vscode"
            / "extensions"
            / "openai.chatgpt-test"
            / "bin"
            / "windows-x86_64"
            / "codex.exe"
        )
        extension_binary.parent.mkdir(parents=True)
        extension_binary.write_text("offline extension stub", encoding="utf-8")
        with (
            patch(
                "agents.coding_agent.providers.openai_codex.shutil.which",
                return_value=None,
            ),
            patch(
                "agents.coding_agent.providers.openai_codex.Path.home",
                return_value=self.workspace,
            ),
            patch(
                "agents.coding_agent.providers.openai_codex.platform.machine",
                return_value="AMD64",
            ),
        ):
            resolved = resolve_codex_executable("codex")
        self.assertEqual(resolved, str(extension_binary.resolve()))

    def test_repair_returns_structured_patch_without_enabling_execution(self) -> None:
        repair_result = {
            "files_created": [],
            "files_modified": [
                {"path": "prime.py", "content": "def is_prime(n): return n >= 2"}
            ],
            "files_deleted": [],
            "summary": "Corrects prime-number handling.",
        }

        def write_repair_response(command: list[str], **kwargs: object):
            cast(BinaryIO, kwargs["stdout"]).write(
                b'{"type":"turn.completed","usage":{"input_tokens":13,"output_tokens":5}}\n'
            )
            schema = Path(command[command.index("--output-schema") + 1])
            self.captured_schema = json.loads(schema.read_text(encoding="utf-8"))
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text(json.dumps(repair_result), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        test_result = TestResult(
            passed=False,
            exit_code=1,
            stdout="AssertionError: untrusted output",
            stderr="",
            duration_seconds=0.1,
            tests_failed=1,
        )
        review_result = ReviewResult(
            verdict=ReviewVerdict.NEEDS_REPAIR,
            summary="A required edge case is missing.",
            specification_satisfied=False,
            code_quality_findings=("Handles the lower boundary incorrectly.",),
            test_analysis="Docker reported FAILED.",
            recommended_actions=("Handle values below 2.",),
            repair_required=True,
            confidence=0.9,
            tests_passed=False,
        )
        current_code = GeneratedCode(
            files={"prime.py": "def is_prime(n): return True"}
        )
        context = RepairInstructions(
            instructions="Apply only the smallest correction.",
            test_result=test_result,
            review_result=review_result,
            attempt_number=1,
        )
        with patch(
            "agents.coding_agent.providers.openai_codex.subprocess.run",
            side_effect=write_repair_response,
        ) as run:
            result = self.agent.repair(self.specification, current_code, context)

        self.assertEqual(
            result.files_modified,
            (RepairFile("prime.py", "def is_prime(n): return n >= 2"),),
        )
        self.assertEqual(result.tokens_used, 18)
        call = run.call_args
        self.assertEqual(
            self.captured_schema["required"],
            ["files_created", "files_modified", "files_deleted", "summary"],
        )
        self.assertEqual(
            call.args[0][call.args[0].index("--sandbox") + 1], "read-only"
        )
        self.assertEqual(
            call.args[0][call.args[0].index("--disable") + 1], "shell_tool"
        )
        prompt = call.kwargs["input"]
        self.assertIn("UNTRUSTED DATA", prompt)
        self.assertIn("AssertionError: untrusted output", prompt)
        self.assertIn('"repair_attempt_number":1', prompt)
        self.assertNotIn("NVIDIA_API_KEY", call.kwargs["env"])

    def test_repair_parser_rejects_malformed_or_empty_patch(self) -> None:
        from agents.coding_agent.providers.openai_codex import _parse_repair_result

        for response in (
            "{bad",
            json.dumps({"files_created": [], "files_modified": [], "files_deleted": []}),
            json.dumps(
                {
                    "files_created": [],
                    "files_modified": [],
                    "files_deleted": [],
                    "summary": "No changes.",
                }
            ),
        ):
            with self.subTest(response=response):
                with self.assertRaises(CodexResponseError):
                    _parse_repair_result(
                        response,
                        model=None,
                        tokens_used=None,
                        duration_seconds=0,
                    )

    @staticmethod
    def _write_text_response(
        response: str, command: list[str], **_: object
    ) -> subprocess.CompletedProcess:
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(response, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)


class GeneratedProjectWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.writer = GeneratedProjectWriter(self.root)
        self.run_id = "12345678-1234-1234-1234-123456789abc"
        self.run_directory = self.writer.create_run_directory(self.run_id)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_writes_multiple_files_under_dedicated_run_directory(self) -> None:
        generated = GeneratedCode(
            files={
                "src/main.py": "def main(): return 0",
                "tests/test_main.py": "def test_main(): assert True",
            },
            provider="openai-codex-cli",
            language="Python",
            summary="Small application with a unit test.",
        )

        written = self.writer.write(self.run_id, generated)

        self.assertEqual(
            written.files_created,
            ("src/main.py", "tests/test_main.py"),
        )
        self.assertEqual(written.files_modified, ())
        self.assertTrue((self.run_directory / "src" / "main.py").is_file())
        self.assertFalse((self.root / "main.py").exists())

    def test_reports_files_modified_within_existing_run_workspace(self) -> None:
        existing = self.run_directory / "main.py"
        existing.write_text("old", encoding="utf-8")

        written = self.writer.write(
            self.run_id,
            GeneratedCode(files={"main.py": "new"}),
        )

        self.assertEqual(written.files_created, ())
        self.assertEqual(written.files_modified, ("main.py",))
        self.assertEqual(existing.read_text(encoding="utf-8"), "new")

    def test_rejects_traversal_absolute_secret_and_docker_paths(self) -> None:
        unsafe_paths = (
            "../main.py",
            "src/../../outside.py",
            "C:/outside.py",
            "/outside.py",
            ".env",
            ".env.production",
            ".git/config",
            "Dockerfile",
            "docker-compose.yml",
            "src\\..\\main.py",
            "subdir/../.env",
        )
        for path in unsafe_paths:
            with self.subTest(path=path):
                with self.assertRaises(GeneratedProjectError):
                    self.writer.write(
                        self.run_id,
                        GeneratedCode(files={path: "unsafe"}),
                    )

    def test_rejects_invalid_or_reused_run_id(self) -> None:
        with self.assertRaises(GeneratedProjectError):
            self.writer.create_run_directory("../outside")
        with self.assertRaises(GeneratedProjectError):
            self.writer.create_run_directory(self.run_id)

    def test_applies_valid_repair_changes_only_inside_assigned_workspace(self) -> None:
        (self.run_directory / "main.py").write_text("old", encoding="utf-8")
        (self.run_directory / "remove.py").write_text("obsolete", encoding="utf-8")
        repair = CodeRepair(
            files_created=(RepairFile("tests/test_main.py", "assert True"),),
            files_modified=(RepairFile("main.py", "new"),),
            files_deleted=("remove.py",),
            summary="Fixes the test failure.",
        )

        applied = self.writer.apply_repair(self.run_id, repair)

        self.assertEqual(applied.changed_paths, ("tests/test_main.py", "main.py", "remove.py"))
        self.assertEqual((self.run_directory / "main.py").read_text(encoding="utf-8"), "new")
        self.assertTrue((self.run_directory / "tests" / "test_main.py").is_file())
        self.assertFalse((self.run_directory / "remove.py").exists())

    def test_rejects_unsafe_repair_paths_without_partial_changes(self) -> None:
        original = self.run_directory / "main.py"
        original.write_text("stable", encoding="utf-8")
        unsafe_patches = (
            CodeRepair(
                files_created=(RepairFile("../outside.py", "bad"),),
                files_modified=(),
                files_deleted=(),
                summary="Unsafe traversal.",
            ),
            CodeRepair(
                files_created=(RepairFile(".env", "SECRET=value"),),
                files_modified=(),
                files_deleted=(),
                summary="Unsafe secret file.",
            ),
        )

        for repair in unsafe_patches:
            with self.subTest(repair=repair.summary):
                with self.assertRaises(GeneratedProjectError):
                    self.writer.apply_repair(self.run_id, repair)
                self.assertEqual(original.read_text(encoding="utf-8"), "stable")
                self.assertFalse((self.root / "outside.py").exists())

if __name__ == "__main__":
    unittest.main()
