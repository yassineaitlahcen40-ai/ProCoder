import io
import logging
import tempfile
import unittest
from uuid import UUID
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from agents.prompt_agent.errors import GeminiTemporarilyUnavailableError
from core.config import Settings
from core.models import (
    GeneratedCode,
    ReviewResult,
    ReviewVerdict,
    Specification,
    TestResult,
)
from main import main


class PromptOnlyCliTests(unittest.TestCase):
    def test_prompt_command_only_invokes_prompt_agent_and_displays_specification(
        self,
    ) -> None:
        request = "Create a Python prime checker."
        settings = Settings.from_environment({"GEMINI_API_KEY": "unit-test-key"})
        prompt_agent = Mock()
        prompt_agent.return_value.create_specification.return_value = Specification(
            request=request,
            summary="Create a prime-number checker.",
            requirements=("Accept an integer.", "Determine whether it is prime."),
            acceptance_criteria=("2 is reported as prime.", "4 is not prime."),
            provider="google-gemini",
            model="gemini-test-model",
        )
        logger = Mock(spec=logging.Logger)
        output = io.StringIO()
        errors = io.StringIO()

        with (
            patch("main.Settings.from_environment", return_value=settings),
            patch("main.configure_logging", return_value=logger),
            patch("main.GeminiPromptAgent", prompt_agent),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            result = main(["prompt", request])

        self.assertEqual(result, 0)
        prompt_agent.assert_called_once_with(settings)
        prompt_agent.return_value.create_specification.assert_called_once_with(request)
        self.assertIn(f"REQUEST:\n{request}", output.getvalue())
        self.assertIn("SUMMARY:\nCreate a prime-number checker.", output.getvalue())
        self.assertIn("REQUIREMENTS:\n1. Accept an integer.", output.getvalue())
        self.assertIn("ACCEPTANCE CRITERIA:\n1. 2 is reported as prime.", output.getvalue())
        self.assertEqual(errors.getvalue(), "")
        logged = repr(logger.warning.call_args_list + logger.info.call_args_list)
        self.assertNotIn(request, logged)
        self.assertNotIn("unit-test-key", logged)

    def test_missing_key_command_shows_safe_message_and_stops(self) -> None:
        settings = Settings.from_environment({})
        logger = Mock(spec=logging.Logger)
        output = io.StringIO()
        errors = io.StringIO()

        with (
            patch("main.Settings.from_environment", return_value=settings),
            patch("main.configure_logging", return_value=logger),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            result = main(["prompt", "Build a calculator."])

        self.assertEqual(result, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("GEMINI_API_KEY is missing", errors.getvalue())
        self.assertNotIn("orchestrator", __import__("main").__dict__)

    def test_check_config_does_not_create_gemini_agent(self) -> None:
        logger = Mock(spec=logging.Logger)
        output = io.StringIO()
        settings = Settings.from_environment({})

        with (
            patch("main.Settings.from_environment", return_value=settings),
            patch("main.configure_logging", return_value=logger),
            patch("main.GeminiPromptAgent") as agent_factory,
            redirect_stdout(output),
        ):
            result = main(["--check-config"])

        self.assertEqual(result, 0)
        self.assertIn("No AI provider calls were made", output.getvalue())
        agent_factory.assert_not_called()

    def test_transient_gemini_failure_shows_safe_message_without_traceback(self) -> None:
        settings = Settings.from_environment({"GEMINI_API_KEY": "unit-test-key"})
        logger = Mock(spec=logging.Logger)
        output = io.StringIO()
        errors = io.StringIO()
        error = GeminiTemporarilyUnavailableError(3)

        with (
            patch("main.Settings.from_environment", return_value=settings),
            patch("main.configure_logging", return_value=logger),
            patch("main.GeminiPromptAgent") as agent_factory,
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            agent_factory.return_value.create_specification.side_effect = error
            result = main(["prompt", "Build a calculator."])

        self.assertEqual(result, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(
            errors.getvalue().strip(),
            "Gemini is temporarily unavailable after 3 attempts. Please try again later.",
        )
        self.assertNotIn("Traceback", errors.getvalue())
        logged = repr(logger.warning.call_args)
        self.assertIn("GeminiTemporarilyUnavailableError", logged)
        self.assertIn("3", logged)
        self.assertNotIn("unit-test-key", logged)
        self.assertNotIn("Build a calculator.", logged)


class GenerateCliTests(unittest.TestCase):
    def test_test_command_uses_only_an_existing_run(self) -> None:
        run_id = "378a99a9-6bf1-4ef7-bfce-9f720d1e40cd"
        settings = Settings.from_environment({})
        logger = Mock(spec=logging.Logger)
        sandbox_result = TestResult(
            passed=False,
            exit_code=1,
            stdout="",
            stderr="AssertionError",
            duration_seconds=0.4,
            tests_passed=0,
            tests_failed=1,
            runtime="docker",
            container_image="procoder-sandbox:local",
        )
        output = io.StringIO()
        errors = io.StringIO()

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch("main.PROJECT_ROOT", Path(temporary_directory)),
            patch("main.Settings.from_environment", return_value=settings),
            patch("main.configure_logging", return_value=logger),
            patch("main.DockerSandboxRunner") as sandbox_factory,
            patch("main.GeminiPromptAgent") as prompt_factory,
            patch("main.OpenAICodexCodingAgent") as coding_factory,
            patch("main.resolve_codex_executable") as codex_resolver,
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            sandbox_factory.return_value.run_project.return_value = sandbox_result
            result = main(["test", run_id])

        self.assertEqual(result, 1)
        sandbox_factory.return_value.run_project.assert_called_once_with(
            Path(temporary_directory)
            / "workspace"
            / "generated_code"
            / run_id
        )
        self.assertIn("TEST RESULT: FAILED", output.getvalue())
        self.assertIn("AssertionError", output.getvalue())
        self.assertEqual(errors.getvalue(), "")
        prompt_factory.assert_not_called()
        coding_factory.assert_not_called()
        codex_resolver.assert_not_called()
        logged = repr(logger.info.call_args)
        self.assertIn("sandbox_test_completed", logged)
        self.assertIn(run_id, logged)

    def test_invalid_test_identifier_is_not_written_to_logs(self) -> None:
        sensitive_identifier = "not-a-run-id-secret-marker"
        settings = Settings.from_environment({})
        logger = Mock(spec=logging.Logger)
        failure = TestResult(
            passed=False,
            exit_code=None,
            stdout="",
            stderr="Project rejected: run ID is invalid.",
            duration_seconds=0.0,
        )

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch("main.PROJECT_ROOT", Path(temporary_directory)),
            patch("main.Settings.from_environment", return_value=settings),
            patch("main.configure_logging", return_value=logger),
            patch("main.DockerSandboxRunner") as sandbox_factory,
            redirect_stdout(io.StringIO()),
        ):
            sandbox_factory.return_value.run_project.return_value = failure
            result = main(["test", sensitive_identifier])

        self.assertEqual(result, 1)
        event_data = logger.info.call_args.kwargs["extra"]["event_data"]
        self.assertIsNone(event_data["run_id"])
        self.assertNotIn(sensitive_identifier, repr(logger.info.call_args))

    def test_run_command_generates_then_tests_in_docker(self) -> None:
        request = "Create a Python prime checker."
        run_id = "378a99a9-6bf1-4ef7-bfce-9f720d1e40cd"
        specification = Specification(
            request=request,
            summary="Create a prime checker.",
            provider="google-gemini",
        )
        generated = GeneratedCode(
            files={
                "prime.py": "def is_prime(number): return number > 1",
                "test_prime.py": "import unittest",
            },
            provider="openai-codex-cli",
            language="Python",
        )
        test_result = TestResult(
            passed=True,
            exit_code=0,
            stdout="Ran 1 test",
            stderr="OK",
            duration_seconds=0.5,
            tests_passed=1,
            tests_failed=0,
            runtime="docker",
            container_image="procoder-sandbox:local",
        )
        settings = Settings.from_environment(
            {"GEMINI_API_KEY": "unit-test-key", "NVIDIA_API_KEY": "nvidia-test-key"}
        )
        logger = Mock(spec=logging.Logger)
        output = io.StringIO()
        errors = io.StringIO()

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            with (
                patch("main.PROJECT_ROOT", project_root),
                patch("main.uuid.uuid4", return_value=UUID(run_id)),
                patch("main.Settings.from_environment", return_value=settings),
                patch("main.configure_logging", return_value=logger),
                patch("main.GeminiPromptAgent") as prompt_factory,
                patch("main.OpenAICodexCodingAgent") as coding_factory,
                patch("main.resolve_codex_executable", return_value="codex"),
                patch("main.DockerSandboxRunner") as sandbox_factory,
                patch("main.NvidiaNimReviewAgent") as review_factory,
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                prompt_factory.return_value.create_specification.return_value = specification
                coding_factory.return_value.generate.return_value = generated
                sandbox_factory.return_value.run_project.return_value = test_result
                review_factory.return_value.review.return_value = ReviewResult(
                    verdict=ReviewVerdict.PASS,
                    summary="Implementation satisfies the specification.",
                    specification_satisfied=True,
                    code_quality_findings=(),
                    test_analysis="Docker reported PASSED.",
                    recommended_actions=(),
                    repair_required=False,
                    confidence=0.9,
                    tests_passed=True,
                )
                result = main(["run", request])

        self.assertEqual(result, 0)
        prompt_factory.return_value.create_specification.assert_called_once_with(request)
        coding_factory.return_value.generate.assert_called_once_with(specification)
        sandbox_factory.return_value.run_project.assert_called_once_with(
            project_root / "workspace" / "generated_code" / run_id,
            language="Python",
        )
        self.assertIn("INITIAL TEST: PASSED", output.getvalue())
        self.assertNotIn("Generated files were not executed.", output.getvalue())
        self.assertIn("FINAL NVIDIA: PASS", output.getvalue())
        review_factory.return_value.review.assert_called_once()
        self.assertEqual(errors.getvalue(), "")
        self.assertNotIn("unit-test-key", repr(logger.info.call_args_list))

    def test_generate_calls_prompt_then_coding_and_only_writes_files(self) -> None:
        request = "Create a Python prime checker."
        specification = Specification(
            request=request,
            summary="Create a prime checker.",
            requirements=("Accept an integer.",),
            acceptance_criteria=("Report whether the integer is prime.",),
            provider="google-gemini",
            model="gemini-test",
        )
        generated = GeneratedCode(
            files={
                "prime.py": "def is_prime(number): return number > 1",
                "test_prime.py": "def test_prime(): assert True",
            },
            provider="openai-codex-cli",
            model="codex-test",
            language="Python",
            summary="A prime checker and unit tests.",
            duration_seconds=0.25,
        )
        settings = Settings.from_environment({"GEMINI_API_KEY": "test-secret"})
        logger = Mock(spec=logging.Logger)
        output = io.StringIO()
        errors = io.StringIO()

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            with (
                patch("main.PROJECT_ROOT", project_root),
                patch("main.Settings.from_environment", return_value=settings),
                patch("main.configure_logging", return_value=logger),
                patch("main.GeminiPromptAgent") as prompt_factory,
                patch("main.OpenAICodexCodingAgent") as coding_factory,
                patch("main.resolve_codex_executable", return_value="codex"),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                prompt_factory.return_value.create_specification.return_value = (
                    specification
                )
                coding_factory.return_value.generate.return_value = generated
                result = main(["generate", request])

            self.assertEqual(result, 0)
            prompt_factory.return_value.create_specification.assert_called_once_with(
                request
            )
            coding_factory.return_value.generate.assert_called_once_with(specification)
            self.assertIn("prime.py", output.getvalue())
            self.assertIn("test_prime.py", output.getvalue())
            self.assertIn("Generated files were not executed.", output.getvalue())
            self.assertEqual(errors.getvalue(), "")
            self.assertEqual(
                len(list((project_root / "workspace" / "generated_code").iterdir())),
                1,
            )
            self.assertNotIn("orchestrator", __import__("main").__dict__)
            self.assertNotIn("sandbox", __import__("main").__dict__)
            logged = repr(logger.info.call_args_list + logger.warning.call_args_list)
            self.assertIn("openai-codex-cli", logged)
            self.assertIn("files_created", logged)
            self.assertNotIn(request, logged)
            self.assertNotIn("test-secret", logged)

    def test_missing_codex_cli_shows_safe_error_without_traceback(self) -> None:
        settings = Settings.from_environment({"GEMINI_API_KEY": "test-secret"})
        logger = Mock(spec=logging.Logger)
        output = io.StringIO()
        errors = io.StringIO()
        specification = Specification(
            request="Create a small app.",
            summary="A small app.",
            provider="google-gemini",
            model="gemini-test",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch("main.PROJECT_ROOT", Path(temporary_directory)),
                patch("main.Settings.from_environment", return_value=settings),
                patch("main.configure_logging", return_value=logger),
                patch("main.GeminiPromptAgent") as prompt_factory,
                patch("main.OpenAICodexCodingAgent") as coding_factory,
                patch("main.resolve_codex_executable", return_value=None),
                redirect_stdout(output),
                redirect_stderr(errors),
            ):
                prompt_factory.return_value.create_specification.return_value = (
                    specification
                )
                result = main(["generate", "Create a small app."])

        self.assertEqual(result, 1)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("Codex CLI was not found", errors.getvalue())
        self.assertNotIn("Traceback", errors.getvalue())
        prompt_factory.return_value.create_specification.assert_not_called()
        coding_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
