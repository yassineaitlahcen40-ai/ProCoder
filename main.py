"""ProCoder command-line entry point."""

import argparse
import json
import logging
import re
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from agents.coding_agent.errors import CodingAgentError, CodexCliNotFoundError
from agents.coding_agent.providers.openai_codex import (
    OpenAICodexCodingAgent,
    resolve_codex_executable,
)
from agents.prompt_agent.errors import PromptAgentError
from agents.prompt_agent.providers.gemini import GeminiPromptAgent
from agents.review_agent.errors import ReviewAgentError
from agents.review_agent.providers.nvidia_nim import NvidiaNimReviewAgent
from core.config import PROJECT_ROOT, Settings
from core.logging import configure_logging
from core.models import (
    FinalRunResult,
    GeneratedCode,
    ReviewResult,
    Specification,
    TestResult,
)
from orchestrator.repair_controller import RepairController
from sandbox.runner import DockerSandboxRunner
from workspace.generated_project import GeneratedProjectWriter
from workspace.run_metadata import RunMetadataError, RunMetadataStore


_RUN_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ProCoder text-based agent pipeline")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate local settings without making provider calls",
    )
    commands = parser.add_subparsers(dest="command")
    prompt_parser = commands.add_parser(
        "prompt",
        help="turn a user request into a structured specification with Gemini",
    )
    prompt_parser.add_argument("request", help="informal software request")
    generate_parser = commands.add_parser(
        "generate",
        help="create a specification and generate files with the local Codex CLI",
    )
    generate_parser.add_argument("request", help="informal software request")
    test_parser = commands.add_parser(
        "test",
        help="test an existing generated project in the Docker sandbox",
    )
    test_parser.add_argument("run_id", help="UUID of an existing generated project")
    review_parser = commands.add_parser(
        "review",
        help="review an existing run and saved Docker results with NVIDIA NIM",
    )
    review_parser.add_argument("run_id", help="UUID of an existing generated project")
    run_parser = commands.add_parser(
        "run",
        help="generate a project and test it in Docker",
    )
    run_parser.add_argument("request", help="informal software request")
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_environment()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    logger = configure_logging(PROJECT_ROOT / "logs" / "procoder.jsonl")
    if args.check_config:
        logger.info(
            "configuration_validated",
            extra={
                "event_data": {
                    "event": "configuration_validated",
                    "max_repair_attempts": settings.max_repair_attempts,
                    "gemini_model": settings.gemini_model,
                    "gemini_fallback_model": settings.gemini_fallback_model,
                    "gemini_timeout": settings.gemini_timeout,
                    "gemini_max_retries": settings.gemini_max_retries,
                    "nvidia_model": settings.nvidia_model,
                    "nvidia_base_url": settings.nvidia_base_url,
                    "nvidia_timeout": settings.nvidia_timeout,
                    "nvidia_max_retries": settings.nvidia_max_retries,
                    "sandbox_timeout_seconds": settings.sandbox_timeout_seconds,
                    "sandbox_memory_limit": settings.sandbox_memory_limit,
                    "sandbox_cpu_limit": settings.sandbox_cpu_limit,
                    "sandbox_pids_limit": settings.sandbox_pids_limit,
                },
            },
        )
        print("Configuration is valid. No AI provider calls were made.")
        return 0

    if args.command == "prompt":
        return run_prompt_command(args.request, settings, logger)
    if args.command == "generate":
        return run_generate_command(args.request, settings, logger)
    if args.command == "test":
        return run_test_command(args.run_id, settings, logger)
    if args.command == "review":
        return run_review_command(args.run_id, settings, logger)
    if args.command == "run":
        return run_generate_command(
            args.request, settings, logger, execute_tests=True
        )

    print(
        "The full AI workflow is not available yet. Use 'prompt', 'generate', "
        "'test', or 'run', or --check-config to validate settings.",
        file=sys.stderr,
    )
    return 2


def run_prompt_command(
    request: str,
    settings: Settings,
    logger: logging.Logger,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout if stdout is not None else sys.stdout
    error_output = stderr if stderr is not None else sys.stderr
    try:
        specification = GeminiPromptAgent(settings).create_specification(request)
    except PromptAgentError as exc:
        logger.warning(
            "gemini_prompt_failed",
            extra={
                "event_data": {
                    "event": "gemini_prompt_failed",
                    "error_code": type(exc).__name__,
                    "attempts": getattr(exc, "attempts", None),
                    "status_code": getattr(exc, "status_code", None),
                    "category": getattr(exc, "category", None),
                    "model": getattr(exc, "model", None),
                    "model_role": getattr(exc, "model_role", None),
                    "fallback_attempted": getattr(
                        exc, "fallback_attempted", False
                    ),
                    "provider_reason": getattr(exc, "provider_reason", None),
                }
            },
        )
        print(exc.safe_message, file=error_output)
        return 1

    logger.info(
        "gemini_specification_created",
        extra={
            "event_data": {
                "event": "gemini_specification_created",
                "provider": specification.provider,
                "model": specification.model,
                "requirement_count": len(specification.requirements),
                "acceptance_criteria_count": len(specification.acceptance_criteria),
                "tokens_used": specification.tokens_used,
            }
        },
    )
    print(format_specification(specification), file=output)
    return 0


def run_generate_command(
    request: str,
    settings: Settings,
    logger: logging.Logger,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    project_root: Path | None = None,
    execute_tests: bool = False,
) -> int:
    output = stdout if stdout is not None else sys.stdout
    error_output = stderr if stderr is not None else sys.stderr
    run_id = str(uuid.uuid4())
    started = time.perf_counter()
    phase = "prompt"
    specification: Specification | None = None
    generated: GeneratedCode | None = None
    if resolve_codex_executable(settings.codex_cli) is None:
        error = CodexCliNotFoundError()
        logger.warning(
            "code_generation_failed",
            extra={
                "event_data": {
                    "event": "code_generation_failed",
                    "run_id": run_id,
                    "phase": "codex_preflight",
                    "generation_duration_seconds": time.perf_counter() - started,
                    "success": False,
                    "error_code": type(error).__name__,
                },
            },
        )
        print(error.safe_message, file=error_output)
        return 1

    try:
        specification = GeminiPromptAgent(settings).create_specification(request)
        phase = "coding"
        writer = GeneratedProjectWriter(project_root or PROJECT_ROOT)
        run_directory = writer.create_run_directory(run_id)
        generated = OpenAICodexCodingAgent(settings, run_directory).generate(
            specification
        )
        phase = "writing"
        result = writer.write(run_id, generated)
        metadata_store = RunMetadataStore(project_root or PROJECT_ROOT)
        metadata_store.save_specification(run_id, specification)
        metadata_store.save_generated(run_id, result)
    except (PromptAgentError, CodingAgentError, RunMetadataError) as exc:
        duration = time.perf_counter() - started
        logger.warning(
            "code_generation_failed",
            extra={
                "event_data": {
                    "event": "code_generation_failed",
                    "run_id": run_id,
                    "phase": phase,
                    "prompt_provider": (
                        specification.provider if specification is not None else None
                    ),
                    "prompt_model": (
                        specification.model if specification is not None else None
                    ),
                    "prompt_tokens_used": (
                        specification.tokens_used
                        if specification is not None
                        else None
                    ),
                    "coding_provider": (
                        generated.provider if generated is not None else None
                    ),
                    "coding_model": (
                        generated.model if generated is not None else None
                    ),
                    "generation_duration_seconds": duration,
                    "coding_duration_seconds": (
                        generated.duration_seconds if generated is not None else None
                    ),
                    "coding_tokens_used": (
                        generated.tokens_used if generated is not None else None
                    ),
                    "success": False,
                    "error_code": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                    "category": getattr(exc, "category", None),
                    "model": getattr(exc, "model", None),
                    "model_role": getattr(exc, "model_role", None),
                    "fallback_attempted": getattr(
                        exc, "fallback_attempted", False
                    ),
                    "provider_reason": getattr(exc, "provider_reason", None),
                },
            },
        )
        print(getattr(exc, "safe_message", RunMetadataError.safe_message), file=error_output)
        return 1

    duration = time.perf_counter() - started
    logger.info(
        "code_generation_completed",
        extra={
            "event_data": {
                "event": "code_generation_completed",
                "run_id": run_id,
                "prompt_provider": specification.provider,
                "prompt_model": specification.model,
                "prompt_tokens_used": specification.tokens_used,
                "coding_provider": result.provider,
                "coding_model": result.model,
                "files_created": list(result.files_created),
                "files_modified": list(result.files_modified),
                "generation_duration_seconds": duration,
                "coding_duration_seconds": result.duration_seconds,
                "coding_tokens_used": result.tokens_used,
                "success": True,
            },
        },
    )
    print(f"RUN ID:\n{run_id}\n", file=output)
    print(f"SUMMARY:\n{result.summary or specification.summary}\n", file=output)
    print(f"LANGUAGE:\n{result.language or 'Not specified'}\n", file=output)
    print(
        "PROMPT AGENT:\n"
        f"{specification.provider or 'unknown'}"
        f"{_model_suffix(specification.model)}\n",
        file=output,
    )
    print(
        "CODING AGENT:\n"
        f"{result.provider or 'unknown'}{_model_suffix(result.model)}\n",
        file=output,
    )
    print("FILES CREATED:", file=output)
    for path in result.files_created:
        print(f"- {path}", file=output)
    print("FILES MODIFIED:", file=output)
    if result.files_modified:
        for path in result.files_modified:
            print(f"- {path}", file=output)
    else:
        print("- None", file=output)
    print(
        f"\nPROJECT DIRECTORY:\nworkspace/generated_code/{run_id}\n"
        f"GENERATION TIME:\n{duration:.2f}s\n"
        + (
            "Generated files will now be tested inside Docker."
            if execute_tests
            else "Generated files were not executed."
        ),
        file=output,
    )
    if not execute_tests:
        return 0
    try:
        root = project_root or PROJECT_ROOT
        repair_controller = RepairController(
            settings,
            OpenAICodexCodingAgent(
                settings, root / "workspace" / "generated_code" / run_id
            ),
            lambda: NvidiaNimReviewAgent(settings, logger=logger),
            DockerSandboxRunner(settings, project_root=root),
            GeneratedProjectWriter(root),
            RunMetadataStore(root),
            project_root=root,
            logger=logger,
        )
        final_result = repair_controller.run(
            run_id,
            specification,
            result,
            workflow_started=started,
            generation_duration_seconds=duration,
        )
    except (CodingAgentError, ReviewAgentError, RunMetadataError) as exc:
        logger.error(
            "repair_workflow_start_failed",
            extra={
                "event_data": {
                    "event": "repair_workflow_start_failed",
                    "run_id": run_id,
                    "success": False,
                    "error_code": type(exc).__name__,
                }
            },
        )
        print(getattr(exc, "safe_message", RunMetadataError.safe_message), file=error_output)
        return 1
    _print_final_run_result(final_result, output)
    return 0 if final_result.success else 1


def run_test_command(
    run_id: str,
    settings: Settings,
    logger: logging.Logger,
    *,
    stdout: TextIO | None = None,
    project_root: Path | None = None,
) -> int:
    output = stdout if stdout is not None else sys.stdout
    root = project_root or PROJECT_ROOT
    test_result = DockerSandboxRunner(
        settings, project_root=root
    ).run_project(root / "workspace" / "generated_code" / run_id)
    if _RUN_ID_PATTERN.fullmatch(run_id):
        try:
            RunMetadataStore(root).save_test_result(run_id, test_result)
        except RunMetadataError as exc:
            print_test_result(test_result, output)
            print(exc.safe_message, file=sys.stderr)
            return 1
    _log_test_result(logger, run_id, test_result)
    print_test_result(test_result, output)
    return 0 if test_result.passed else 1


def run_review_command(
    run_id: str,
    settings: Settings,
    logger: logging.Logger,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    project_root: Path | None = None,
) -> int:
    output = stdout if stdout is not None else sys.stdout
    error_output = stderr if stderr is not None else sys.stderr
    root = project_root or PROJECT_ROOT
    try:
        metadata = RunMetadataStore(root)
        specification = metadata.load_specification(run_id)
        generated_metadata = metadata.load_generated_metadata(run_id)
        test_result = metadata.load_test_result(run_id)
        files = GeneratedProjectWriter(root).read_files(run_id)
        generated = GeneratedCode(
            files=files,
            provider=generated_metadata.provider,
            model=generated_metadata.model,
            tokens_used=generated_metadata.tokens_used,
            language=generated_metadata.language,
            summary=generated_metadata.summary,
            files_created=generated_metadata.files_created,
            files_modified=generated_metadata.files_modified,
            duration_seconds=generated_metadata.duration_seconds,
        )
        review = NvidiaNimReviewAgent(settings, logger=logger).review(
            specification, generated, test_result
        )
    except (RunMetadataError, CodingAgentError, ReviewAgentError) as exc:
        logger.warning(
            "nvidia_review_failed",
            extra={
                "event_data": {
                    "event": "nvidia_review_failed",
                    "run_id": run_id if _RUN_ID_PATTERN.fullmatch(run_id) else None,
                    "provider": "nvidia-nim",
                    "success": False,
                    "error_code": type(exc).__name__,
                }
            },
        )
        print(exc.safe_message, file=error_output)
        return 1

    _log_review_result(logger, run_id, test_result, review)
    print(
        f"RUN ID: {run_id}\n"
        f"REVIEW RESULT:\n"
        f"{json.dumps(review.model_dump(mode='json'), indent=2, ensure_ascii=True)}",
        file=output,
    )
    return 0 if review.passed else 1


def _log_review_result(
    logger: logging.Logger,
    run_id: str,
    test_result: TestResult,
    review: ReviewResult,
) -> None:
    logger.info(
        "nvidia_review_completed",
        extra={
            "event_data": {
                "event": "nvidia_review_completed",
                "run_id": run_id if _RUN_ID_PATTERN.fullmatch(run_id) else None,
                "provider": review.provider,
                "model": review.model,
                "duration_seconds": review.duration_seconds,
                "verdict": review.verdict.value,
                "repair_required": review.repair_required,
                "tokens_used": review.tokens_used,
                "docker_passed": test_result.passed,
                "docker_exit_code": test_result.exit_code,
                "source_truncated": review.source_truncated,
                "stdout_truncated": review.stdout_truncated,
                "stderr_truncated": review.stderr_truncated,
                "success": True,
            }
        },
    )


def _print_final_run_result(result: FinalRunResult, output: TextIO) -> None:
    print(
        f"INITIAL TEST: {'PASSED' if result.initial_docker_result.passed else 'FAILED'}",
        file=output,
    )
    if result.initial_nvidia_verdict is not None:
        print(f"INITIAL NVIDIA: {result.initial_nvidia_verdict.value}", file=output)
    for attempt in result.repair_history:
        print(f"REPAIR ATTEMPT {attempt.attempt_number}", file=output)
        print(
            "FILES CHANGED: "
            + (
                ", ".join(
                    (*attempt.files_created, *attempt.files_modified, *attempt.files_deleted)
                )
                or "None"
            ),
            file=output,
        )
        print(
            f"RETEST: {'PASSED' if attempt.docker_after and attempt.docker_after.passed else 'FAILED'}",
            file=output,
        )
        if attempt.review_after is not None:
            print(f"NVIDIA: {attempt.review_after.verdict.value}", file=output)
    print(
        f"\nFINAL RESULT: {'SUCCESS' if result.success else 'FAILED'}\n"
        f"FINAL DOCKER: {'PASSED' if result.final_docker_result.passed else 'FAILED'}\n"
        f"FINAL NVIDIA: "
        f"{result.final_nvidia_verdict.value if result.final_nvidia_verdict else 'NOT AVAILABLE'}\n"
        f"REPAIR ATTEMPTS: {result.total_repair_attempts}\n"
        f"TERMINATION: {result.termination_reason.value}\n"
        f"DURATION: {result.total_duration_seconds:.2f}s\n"
        f"TOKENS: {result.total_tokens_used if result.total_tokens_used is not None else 'Unknown'}",
        file=output,
    )


def _log_test_result(
    logger: logging.Logger, run_id: str, result: TestResult
) -> None:
    logger.info(
        "sandbox_test_completed",
        extra={
            "event_data": {
                "event": "sandbox_test_completed",
                "run_id": run_id if _RUN_ID_PATTERN.fullmatch(run_id) else None,
                "passed": result.passed,
                "exit_code": result.exit_code,
                "command": list(result.command),
                "duration_seconds": result.duration_seconds,
                "timed_out": result.timed_out,
                "output_truncated": result.output_truncated,
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
                "tests_skipped": result.tests_skipped,
                "runtime": result.runtime,
                "container_image": result.container_image,
            }
        },
    )


def print_test_result(result: TestResult, output: TextIO) -> None:
    command = " ".join(result.command) if result.command else "Not run"
    print(
        f"TEST RESULT: {'PASSED' if result.passed else 'FAILED'}\n"
        f"EXIT CODE: {result.exit_code if result.exit_code is not None else 'Not available'}\n"
        f"TEST COMMAND: {command}\n"
        f"TESTS PASSED: {result.tests_passed if result.tests_passed is not None else 'Unknown'}\n"
        f"TESTS FAILED: {result.tests_failed if result.tests_failed is not None else 'Unknown'}\n"
        f"TIMEOUT: {'Yes' if result.timed_out else 'No'}\n"
        f"DURATION: {result.duration_seconds:.2f}s\n"
        f"RUNTIME: {result.runtime or 'Not started'}\n"
        f"CONTAINER IMAGE: {result.container_image or 'Not available'}\n"
        f"STDOUT:\n{result.stdout or '(empty)'}\n"
        f"STDERR:\n{result.stderr or '(empty)'}",
        file=output,
    )


def _model_suffix(model: str | None) -> str:
    return f" ({model})" if model else ""


def format_specification(specification: Specification) -> str:
    requirements = "\n".join(
        f"{index}. {item}"
        for index, item in enumerate(specification.requirements, start=1)
    )
    criteria = "\n".join(
        f"{index}. {item}"
        for index, item in enumerate(specification.acceptance_criteria, start=1)
    )
    return (
        f"REQUEST:\n{specification.request}\n\n"
        f"SUMMARY:\n{specification.summary}\n\n"
        f"REQUIREMENTS:\n{requirements}\n\n"
        f"ACCEPTANCE CRITERIA:\n{criteria}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
