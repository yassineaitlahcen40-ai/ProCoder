"""Bounded, provider-independent prompt/code/test/review/repair workflow."""

import logging
import time
import uuid
from typing import Protocol

from agents.coding_agent.base import CodingAgent
from agents.prompt_agent.base import PromptAgent
from agents.repair_agent.base import RepairAgent
from agents.review_agent.base import ReviewAgent
from core.models import (
    GeneratedCode,
    ReviewDecision,
    Specification,
    TestResult,
    WorkflowResult,
)


class SandboxRunner(Protocol):
    def run_project(
        self, code: GeneratedCode, specification: Specification
    ) -> TestResult:
        """Run deterministic checks in an isolated environment."""


class Orchestrator:
    def __init__(
        self,
        prompt_agent: PromptAgent,
        coding_agent: CodingAgent,
        sandbox_runner: SandboxRunner,
        review_agent: ReviewAgent,
        repair_agent: RepairAgent,
        *,
        max_repair_attempts: int,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must not be negative.")
        self._prompt_agent = prompt_agent
        self._coding_agent = coding_agent
        self._sandbox_runner = sandbox_runner
        self._review_agent = review_agent
        self._repair_agent = repair_agent
        self._max_repair_attempts = max_repair_attempts
        self._logger = logger or logging.getLogger("procoder")

    def run(self, request: str) -> WorkflowResult:
        if not request.strip():
            raise ValueError("The user request must not be empty.")

        run_id = str(uuid.uuid4())
        started = time.perf_counter()
        tokens_used = 0
        has_token_usage = False
        history: list[TestResult] = []
        self._emit(
            "workflow_started",
            run_id=run_id,
            max_repair_attempts=self._max_repair_attempts,
        )

        specification = self._prompt_agent.create_specification(request)
        tokens_used, has_token_usage = self._add_usage(
            specification.tokens_used, tokens_used, has_token_usage
        )
        self._emit(
            "specification_created",
            run_id=run_id,
            provider=specification.provider,
            model=specification.model,
            tokens_used=specification.tokens_used,
        )

        code = self._coding_agent.generate(specification)
        tokens_used, has_token_usage = self._add_usage(
            code.tokens_used, tokens_used, has_token_usage
        )
        self._emit(
            "code_generated",
            run_id=run_id,
            provider=code.provider,
            model=code.model,
            tokens_used=code.tokens_used,
            file_count=len(code.files),
        )

        repair_attempts = 0
        review: ReviewDecision
        while True:
            test_result = self._sandbox_runner.run_project(code, specification)
            history.append(test_result)
            review = self._review_agent.review(specification, code, test_result)
            tokens_used, has_token_usage = self._add_usage(
                review.tokens_used, tokens_used, has_token_usage
            )
            iteration = len(history)
            passed = test_result.passed and review.passed
            self._emit(
                "iteration_completed",
                run_id=run_id,
                iteration=iteration,
                repair_attempts=repair_attempts,
                passed=passed,
                tests_passed=test_result.passed,
                review_passed=review.passed,
                exit_code=test_result.exit_code,
                timed_out=test_result.timed_out,
                output_truncated=test_result.output_truncated,
                duration_seconds=test_result.duration_seconds,
                stdout_bytes=len(test_result.stdout.encode("utf-8")),
                stderr_bytes=len(test_result.stderr.encode("utf-8")),
                provider=review.provider,
                model=review.model,
                tokens_used=review.tokens_used,
            )

            if passed:
                return self._finish(
                    run_id,
                    True,
                    "passed",
                    started,
                    history,
                    code,
                    review,
                    repair_attempts,
                    tokens_used if has_token_usage else None,
                )

            if repair_attempts >= self._max_repair_attempts:
                return self._finish(
                    run_id,
                    False,
                    "max_repair_attempts_reached",
                    started,
                    history,
                    code,
                    review,
                    repair_attempts,
                    tokens_used if has_token_usage else None,
                )

            instructions = self._repair_agent.create_instructions(
                specification, code, test_result, review
            )
            tokens_used, has_token_usage = self._add_usage(
                instructions.tokens_used, tokens_used, has_token_usage
            )
            code = self._coding_agent.repair(specification, code, instructions)
            tokens_used, has_token_usage = self._add_usage(
                code.tokens_used, tokens_used, has_token_usage
            )
            repair_attempts += 1
            self._emit(
                "repair_created",
                run_id=run_id,
                repair_attempt=repair_attempts,
                repair_provider=instructions.provider,
                repair_model=instructions.model,
                repair_tokens_used=instructions.tokens_used,
                coding_provider=code.provider,
                coding_model=code.model,
                coding_tokens_used=code.tokens_used,
            )

    def _finish(
        self,
        run_id: str,
        passed: bool,
        status: str,
        started: float,
        history: list[TestResult],
        code: GeneratedCode,
        review: ReviewDecision,
        repair_attempts: int,
        tokens_used: int | None,
    ) -> WorkflowResult:
        duration = time.perf_counter() - started
        result = WorkflowResult(
            run_id=run_id,
            passed=passed,
            status=status,
            iterations=len(history),
            repair_attempts=repair_attempts,
            duration_seconds=duration,
            test_results=tuple(history),
            final_code=code,
            final_review=review,
            tokens_used=tokens_used,
        )
        self._emit(
            "workflow_completed",
            run_id=run_id,
            status=status,
            passed=passed,
            iterations=result.iterations,
            repair_attempts=repair_attempts,
            duration_seconds=duration,
            tokens_used=tokens_used,
        )
        return result

    def _emit(self, event: str, **fields: object) -> None:
        self._logger.info(event, extra={"event_data": {"event": event, **fields}})

    @staticmethod
    def _add_usage(
        usage: int | None, total: int, has_usage: bool
    ) -> tuple[int, bool]:
        if usage is None:
            return total, has_usage
        if usage < 0:
            raise ValueError("Agent token usage must not be negative.")
        return total + usage, True
