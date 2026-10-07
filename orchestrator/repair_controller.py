"""Deterministic, bounded control loop for Docker testing and code repair."""

import logging
import time
from dataclasses import replace
from collections.abc import Callable
from pathlib import Path
from typing import cast

from agents.coding_agent.base import CodingAgent
from agents.coding_agent.errors import (
    CodingAgentError,
    CodexResponseError,
)
from agents.review_agent.base import ReviewAgent
from agents.review_agent.errors import ReviewAgentError
from core.config import Settings
from core.models import (
    CodeRepair,
    FinalRunResult,
    GeneratedCode,
    RepairAttemptRecord,
    RepairInstructions,
    ReviewResult,
    ReviewVerdict,
    Specification,
    TerminationReason,
    TestResult,
)
from sandbox.runner import DockerSandboxRunner
from workspace.generated_project import GeneratedProjectError, GeneratedProjectWriter
from workspace.run_metadata import RunMetadataError, RunMetadataStore


class RepairController:
    def __init__(
        self,
        settings: Settings,
        coding_agent: CodingAgent,
        review_agent: ReviewAgent | Callable[[], ReviewAgent],
        sandbox_runner: DockerSandboxRunner,
        writer: GeneratedProjectWriter,
        metadata: RunMetadataStore,
        *,
        project_root: Path,
        logger: logging.Logger,
    ) -> None:
        self._settings = settings
        self._coding_agent = coding_agent
        self._review_agent = review_agent
        self._review_provider: ReviewAgent | None = (
            cast(ReviewAgent, review_agent)
            if callable(getattr(review_agent, "review", None))
            else None
        )
        self._sandbox_runner = sandbox_runner
        self._writer = writer
        self._metadata = metadata
        self._project_root = project_root
        self._logger = logger

    def run(
        self,
        run_id: str,
        specification: Specification,
        generated: GeneratedCode,
        *,
        workflow_started: float | None = None,
        generation_duration_seconds: float | None = None,
    ) -> FinalRunResult:
        started = workflow_started or time.perf_counter()
        run_directory = (
            self._project_root / "workspace" / "generated_code" / run_id
        )
        current_code = GeneratedCode(
            files=self._writer.read_files(run_id),
            provider=generated.provider,
            model=generated.model,
            tokens_used=generated.tokens_used,
            language=generated.language,
            summary=generated.summary,
            files_created=generated.files_created,
            files_modified=generated.files_modified,
            duration_seconds=generated.duration_seconds,
        )
        initial_test = self._sandbox_runner.run_project(
            run_directory, language=current_code.language
        )
        try:
            self._metadata.save_test_result(run_id, initial_test)
        except RunMetadataError as exc:
            self._log_termination(
                run_id, TerminationReason.PERSISTENCE_ERROR, exc
            )
            return self._finish(
                run_id,
                specification,
                generated,
                initial_test,
                initial_test,
                [],
                [],
                None,
                0.0,
                started,
                TerminationReason.PERSISTENCE_ERROR,
                generation_duration_seconds,
            )
        history: list[RepairAttemptRecord] = []
        review_duration = 0.0
        repair_duration = 0.0
        repair_tokens = 0
        has_repair_tokens = False
        reviews: list[ReviewResult] = []
        latest_review: ReviewResult | None = None
        final_test = initial_test
        termination = (
            TerminationReason.SANDBOX_ERROR
            if initial_test.infrastructure_error
            else TerminationReason.PROVIDER_ERROR
        )

        if not initial_test.infrastructure_error:
            try:
                review = self._get_review_agent().review(
                    specification, current_code, initial_test
                )
                reviews.append(review)
                latest_review = review
                review_duration += review.duration_seconds or 0.0
            except ReviewAgentError as exc:
                self._log_termination(run_id, TerminationReason.PROVIDER_ERROR, exc)
            else:
                if _is_success(initial_test, review):
                    termination = TerminationReason.SUCCESS
                else:
                    termination = TerminationReason.MAX_REPAIR_ATTEMPTS

                while (
                    termination is not TerminationReason.SUCCESS
                    and len(history) < self._settings.max_repair_attempts
                    and not final_test.infrastructure_error
                ):
                    attempt_number = len(history) + 1
                    before_test = final_test
                    before_review = reviews[-1]
                    repair_started = time.perf_counter()
                    self._emit_attempt_started(run_id, attempt_number)
                    instructions = RepairInstructions(
                        instructions="Repair only the demonstrated specification or test failures.",
                        test_result=before_test,
                        review_result=before_review,
                        attempt_number=attempt_number,
                    )
                    repair: CodeRepair | None = None
                    patch_applied = False
                    repair_elapsed = 0.0
                    after_review: ReviewResult | None = None
                    try:
                        repair = self._coding_agent.repair(
                            specification, current_code, instructions
                        )
                        if repair.tokens_used is not None:
                            repair_tokens += repair.tokens_used
                            has_repair_tokens = True
                        self._writer.apply_repair(run_id, repair)
                        patch_applied = True
                        current_code = self._read_updated_code(
                            run_id, current_code, repair
                        )
                        latest_review = None
                        repair_elapsed = time.perf_counter() - repair_started
                        repair_duration += repair.duration_seconds or repair_elapsed
                    except (GeneratedProjectError, CodexResponseError, ValueError) as exc:
                        termination = TerminationReason.INVALID_REPAIR
                        self._log_termination(run_id, termination, exc)
                    except CodingAgentError as exc:
                        termination = TerminationReason.PROVIDER_ERROR
                        self._log_termination(run_id, termination, exc)
                    else:
                        try:
                            final_test = self._sandbox_runner.run_project(
                                run_directory, language=current_code.language
                            )
                            self._metadata.save_test_result(run_id, final_test)
                            if not final_test.infrastructure_error:
                                after_review = self._get_review_agent().review(
                                    specification, current_code, final_test
                                )
                                reviews.append(after_review)
                                latest_review = after_review
                                review_duration += after_review.duration_seconds or 0.0
                        except ReviewAgentError as exc:
                            termination = TerminationReason.PROVIDER_ERROR
                            self._log_termination(run_id, termination, exc)
                        except RunMetadataError as exc:
                            termination = TerminationReason.PERSISTENCE_ERROR
                            self._log_termination(run_id, termination, exc)

                        if termination in {
                            TerminationReason.PROVIDER_ERROR,
                            TerminationReason.PERSISTENCE_ERROR,
                        }:
                            pass
                        elif final_test.infrastructure_error:
                            termination = TerminationReason.SANDBOX_ERROR
                        elif after_review is not None and _is_success(
                            final_test, after_review
                        ):
                            termination = TerminationReason.SUCCESS
                        else:
                            termination = TerminationReason.MAX_REPAIR_ATTEMPTS

                    if not patch_applied:
                        repair_elapsed = time.perf_counter() - repair_started
                        repair_duration += repair_elapsed
                    record = RepairAttemptRecord(
                        attempt_number=attempt_number,
                        files_created=(
                            tuple(item.path for item in repair.files_created)
                            if repair
                            else ()
                        ),
                        files_modified=(
                            tuple(item.path for item in repair.files_modified)
                            if repair
                            else ()
                        ),
                        files_deleted=repair.files_deleted if repair else (),
                        repair_summary=(
                            repair.summary
                            if repair
                            else "Codex repair did not produce an applicable patch."
                        ),
                        docker_before=before_test,
                        review_before=before_review,
                        docker_after=(
                            final_test if repair is not None else None
                        ),
                        review_after=after_review,
                        repair_duration_seconds=repair_elapsed,
                        tokens_used=repair.tokens_used if repair else None,
                    )
                    history.append(record)
                    try:
                        if repair is not None:
                            self._metadata.save_generated(run_id, current_code)
                            self._metadata.save_test_result(run_id, final_test)
                        self._metadata.save_repair_history(run_id, tuple(history))
                    except RunMetadataError as exc:
                        termination = TerminationReason.PERSISTENCE_ERROR
                        self._log_termination(run_id, termination, exc)

                    if termination in {
                        TerminationReason.INVALID_REPAIR,
                        TerminationReason.PROVIDER_ERROR,
                        TerminationReason.PERSISTENCE_ERROR,
                        TerminationReason.SANDBOX_ERROR,
                    }:
                        break

        return self._finish(
            run_id,
            specification,
            generated,
            initial_test,
            final_test,
            history,
            reviews,
            latest_review,
            review_duration,
            started,
            termination,
            repair_duration,
            repair_tokens if has_repair_tokens else None,
            generation_duration_seconds,
        )

    def _finish(
        self,
        run_id: str,
        specification: Specification,
        generated: GeneratedCode,
        initial_test: TestResult,
        final_test: TestResult,
        history: list[RepairAttemptRecord],
        reviews: list[ReviewResult],
        final_review: ReviewResult | None,
        review_duration: float,
        started: float,
        termination: TerminationReason,
        repair_duration: float = 0.0,
        repair_tokens: int | None = None,
        generation_duration_seconds: float | None = None,
    ) -> FinalRunResult:
        initial_review = reviews[0] if reviews else None
        token_values = [
            specification.tokens_used,
            generated.tokens_used,
            *(review.tokens_used for review in reviews),
            repair_tokens,
        ]
        known_tokens = [value for value in token_values if value is not None]
        final = FinalRunResult(
            run_id=run_id,
            success=termination is TerminationReason.SUCCESS,
            final_verdict=final_review.verdict if final_review else None,
            total_repair_attempts=len(history),
            initial_docker_result=initial_test,
            final_docker_result=final_test,
            initial_nvidia_verdict=(
                initial_review.verdict if initial_review else None
            ),
            final_nvidia_verdict=final_review.verdict if final_review else None,
            repair_history=tuple(history),
            generation_duration_seconds=(
                generation_duration_seconds
                if generation_duration_seconds is not None
                else generated.duration_seconds
            ),
            review_duration_seconds=review_duration,
            repair_duration_seconds=repair_duration,
            total_duration_seconds=time.perf_counter() - started,
            total_tokens_used=sum(known_tokens) if known_tokens else None,
            termination_reason=termination,
        )
        try:
            self._metadata.save_repair_history(run_id, tuple(history))
            self._metadata.save_final_result(final)
        except RunMetadataError as exc:
            final = replace(
                final,
                success=False,
                termination_reason=TerminationReason.PERSISTENCE_ERROR,
            )
            self._log_termination(run_id, TerminationReason.PERSISTENCE_ERROR, exc)
        self._emit(
            "repair_workflow_completed",
            run_id=run_id,
            success=final.success,
            repair_attempts=final.total_repair_attempts,
            termination_reason=final.termination_reason.value,
            final_verdict=(
                final.final_verdict.value if final.final_verdict else None
            ),
            total_duration_seconds=final.total_duration_seconds,
            total_tokens_used=final.total_tokens_used,
        )
        return final

    def _read_updated_code(
        self, run_id: str, original: GeneratedCode, repair: CodeRepair
    ) -> GeneratedCode:
        files = self._writer.read_files(run_id)
        return GeneratedCode(
            files=files,
            provider=repair.provider or original.provider,
            model=repair.model or original.model,
            tokens_used=repair.tokens_used,
            language=original.language,
            summary=repair.summary,
            files_created=tuple(item.path for item in repair.files_created),
            files_modified=tuple(item.path for item in repair.files_modified),
            duration_seconds=repair.duration_seconds,
        )

    def _emit_attempt_started(self, run_id: str, attempt: int) -> None:
        self._emit(
            "repair_attempt_started",
            run_id=run_id,
            attempt=attempt,
            max_attempts=self._settings.max_repair_attempts,
        )

    def _get_review_agent(self) -> ReviewAgent:
        if self._review_provider is None:
            if not callable(self._review_agent):
                raise TypeError("Review provider factory is invalid.")
            self._review_provider = self._review_agent()
        return self._review_provider

    def _log_termination(
        self, run_id: str, reason: TerminationReason, error: Exception
    ) -> None:
        self._logger.warning(
            "repair_workflow_terminated",
            extra={
                "event_data": {
                    "event": "repair_workflow_terminated",
                    "run_id": run_id,
                    "termination_reason": reason.value,
                    "error_code": type(error).__name__,
                }
            },
        )

    def _emit(self, event: str, **fields: object) -> None:
        self._logger.info(event, extra={"event_data": {"event": event, **fields}})


def _is_success(test_result: TestResult, review: ReviewResult) -> bool:
    return (
        test_result.passed
        and test_result.exit_code == 0
        and not test_result.timed_out
        and not test_result.infrastructure_error
        and review.verdict is ReviewVerdict.PASS
        and not review.repair_required
        and review.specification_satisfied
        and review.tests_passed
    )
