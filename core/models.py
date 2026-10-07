"""Provider-independent data structures exchanged between workflow stages."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


@dataclass(frozen=True)
class Specification:
    request: str
    summary: str
    requirements: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    provider: str | None = None
    model: str | None = None
    tokens_used: int | None = None


@dataclass(frozen=True)
class GeneratedCode:
    files: Mapping[str, str]
    provider: str | None = None
    model: str | None = None
    tokens_used: int | None = None
    language: str | None = None
    summary: str | None = None
    files_created: tuple[str, ...] = ()
    files_modified: tuple[str, ...] = ()
    duration_seconds: float | None = None


@dataclass(frozen=True)
class TestResult:
    passed: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    output_truncated: bool = False
    command: tuple[str, ...] = ()
    tests_passed: int | None = None
    tests_failed: int | None = None
    tests_skipped: int | None = None
    runtime: str | None = None
    container_image: str | None = None
    infrastructure_error: bool = False


@dataclass(frozen=True)
class ReviewDecision:
    passed: bool
    summary: str
    findings: tuple[str, ...] = ()
    provider: str | None = None
    model: str | None = None
    tokens_used: int | None = None


class ReviewVerdict(str, Enum):
    PASS = "PASS"
    NEEDS_REPAIR = "NEEDS_REPAIR"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: ReviewVerdict
    summary: str = Field(min_length=1, max_length=2000)
    specification_satisfied: bool
    code_quality_findings: tuple[str, ...]
    test_analysis: str = Field(min_length=1, max_length=4000)
    likely_root_cause: str | None = Field(default=None, max_length=2000)
    recommended_actions: tuple[str, ...]
    repair_required: bool
    confidence: float = Field(ge=0.0, le=1.0)
    provider: str | None = None
    model: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    tokens_used: int | None = Field(default=None, ge=0)
    source_truncated: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    tests_passed: bool

    @field_validator("code_quality_findings", "recommended_actions")
    @classmethod
    def validate_items(cls, items: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in items)
        if any(not item for item in cleaned):
            raise ValueError("review list entries must not be empty")
        if any(len(item) > 1000 for item in cleaned):
            raise ValueError("review list entries are too long")
        return cleaned

    @property
    def passed(self) -> bool:
        return (
            self.tests_passed
            and self.verdict is ReviewVerdict.PASS
            and self.specification_satisfied
            and not self.repair_required
        )

    @property
    def findings(self) -> tuple[str, ...]:
        return self.code_quality_findings


@dataclass(frozen=True)
class RepairInstructions:
    instructions: str
    provider: str | None = None
    model: str | None = None
    tokens_used: int | None = None
    test_result: TestResult | None = None
    review_result: ReviewResult | None = None
    attempt_number: int | None = None


@dataclass(frozen=True)
class RepairFile:
    path: str
    content: str


@dataclass(frozen=True)
class CodeRepair:
    files_created: tuple[RepairFile, ...]
    files_modified: tuple[RepairFile, ...]
    files_deleted: tuple[str, ...]
    summary: str
    provider: str | None = None
    model: str | None = None
    tokens_used: int | None = None
    duration_seconds: float = 0.0

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return (
            *(item.path for item in self.files_created),
            *(item.path for item in self.files_modified),
            *self.files_deleted,
        )


class TerminationReason(str, Enum):
    SUCCESS = "SUCCESS"
    MAX_REPAIR_ATTEMPTS = "MAX_REPAIR_ATTEMPTS"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    SANDBOX_ERROR = "SANDBOX_ERROR"
    INVALID_REPAIR = "INVALID_REPAIR"
    PERSISTENCE_ERROR = "PERSISTENCE_ERROR"


@dataclass(frozen=True)
class RepairAttemptRecord:
    attempt_number: int
    files_created: tuple[str, ...]
    files_modified: tuple[str, ...]
    files_deleted: tuple[str, ...]
    repair_summary: str
    docker_before: TestResult
    review_before: ReviewResult
    docker_after: TestResult | None
    review_after: ReviewResult | None
    repair_duration_seconds: float
    tokens_used: int | None = None


@dataclass(frozen=True)
class FinalRunResult:
    run_id: str
    success: bool
    final_verdict: ReviewVerdict | None
    total_repair_attempts: int
    initial_docker_result: TestResult
    final_docker_result: TestResult
    initial_nvidia_verdict: ReviewVerdict | None
    final_nvidia_verdict: ReviewVerdict | None
    repair_history: tuple[RepairAttemptRecord, ...]
    generation_duration_seconds: float | None
    review_duration_seconds: float
    repair_duration_seconds: float
    total_duration_seconds: float
    total_tokens_used: int | None
    termination_reason: TerminationReason


@dataclass(frozen=True)
class WorkflowResult:
    run_id: str
    passed: bool
    status: str
    iterations: int
    repair_attempts: int
    duration_seconds: float
    test_results: tuple[TestResult, ...]
    final_code: GeneratedCode
    final_review: ReviewDecision
    tokens_used: int | None = None
