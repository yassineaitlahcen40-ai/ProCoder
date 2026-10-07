"""Interface for turning test and review failures into repair instructions."""

from abc import ABC, abstractmethod

from core.models import (
    GeneratedCode,
    RepairInstructions,
    ReviewDecision,
    Specification,
    TestResult,
)


class RepairAgent(ABC):
    @abstractmethod
    def create_instructions(
        self,
        specification: Specification,
        code: GeneratedCode,
        test_result: TestResult,
        review: ReviewDecision,
    ) -> RepairInstructions:
        """Create focused instructions from concrete failures and findings."""
