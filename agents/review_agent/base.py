"""Interface for analysis of code and deterministic test evidence."""

from abc import ABC, abstractmethod

from core.models import GeneratedCode, ReviewResult, Specification, TestResult


class ReviewAgent(ABC):
    @abstractmethod
    def review(
        self,
        specification: Specification,
        code: GeneratedCode,
        test_result: TestResult,
    ) -> ReviewResult:
        """Assess an artifact using the actual sandbox result as evidence."""
