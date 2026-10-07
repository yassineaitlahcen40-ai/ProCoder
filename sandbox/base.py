"""Provider-independent interface for executing and testing generated projects."""

from abc import ABC, abstractmethod
from pathlib import Path

from core.models import TestResult


class SandboxRunner(ABC):
    """Execute a generated project under a concrete isolation runtime."""

    @abstractmethod
    def run_project(
        self, project_directory: Path, *, language: str | None = None
    ) -> TestResult:
        """Run supported tests in the generated project and return their result."""

