"""Interface for turning user requests into structured specifications."""

from abc import ABC, abstractmethod

from core.models import Specification


class PromptAgent(ABC):
    @abstractmethod
    def create_specification(self, request: str) -> Specification:
        """Convert a natural-language request into a testable specification."""
