"""Interface for creating and updating code artifacts."""

from abc import ABC, abstractmethod

from core.models import CodeRepair, GeneratedCode, RepairInstructions, Specification


class CodingAgent(ABC):
    @abstractmethod
    def generate(self, specification: Specification) -> GeneratedCode:
        """Produce an initial code artifact for a specification."""

    @abstractmethod
    def repair(
        self,
        specification: Specification,
        current_code: GeneratedCode,
        instructions: RepairInstructions,
    ) -> CodeRepair:
        """Return validated file changes without executing or writing generated code."""
