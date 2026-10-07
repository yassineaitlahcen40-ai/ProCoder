"""TODO: implement against current official OpenAI SDK documentation."""

from core.models import Specification

from agents.prompt_agent.base import PromptAgent


class OpenAIChatGPTPromptAgent(PromptAgent):
    def create_specification(self, request: str) -> Specification:
        raise NotImplementedError(
            "OpenAI prompt-agent integration is not implemented. "
            "Confirm the current official SDK and API before adding it."
        )
