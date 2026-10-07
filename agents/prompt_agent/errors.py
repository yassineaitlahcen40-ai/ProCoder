"""Safe, provider-specific errors surfaced by the Gemini Prompt Agent."""


class PromptAgentError(RuntimeError):
    """Base class for errors with messages safe to show to a CLI user."""

    safe_message = "The prompt agent could not complete the request."


class MissingGeminiApiKeyError(PromptAgentError):
    safe_message = (
        "GEMINI_API_KEY is missing. Add it to the local .env file or environment."
    )


class GeminiProviderError(PromptAgentError):
    safe_message = "The Gemini service could not complete the request."

    def __init__(
        self,
        *,
        status_code: int | None = None,
        category: str = "provider_error",
        model: str | None = None,
        model_role: str | None = None,
        provider_reason: str | None = None,
        fallback_attempted: bool = False,
        attempts: int = 0,
    ) -> None:
        self.status_code = status_code
        self.category = category
        self.model = model
        self.model_role = model_role
        self.provider_reason = provider_reason
        self.fallback_attempted = fallback_attempted
        self.attempts = attempts
        super().__init__(self.safe_message)


class GeminiAuthenticationError(GeminiProviderError):
    safe_message = "Gemini authentication failed. Check the API key configuration."


class GeminiRateLimitError(GeminiProviderError):
    safe_message = "Gemini quota or rate limit was reached. Try again later."


class GeminiTimeoutError(GeminiProviderError):
    safe_message = "The Gemini request timed out. Try again."


class GeminiNetworkError(GeminiProviderError):
    safe_message = "Gemini could not be reached because of a network error."


class GeminiProviderRefusalError(PromptAgentError):
    safe_message = "Gemini declined or could not complete this specification request."


class GeminiMalformedResponseError(PromptAgentError):
    safe_message = "Gemini returned a response that was not valid JSON."


class GeminiSchemaValidationError(PromptAgentError):
    safe_message = (
        "Gemini returned JSON that does not match ProCoder's specification schema."
    )


class GeminiTemporarilyUnavailableError(GeminiProviderError):
    def __init__(
        self,
        attempts: int,
        status_code: int | None = None,
        *,
        category: str = "availability",
        model: str | None = None,
        model_role: str | None = None,
        provider_reason: str | None = None,
        fallback_attempted: bool = False,
    ) -> None:
        self.attempts = attempts
        super().__init__(
            status_code=status_code,
            category=category,
            model=model,
            model_role=model_role,
            provider_reason=provider_reason,
            fallback_attempted=fallback_attempted,
            attempts=attempts,
        )

    @property
    def safe_message(self) -> str:
        return (
            f"Gemini is temporarily unavailable after {self.attempts} attempts. "
            "Please try again later."
        )
