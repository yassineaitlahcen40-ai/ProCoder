"""Safe errors surfaced by the NVIDIA NIM review adapter."""


class ReviewAgentError(RuntimeError):
    """Base class for errors safe to show to a CLI user."""

    safe_message = "The code review service could not complete the request."


class MissingNvidiaApiKeyError(ReviewAgentError):
    safe_message = "NVIDIA_API_KEY is missing. Configure it before requesting a review."


class NvidiaAuthenticationError(ReviewAgentError):
    safe_message = "NVIDIA authentication failed. Check the API key configuration."


class NvidiaRateLimitError(ReviewAgentError):
    safe_message = "NVIDIA API rate limit or quota was reached. Try again later."


class NvidiaTimeoutError(ReviewAgentError):
    safe_message = "The NVIDIA review request timed out. Try again."


class NvidiaNetworkError(ReviewAgentError):
    safe_message = "The NVIDIA review service could not be reached."


class NvidiaUnavailableError(ReviewAgentError):
    safe_message = "The NVIDIA review service is temporarily unavailable."


class NvidiaInvalidResponseError(ReviewAgentError):
    safe_message = "NVIDIA returned a response that could not be validated."


class NvidiaInvalidRequestError(ReviewAgentError):
    safe_message = "NVIDIA rejected the review request. Check the model and request settings."
