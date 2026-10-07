"""Google Gemini implementation of ProCoder's generic PromptAgent interface."""

import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agents.prompt_agent.base import PromptAgent
from agents.prompt_agent.errors import (
    GeminiAuthenticationError,
    GeminiMalformedResponseError,
    GeminiProviderError,
    GeminiProviderRefusalError,
    GeminiRateLimitError,
    GeminiSchemaValidationError,
    GeminiTemporarilyUnavailableError,
    GeminiTimeoutError,
    MissingGeminiApiKeyError,
)
from agents.prompt_agent.system_instruction import PROMPT_AGENT_SYSTEM_INSTRUCTION
from core.config import Settings
from core.models import Specification


class _SpecificationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="A concise description of the requested work.")
    requirements: list[str] = Field(
        description="Actionable functional and relevant technical requirements."
    )
    acceptance_criteria: list[str] = Field(
        description="Concrete, verifiable outcomes that define completion."
    )

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("summary must not be empty")
        return cleaned

    @field_validator("requirements", "acceptance_criteria")
    @classmethod
    def validate_items(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if not cleaned or any(not value for value in cleaned):
            raise ValueError("lists must contain non-empty strings")
        return cleaned


@dataclass(frozen=True)
class _ProviderDiagnostic:
    status_code: int | None
    category: str
    reason: str | None

class GeminiPromptAgent(PromptAgent):
    """Turn a user request into a validated Specification using Gemini."""

    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: Callable[..., Any] = genai.Client,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        if not settings.gemini_api_key:
            raise MissingGeminiApiKeyError()
        self._api_key = settings.gemini_api_key
        self._model = settings.gemini_model
        self._fallback_model = settings.gemini_fallback_model
        self._timeout = settings.gemini_timeout
        self._max_retries = settings.gemini_max_retries
        self._client_factory = client_factory
        self._sleep = sleep
        self._logger = logger or logging.getLogger("procoder")

    def create_specification(self, request: str) -> Specification:
        if not request.strip():
            raise ValueError("The user request must not be empty.")

        client = self._client_factory(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        total_attempts = 0
        interaction: Any
        selected_model: str | None = None
        try:
            models = [self._model]
            if self._fallback_model:
                models.append(self._fallback_model)
            for model_index, model in enumerate(models):
                model_role = "primary" if model_index == 0 else "fallback"
                try:
                    interaction, model_attempts = self._generate_with_retry(
                        client, model, request, model_role=model_role
                    )
                    total_attempts += model_attempts
                    selected_model = model
                    break
                except GeminiProviderError as exc:
                    total_attempts += exc.attempts
                    should_fallback = bool(
                        model_index == 0
                        and len(models) > 1
                        and exc.category
                        and _is_fallback_eligible(exc.category)
                    )
                    self._log_provider_failure(
                        exc, fallback_attempted=model_role == "fallback"
                    )
                    if not should_fallback:
                        if isinstance(exc, GeminiTemporarilyUnavailableError):
                            raise GeminiTemporarilyUnavailableError(
                                total_attempts,
                                status_code=exc.status_code,
                                category=exc.category,
                                model=exc.model or model,
                                model_role=model_role,
                                provider_reason=exc.provider_reason,
                                fallback_attempted=model_role == "fallback",
                            ) from None
                        raise
                    self._logger.warning(
                        "gemini_fallback_model_selected",
                        extra={
                            "event_data": {
                                "event": "gemini_fallback_model_selected",
                                "primary_model": model,
                                "fallback_model": models[model_index + 1],
                                "primary_attempts": exc.attempts,
                                "status_code": exc.status_code,
                                "category": exc.category,
                                "reason": exc.provider_reason,
                                "fallback_attempted": True,
                            }
                        },
                    )
            if selected_model is None:
                raise GeminiProviderError()
        finally:
            client.close()

        if _is_refusal(interaction):
            raise GeminiProviderRefusalError()
        _validate_interaction_status(interaction)

        output_text = getattr(interaction, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise GeminiMalformedResponseError()

        try:
            raw_payload = json.loads(output_text)
        except json.JSONDecodeError:
            raise GeminiMalformedResponseError() from None

        try:
            payload = _SpecificationPayload.model_validate(raw_payload)
        except ValidationError:
            raise GeminiSchemaValidationError() from None

        return Specification(
            request=request,
            summary=payload.summary,
            requirements=tuple(payload.requirements),
            acceptance_criteria=tuple(payload.acceptance_criteria),
            provider="google-gemini",
            model=selected_model,
            tokens_used=_get_total_tokens(interaction),
        )

    def _generate_with_retry(
        self,
        client: Any,
        model: str,
        request: str,
        *,
        model_role: str,
    ) -> tuple[Any, int]:
        for attempt in range(1, self._max_retries + 2):
            try:
                interaction = client.interactions.create(
                    model=model,
                    input=request,
                    system_instruction=PROMPT_AGENT_SYSTEM_INSTRUCTION,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": _SpecificationPayload.model_json_schema(),
                    },
                    store=False,
                    timeout=self._timeout,
                )
                return interaction, attempt
            except (httpx.TimeoutException, TimeoutError):
                diagnostic = _ProviderDiagnostic(None, "timeout", None)
                if attempt <= self._max_retries:
                    self._schedule_retry(
                        attempt=attempt,
                        reason=diagnostic.category,
                        model=model,
                        model_role=model_role,
                    )
                    continue
                raise GeminiTemporarilyUnavailableError(
                    attempt,
                    category=diagnostic.category,
                    model=model,
                    model_role=model_role,
                ) from None
            except httpx.RequestError:
                diagnostic = _ProviderDiagnostic(None, "network_error", None)
                if attempt <= self._max_retries:
                    self._schedule_retry(
                        attempt=attempt,
                        reason=diagnostic.category,
                        model=model,
                        model_role=model_role,
                    )
                    continue
                raise GeminiTemporarilyUnavailableError(
                    attempt,
                    category=diagnostic.category,
                    model=model,
                    model_role=model_role,
                ) from None
            except Exception as exc:
                diagnostic = _get_provider_diagnostic(
                    exc,
                    request=request,
                    api_key=self._api_key,
                )
                status_code = diagnostic.status_code
                if status_code is None:
                    raise GeminiProviderError(
                        category=diagnostic.category,
                        model=model,
                        model_role=model_role,
                        provider_reason=diagnostic.reason,
                        attempts=attempt,
                    ) from None
                if _is_retryable_status_code(status_code):
                    if attempt <= self._max_retries:
                        self._schedule_retry(
                            attempt=attempt,
                            reason=diagnostic.category,
                            status_code=status_code,
                            model=model,
                            model_role=model_role,
                            provider_reason=diagnostic.reason,
                        )
                        continue
                    raise GeminiTemporarilyUnavailableError(
                        attempt,
                        status_code=status_code,
                        category=diagnostic.category,
                        model=model,
                        model_role=model_role,
                        provider_reason=diagnostic.reason,
                    ) from None
                raise _translate_api_error(
                    status_code,
                    category=diagnostic.category,
                    model=model,
                    model_role=model_role,
                    provider_reason=diagnostic.reason,
                    attempts=attempt,
                ) from None

    def _schedule_retry(
        self,
        *,
        attempt: int,
        reason: str,
        model: str,
        model_role: str,
        status_code: int | None = None,
        provider_reason: str | None = None,
    ) -> None:
        delay_seconds = min(0.5 * (2 ** (attempt - 1)), 2.0)
        self._logger.warning(
            "gemini_retry_scheduled",
            extra={
                "event_data": {
                    "event": "gemini_retry_scheduled",
                    "attempt": attempt,
                    "max_attempts": self._max_retries + 1,
                    "reason": reason,
                    "model": model,
                    "model_role": model_role,
                    "status_code": status_code,
                    "provider_reason": provider_reason,
                    "delay_seconds": delay_seconds,
                }
            },
        )
        self._sleep(delay_seconds)

    def _log_provider_failure(
        self, error: GeminiProviderError, *, fallback_attempted: bool
    ) -> None:
        self._logger.warning(
            "gemini_provider_failure",
            extra={
                "event_data": {
                    "event": "gemini_provider_failure",
                    "status_code": error.status_code,
                    "category": error.category,
                    "model": error.model,
                    "model_role": error.model_role,
                    "fallback_attempted": fallback_attempted,
                    "provider_reason": error.provider_reason,
                    "attempts": error.attempts,
                }
            },
        )


def _translate_api_error(
    status_code: int,
    *,
    category: str,
    model: str,
    model_role: str,
    provider_reason: str | None,
    attempts: int,
) -> GeminiProviderError:
    diagnostics = {
        "status_code": status_code,
        "category": category,
        "model": model,
        "model_role": model_role,
        "provider_reason": provider_reason,
        "attempts": attempts,
    }
    if status_code in (401, 403):
        return GeminiAuthenticationError(**diagnostics)
    if status_code == 429:
        return GeminiRateLimitError(**diagnostics)
    if status_code in (408, 504):
        return GeminiTimeoutError(**diagnostics)
    return GeminiProviderError(**diagnostics)


def _get_http_status_code(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status

    error_code = getattr(error, "code", None)
    return error_code if isinstance(error_code, int) else None


def _get_provider_diagnostic(
    error: Exception, *, request: str, api_key: str
) -> _ProviderDiagnostic:
    status_code = _get_http_status_code(error)
    if status_code is None and isinstance(error, genai_errors.APIError):
        status_code = error.code

    payload = getattr(error, "body", None)
    if not isinstance(payload, Mapping):
        response = getattr(error, "response", None)
        response_json = getattr(response, "json", None)
        if callable(response_json):
            try:
                possible_payload = response_json()
            except (ValueError, TypeError):
                possible_payload = None
            if isinstance(possible_payload, Mapping):
                payload = possible_payload

    error_payload = payload.get("error", payload) if isinstance(payload, Mapping) else {}
    if not isinstance(error_payload, Mapping):
        error_payload = {}
    raw_reason = _first_string(
        error_payload.get("status"),
        error_payload.get("reason"),
        getattr(error, "reason", None),
    )
    raw_message = _first_string(
        error_payload.get("message"),
        getattr(error, "message", None),
        str(error),
    )
    classification_text = " ".join(
        value.casefold() for value in (raw_reason, raw_message) if value
    )

    category = _classify_provider_failure(status_code, classification_text)
    reason = _safe_provider_reason(raw_reason, raw_message, request, api_key)
    return _ProviderDiagnostic(status_code, category, reason)


def _classify_provider_failure(status_code: int | None, detail: str) -> str:
    if status_code in (401, 403):
        return "authentication"
    if _mentions_model_selection_failure(detail):
        if status_code == 404 or "not found" in detail or "not_found" in detail:
            return "model_not_found"
        return "unsupported_model"
    if status_code == 429:
        return "rate_limit"
    if status_code in (408, 504):
        return "timeout"
    if status_code is not None and 500 <= status_code < 600:
        return "availability"
    if status_code == 400:
        return "invalid_request"
    return "provider_error"


def _mentions_model_selection_failure(detail: str) -> bool:
    if "model" not in detail:
        return False
    return any(
        marker in detail
        for marker in (
            "not found",
            "not_found",
            "unknown model",
            "unsupported model",
            "unsupported",
            "model_not_supported",
            "model not supported",
            "is not supported",
            "does not support",
            "not supported for",
        )
    )


def _safe_provider_reason(
    raw_reason: str | None,
    raw_message: str | None,
    request: str,
    api_key: str,
) -> str | None:
    reason_parts = [
        value for value in (raw_reason, raw_message) if value and value != raw_reason
    ]
    if raw_reason:
        reason_parts.insert(0, raw_reason)
    if not reason_parts:
        return None
    reason = ": ".join(reason_parts)
    if request:
        reason = re.sub(
            re.escape(request), "[request redacted]", reason, flags=re.IGNORECASE
        )
    if api_key:
        reason = re.sub(
            re.escape(api_key), "[credential redacted]", reason, flags=re.IGNORECASE
        )
    reason = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [redacted]", reason)
    reason = re.sub(
        r"(?i)(?:AIza)[A-Za-z0-9_-]{20,}",
        "[credential redacted]",
        reason,
    )
    reason = re.sub(
        r"(?i)(?:api[_ -]?key|authorization|x-goog-api-key)\s*[:=]\s*[^,\s;]+",
        "[credential redacted]",
        reason,
    )
    reason = re.sub(r"https?://\S+", "[url redacted]", reason)
    return " ".join(reason.split())[:200] or None


def _first_string(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_fallback_eligible(category: str) -> bool:
    return category in {
        "availability",
        "model_not_found",
        "rate_limit",
        "timeout",
        "unsupported_model",
    }


def _is_retryable_status_code(status_code: int) -> bool:
    return status_code in (408, 429) or 500 <= status_code < 600


def _is_refusal(interaction: Any) -> bool:
    if getattr(interaction, "status", None) == "incomplete":
        return True

    for step in getattr(interaction, "steps", None) or ():
        if getattr(step, "type", None) != "model_output":
            continue
        error = getattr(step, "error", None)
        code = getattr(error, "code", None)
        if isinstance(code, str) and any(
            marker in code.casefold() for marker in ("safety", "refus", "blocked")
        ):
            return True
    return False


def _validate_interaction_status(interaction: Any) -> None:
    status = getattr(interaction, "status", None)
    if status is not None and status != "completed":
        raise GeminiProviderError()


def _get_total_tokens(interaction: Any) -> int | None:
    usage = getattr(interaction, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    return total_tokens if isinstance(total_tokens, int) and total_tokens >= 0 else None
