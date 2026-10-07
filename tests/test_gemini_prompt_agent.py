import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import errors as genai_errors
import httpx
from google.genai._gaos.lib.compat_errors import InternalServerError, NotFoundError

from agents.prompt_agent.errors import (
    GeminiAuthenticationError,
    GeminiMalformedResponseError,
    GeminiProviderError,
    GeminiProviderRefusalError,
    GeminiSchemaValidationError,
    GeminiTemporarilyUnavailableError,
    MissingGeminiApiKeyError,
)
from agents.prompt_agent.providers.gemini import GeminiPromptAgent
from core.config import Settings


class GeminiPromptAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings.from_environment(
            {
                "GEMINI_API_KEY": "unit-test-only-key",
                "GEMINI_MODEL": "gemini-test-model",
                "GEMINI_FALLBACK_MODEL": "",
                "GEMINI_TIMEOUT": "7",
            }
        )
        self.client = Mock()
        self.sleep = Mock()
        self.logger = Mock()
        self.client_factory = Mock(return_value=self.client)
        self.client.interactions.create.return_value = SimpleNamespace(
            status="completed",
            output_text=json.dumps(
                {
                    "summary": "Create a Python prime-number function.",
                    "requirements": [
                        "Accept an integer input.",
                        "Return whether the input is prime.",
                    ],
                    "acceptance_criteria": [
                        "Return true for 2 and other prime numbers.",
                        "Return false for 1 and composite numbers.",
                    ],
                }
            ),
            usage=SimpleNamespace(total_tokens=42),
        )

    def make_agent(self, settings: Settings | None = None) -> GeminiPromptAgent:
        return GeminiPromptAgent(
            settings or self.settings,
            client_factory=self.client_factory,
            sleep=self.sleep,
            logger=self.logger,
        )

    def test_success_returns_validated_specification_and_uses_schema(self) -> None:
        request = "Create a Python function that determines whether a number is prime."

        result = self.make_agent().create_specification(request)

        self.assertEqual(result.request, request)
        self.assertEqual(result.summary, "Create a Python prime-number function.")
        self.assertEqual(len(result.requirements), 2)
        self.assertEqual(len(result.acceptance_criteria), 2)
        self.assertEqual(result.provider, "google-gemini")
        self.assertEqual(result.model, "gemini-test-model")
        self.assertEqual(result.tokens_used, 42)
        call = self.client.interactions.create.call_args.kwargs
        self.assertEqual(call["model"], "gemini-test-model")
        self.assertEqual(call["input"], request)
        self.assertEqual(call["timeout"], 7)
        self.assertFalse(call["store"])
        self.assertEqual(call["response_format"]["mime_type"], "application/json")
        self.assertIn("requirements", call["response_format"]["schema"]["properties"])
        self.assertIn("Your only\nresponsibility", call["system_instruction"])
        self.client.close.assert_called_once()
        retry_options = self.client_factory.call_args.kwargs[
            "http_options"
        ].retry_options
        self.assertEqual(retry_options.attempts, 1)

    def test_missing_api_key_fails_before_creating_a_client(self) -> None:
        factory = Mock()
        settings = Settings.from_environment({})

        with self.assertRaises(MissingGeminiApiKeyError):
            GeminiPromptAgent(settings, client_factory=factory)

        factory.assert_not_called()

    def test_rejects_invalid_json_without_parsing_markdown(self) -> None:
        self.client.interactions.create.return_value = SimpleNamespace(
            status="completed", output_text="```json\n{}\n```"
        )

        with self.assertRaises(GeminiMalformedResponseError):
            self.make_agent().create_specification("Build a calculator.")

    def test_rejects_json_that_does_not_match_the_schema(self) -> None:
        self.client.interactions.create.return_value = SimpleNamespace(
            status="completed",
            output_text=json.dumps(
                {"summary": "A calculator.", "requirements": ["Add numbers"]}
            ),
        )

        with self.assertRaises(GeminiSchemaValidationError):
            self.make_agent().create_specification("Build a calculator.")

    def test_provider_errors_are_mapped_without_showing_provider_details(self) -> None:
        self.client.interactions.create.side_effect = genai_errors.APIError(
            401, {"error": {"message": "sensitive provider diagnostic"}}
        )

        with self.assertRaises(GeminiAuthenticationError) as raised:
            self.make_agent().create_specification("Build a calculator.")

        self.assertNotIn("sensitive provider diagnostic", raised.exception.safe_message)
        self.client.interactions.create.assert_called_once()
        self.sleep.assert_not_called()

    def test_quota_errors_are_mapped(self) -> None:
        self.client.interactions.create.side_effect = genai_errors.APIError(
            429, {"error": {"message": "quota exceeded"}}
        )

        with self.assertRaises(GeminiTemporarilyUnavailableError):
            self.make_agent().create_specification("Build a calculator.")
        self.assertEqual(self.client.interactions.create.call_count, 3)

    def test_retries_transient_server_errors_with_exponential_backoff(self) -> None:
        self.client.interactions.create.side_effect = [
            genai_errors.APIError(503, {"error": {"status": "UNAVAILABLE"}}),
            genai_errors.APIError(500, {"error": {"status": "INTERNAL"}}),
            self.client.interactions.create.return_value,
        ]

        result = self.make_agent().create_specification("Build a calculator.")

        self.assertEqual(result.provider, "google-gemini")
        self.assertEqual(self.client.interactions.create.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in self.sleep.call_args_list], [0.5, 1.0]
        )
        self.assertEqual(self.logger.warning.call_count, 2)
        logged = repr(self.logger.warning.call_args_list)
        self.assertNotIn("unit-test-only-key", logged)
        self.assertNotIn("Build a calculator.", logged)

    def test_retries_actual_google_genai_228_internal_server_error(self) -> None:
        request = httpx.Request("POST", "https://example.invalid")
        response = httpx.Response(503, request=request)
        actual_sdk_exception = InternalServerError(
            "Error code: 503 - service_unavailable",
            response=response,
            body={
                "error": {
                    "code": 503,
                    "status": "service_unavailable",
                    "message": "high demand",
                }
            },
        )
        self.assertNotIsInstance(actual_sdk_exception, genai_errors.APIError)
        self.client.interactions.create.side_effect = [
            actual_sdk_exception,
            actual_sdk_exception,
            self.client.interactions.create.return_value,
        ]

        result = self.make_agent().create_specification("Build a calculator.")

        self.assertEqual(result.model, "gemini-test-model")
        self.assertEqual(self.client.interactions.create.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in self.sleep.call_args_list], [0.5, 1.0]
        )

    def test_exhausted_transient_retries_have_concise_safe_error(self) -> None:
        self.client.interactions.create.side_effect = genai_errors.APIError(
            503, {"error": {"message": "sensitive provider diagnostic"}}
        )

        with self.assertRaises(GeminiTemporarilyUnavailableError) as raised:
            self.make_agent().create_specification("Build a calculator.")

        self.assertEqual(self.client.interactions.create.call_count, 3)
        self.assertEqual(
            raised.exception.safe_message,
            "Gemini is temporarily unavailable after 3 attempts. Please try again later.",
        )
        self.assertNotIn("sensitive provider diagnostic", str(raised.exception))
        self.assertEqual(
            [call.args[0] for call in self.sleep.call_args_list], [0.5, 1.0]
        )

    def test_retries_rate_limits_but_not_invalid_requests(self) -> None:
        self.client.interactions.create.side_effect = [
            genai_errors.APIError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}}),
            self.client.interactions.create.return_value,
        ]

        self.make_agent().create_specification("Build a calculator.")

        self.assertEqual(self.client.interactions.create.call_count, 2)
        self.sleep.assert_called_once_with(0.5)

        self.client.interactions.create.reset_mock()
        self.client.interactions.create.side_effect = genai_errors.APIError(
            400, {"error": {"message": "invalid request"}}
        )
        self.sleep.reset_mock()

        with self.assertRaises(GeminiProviderError):
            self.make_agent().create_specification("Build a calculator.")

        self.client.interactions.create.assert_called_once()
        self.sleep.assert_not_called()

    def test_retries_network_and_timeout_failures_and_stops_at_attempt_limit(self) -> None:
        self.client.interactions.create.side_effect = httpx.ReadTimeout(
            "sensitive timeout detail"
        )

        with self.assertRaises(GeminiTemporarilyUnavailableError) as raised:
            self.make_agent().create_specification("Build a calculator.")

        self.assertEqual(self.client.interactions.create.call_count, 3)
        self.assertEqual(raised.exception.attempts, 3)
        self.assertEqual(
            [call.args[0] for call in self.sleep.call_args_list], [0.5, 1.0]
        )

    def test_configured_fallback_model_is_used_after_primary_5xx_retries(self) -> None:
        request = httpx.Request("POST", "https://example.invalid")
        response = httpx.Response(503, request=request)
        primary_error = InternalServerError(
            "Error code: 503 - service_unavailable",
            response=response,
            body={"error": {"code": 503, "status": "service_unavailable"}},
        )
        self.client.interactions.create.side_effect = [
            primary_error,
            primary_error,
            primary_error,
            self.client.interactions.create.return_value,
        ]
        settings = replace(
            self.settings, gemini_fallback_model="gemini-3.1-flash-lite"
        )

        result = self.make_agent(settings).create_specification("Build a calculator.")

        self.assertEqual(result.model, "gemini-3.1-flash-lite")
        self.assertEqual(self.client.interactions.create.call_count, 4)
        calls = self.client.interactions.create.call_args_list
        self.assertEqual([call.kwargs["model"] for call in calls], [
            "gemini-test-model",
            "gemini-test-model",
            "gemini-test-model",
            "gemini-3.1-flash-lite",
        ])
        self.assertTrue(
            all(
                call.kwargs["response_format"]["mime_type"] == "application/json"
                for call in calls
            )
        )
        retry_events = [
            call
            for call in self.logger.warning.call_args_list
            if call.args[0] == "gemini_retry_scheduled"
        ]
        self.assertEqual(len(retry_events), 2)
        logged = repr(self.logger.warning.call_args_list)
        self.assertIn("gemini_fallback_model_selected", logged)
        self.assertNotIn("unit-test-only-key", logged)

    def test_actual_sdk_model_not_found_error_uses_configured_fallback(self) -> None:
        request = "Build a concise prime-number helper."
        response = httpx.Response(
            404, request=httpx.Request("POST", "https://example.invalid")
        )
        model_not_found = NotFoundError(
            "model not found",
            response=response,
            body={
                "error": {
                    "code": 404,
                    "status": "NOT_FOUND",
                    "message": "Model gemini-primary-test was not found.",
                }
            },
        )
        self.assertEqual(model_not_found.status_code, 404)
        self.client.interactions.create.side_effect = [
            model_not_found,
            self.client.interactions.create.return_value,
        ]
        settings = replace(
            self.settings, gemini_fallback_model="gemini-3.1-flash-lite"
        )

        result = self.make_agent(settings).create_specification(request)

        self.assertEqual(result.model, "gemini-3.1-flash-lite")
        self.assertEqual(self.client.interactions.create.call_count, 2)
        failure_event = next(
            call.kwargs["extra"]["event_data"]
            for call in self.logger.warning.call_args_list
            if call.args[0] == "gemini_provider_failure"
        )
        self.assertEqual(failure_event["status_code"], 404)
        self.assertEqual(failure_event["category"], "model_not_found")
        self.assertEqual(failure_event["model_role"], "primary")
        self.assertEqual(failure_event["fallback_attempted"], False)
        self.assertIn("Model gemini-primary-test was not found", failure_event["provider_reason"])

    def test_unsupported_model_error_falls_back_but_invalid_request_does_not(self) -> None:
        settings = replace(
            self.settings, gemini_fallback_model="gemini-3.1-flash-lite"
        )
        self.client.interactions.create.side_effect = [
            genai_errors.APIError(
                400,
                {
                    "error": {
                        "status": "INVALID_ARGUMENT",
                        "message": "Model gemini-primary-test is not supported.",
                    }
                },
            ),
            self.client.interactions.create.return_value,
        ]

        result = self.make_agent(settings).create_specification("Build a calculator.")

        self.assertEqual(result.model, "gemini-3.1-flash-lite")
        self.assertEqual(self.client.interactions.create.call_count, 2)

        self.client.interactions.create.reset_mock()
        self.sleep.reset_mock()
        self.client.interactions.create.side_effect = genai_errors.APIError(
            400, {"error": {"status": "INVALID_ARGUMENT", "message": "invalid request"}}
        )
        with self.assertRaises(GeminiProviderError) as raised:
            self.make_agent(settings).create_specification("Build a calculator.")
        self.assertEqual(raised.exception.category, "invalid_request")
        self.assertEqual(self.client.interactions.create.call_count, 1)
        self.sleep.assert_not_called()

    def test_rate_limit_retries_are_exhausted_before_model_fallback(self) -> None:
        settings = replace(
            self.settings, gemini_fallback_model="gemini-3.1-flash-lite"
        )
        rate_limit = genai_errors.APIError(
            429, {"error": {"status": "RESOURCE_EXHAUSTED", "message": "quota"}}
        )
        self.client.interactions.create.side_effect = [
            rate_limit,
            rate_limit,
            rate_limit,
            self.client.interactions.create.return_value,
        ]

        result = self.make_agent(settings).create_specification("Build a calculator.")

        self.assertEqual(result.model, "gemini-3.1-flash-lite")
        self.assertEqual(self.client.interactions.create.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in self.sleep.call_args_list],
            [0.5, 1.0],
        )
        fallback_event = next(
            call.kwargs["extra"]["event_data"]
            for call in self.logger.warning.call_args_list
            if call.args[0] == "gemini_fallback_model_selected"
        )
        self.assertEqual(fallback_event["status_code"], 429)
        self.assertEqual(fallback_event["category"], "rate_limit")

    def test_provider_failure_diagnostics_redact_key_and_request(self) -> None:
        request = "Build a calculator using request-sensitive-marker."
        self.client.interactions.create.side_effect = genai_errors.APIError(
            400,
            {
                "error": {
                    "status": "INVALID_ARGUMENT",
                    "message": (
                        f"Rejected {request}; api_key=unit-test-only-key"
                    ),
                }
            },
        )

        with self.assertRaises(GeminiProviderError):
            self.make_agent().create_specification(request)

        event = next(
            call.kwargs["extra"]["event_data"]
            for call in self.logger.warning.call_args_list
            if call.args[0] == "gemini_provider_failure"
        )
        self.assertEqual(event["status_code"], 400)
        self.assertEqual(event["category"], "invalid_request")
        self.assertEqual(event["model"], "gemini-test-model")
        self.assertEqual(event["model_role"], "primary")
        self.assertFalse(event["fallback_attempted"])
        self.assertIn("[request redacted]", event["provider_reason"])
        self.assertIn("[credential redacted]", event["provider_reason"])
        serialized_logs = repr(self.logger.warning.call_args_list)
        self.assertNotIn(request, serialized_logs)
        self.assertNotIn("unit-test-only-key", serialized_logs)

    def test_fallback_model_gets_its_own_bounded_retry_budget(self) -> None:
        request = httpx.Request("POST", "https://example.invalid")
        response = httpx.Response(503, request=request)
        unavailable = InternalServerError(
            "Error code: 503 - service_unavailable",
            response=response,
            body={"error": {"code": 503, "status": "service_unavailable"}},
        )
        self.client.interactions.create.side_effect = [
            unavailable,
            unavailable,
            unavailable,
            unavailable,
            unavailable,
            self.client.interactions.create.return_value,
        ]
        settings = replace(
            self.settings, gemini_fallback_model="gemini-3.1-flash-lite"
        )

        result = self.make_agent(settings).create_specification("Build a calculator.")

        self.assertEqual(result.model, "gemini-3.1-flash-lite")
        self.assertEqual(self.client.interactions.create.call_count, 6)
        models = [call.kwargs["model"] for call in self.client.interactions.create.call_args_list]
        self.assertEqual(
            models,
            [
                "gemini-test-model",
                "gemini-test-model",
                "gemini-test-model",
                "gemini-3.1-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-3.1-flash-lite",
            ],
        )
        self.assertEqual(
            [call.args[0] for call in self.sleep.call_args_list],
            [0.5, 1.0, 0.5, 1.0],
        )
        retry_events = [
            call.kwargs["extra"]["event_data"]
            for call in self.logger.warning.call_args_list
            if call.args[0] == "gemini_retry_scheduled"
        ]
        self.assertEqual(
            [event["model"] for event in retry_events],
            [
                "gemini-test-model",
                "gemini-test-model",
                "gemini-3.1-flash-lite",
                "gemini-3.1-flash-lite",
            ],
        )
        self.assertNotIn("unit-test-only-key", repr(self.logger.warning.call_args_list))

    def test_fallback_is_not_used_for_authentication_failure(self) -> None:
        self.client.interactions.create.side_effect = genai_errors.APIError(
            401, {"error": {"message": "invalid key"}}
        )
        settings = replace(
            self.settings, gemini_fallback_model="gemini-3.1-flash-lite"
        )

        with self.assertRaises(GeminiAuthenticationError):
            self.make_agent(settings).create_specification("Build a calculator.")

        self.client.interactions.create.assert_called_once()

    def test_network_and_timeout_errors_are_retried_safely(self) -> None:
        self.client.interactions.create.side_effect = httpx.ConnectError(
            "sensitive connection detail"
        )
        with self.assertRaises(GeminiTemporarilyUnavailableError):
            self.make_agent().create_specification("Build a calculator.")
        self.assertEqual(self.client.interactions.create.call_count, 3)

        self.client.interactions.create.side_effect = httpx.ReadTimeout(
            "sensitive timeout detail"
        )
        self.client.interactions.create.reset_mock()
        self.sleep.reset_mock()
        with self.assertRaises(GeminiTemporarilyUnavailableError):
            self.make_agent().create_specification("Build a calculator.")
        self.assertEqual(self.client.interactions.create.call_count, 3)

    def test_incomplete_provider_response_is_treated_as_refusal(self) -> None:
        self.client.interactions.create.return_value = SimpleNamespace(
            status="incomplete", output_text=None
        )

        with self.assertRaises(GeminiProviderRefusalError):
            self.make_agent().create_specification("Build a calculator.")

    def test_failed_interaction_status_is_a_provider_error(self) -> None:
        self.client.interactions.create.return_value = SimpleNamespace(
            status="failed", output_text=None
        )

        with self.assertRaises(GeminiProviderError):
            self.make_agent().create_specification("Build a calculator.")

    def test_other_api_failure_is_safe(self) -> None:
        self.client.interactions.create.side_effect = genai_errors.APIError(
            503, {"error": {"message": "sensitive provider diagnostic"}}
        )

        with self.assertRaises(GeminiTemporarilyUnavailableError):
            self.make_agent().create_specification("Build a calculator.")
        self.assertEqual(self.client.interactions.create.call_count, 3)

    def test_blank_request_does_not_call_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.make_agent().create_specification(" ")

        self.client.interactions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
