import unittest

from core.config import Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_and_secret_placeholders(self) -> None:
        settings = Settings.from_environment({})

        self.assertEqual(settings.max_repair_attempts, 3)
        self.assertEqual(settings.sandbox_timeout_seconds, 30)
        self.assertEqual(settings.sandbox_pids_limit, 64)
        self.assertEqual(settings.gemini_model, "gemini-2.5-flash-lite")
        self.assertEqual(
            settings.gemini_fallback_model, "gemini-3.1-flash-lite"
        )
        self.assertEqual(settings.gemini_timeout, 30)
        self.assertEqual(settings.gemini_max_retries, 2)
        self.assertEqual(settings.codex_cli, "codex")
        self.assertIsNone(settings.codex_model)
        self.assertEqual(settings.codex_timeout, 600)
        self.assertIsNone(settings.gemini_api_key)
        self.assertIsNone(settings.openai_api_key)
        self.assertIsNone(settings.nvidia_api_key)
        self.assertEqual(
            settings.nvidia_model, "nvidia/nemotron-3.5-lightning-30b-a3b"
        )
        self.assertEqual(
            settings.nvidia_base_url, "https://integrate.api.nvidia.com/v1"
        )
        self.assertEqual(settings.nvidia_timeout, 60)
        self.assertEqual(settings.nvidia_max_retries, 2)

    def test_reads_values_without_mutating_environment(self) -> None:
        settings = Settings.from_environment(
            {
                "OPENAI_API_KEY": " example-key ",
                "GEMINI_API_KEY": " gemini-example-key ",
                "GEMINI_MODEL": "gemini-test-model",
                "GEMINI_FALLBACK_MODEL": "gemini-fallback-test-model",
                "GEMINI_TIMEOUT": "12",
                "GEMINI_MAX_RETRIES": "3",
                "CODEX_CLI": "codex-test",
                "CODEX_MODEL": "test-model",
                "CODEX_TIMEOUT": "90",
                "MAX_REPAIR_ATTEMPTS": "2",
                "SANDBOX_CPU_LIMIT": "1.25",
                "SANDBOX_PIDS_LIMIT": "128",
            }
        )

        self.assertEqual(settings.openai_api_key, "example-key")
        self.assertEqual(settings.gemini_api_key, "gemini-example-key")
        self.assertEqual(settings.gemini_model, "gemini-test-model")
        self.assertEqual(
            settings.gemini_fallback_model, "gemini-fallback-test-model"
        )
        self.assertEqual(settings.gemini_timeout, 12)
        self.assertEqual(settings.gemini_max_retries, 3)
        self.assertEqual(settings.codex_cli, "codex-test")
        self.assertEqual(settings.codex_model, "test-model")
        self.assertEqual(settings.codex_timeout, 90)
        self.assertEqual(settings.max_repair_attempts, 2)
        self.assertEqual(settings.sandbox_cpu_limit, 1.25)
        self.assertEqual(settings.sandbox_pids_limit, 128)

    def test_rejects_negative_repair_attempts(self) -> None:
        with self.assertRaisesRegex(ValueError, "MAX_REPAIR_ATTEMPTS"):
            Settings.from_environment({"MAX_REPAIR_ATTEMPTS": "-1"})

    def test_rejects_invalid_resource_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "SANDBOX_MEMORY_LIMIT"):
            Settings.from_environment({"SANDBOX_MEMORY_LIMIT": "unlimited"})
        with self.assertRaisesRegex(ValueError, "SANDBOX_CPU_LIMIT"):
            Settings.from_environment({"SANDBOX_CPU_LIMIT": "inf"})
        with self.assertRaisesRegex(ValueError, "SANDBOX_PIDS_LIMIT"):
            Settings.from_environment({"SANDBOX_PIDS_LIMIT": "0"})

    def test_rejects_invalid_gemini_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "GEMINI_TIMEOUT"):
            Settings.from_environment({"GEMINI_TIMEOUT": "0"})
        with self.assertRaisesRegex(ValueError, "GEMINI_MODEL"):
            Settings.from_environment({"GEMINI_MODEL": " "})
        with self.assertRaisesRegex(ValueError, "GEMINI_MAX_RETRIES"):
            Settings.from_environment({"GEMINI_MAX_RETRIES": "6"})
        with self.assertRaisesRegex(ValueError, "must differ"):
            Settings.from_environment(
                {
                    "GEMINI_MODEL": "gemini-test",
                    "GEMINI_FALLBACK_MODEL": "gemini-test",
                }
            )

    def test_blank_fallback_model_disables_model_fallback(self) -> None:
        settings = Settings.from_environment({"GEMINI_FALLBACK_MODEL": ""})

        self.assertIsNone(settings.gemini_fallback_model)

    def test_nvidia_review_settings_are_configurable_and_validated(self) -> None:
        settings = Settings.from_environment(
            {
                "NVIDIA_API_KEY": "nvidia-example-key",
                "NVIDIA_MODEL": "nvidia/test-model",
                "NVIDIA_BASE_URL": "https://example.invalid/v1/",
                "NVIDIA_TIMEOUT": "15",
                "NVIDIA_MAX_RETRIES": "4",
            }
        )

        self.assertEqual(settings.nvidia_api_key, "nvidia-example-key")
        self.assertEqual(settings.nvidia_model, "nvidia/test-model")
        self.assertEqual(settings.nvidia_base_url, "https://example.invalid/v1")
        self.assertEqual(settings.nvidia_timeout, 15)
        self.assertEqual(settings.nvidia_max_retries, 4)

        with self.assertRaisesRegex(ValueError, "NVIDIA_BASE_URL"):
            Settings.from_environment({"NVIDIA_BASE_URL": "http://example.invalid"})
        with self.assertRaisesRegex(ValueError, "NVIDIA_MAX_RETRIES"):
            Settings.from_environment({"NVIDIA_MAX_RETRIES": "6"})
