import io
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bridge import BridgeServer, PROTOCOL_VERSION
from core.config import Settings
from core.models import GeneratedCode
from workspace.generated_project import GeneratedProjectWriter


class BridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger(f"test.procoder.bridge.{id(self)}")
        self.logger.handlers.clear()
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        self.output = io.StringIO()
        self.server = BridgeServer(
            Settings.from_environment({}),
            self.logger,
            input_stream=io.StringIO(),
            output_stream=self.output,
        )
        self.addCleanup(self.logger.removeHandler, self.server._log_handler)
        self.addCleanup(self.server._log_handler.close)

    def events(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.output.getvalue().splitlines()]

    def test_bridge_emits_versioned_jsonl_ready_event(self) -> None:
        self.server.serve()
        ready = self.events()[0]
        self.assertEqual(ready["type"], "ready")
        self.assertEqual(ready["protocol_version"], PROTOCOL_VERSION)
        capabilities = ready["capabilities"]
        self.assertTrue(capabilities["transcript_confirmation"])
        self.assertFalse(capabilities["workflow_cancellation"])

    def test_bridge_rejects_unversioned_or_future_protocol_commands(self) -> None:
        self.server._dispatch({"type": "shutdown"})
        self.assertEqual(self.events()[-1]["type"], "protocol_error")
        self.server._dispatch({"protocol_version": 2, "type": "shutdown"})
        self.assertEqual(self.events()[-1]["type"], "protocol_error")

    def test_rejected_transcript_never_starts_workflow(self) -> None:
        self.server._pending_transcript = "Build a calculator."
        with patch.object(self.server, "_start_operation") as start:
            self.server._dispatch(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "type": "transcript_decision",
                    "approved": False,
                }
            )
        start.assert_not_called()
        self.assertIsNone(self.server._pending_transcript)
        self.assertEqual(self.events()[-1]["type"], "transcript_rejected")

    def test_approved_transcript_is_validated_before_workflow(self) -> None:
        self.server._pending_transcript = "Build a calculator."
        captured: dict[str, object] = {}

        def capture(operation: str, target: object) -> None:
            captured["operation"] = operation
            captured["target"] = target

        with patch.object(self.server, "_start_operation", side_effect=capture):
            self.server._dispatch(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "type": "transcript_decision",
                    "approved": True,
                    "transcript": "  Build a calculator.  ",
                }
            )
        self.assertEqual(captured["operation"], "workflow")
        self.assertIsNone(self.server._pending_transcript)
        self.assertEqual(self.events()[-1]["type"], "transcript_approved")

    def test_workflow_events_and_final_file_list_are_structured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = "bcf3a60b-f031-41a3-830d-ae941f8b948d"
            writer = GeneratedProjectWriter(root)
            writer.create_run_directory(run_id)
            writer.write(
                run_id,
                GeneratedCode(files={"src/main.py": "print('safe')\n"}),
            )

            def fake_run(_request, _settings, logger, **_kwargs) -> int:
                logger.info(
                    "workflow_started",
                    extra={
                        "event_data": {"event": "workflow_started", "run_id": run_id}
                    },
                )
                logger.info(
                    "testing_completed",
                    extra={
                        "event_data": {
                            "event": "testing_completed",
                            "run_id": run_id,
                            "attempt": 0,
                            "passed": True,
                            "exit_code": 0,
                            "timed_out": False,
                            "infrastructure_error": False,
                        }
                    },
                )
                logger.info(
                    "review_completed",
                    extra={
                        "event_data": {
                            "event": "review_completed",
                            "run_id": run_id,
                            "attempt": 0,
                            "verdict": "PASS",
                            "specification_satisfied": True,
                            "tests_passed": True,
                            "repair_required": False,
                            "confidence": 0.9,
                        }
                    },
                )
                logger.info(
                    "repair_workflow_completed",
                    extra={
                        "event_data": {
                            "event": "repair_workflow_completed",
                            "run_id": run_id,
                            "success": True,
                            "repair_attempts": 0,
                            "termination_reason": "SUCCESS",
                            "final_verdict": "PASS",
                        }
                    },
                )
                return 0

            self.server._project_root = root
            with patch("bridge.run_generate_command", side_effect=fake_run):
                self.server._run_workflow("Do not echo this request")

        events = self.events()
        final = events[-1]
        self.assertEqual(final["type"], "workflow_completed")
        self.assertTrue(final["success"])
        self.assertEqual(final["files"], ["src/main.py"])
        self.assertEqual(final["docker"]["passed"], True)
        self.assertEqual(final["nvidia"]["verdict"], "PASS")
        self.assertNotIn("Do not echo this request", self.output.getvalue())

    def test_run_failure_event_does_not_forward_untrusted_log_fields(self) -> None:
        self.logger.warning(
            "generation failed",
            extra={
                "event_data": {
                    "event": "code_generation_failed",
                    "run_id": "bcf3a60b-f031-41a3-830d-ae941f8b948d",
                    "phase": "coding",
                    "error_code": "CodingAgentError",
                    "secret": "must-not-forward",
                }
            },
        )
        failure = self.events()[-1]
        self.assertEqual(failure["type"], "workflow_stage_failed")
        self.assertEqual(failure["message"], "Code generation failed.")
        self.assertNotIn("secret", failure)
        self.assertNotIn("must-not-forward", self.output.getvalue())

    def test_workflow_cancellation_is_not_reported_as_successful(self) -> None:
        self.server._active_operation = "workflow"
        self.server._dispatch({"protocol_version": PROTOCOL_VERSION, "type": "cancel"})
        event = self.events()[-1]
        self.assertEqual(event["type"], "cancel_unavailable")
        self.assertIn("unavailable", event["message"])


if __name__ == "__main__":
    unittest.main()
