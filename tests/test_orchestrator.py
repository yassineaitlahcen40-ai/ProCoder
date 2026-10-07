import logging
import unittest
from unittest.mock import Mock

from core.models import (
    GeneratedCode,
    RepairInstructions,
    ReviewDecision,
    Specification,
    TestResult,
)
from orchestrator.workflow import Orchestrator


class OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prompt = Mock()
        self.prompt.create_specification.return_value = Specification(
            request="make a program", summary="A test program", tokens_used=10
        )
        self.coder = Mock()
        self.coder.generate.return_value = GeneratedCode(
            files={"main.py": "print('safe test fixture')"}, tokens_used=20
        )
        self.coder.repair.return_value = GeneratedCode(
            files={"main.py": "print('repaired test fixture')"}, tokens_used=30
        )
        self.repair = Mock()
        self.repair.create_instructions.return_value = RepairInstructions(
            instructions="Fix the failing check.", tokens_used=5
        )
        self.review = Mock()
        self.logger = Mock(spec=logging.Logger)

    def make_orchestrator(self, sandbox: Mock, max_attempts: int = 2) -> Orchestrator:
        return Orchestrator(
            self.prompt,
            self.coder,
            sandbox,
            self.review,
            self.repair,
            max_repair_attempts=max_attempts,
            logger=self.logger,
        )

    @staticmethod
    def _make_test_result(passed: bool) -> TestResult:
        return TestResult(
            passed=passed,
            exit_code=0 if passed else 1,
            stdout="real sandbox output",
            stderr="",
            duration_seconds=0.01,
        )

    def test_passes_after_one_repair_using_real_test_evidence(self) -> None:
        sandbox = Mock()
        first_result = self._make_test_result(False)
        sandbox.run_project.side_effect = [
            first_result,
            self._make_test_result(True),
        ]
        self.review.review.side_effect = [
            ReviewDecision(False, "Test failed."),
            ReviewDecision(True, "Tests and review pass."),
        ]

        result = self.make_orchestrator(sandbox).run("make a program")

        self.assertTrue(result.passed)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.repair_attempts, 1)
        self.assertEqual(result.tokens_used, 65)
        self.assertEqual(len(result.test_results), 2)
        self.repair.create_instructions.assert_called_once()
        self.coder.repair.assert_called_once()
        self.review.review.assert_any_call(
            self.prompt.create_specification.return_value,
            self.coder.generate.return_value,
            first_result,
        )

    def test_review_cannot_override_a_failed_test_result(self) -> None:
        sandbox = Mock()
        sandbox.run_project.return_value = self._make_test_result(False)
        self.review.review.return_value = ReviewDecision(True, "Looks correct.")

        result = self.make_orchestrator(sandbox, max_attempts=0).run("make a program")

        self.assertFalse(result.passed)
        self.assertEqual(result.status, "max_repair_attempts_reached")
        self.assertEqual(result.repair_attempts, 0)
        self.repair.create_instructions.assert_not_called()
        self.coder.repair.assert_not_called()

    def test_stops_after_configured_repair_limit(self) -> None:
        sandbox = Mock()
        sandbox.run_project.return_value = self._make_test_result(False)
        self.review.review.return_value = ReviewDecision(False, "Still failing.")

        result = self.make_orchestrator(sandbox, max_attempts=1).run("make a program")

        self.assertFalse(result.passed)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.repair_attempts, 1)
        self.assertEqual(sandbox.run_project.call_count, 2)

    def test_rejects_blank_requests(self) -> None:
        sandbox = Mock()

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.make_orchestrator(sandbox).run("  ")

        self.prompt.create_specification.assert_not_called()
