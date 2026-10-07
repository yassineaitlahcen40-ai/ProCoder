"""Run only the predetermined, non-AI Docker smoke-test program."""

import json
import sys
from dataclasses import asdict
from pathlib import Path

from core.config import PROJECT_ROOT, Settings
from core.logging import configure_logging
from sandbox.runner import DockerSandboxRunner


def main() -> int:
    settings = Settings.from_environment()
    logger = configure_logging(PROJECT_ROOT / "logs" / "procoder.jsonl")
    try:
        result = DockerSandboxRunner(settings).run_smoke_test()
    except (OSError, RuntimeError) as exc:
        logger.error(
            "sandbox_smoke_test_error",
            extra={
                "event_data": {
                    "event": "sandbox_smoke_test_error",
                    "error_type": type(exc).__name__,
                }
            },
        )
        print(f"Sandbox smoke test could not run: {exc}", file=sys.stderr)
        return 1

    logger.info(
        "sandbox_smoke_test_completed",
        extra={
            "event_data": {
                "event": "sandbox_smoke_test_completed",
                "passed": result.passed,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "output_truncated": result.output_truncated,
                "duration_seconds": result.duration_seconds,
            }
        },
    )
    print(json.dumps(asdict(result), ensure_ascii=True, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
