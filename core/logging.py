"""Structured JSON Lines logging without recording prompts, code, or secrets."""

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event_data: dict[str, Any] = getattr(record, "event_data", {})
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": event_data.get("event", record.getMessage()),
            **{key: value for key, value in event_data.items() if key != "event"},
        }
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def configure_logging(
    log_path: Path, *, stream: bool = True
) -> logging.Logger:
    logger = logging.getLogger("procoder")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(JsonFormatter())
        logger.addHandler(stream_handler)
    return logger
