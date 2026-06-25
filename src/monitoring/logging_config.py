"""
Logging Configuration - Structured JSON logging for production.
"""

import logging
import logging.config
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure structured logging for production use.
    Uses JSON format for log aggregation tools (ELK, Datadog, CloudWatch).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = ["console"]
    if log_file:
        handlers.append("file")

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%SZ",
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": format_type,
                "level": log_level,
            },
        },
        "root": {"level": log_level, "handlers": handlers},
        "loggers": {
            "uvicorn": {"level": "INFO", "propagate": True},
            "fastapi": {"level": "INFO", "propagate": True},
            "sentence_transformers": {"level": "WARNING", "propagate": True},
            "chromadb": {"level": "WARNING", "propagate": True},
        },
    }

    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file,
            "maxBytes": 100 * 1024 * 1024,  # 100MB
            "backupCount": 5,
            "formatter": "json",
            "level": log_level,
        }

    logging.config.dictConfig(config)
    logging.getLogger(__name__).info(
        f"Logging configured: level={level}, format={format_type}"
    )
