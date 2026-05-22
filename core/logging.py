"""
core/logging.py — Structured logging setup for the entire process.

Responsibility:
    Configures structlog + stdlib logging once at startup.  JSON output
    in production (for Railway log drain), human-readable colour console
    in development.  Every logger created via get_logger() carries a
    ``component`` key so log lines can be filtered by subsystem.

Blast radius on failure:
    LOW-MEDIUM.  If logging setup fails the process will crash on
    startup.  If a logger call fails at runtime the exception propagates
    to the caller — but structlog is extremely stable so this is
    unlikely.  Worst case: you lose observability, not functionality.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(*, env: str = "development") -> None:
    """
    setup_logging(env: str = "development") -> None

    Configure structlog + stdlib logging for the whole process.
    Call once at startup, before any logger is created.  Selects JSON
    renderer for production and colour console for development.

    On failure: raises if structlog configuration is invalid (should
    never happen with static config).  Safe to call multiple times —
    handlers are cleared before reconfiguration.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if env == "production":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Silence noisy libs
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    """
    get_logger(component: str) -> structlog.stdlib.BoundLogger

    Return a logger pre-bound with the given component name.
    All log lines emitted through this logger will carry
    ``component=<component>`` for easy filtering.

    On failure: never fails — structlog.get_logger() always returns a
    usable logger, even if setup_logging() was never called (falls back
    to stdlib defaults).
    """
    return structlog.get_logger(component=component)
