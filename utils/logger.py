"""
MindSense AI - Structured Logger
==================================
Provides a centralized, structured logging system with:
  - Color-coded console output
  - Rotating file handler (JSON-structured logs)
  - Per-module named loggers
  - Configurable log levels

Usage::

    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Component started")
    logger.error("Something went wrong", exc_info=True)
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import colorlog; fall back gracefully if not installed
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


# ---------------------------------------------------------------------------
# JSON Formatter for structured file logging
# ---------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """
    Custom log formatter that outputs each record as a single JSON line.
    This is ideal for log aggregation systems (ELK, CloudWatch, etc.).
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON object.

        Args:
            record: The log record to format.

        Returns:
            JSON-serialized string representation of the log record.
        """
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Attach exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Attach any extra fields passed via extra={}
        extra_keys = set(record.__dict__.keys()) - {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
        }
        for key in extra_keys:
            log_entry[key] = record.__dict__[key]

        return json.dumps(log_entry, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------
_initialized: bool = False


def _initialize_root_logger() -> None:
    """
    Configure the root logger once for the entire application.
    Idempotent — safe to call multiple times.
    """
    global _initialized
    if _initialized:
        return

    # Import settings lazily to avoid circular imports
    try:
        from config import settings
        log_dir: Path = settings.logging.LOG_DIR
        log_file: str = settings.logging.LOG_FILE
        log_level_str: str = settings.logging.LOG_LEVEL
        max_bytes: int = settings.logging.MAX_BYTES
        backup_count: int = settings.logging.BACKUP_COUNT
    except Exception:
        # Fallback defaults if config isn't available yet
        log_dir = Path("logs")
        log_file = "mindsense.log"
        log_level_str = "INFO"
        max_bytes = 10 * 1024 * 1024
        backup_count = 5

    log_dir.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers to avoid duplicate entries
    root_logger.handlers.clear()

    # ---- Console handler (colored if colorlog is available) ----
    if HAS_COLORLOG:
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s%(reset)s | %(log_color)s%(levelname)-8s%(reset)s | "
            "%(cyan)s%(name)s%(reset)s | %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # ---- Rotating file handler (JSON structured) ----
    log_path = log_dir / log_file
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ["urllib3", "httpx", "httpcore", "transformers", "sentence_transformers"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a named logger for a specific module.

    Args:
        name: The logger name. Use ``__name__`` in each module.
              Defaults to the root logger if not provided.

    Returns:
        A configured :class:`logging.Logger` instance.

    Example::

        logger = get_logger(__name__)
        logger.info("MindSense AI initialized successfully")
    """
    _initialize_root_logger()
    return logging.getLogger(name or "mindsense")
