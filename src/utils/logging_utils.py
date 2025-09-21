"""Structured logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_structured_logging(
    log_dir: Path,
    *,
    log_file: str = "runner.log",
    level: int = logging.INFO,
    fmt: str | None = None,
) -> logging.Logger:
    """Configure logging to write both to console and a dedicated log file."""

    log_dir.mkdir(parents=True, exist_ok=True)
    logfile_path = log_dir / log_file

    # Clear existing handlers to avoid duplicate logs
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt or "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(logfile_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root.addHandler(stream_handler)

    logger = logging.getLogger("runner")
    logger.propagate = True
    logger.setLevel(level)
    return logger
