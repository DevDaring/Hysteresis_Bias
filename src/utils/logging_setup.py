"""
Structured logging setup for the Bias Hysteresis pipeline.

Provides consistent, timestamped logging across all scripts.

# ============================================================
# PAPER CITATIONS
# [1]-[16] — See configs/models.yaml for full citation list
# ============================================================
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.utils.config import get_project_root


def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Create a structured logger with file and console handlers.

    Args:
        name: Logger name (typically __name__ of calling module).
        log_file: Optional path to log file. If None, logs to
                  results/logs/{name}_{timestamp}.log.
        level: Logging level (default: INFO).
        console: Whether to also log to console (default: True).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Formatter
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    if log_file is None:
        log_dir = get_project_root() / "results" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.

    Convenience wrapper — if the logger was already set up, returns it;
    otherwise creates one with default settings.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
