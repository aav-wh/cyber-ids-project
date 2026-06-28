"""
ids.utils.logging_config
------------------------
Structured logging setup for the AI-IDS project.

Call configure_logging() once at application startup to set up
console and (optionally) file handlers with a consistent format.
"""

from __future__ import annotations

import logging
import os
import sys


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    name: str = "ids",
) -> logging.Logger:
    """
    Configure and return the root IDS logger.

    Parameters
    ----------
    level    : logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    log_file : if given, also write logs to this file path
    name     : logger name (default 'ids')

    Returns
    -------
    configured logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "ids") -> logging.Logger:
    """
    Get a child logger under the 'ids' hierarchy.

    Usage
    -----
        logger = get_logger("ids.api")
        logger.info("Request received")
    """
    return logging.getLogger(name)
