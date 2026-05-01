"""Centralised logging setup for Pokemon TCG OCR."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-20s — %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"

# Root logger for the entire src package.
_ROOT = "src"


def setup(log_file: str | None = None, verbose: bool = False) -> None:
    """Configure logging for the application.

    - DEBUG+ goes to *log_file* (when provided).
    - WARNING+ always goes to stderr so the terminal still shows problems.
    - When *verbose* is True, INFO+ also goes to stderr.
    """
    root = logging.getLogger(_ROOT)
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO if verbose else logging.WARNING)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def get(name: str) -> logging.Logger:
    """Return a child logger scoped under the src package."""
    return logging.getLogger(f"{_ROOT}.{name}")
