"""Application-level logging configuration.

Uses the same human-readable format as ``src/run_pipeline.py`` so that API
logs, pipeline logs and CLI logs are visually consistent:

    ``2026-09-27 10:00:00 [INFO] (OilSpillAPI) application startup complete``

The root logger is configured exactly once (``configure_logging`` is
idempotent), which keeps ``pytest`` runs free of duplicated handlers.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict

LOG_FORMAT = "%(asctime)s [%(levelname)s] (%(name)s) %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Environment variable used to override the root log level (default ``INFO``).
LOG_LEVEL_ENV = "OILSPILL_LOG_LEVEL"

_CONFIGURED = False


def configure_logging(
    level: str | int | None = None,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Configure the root logger once and return a small summary dict.

    Args:
        level: Numeric level or level name; falls back to
            ``OILSPILL_LOG_LEVEL`` and then to ``INFO``.
        force: Re-apply the configuration even if it was already applied.

    Returns:
        Mapping with ``level`` (effective level name) and ``configured``
        (whether this call performed the configuration).
    """
    global _CONFIGURED

    resolved: int | str = (
        level if level is not None else os.environ.get(LOG_LEVEL_ENV, "INFO")
    )
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)

    if _CONFIGURED and not force:
        logging.getLogger().setLevel(resolved)
        return {"level": logging.getLevelName(resolved), "configured": False}

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved)

    # Third-party noise control: OpenDrift and rasterio are extremely chatty.
    for noisy, noisy_level in (
        ("opendrift", logging.WARNING),
        ("rasterio", logging.WARNING),
        ("matplotlib", logging.WARNING),
        ("urllib3", logging.WARNING),
    ):
        logging.getLogger(noisy).setLevel(noisy_level)

    _CONFIGURED = True
    logging.getLogger("OilSpillAPI").debug(
        "Logging configured at level %s", logging.getLevelName(resolved)
    )
    return {"level": logging.getLevelName(resolved), "configured": True}
