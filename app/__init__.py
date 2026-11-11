"""FastAPI orchestration layer for the NTRO satellite oil-spill pipeline.

This package exposes the trained SAR segmentation model and every processing
module in ``src/`` through a documented HTTP API, while keeping the actual
orchestration logic importable **without** FastAPI.

Package layout
--------------
``app.config``    - settings resolved from ``config.yaml`` plus env overrides.
``app.schemas``   - Pydantic request/response contracts (the public API types).
``app.errors``    - domain exceptions and the HTTP exception handlers.
``app.services``  - FastAPI-free orchestration services (model, detection,
                    drift, attribution, end-to-end pipeline).
``app.api``       - route modules and dependency-injection wiring.
``app.main``      - ASGI application factory, lifespan and logging bootstrap.

Reusing ``src/``
----------------
The pre-existing processing modules live in ``SIH26143/src`` and are imported
as ``src.<module>`` (an implicit namespace package).  Importing :mod:`app`
therefore guarantees the project root is on ``sys.path`` so that both
``uvicorn app.main:app`` and ``pytest`` resolve those imports identically.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Absolute path of the ``SIH26143`` project root (the directory that holds
#: ``config.yaml``, ``src/``, ``models/`` and ``data/``).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = ["PROJECT_ROOT"]
