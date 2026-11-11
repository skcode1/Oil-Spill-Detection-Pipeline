"""HTTP layer of the oil-spill pipeline.

``routes`` holds the endpoints, ``deps`` the singletons they share.  Only this
package imports FastAPI: everything under ``app/services`` stays usable from
scripts, the CLI in ``src/`` and the pytest suite.
"""

from __future__ import annotations

from app.api.deps import (
    AppServices,
    build_services,
    evaluation_dataset_info,
    get_drift_service,
    get_pipeline_service,
    health_report,
)
from app.api.routes import router

__all__ = [
    "AppServices",
    "build_services",
    "evaluation_dataset_info",
    "get_drift_service",
    "get_pipeline_service",
    "health_report",
    "router",
]
