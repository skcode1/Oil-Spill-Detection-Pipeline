"""Services layer: FastAPI-free orchestration of the whole pipeline.

Modules
-------
``geo``                   geodesy, error ellipses, polygon helpers
``image_io``              SAR scene loading, validation, artifact export
``model_service``         the trained U-Net (lifecycle + batched inference)
``detection_service``     segmentation -> geometry -> statistics -> age
``age_estimator``         heuristic spill-age inversion from slick extent
``drift_service``         OpenDrift (``src/``) + built-in Lagrangian engine
``attribution_service``   AIS traffic reconstruction, pruning, ranking
``pipeline_service``      end-to-end detection + drift + attribution workflow

Every service is importable and runnable without FastAPI, e.g.::

    from app.config import load_settings
    from app.services import DetectionService, SpillSegmentationModel

    settings = load_settings()
    model = SpillSegmentationModel(settings).load_sync()
"""

from __future__ import annotations

from app.services.age_estimator import estimate_spill_age
from app.services.attribution_service import (
    AttributionService,
    load_pruner,
    load_scorer,
)
from app.services.briefing_service import BriefingService
from app.services.detection_service import DetectionOutcome, DetectionService
from app.services.drift_service import DriftService
from app.services.model_service import InferenceResult, SpillSegmentationModel
from app.services.pipeline_service import PipelineService

__all__ = [
    "AttributionService",
    "BriefingService",
    "DetectionOutcome",
    "DetectionService",
    "DriftService",
    "InferenceResult",
    "PipelineService",
    "SpillSegmentationModel",
    "estimate_spill_age",
    "load_pruner",
    "load_scorer",
]
