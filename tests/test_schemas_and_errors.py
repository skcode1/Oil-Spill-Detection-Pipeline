from __future__ import annotations
import pytest
from app.config import InferenceSettings
from app.errors import AisSourceError, DriftEngineError, EvaluationDataUnavailableError, InvalidInputError, ModelNotReadyError, PayloadTooLargeError, SceneNotFoundError, UnsupportedImageryError
from app.schemas import DetectionOptions, DriftRequest, PipelineOptions

def test_detection_options_reject_empty_confidence_range():
    with pytest.raises(ValueError):
        DetectionOptions(confidence_range=[0.9, 0.1])

def test_pipeline_options_require_ais_for_attribution():
    with pytest.raises(ValueError):
        PipelineOptions(run_attribution=True)

def test_pipeline_options_attribution_free_ok():
    opts = PipelineOptions(run_attribution=False)
    assert opts.run_drift is True and opts.run_forecast is True

def test_drift_request_rejects_blank_engine():
    with pytest.raises(ValueError):
        DriftRequest(engine=' ')

def test_error_status_contract():
    assert InvalidInputError('x').status_code == 400
    assert SceneNotFoundError('x').status_code == 404
    assert ModelNotReadyError('x').status_code == 503
    assert PayloadTooLargeError('x').status_code == 413
    assert UnsupportedImageryError('x').code == 'invalid_image'
    assert AisSourceError('x').status_code == 400
    assert DriftEngineError('x').status_code == 502
    assert EvaluationDataUnavailableError('x').status_code == 404

def test_inference_defaults_validate():
    InferenceSettings().validate()
    with pytest.raises(ValueError):
        InferenceSettings(morphology_kernel_size=4).validate()
    with pytest.raises(ValueError):
        InferenceSettings(contour_epsilon_fraction=0.5).validate()