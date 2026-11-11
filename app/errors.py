"""Domain exceptions and the FastAPI exception handlers.

The service layer raises the exceptions defined here (it never imports
``fastapi.HTTPException``), so the orchestration logic stays usable outside
FastAPI.  :func:`register_exception_handlers` maps them onto HTTP status codes
and a single, consistent error envelope::

    {"success": false,
     "message": "...",
     "request_id": "...",
     "error": {"code": "invalid_image", "message": "...", "details": null}}
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineError",
    "InvalidInputError",
    "SceneNotFoundError",
    "UnsupportedImageryError",
    "ModelNotReadyError",
    "AisSourceError",
    "DriftEngineError",
    "AttributionError",
    "EvaluationDataUnavailableError",
    "register_exception_handlers",
    "error_payload",
]


class PipelineError(Exception):
    """Base class for every recoverable, client-meaningful pipeline failure."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    default_message: str = "Internal pipeline error."

    def __init__(
        self,
        message: Optional[str] = None,
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Any = None,
        warnings: Optional[Sequence[str]] = None,
    ) -> None:
        self.message = message or self.default_message
        self.details = details
        self.warnings: list[str] = list(warnings or [])
        if code:
            self.code = code
        if status_code:
            self.status_code = status_code
        super().__init__(self.message)


class InvalidInputError(PipelineError):
    """Malformed or physically implausible request payload (HTTP 400)."""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "invalid_input"
    default_message = "The request payload is invalid."


class PayloadTooLargeError(InvalidInputError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "payload_too_large"
    default_message = "The uploaded scene exceeds the size limit."


class SceneNotFoundError(PipelineError):
    """Referenced scene/AIS artifact does not exist on the server (HTTP 404)."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "artifact_not_found"
    default_message = "The requested artifact was not found on the server."


class UnsupportedImageryError(InvalidInputError):
    """Input is not usable SAR/Sentinel backscatter imagery (HTTP 400)."""

    code = "invalid_image"
    default_message = "The uploaded raster is not usable SAR imagery."


class ModelNotReadyError(PipelineError):
    """Segmentation checkpoint could not be loaded (HTTP 503)."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_unavailable"
    default_message = (
        "The segmentation model is not loaded. Check the checkpoint path and "
        "application logs."
    )


class AisSourceError(InvalidInputError):
    """AIS source cannot be parsed or lacks required fields (HTTP 400)."""

    code = "invalid_ais_source"
    default_message = "The supplied AIS source could not be processed."


class DriftEngineError(PipelineError):
    """Drift engine failed upstream (bad NetCDF, OpenDrift error) (HTTP 502)."""

    status_code = status.HTTP_502_BAD_GATEWAY
    code = "drift_engine_failure"
    default_message = "The requested drift simulation could not be completed."


class AttributionError(PipelineError):
    """Attribution scoring failed (HTTP 500)."""

    code = "attribution_failure"
    default_message = "Vessel attribution could not be computed."


class EvaluationDataUnavailableError(SceneNotFoundError):
    """The unseen evaluation corpus is absent (HTTP 404)."""

    code = "evaluation_data_unavailable"
    default_message = "The authentic_testing evaluation corpus was not found."


def error_payload(exc: PipelineError) -> Mapping[str, Any]:
    """Dict form of a :class:`PipelineError`, useful for standalone scripts."""
    return {
        "code": exc.code,
        "message": exc.message,
        "details": exc.details,
        "status_code": exc.status_code,
    }


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
) -> JSONResponse:
    """Build the canonical error envelope for a request."""
    request_id = str(getattr(request.state, "request_id", "") or "")
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "request_id": request_id,
            "error": {"code": code, "message": message, "details": details},
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the error handlers to ``app`` (called once by the factory)."""

    @app.exception_handler(PipelineError)
    async def _pipeline_error(request: Request, exc: PipelineError) -> JSONResponse:
        log = logger.error if exc.status_code >= 500 else logger.warning
        log(
            "PipelineError [%s] %s (status=%s) path=%s",
            exc.code,
            exc.message,
            exc.status_code,
            request.url.path,
        )
        return _error_response(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Reshape FastAPI's 422 body into the shared error envelope."""
        errors = [
            {
                "loc": list(error.get("loc", [])),
                "msg": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        first = errors[0]["msg"] if errors else "Request validation failed."
        logger.warning("Request validation failed on %s: %s", request.url.path, first)
        return _error_response(
            request,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="validation_error",
            message=f"Request validation failed: {first}",
            details=errors,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = {
            400: "bad_request",
            404: "not_found",
            405: "method_not_allowed",
            413: "payload_too_large",
            415: "unsupported_media_type",
            503: "service_unavailable",
        }.get(exc.status_code, f"http_{exc.status_code}")
        return _error_response(
            request,
            status_code=exc.status_code,
            code=code,
            message=str(exc.detail),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception on %s: %s", request.url.path, exc, exc_info=True
        )
        return _error_response(
            request,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="internal_error",
            message="Unexpected server error. See server logs for details.",
            details={"exception": type(exc).__name__},
        )

