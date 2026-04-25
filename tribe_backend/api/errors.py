"""Uniform error envelope + FastAPI exception handlers.

Shape (per plan §9 Error Catalog):

    {"error": {"code": "<machine-readable>", "message": "<human-readable>"}}

Tracebacks are never leaked to the client. The plan's machine-readable codes
are reused here verbatim so the §9 docs stay accurate.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def error_body(*, code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


class APIError(Exception):
    """Convenience for raising structured HTTP errors from handlers."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Compose a single human-readable message from the first error.
    errs = exc.errors()
    if errs:
        first = errs[0]
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        msg = first.get("msg", "validation error")
        message = f"{loc}: {msg}" if loc else msg
    else:
        message = "validation error"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_body(code="validation_error", message=message),
    )


def _http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    # If the detail is already an envelope, pass it through; else wrap.
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    code = _default_code_for(exc.status_code)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code=code, message=str(detail)),
    )


def _api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code=exc.code, message=exc.message),
    )


def _inference_failure_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("InferenceFailure during /v1/runs")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_body(code="INFERENCE_FAILURE", message=str(exc) or "inference failed"),
    )


def _generic_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled exception in API handler")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_body(code="internal_error", message="internal server error"),
    )


def _default_code_for(status_code: int) -> str:
    return {
        400: "bad_request",
        404: "not_found",
        409: "conflict",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        503: "service_unavailable",
    }.get(status_code, "error")


def install_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the given FastAPI app."""
    # InferenceFailure is re-exported by the control composition root, so the
    # api package never imports tribe_backend.inference directly.
    from tribe_backend.control.factory import InferenceFailure

    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_handler)
    app.add_exception_handler(APIError, _api_error_handler)
    app.add_exception_handler(InferenceFailure, _inference_failure_handler)
    app.add_exception_handler(Exception, _generic_handler)
