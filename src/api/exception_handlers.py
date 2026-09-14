from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.request_context import REQUEST_ID_HEADER, get_request_id
from src.api.schemas.common import ErrorResponse


logger = logging.getLogger("persian_cultural_rag.api")


def _error_response(
    *,
    status_code: int,
    error: str,
    message: str,
    request_id: str | None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=error,
        message=message,
        request_id=request_id,
    )

    headers = (
        {REQUEST_ID_HEADER: request_id}
        if request_id
        else None
    )

    return JSONResponse(
        status_code=status_code,
        content=payload.model_dump(),
        headers=headers,
    )


def register_exception_handlers(
    app: FastAPI,
) -> None:
    """
    Register stable public error contracts.

    Validation details and internal exceptions are written to application
    logs for debugging, while public API responses avoid leaking internals.
    """

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = get_request_id(request)
        errors = exc.errors()

        logger.warning(
            "request_validation_failed "
            "request_id=%s "
            "method=%s "
            "path=%s "
            "error_count=%s "
            "errors=%s",
            request_id,
            request.method,
            request.url.path,
            len(errors),
            errors,
        )

        return _error_response(
            status_code=422,
            error="validation_error",
            message="Request validation failed.",
            request_id=request_id,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        request_id = get_request_id(request)

        if isinstance(exc.detail, str):
            message = exc.detail
        else:
            message = "HTTP request failed."

        logger.warning(
            "http_exception "
            "request_id=%s "
            "method=%s "
            "path=%s "
            "status_code=%s "
            "detail=%s",
            request_id,
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )

        return _error_response(
            status_code=exc.status_code,
            error="http_error",
            message=message,
            request_id=request_id,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        request_id = get_request_id(request)

        logger.exception(
            "unhandled_exception "
            "request_id=%s "
            "method=%s "
            "path=%s "
            "exception_type=%s",
            request_id,
            request.method,
            request.url.path,
            type(exc).__name__,
        )

        return _error_response(
            status_code=500,
            error="internal_server_error",
            message="An internal server error occurred.",
            request_id=request_id,
        )
