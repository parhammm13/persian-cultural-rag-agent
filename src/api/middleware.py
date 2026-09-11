from __future__ import annotations

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from src.api.request_context import (
    REQUEST_ID_HEADER,
    REQUEST_ID_STATE_KEY,
    reset_current_request_id,
    set_current_request_id,
)


logger = logging.getLogger("persian_cultural_rag.api")


def register_http_middleware(app: FastAPI) -> None:
    """Register request correlation and access logging middleware."""

    @app.middleware("http")
    async def request_context_middleware(
        request: Request,
        call_next,
    ) -> Response:
        request_id = uuid4().hex
        setattr(request.state, REQUEST_ID_STATE_KEY, request_id)
        request_id_token = set_current_request_id(request_id)
        started_at = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (perf_counter() - started_at) * 1_000
            logger.info(
                "request_completed request_id=%s method=%s path=%s "
                "status_code=500 latency_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                latency_ms,
            )
            raise
        else:
            latency_ms = (perf_counter() - started_at) * 1_000
            response.headers[REQUEST_ID_HEADER] = request_id
            logger.info(
                "request_completed request_id=%s method=%s path=%s "
                "status_code=%s latency_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                latency_ms,
            )
            return response
        finally:
            reset_current_request_id(request_id_token)
