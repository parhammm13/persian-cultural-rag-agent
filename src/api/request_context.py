from __future__ import annotations

from contextvars import ContextVar, Token

from fastapi import Request


REQUEST_ID_STATE_KEY = "request_id"
REQUEST_ID_HEADER = "X-Request-ID"

_current_request_id: ContextVar[str | None] = ContextVar(
    "current_request_id",
    default=None,
)


def set_current_request_id(request_id: str) -> Token[str | None]:
    """Bind a request ID to the current execution context."""
    return _current_request_id.set(request_id)


def reset_current_request_id(token: Token[str | None]) -> None:
    """Restore the previous request ID context."""
    _current_request_id.reset(token)


def get_current_request_id() -> str | None:
    """Return the request ID available to services/loggers in this context."""
    return _current_request_id.get()


def get_request_id(request: Request) -> str | None:
    """Return the request correlation ID set by the HTTP middleware."""
    return getattr(request.state, REQUEST_ID_STATE_KEY, None)
