from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints


QueryText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=2,
        max_length=1_000,
    ),
]


class StrictAPIModel(BaseModel):
    """Base class for public API contracts."""

    model_config = ConfigDict(
        extra="forbid",
    )


class HealthResponse(StrictAPIModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str


class ErrorResponse(StrictAPIModel):
    error: str
    message: str
    request_id: str | None = None
