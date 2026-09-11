from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from .common import QueryText, StrictAPIModel


class QueryRequest(StrictAPIModel):
    query: QueryText


class SourceResponse(StrictAPIModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )

    rank: int = Field(ge=1)
    chunk_id: str
    page_title: str
    section_heading: str | None = None
    source_url: str | None = None
    score: float


class QueryResponse(StrictAPIModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )

    answer: str
    sources: list[SourceResponse]
    retrieved_count: int = Field(ge=0)
    reranked_count: int = Field(ge=0)
    model: str
    usage: dict[str, Any] | None = None
