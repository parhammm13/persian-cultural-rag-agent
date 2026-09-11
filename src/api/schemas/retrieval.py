from __future__ import annotations

from pydantic import ConfigDict, Field

from .common import QueryText, StrictAPIModel


class RetrieveRequest(StrictAPIModel):
    query: QueryText
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalResultResponse(StrictAPIModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )

    rank: int = Field(ge=1)
    chunk_id: str
    text: str
    page_title: str
    section_heading: str | None = None
    source_url: str | None = None
    score: float


class RetrieveResponse(StrictAPIModel):
    query: str
    results: list[RetrievalResultResponse]
    result_count: int = Field(ge=0)
