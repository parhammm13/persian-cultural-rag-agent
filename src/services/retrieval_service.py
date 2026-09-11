from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from src.api.request_context import get_current_request_id


logger = logging.getLogger("persian_cultural_rag.api")


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, top_k: int) -> Sequence[Any]:
        ...


class RerankerProtocol(Protocol):
    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        top_k: int,
    ) -> Sequence[Any]:
        ...


@dataclass(frozen=True, slots=True)
class RankedRetrievalResult:
    rank: int
    chunk_id: str
    text: str
    page_title: str
    section_heading: str | None
    source_url: str | None
    score: float


class RetrievalService:
    """Run hybrid retrieval + reranking without generation."""

    def __init__(
        self,
        retriever: RetrieverProtocol,
        reranker: RerankerProtocol,
        *,
        candidate_k: int,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be > 0")

        self._retriever = retriever
        self._reranker = reranker
        self._candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
    ) -> tuple[RankedRetrievalResult, ...]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        candidate_count = max(self._candidate_k, top_k)
        candidates = tuple(
            self._retriever.retrieve(
                normalized_query,
                top_k=candidate_count,
            )
        )

        reranked = tuple(
            self._reranker.rerank(
                normalized_query,
                candidates,
                top_k=top_k,
            )
        )

        results = tuple(
            RankedRetrievalResult(
                rank=rank,
                chunk_id=item.chunk_id,
                text=item.text,
                page_title=item.page_title,
                section_heading=item.section_heading,
                source_url=item.source_url,
                score=float(item.score),
            )
            for rank, item in enumerate(reranked, start=1)
        )

        logger.info(
            "retrieval_completed request_id=%s candidate_count=%s "
            "reranked_count=%s requested_top_k=%s",
            get_current_request_id(),
            len(candidates),
            len(results),
            top_k,
        )

        return results
