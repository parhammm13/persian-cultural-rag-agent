from __future__ import annotations

import logging
from typing import Any, Protocol

from src.api.request_context import get_current_request_id


logger = logging.getLogger("persian_cultural_rag.api")


class RAGPipelineProtocol(Protocol):
    """Minimum contract required from the existing RAGPipeline."""

    def run(self, query: str) -> Any:
        ...


class RAGService:
    """Thin application service around the existing RAG pipeline."""

    def __init__(self, pipeline: RAGPipelineProtocol) -> None:
        self._pipeline = pipeline

    def query(self, query: str) -> Any:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        result = self._pipeline.run(normalized_query)

        logger.info(
            "rag_query_completed request_id=%s retrieved_count=%s "
            "reranked_count=%s model=%s",
            get_current_request_id(),
            getattr(result, "retrieved_count", None),
            getattr(result, "reranked_count", None),
            getattr(result, "model", None),
        )

        return result
