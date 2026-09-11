from __future__ import annotations

from typing import Any, Protocol


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

        return self._pipeline.run(normalized_query)
