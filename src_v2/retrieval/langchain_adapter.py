"""LangChain adapters for the existing retrieval implementation.

This module deliberately wraps the proven dense/BM25/parent-RRF pipeline
instead of replacing its ranking behavior.  The rest of the application can
use LangChain's standard ``BaseRetriever`` and ``Document`` contracts while
the retrieval backend is migrated incrementally.
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from .types import RetrievalResult


class RetrievalBackend(Protocol):
    """Minimal contract implemented by the existing hybrid retriever."""

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]: ...


def to_langchain_document(result: RetrievalResult) -> Document:
    """Convert one canonical retrieval result to a LangChain document."""
    if not isinstance(result, RetrievalResult):
        raise TypeError("result must be a RetrievalResult object.")

    metadata = dict(result.metadata)
    metadata.update(
        {
            "chunk_id": result.chunk_id,
            "page_title": result.page_title,
            "section_heading": result.section_heading,
            "source_url": result.source_url,
            "score": float(result.score),
        }
    )

    # Keep the old key available for code and stored payloads that still use it.
    metadata.setdefault("page_url", result.source_url)

    return Document(
        page_content=result.text,
        metadata=metadata,
    )


def from_langchain_document(
    document: Document,
    *,
    score: float | None = None,
) -> RetrievalResult:
    """Convert a LangChain document back to the canonical project result."""
    if not isinstance(document, Document):
        raise TypeError("document must be a LangChain Document object.")

    metadata: dict[str, Any] = dict(document.metadata)
    chunk_id = str(metadata.get("chunk_id", "")).strip()
    if not chunk_id:
        raise ValueError("document metadata must include a non-empty chunk_id.")

    resolved_score = metadata.get("score", 0.0) if score is None else score
    if isinstance(resolved_score, bool) or not isinstance(
        resolved_score,
        (int, float),
    ):
        raise TypeError("score must be numeric.")

    section_heading = metadata.get("section_heading")
    source_url = metadata.get("source_url") or metadata.get("page_url") or ""

    return RetrievalResult(
        score=float(resolved_score),
        chunk_id=chunk_id,
        text=document.page_content,
        page_title=str(metadata.get("page_title", "")),
        section_heading=(
            str(section_heading)
            if section_heading is not None
            else None
        ),
        source_url=str(source_url),
        metadata=metadata,
    )


class HybridParentRetriever(BaseRetriever):
    """Expose the existing hybrid parent-level retriever to LangChain."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    backend: RetrievalBackend
    top_k: int = Field(default=20, gt=0)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = self.backend.retrieve(
            query=query,
            top_k=self.top_k,
        )
        return [
            to_langchain_document(result)
            for result in results
        ]


__all__ = [
    "HybridParentRetriever",
    "RetrievalBackend",
    "from_langchain_document",
    "to_langchain_document",
]
