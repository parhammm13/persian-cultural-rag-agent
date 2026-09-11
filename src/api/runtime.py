from __future__ import annotations

from typing import Protocol

from src.services.rag_service import RAGPipelineProtocol
from src.services.retrieval_service import RerankerProtocol, RetrieverProtocol


class ApplicationRuntimeProtocol(Protocol):
    """Shared long-lived resources created once during application startup."""

    pipeline: RAGPipelineProtocol
    retriever: RetrieverProtocol
    reranker: RerankerProtocol
    candidate_k: int
