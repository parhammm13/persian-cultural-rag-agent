from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from src.api.runtime import ApplicationRuntimeProtocol
from src.services.rag_service import RAGService
from src.services.retrieval_service import RetrievalService


def get_runtime(request: Request) -> ApplicationRuntimeProtocol:
    runtime = getattr(request.app.state, "rag_runtime", None)
    if runtime is None:
        raise RuntimeError("RAG runtime is not initialized.")

    return cast(ApplicationRuntimeProtocol, runtime)


def get_rag_service(
    runtime: Annotated[
        ApplicationRuntimeProtocol,
        Depends(get_runtime),
    ],
) -> RAGService:
    return RAGService(runtime.pipeline)


def get_retrieval_service(
    runtime: Annotated[
        ApplicationRuntimeProtocol,
        Depends(get_runtime),
    ],
) -> RetrievalService:
    return RetrievalService(
        runtime.retriever,
        runtime.reranker,
        candidate_k=runtime.candidate_k,
    )
