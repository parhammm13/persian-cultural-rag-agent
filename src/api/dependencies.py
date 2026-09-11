from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from src.services.rag_service import RAGPipelineProtocol, RAGService


def get_rag_pipeline(request: Request) -> RAGPipelineProtocol:
    pipeline = getattr(
        request.app.state,
        "rag_pipeline",
        None,
    )
    if pipeline is None:
        raise RuntimeError("RAG pipeline is not initialized.")

    return cast(RAGPipelineProtocol, pipeline)


def get_rag_service(
    pipeline: Annotated[
        RAGPipelineProtocol,
        Depends(get_rag_pipeline),
    ],
) -> RAGService:
    return RAGService(pipeline)
