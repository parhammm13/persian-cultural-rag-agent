from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from src.api.routes import health_router, query_router


PipelineBuilder = Callable[[], Any]


def build_default_pipeline() -> Any:
    """Import the heavy RAG factory only when the application starts."""
    from src.rag.factory import build_rag_pipeline

    return build_rag_pipeline()


def create_app(
    pipeline_builder: PipelineBuilder = build_default_pipeline,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag_pipeline = pipeline_builder()

        try:
            yield
        finally:
            app.state.rag_pipeline = None

    application = FastAPI(
        title="Persian Cultural RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.include_router(health_router)
    application.include_router(query_router)

    return application


app = create_app()
