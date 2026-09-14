from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.exception_handlers import register_exception_handlers
from src.api.middleware import register_http_middleware
from src.api.routes import health_router, query_router, retrieve_router
from src.api.routes.chat import router as chat_router
from src.api.routes.conversations import router as conversations_router
from src.api.runtime import ApplicationRuntimeProtocol
from src.core.logging import configure_api_logging


RuntimeBuilder = Callable[[], ApplicationRuntimeProtocol]


def build_default_runtime() -> ApplicationRuntimeProtocol:
    """Import and construct heavy RAG resources only at application startup."""
    from src.rag.factory import build_rag_runtime

    return build_rag_runtime()


def create_app(
    runtime_builder: RuntimeBuilder = build_default_runtime,
) -> FastAPI:
    configure_api_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag_runtime = runtime_builder()

        try:
            yield
        finally:
            app.state.rag_runtime = None

    application = FastAPI(
        title="Persian Cultural RAG API",
        version="0.3.0",
        lifespan=lifespan,
    )

    register_http_middleware(application)
    register_exception_handlers(application)

    application.include_router(health_router)
    application.include_router(query_router)
    application.include_router(retrieve_router)
    application.include_router(conversations_router)
    application.include_router(chat_router)

    return application


app = create_app()
