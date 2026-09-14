from __future__ import annotations

from collections.abc import Generator
from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from src.api.runtime import ApplicationRuntimeProtocol
from src.database.session import get_session_factory
from src.services.chat_service import ChatService
from src.services.conversation_service import ConversationService
from src.services.rag_service import RAGService
from src.services.retrieval_service import RetrievalService


def get_runtime(request: Request) -> ApplicationRuntimeProtocol:
    runtime = getattr(request.app.state, "rag_runtime", None)
    if runtime is None:
        raise RuntimeError("RAG runtime is not initialized.")

    return cast(ApplicationRuntimeProtocol, runtime)


def get_db() -> Generator[Session, None, None]:
    """Provide one SQLAlchemy Session per FastAPI request."""
    session_factory = get_session_factory()
    db = session_factory()

    try:
        yield db
    finally:
        db.close()


DatabaseSession = Annotated[
    Session,
    Depends(get_db),
]


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


def get_conversation_service(
    db: DatabaseSession,
) -> ConversationService:
    return ConversationService(db)


def get_chat_service(
    db: DatabaseSession,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> ChatService:
    return ChatService(
        db,
        rag_service,
    )
