from __future__ import annotations

from collections.abc import Generator
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from src.api.runtime import ApplicationRuntimeProtocol
from src.core.security import (
    InvalidTokenError,
    JWTSettings,
    decode_access_token,
)
from src.database.models.user import User
from src.database.repositories.users import UserRepository
from src.database.session import get_session_factory
from functools import lru_cache

from src.services.auth_service import AuthService
from src.services.chat_service import ChatService
from src.services.conversation_router import ConversationRouter
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

@lru_cache(maxsize=1)
def get_conversation_router() -> ConversationRouter:
    return ConversationRouter.from_env()


@lru_cache(maxsize=1)
def get_jwt_settings() -> JWTSettings:
    return JWTSettings.from_env()


def get_auth_service(
    db: DatabaseSession,
) -> AuthService:
    return AuthService(db, get_jwt_settings())


_bearer_scheme = HTTPBearer(
    bearerFormat="JWT",
    description="Paste the access token from POST /auth/login.",
    # auto_error=False so a missing token yields our stable 401 contract
    # instead of FastAPI's default 403.
    auto_error=False,
)


def get_current_user(
    db: DatabaseSession,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(_bearer_scheme),
    ],
) -> User:
    """Resolve the authenticated user from a Bearer JWT access token."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="not_authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = decode_access_token(
            token=credentials.credentials,
            settings=get_jwt_settings(),
        )
    except InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_or_expired_token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user = UserRepository(db).get_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_or_expired_token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_chat_service(
    db: DatabaseSession,
    rag_service: Annotated[
        RAGService,
        Depends(get_rag_service),
    ],
) -> ChatService:
    from src.rag.observability import setup_phoenix_tracer

    pipeline = getattr(rag_service, "_pipeline", None)
    tracer = getattr(pipeline, "tracer", None)
    if tracer is None:
        tracer = setup_phoenix_tracer()

    return ChatService(
        db=db,
        rag_service=rag_service,
        router=get_conversation_router(),
        tracer=tracer,
    )