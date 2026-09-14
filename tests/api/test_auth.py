from __future__ import annotations

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api import dependencies as deps
from src.api.main import create_app
from src.core.security import JWTSettings, create_access_token
from src.database.base import Base
from src.database.models import User
from src.services.chat_service import ChatService
from src.services.conversation_router import (
    ConversationRoute,
    RouteDecision,
)


TEST_JWT_SECRET = "test-secret-key-0123456789abcdef-0123456789abcdef-xyz"


class FakePipeline:
    tracer = None

    def run(self, query: str):
        raise AssertionError("pipeline should not be called")


class FakeRetriever:
    def retrieve(self, query: str, top_k: int):
        return ()


class FakeReranker:
    def rerank(self, query: str, candidates, top_k: int):
        return ()


class FakeRuntime:
    def __init__(self) -> None:
        self.pipeline = FakePipeline()
        self.retriever = FakeRetriever()
        self.reranker = FakeReranker()
        self.candidate_k = 20


class StubRouter:
    capture_content = False

    def __init__(self, standalone_query: str = "standalone query") -> None:
        self.standalone_query = standalone_query
        self.seen: list[tuple[str, object]] = []

    def route(self, *, message: str, history) -> RouteDecision:
        self.seen.append((message, history))
        return RouteDecision(
            route=ConversationRoute.RAG_REQUIRED,
            standalone_query=self.standalone_query,
        )


class StubRAGResult:
    answer = "پاسخ تستی"
    sources = ()


class StubRAGService:
    def __init__(self) -> None:
        self.received_queries: list[str] = []

    def query(self, query: str) -> StubRAGResult:
        self.received_queries.append(query)
        return StubRAGResult()


@pytest.fixture()
def app_client(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", TEST_JWT_SECRET)
    monkeypatch.setenv("JWT_ALGORITHM", "HS256")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    monkeypatch.setenv("PHOENIX_TRACING_ENABLED", "0")
    deps.get_jwt_settings.cache_clear()

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    def override_get_db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    application = create_app(FakeRuntime)
    application.dependency_overrides[deps.get_db] = override_get_db

    with TestClient(application) as client:
        yield client, session_factory

    deps.get_jwt_settings.cache_clear()


def register(client: TestClient, email: str, password: str = "s3cret-Pass") -> dict:
    response = client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )
    assert response.status_code == 201, response.text
    return response.json()


def login(client: TestClient, email: str, password: str = "s3cret-Pass") -> str:
    response = client.post(
        "/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["token_type"] == "bearer"
    return body["access_token"]


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_register_stores_hash_not_plaintext(app_client) -> None:
    client, session_factory = app_client

    body = register(client, "user@example.com")

    assert body["id"] > 0
    assert body["email"] == "user@example.com"

    with session_factory() as db:
        stored = db.scalar(
            select(User).where(User.email == "user@example.com")
        )
        assert stored is not None
        assert stored.password_hash != "s3cret-Pass"
        assert stored.password_hash.startswith("$2")


def test_register_rejects_duplicate_email(app_client) -> None:
    client, _ = app_client

    register(client, "dupe@example.com")
    response = client.post(
        "/auth/register",
        json={"email": "dupe@example.com", "password": "s3cret-Pass"},
    )

    assert response.status_code == 409
    assert response.json()["message"] == "email_already_registered"


def test_register_rejects_invalid_payload(app_client) -> None:
    client, _ = app_client

    assert (
        client.post(
            "/auth/register",
            json={"email": "not-an-email", "password": "s3cret-Pass"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/auth/register",
            json={"email": "short@example.com", "password": "short"},
        ).status_code
        == 422
    )


def test_login_returns_token_with_user_subject(app_client) -> None:
    client, _ = app_client

    created = register(client, "login@example.com")
    token = login(client, "login@example.com")

    settings = JWTSettings(
        secret_key=TEST_JWT_SECRET,
        algorithm="HS256",
        access_token_expire_minutes=60,
    )
    from src.core.security import decode_access_token

    assert decode_access_token(token=token, settings=settings) == created["id"]


def test_login_rejects_bad_credentials(app_client) -> None:
    client, _ = app_client

    register(client, "cred@example.com")

    wrong_password = client.post(
        "/auth/login",
        json={"email": "cred@example.com", "password": "wrong-Pass1"},
    )
    assert wrong_password.status_code == 401
    assert wrong_password.json()["message"] == "invalid_credentials"

    unknown_email = client.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "s3cret-Pass"},
    )
    assert unknown_email.status_code == 401
    assert unknown_email.json()["message"] == "invalid_credentials"


def test_me_needs_token_and_returns_current_user(app_client) -> None:
    client, _ = app_client

    assert client.get("/auth/me").status_code == 401

    register(client, "me@example.com")
    token = login(client, "me@example.com")

    response = client.get("/auth/me", headers=auth_headers(token))
    assert response.status_code == 200
    assert response.json()["email"] == "me@example.com"


def test_malformed_and_expired_tokens_rejected(app_client) -> None:
    client, _ = app_client

    register(client, "tok@example.com")
    token = login(client, "tok@example.com")

    assert (
        client.get(
            "/auth/me", headers=auth_headers(token + "tampered")
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/auth/me", headers={"Authorization": "Bearer not-a-jwt"}
        ).status_code
        == 401
    )

    settings = JWTSettings(
        secret_key=TEST_JWT_SECRET,
        algorithm="HS256",
        access_token_expire_minutes=60,
    )
    expired = create_access_token(
        user_id=1,
        settings=settings,
        expires_minutes=-1,
    )
    assert (
        client.get("/auth/me", headers=auth_headers(expired)).status_code
        == 401
    )


def test_conversation_endpoints_require_authentication(app_client) -> None:
    client, _ = app_client

    assert client.post("/conversations", json={}).status_code == 401
    assert client.get("/conversations").status_code == 401
    assert client.get("/conversations/1").status_code == 401
    assert client.get("/conversations/1/messages").status_code == 401
    assert client.delete("/conversations/1").status_code == 401
    assert (
        client.post(
            "/chat", json={"conversation_id": 1, "message": "سلام"}
        ).status_code
        == 401
    )


def test_conversations_scoped_to_current_user(app_client) -> None:
    client, _ = app_client

    register(client, "a@example.com")
    register(client, "b@example.com")
    token_a = login(client, "a@example.com")
    token_b = login(client, "b@example.com")

    created = client.post(
        "/conversations",
        json={"title": "شیراز"},
        headers=auth_headers(token_a),
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    # B cannot read, list messages for, or delete A's conversation.
    assert (
        client.get(
            f"/conversations/{conversation_id}",
            headers=auth_headers(token_b),
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/conversations/{conversation_id}/messages",
            headers=auth_headers(token_b),
        ).status_code
        == 404
    )
    assert (
        client.delete(
            f"/conversations/{conversation_id}",
            headers=auth_headers(token_b),
        ).status_code
        == 404
    )

    # B's own list is empty; A's list contains the conversation.
    list_b = client.get("/conversations", headers=auth_headers(token_b))
    assert list_b.status_code == 200
    assert list_b.json()["count"] == 0

    list_a = client.get("/conversations", headers=auth_headers(token_a))
    assert list_a.status_code == 200
    assert list_a.json()["count"] == 1


def test_chat_uses_token_identity_and_rejects_user_id(app_client) -> None:
    client, _ = app_client

    register(client, "chat-a@example.com")
    register(client, "chat-b@example.com")
    token_a = login(client, "chat-a@example.com")
    token_b = login(client, "chat-b@example.com")

    created = client.post(
        "/conversations",
        json={},
        headers=auth_headers(token_a),
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]

    stub_rag = StubRAGService()
    stub_router = StubRouter(
        standalone_query="تخت جمشید را چه کسی ساخت؟"
    )

    def override_chat_service(
        db=Depends(deps.get_db),
    ) -> ChatService:
        return ChatService(
            db=db,
            rag_service=stub_rag,
            router=stub_router,
            tracer=None,
        )

    client.app.dependency_overrides[deps.get_chat_service] = (
        override_chat_service
    )
    try:
        ok = client.post(
            "/chat",
            json={"conversation_id": conversation_id, "message": "چه کسی اونو ساخت"},
            headers=auth_headers(token_a),
        )
        assert ok.status_code == 200, ok.text
        assert stub_rag.received_queries == ["تخت جمشید را چه کسی ساخت؟"]

        # B cannot write to A's conversation.
        denied = client.post(
            "/chat",
            json={"conversation_id": conversation_id, "message": "سلام"},
            headers=auth_headers(token_b),
        )
        assert denied.status_code == 404

        # user_id is no longer part of the contract.
        legacy = client.post(
            "/chat",
            json={
                "user_id": 999,
                "conversation_id": conversation_id,
                "message": "سلام",
            },
            headers=auth_headers(token_a),
        )
        assert legacy.status_code == 422
    finally:
        del client.app.dependency_overrides[deps.get_chat_service]


def test_public_user_id_removed_from_conversation_contract(app_client) -> None:
    client, _ = app_client

    register(client, "c@example.com")
    token = login(client, "c@example.com")

    legacy = client.post(
        "/conversations",
        json={"user_id": 1, "title": "x"},
        headers=auth_headers(token),
    )
    assert legacy.status_code == 422


def test_dependency_injection_has_no_plain_service_params(app_client) -> None:
    client, _ = app_client

    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    paths = spec.json()["paths"]

    chat_params = paths["/chat"]["post"].get("parameters", [])
    chat_param_names = {param["name"] for param in chat_params}
    assert "db" not in chat_param_names
    assert "rag_service" not in chat_param_names

    for path in (
        "/conversations",
        "/conversations/{conversation_id}",
        "/conversations/{conversation_id}/messages",
    ):
        for operation in paths[path].values():
            names = {param["name"] for param in operation.get("parameters", [])}
            assert "user_id" not in names, path
