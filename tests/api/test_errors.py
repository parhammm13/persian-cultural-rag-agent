from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.request_context import REQUEST_ID_HEADER

from dataclasses import dataclass

class NoopRetriever:
    def retrieve(self, query: str, top_k: int):
        return ()

class NoopReranker:
    def rerank(self, query: str, candidates, top_k: int):
        return ()

@dataclass
class FakeRuntime:
    pipeline: object
    retriever: object = None
    reranker: object = None
    candidate_k: int = 20

    def __post_init__(self):
        if self.retriever is None:
            self.retriever = NoopRetriever()
        if self.reranker is None:
            self.reranker = NoopReranker()



class ExplodingPipeline:
    def run(self, query: str):
        raise RuntimeError("OPENROUTER_API_KEY=super-secret-value")


def test_validation_error_uses_public_error_contract() -> None:
    app = create_app(lambda: FakeRuntime(ExplodingPipeline()))

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "   "},
        )

    assert response.status_code == 422

    body = response.json()
    assert body["error"] == "validation_error"
    assert body["message"] == "Request validation failed."
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_unhandled_exception_is_not_leaked_to_client() -> None:
    app = create_app(lambda: FakeRuntime(ExplodingPipeline()))

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/query",
            json={"query": "ارگ بم کجاست؟"},
        )

    assert response.status_code == 500

    body = response.json()
    assert body == {
        "error": "internal_server_error",
        "message": "An internal server error occurred.",
        "request_id": response.headers[REQUEST_ID_HEADER],
    }

    assert "super-secret-value" not in response.text


def test_not_found_uses_http_error_contract() -> None:
    app = create_app(lambda: FakeRuntime(ExplodingPipeline()))

    with TestClient(app) as client:
        response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {
        "error": "http_error",
        "message": "Not Found",
        "request_id": response.headers[REQUEST_ID_HEADER],
    }
