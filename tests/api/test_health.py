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



class FakePipeline:
    def run(self, query: str):
        raise AssertionError("health endpoint must not call the RAG pipeline")


def test_health() -> None:
    app = create_app(lambda: FakeRuntime(FakePipeline()))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "persian-cultural-rag-api",
        "version": "0.2.0",
    }
    assert response.headers[REQUEST_ID_HEADER]
