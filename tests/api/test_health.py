from fastapi.testclient import TestClient

from src.api.main import create_app


class FakePipeline:
    def run(self, query: str):
        raise AssertionError("health endpoint must not call the RAG pipeline")


def test_health() -> None:
    app = create_app(lambda: FakePipeline())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "persian-cultural-rag-api",
        "version": "0.1.0",
    }
