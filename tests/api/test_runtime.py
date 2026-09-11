from dataclasses import dataclass

from fastapi.testclient import TestClient

from src.api.main import create_app


class FakePipeline:
    def run(self, query: str):
        raise AssertionError("pipeline should not be called")


class FakeRetriever:
    def retrieve(self, query: str, top_k: int):
        return ()


class FakeReranker:
    def rerank(self, query: str, candidates, top_k: int):
        return ()


@dataclass
class FakeRuntime:
    pipeline: object
    retriever: object
    reranker: object
    candidate_k: int = 20


def test_runtime_is_built_once_for_multiple_requests() -> None:
    builds = 0

    def builder():
        nonlocal builds
        builds += 1
        return FakeRuntime(FakePipeline(), FakeRetriever(), FakeReranker())

    app = create_app(builder)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200

    assert builds == 1
