from dataclasses import dataclass

from fastapi.testclient import TestClient

from src.api.main import create_app


@dataclass(frozen=True, slots=True)
class FakeRetrievalResult:
    chunk_id: str
    text: str
    page_title: str
    section_heading: str | None
    source_url: str | None
    score: float


class FakePipeline:
    def run(self, query: str):
        raise AssertionError("/retrieve must not call generation pipeline")


class FakeRetriever:
    def __init__(self) -> None:
        self.last_query = None
        self.last_top_k = None

    def retrieve(self, query: str, top_k: int):
        self.last_query = query
        self.last_top_k = top_k
        return (
            FakeRetrievalResult(
                chunk_id="parent-1",
                text="first",
                page_title="مسجد آقابزرگ",
                section_heading="تاریخچه",
                source_url="https://example.com/1",
                score=0.1,
            ),
            FakeRetrievalResult(
                chunk_id="parent-2",
                text="second",
                page_title="مسجد آقابزرگ",
                section_heading=None,
                source_url=None,
                score=0.2,
            ),
        )


class FakeReranker:
    def __init__(self) -> None:
        self.last_top_k = None

    def rerank(self, query: str, candidates, top_k: int):
        self.last_top_k = top_k
        return tuple(reversed(candidates))[:top_k]


@dataclass
class FakeRuntime:
    pipeline: object
    retriever: object
    reranker: object
    candidate_k: int = 20


def test_retrieve_runs_hybrid_retrieval_and_reranking() -> None:
    retriever = FakeRetriever()
    reranker = FakeReranker()
    runtime = FakeRuntime(FakePipeline(), retriever, reranker)
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/retrieve",
            json={"query": "  مسجد آقابزرگ  ", "top_k": 2},
        )

    assert response.status_code == 200
    assert retriever.last_query == "مسجد آقابزرگ"
    assert retriever.last_top_k == 20
    assert reranker.last_top_k == 2

    body = response.json()
    assert body["query"] == "مسجد آقابزرگ"
    assert body["result_count"] == 2
    assert body["results"][0]["rank"] == 1
    assert body["results"][0]["chunk_id"] == "parent-2"
    assert body["results"][1]["rank"] == 2


def test_retrieve_rejects_invalid_top_k() -> None:
    runtime = FakeRuntime(FakePipeline(), FakeRetriever(), FakeReranker())
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/retrieve",
            json={"query": "ارگ بم", "top_k": 21},
        )

    assert response.status_code == 422


def test_retrieve_rejects_unknown_fields() -> None:
    runtime = FakeRuntime(FakePipeline(), FakeRetriever(), FakeReranker())
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/retrieve",
            json={"query": "ارگ بم", "debug": True},
        )

    assert response.status_code == 422
