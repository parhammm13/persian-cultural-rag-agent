from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from src.api.main import create_app


@dataclass(frozen=True, slots=True)
class FakeSource:
    rank: int
    chunk_id: str
    page_title: str
    section_heading: str | None
    source_url: str | None
    score: float


@dataclass(frozen=True, slots=True)
class FakeRAGResponse:
    answer: str
    sources: tuple[FakeSource, ...]
    retrieved_count: int
    reranked_count: int
    model: str
    usage: dict[str, Any] | None


class FakePipeline:
    def __init__(self) -> None:
        self.last_query: str | None = None

    def run(self, query: str) -> FakeRAGResponse:
        self.last_query = query

        return FakeRAGResponse(
            answer="مسجد آقابزرگ در کاشان قرار دارد. [1]",
            sources=(
                FakeSource(
                    rank=1,
                    chunk_id="parent-123",
                    page_title="مسجد آقابزرگ",
                    section_heading="معرفی",
                    source_url="https://fa.wikipedia.org/example",
                    score=0.91,
                ),
            ),
            retrieved_count=20,
            reranked_count=5,
            model="example/model",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        )


def test_query_maps_rag_dataclass_to_api_response() -> None:
    pipeline = FakePipeline()
    app = create_app(lambda: pipeline)

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "query": "  مسجد آقابزرگ کجاست؟  ",
            },
        )

    assert response.status_code == 200
    assert pipeline.last_query == "مسجد آقابزرگ کجاست؟"

    body = response.json()
    assert body["answer"] == "مسجد آقابزرگ در کاشان قرار دارد. [1]"
    assert body["retrieved_count"] == 20
    assert body["reranked_count"] == 5
    assert body["sources"][0]["chunk_id"] == "parent-123"


def test_query_rejects_blank_input_before_pipeline_call() -> None:
    pipeline = FakePipeline()
    app = create_app(lambda: pipeline)

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "query": "     ",
            },
        )

    assert response.status_code == 422
    assert pipeline.last_query is None


def test_query_rejects_unknown_request_fields() -> None:
    pipeline = FakePipeline()
    app = create_app(lambda: pipeline)

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "query": "مسجد آقابزرگ کجاست؟",
                "unexpected": True,
            },
        )

    assert response.status_code == 422
    assert pipeline.last_query is None
