"""End-to-end RAG orchestration: retrieve, rerank, then generate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..retrieval.dense_retriever import RetrievalResult
from .generator import GenerationResult


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]: ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]: ...


class Generator(Protocol):
    def generate(
        self,
        query: str,
        context: str,
    ) -> GenerationResult: ...


@dataclass(frozen=True, slots=True)
class Source:
    rank: int
    chunk_id: str
    page_title: str
    section_heading: str | None
    source_url: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "page_title": self.page_title,
            "section_heading": self.section_heading,
            "source_url": self.source_url,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class RAGResponse:
    query: str
    answer: str
    sources: tuple[Source, ...]
    retrieved_count: int
    reranked_count: int
    model: str | None
    usage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "retrieved_count": self.retrieved_count,
            "reranked_count": self.reranked_count,
            "model": self.model,
            "usage": self.usage,
        }


class RAGPipeline:
    """Coordinate retrieval, cross-encoder reranking, and generation."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        candidate_k: int = 20,
        final_k: int = 5,
        max_context_chars: int = 18_000,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be > 0.")
        if final_k <= 0:
            raise ValueError("final_k must be > 0.")
        if final_k > candidate_k:
            raise ValueError("final_k must be <= candidate_k.")
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be > 0.")

        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.max_context_chars = max_context_chars

    def run(self, query: str) -> RAGResponse:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()
        if not query:
            raise ValueError("query must not be empty.")

        candidates = self.retriever.retrieve(
            query=query,
            top_k=self.candidate_k,
        )

        if not candidates:
            return RAGResponse(
                query=query,
                answer="اطلاعات کافی در منابع پیدا نشد.",
                sources=(),
                retrieved_count=0,
                reranked_count=0,
                model=None,
                usage={},
            )

        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=self.final_k,
        )

        context, used_results = self._build_context(reranked)

        if not used_results:
            return RAGResponse(
                query=query,
                answer="اطلاعات کافی در منابع پیدا نشد.",
                sources=(),
                retrieved_count=len(candidates),
                reranked_count=0,
                model=None,
                usage={},
            )

        generation = self.generator.generate(
            query=query,
            context=context,
        )

        sources = tuple(
            Source(
                rank=rank,
                chunk_id=result.chunk_id,
                page_title=result.page_title,
                section_heading=result.section_heading,
                source_url=result.source_url,
                score=float(result.score),
            )
            for rank, result in enumerate(used_results, start=1)
        )

        return RAGResponse(
            query=query,
            answer=generation.answer,
            sources=sources,
            retrieved_count=len(candidates),
            reranked_count=len(used_results),
            model=generation.model,
            usage=generation.usage,
        )

    def _build_context(
        self,
        results: list[RetrievalResult],
    ) -> tuple[str, list[RetrievalResult]]:
        blocks: list[str] = []
        used_results: list[RetrievalResult] = []
        remaining = self.max_context_chars

        for result in results:
            rank = len(used_results) + 1
            heading = result.section_heading or "بدون عنوان بخش"
            block = (
                f"[{rank}]\n"
                f"عنوان صفحه: {result.page_title}\n"
                f"عنوان بخش: {heading}\n"
                f"نشانی: {result.source_url}\n"
                f"متن:\n{result.text.strip()}"
            )

            if len(block) > remaining:
                block = block[:remaining].rstrip()

            if not block:
                break

            blocks.append(block)
            used_results.append(result)
            remaining -= len(block) + 2

            if remaining <= 0:
                break

        return "\n\n".join(blocks), used_results


__all__ = [
    "RAGPipeline",
    "RAGResponse",
    "Source",
]

