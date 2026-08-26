"""LCEL RAG orchestration with Phoenix stage-level tracing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langchain_core.runnables import Runnable, RunnableLambda

from ..retrieval.types import RetrievalResult
from .generator import GenerationResult
from .observability import NoOpTracer, capture_content_enabled


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


class _RAGState(TypedDict, total=False):
    query: str
    candidates: list[RetrievalResult]
    reranked: list[RetrievalResult]
    context: str
    used_results: list[RetrievalResult]
    generation: GenerationResult | None


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
    """Coordinate retrieval, reranking, context building, and generation.

    The public ``chain`` is a LangChain LCEL runnable. Phoenix remains the
    observability backend through explicit OpenInference stage spans. No
    LangSmith configuration or callback is required.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: Generator,
        candidate_k: int = 20,
        final_k: int = 5,
        max_context_chars: int = 18_000,
        tracer: Any | None = None,
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
        self.tracer = tracer or NoOpTracer()
        self.capture_content = capture_content_enabled()

        self.chain: Runnable[dict[str, str], RAGResponse] = (
            RunnableLambda(self._prepare_stage).with_config(
                {"run_name": "rag.validate"}
            )
            | RunnableLambda(self._retrieve_stage).with_config(
                {"run_name": "rag.retrieve"}
            )
            | RunnableLambda(self._rerank_stage).with_config(
                {"run_name": "rag.rerank"}
            )
            | RunnableLambda(self._context_stage).with_config(
                {"run_name": "rag.context"}
            )
            | RunnableLambda(self._generate_stage).with_config(
                {"run_name": "rag.generate"}
            )
            | RunnableLambda(self._finalize_stage).with_config(
                {"run_name": "rag.finalize"}
            )
        ).with_config({"run_name": "persian_cultural_rag"})

    def run(self, query: str) -> RAGResponse:
        query = self._validate_query(query)

        with self.tracer.start_as_current_span(
            "rag.pipeline",
            openinference_span_kind="chain",
        ) as pipeline_span:
            pipeline_span.set_input(
                query
                if self.capture_content
                else {"query_length": len(query)}
            )
            pipeline_span.set_attribute("rag.candidate_k", self.candidate_k)
            pipeline_span.set_attribute("rag.final_k", self.final_k)
            pipeline_span.set_attribute("rag.framework", "langchain_lcel")
            pipeline_span.set_attribute("rag.observability", "phoenix")

            try:
                response = self.chain.invoke({"query": query})
            except Exception as error:
                pipeline_span.record_exception(error)
                raise

            pipeline_span.set_attribute(
                "rag.retrieved_count",
                response.retrieved_count,
            )
            pipeline_span.set_attribute(
                "rag.reranked_count",
                response.reranked_count,
            )
            if response.model:
                pipeline_span.set_attribute("rag.model", response.model)

            pipeline_span.set_output(
                response.to_dict()
                if self.capture_content
                else {
                    "answer_length": len(response.answer),
                    "source_count": len(response.sources),
                }
            )
            return response

    def invoke(self, query: str) -> RAGResponse:
        """LangChain-style convenience alias for ``run``."""
        return self.run(query)

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()
        if not query:
            raise ValueError("query must not be empty.")
        return query

    def _prepare_stage(self, state: dict[str, str]) -> _RAGState:
        if not isinstance(state, dict):
            raise TypeError("chain input must be a dictionary.")
        return {
            "query": self._validate_query(state.get("query", "")),
        }

    def _retrieve_stage(self, state: _RAGState) -> _RAGState:
        query = state["query"]

        with self.tracer.start_as_current_span(
            "rag.retrieve",
            openinference_span_kind="retriever",
        ) as span:
            span.set_input(
                query
                if self.capture_content
                else {"query_length": len(query)}
            )
            candidates = self.retriever.retrieve(
                query=query,
                top_k=self.candidate_k,
            )
            span.set_attribute(
                "retrieval.document_count",
                len(candidates),
            )
            self._set_document_attributes(
                span,
                prefix="retrieval.documents",
                results=candidates,
            )
            span.set_output(
                json.dumps(
                    self._trace_results(candidates),
                    ensure_ascii=False,
                )
            )

        return {
            **state,
            "candidates": candidates,
        }

    def _rerank_stage(self, state: _RAGState) -> _RAGState:
        query = state["query"]
        candidates = state.get("candidates", [])

        if not candidates:
            return {
                **state,
                "reranked": [],
            }

        with self.tracer.start_as_current_span(
            "rag.rerank",
            openinference_span_kind="reranker",
        ) as span:
            rerank_input = (
                {
                    "query": query,
                    "candidates": self._trace_results(candidates),
                }
                if self.capture_content
                else {
                    "query_length": len(query),
                    "candidate_count": len(candidates),
                }
            )
            span.set_input(json.dumps(rerank_input, ensure_ascii=False))
            if self.capture_content:
                span.set_attribute("reranker.query", query)
            else:
                span.set_attribute("reranker.query_length", len(query))
            span.set_attribute("reranker.top_k", self.final_k)
            self._set_document_attributes(
                span,
                prefix="reranker.input_documents",
                results=candidates,
            )

            reranked = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=self.final_k,
            )
            span.set_attribute("reranker.document_count", len(reranked))
            self._set_document_attributes(
                span,
                prefix="reranker.output_documents",
                results=reranked,
            )

            if reranked:
                reranker_model = reranked[0].metadata.get("reranker_model")
                if reranker_model:
                    span.set_attribute(
                        "reranker.model_name",
                        str(reranker_model),
                    )

            span.set_output(
                json.dumps(
                    self._trace_results(reranked),
                    ensure_ascii=False,
                )
            )

        return {
            **state,
            "reranked": reranked,
        }

    def _context_stage(self, state: _RAGState) -> _RAGState:
        context, used_results = self._build_context(
            state.get("reranked", [])
        )
        return {
            **state,
            "context": context,
            "used_results": used_results,
        }

    def _generate_stage(self, state: _RAGState) -> _RAGState:
        query = state["query"]
        context = state.get("context", "")
        used_results = state.get("used_results", [])

        if not used_results or not context:
            return {
                **state,
                "generation": None,
            }

        with self.tracer.start_as_current_span(
            "rag.generate",
            openinference_span_kind="llm",
        ) as span:
            span.set_input(
                {"query": query, "context": context}
                if self.capture_content
                else {
                    "query_length": len(query),
                    "context_length": len(context),
                }
            )
            generation = self.generator.generate(
                query=query,
                context=context,
            )
            span.set_attribute("llm.model_name", generation.model)
            self._set_usage_attributes(span, generation.usage)
            span.set_output(
                generation.answer
                if self.capture_content
                else {"answer_length": len(generation.answer)}
            )

        return {
            **state,
            "generation": generation,
        }

    def _finalize_stage(self, state: _RAGState) -> RAGResponse:
        query = state["query"]
        candidates = state.get("candidates", [])
        used_results = state.get("used_results", [])
        generation = state.get("generation")

        if generation is None:
            return self._empty_response(
                query,
                retrieved_count=len(candidates),
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

    @staticmethod
    def _empty_response(
        query: str,
        retrieved_count: int,
    ) -> RAGResponse:
        return RAGResponse(
            query=query,
            answer="اطلاعات کافی در منابع پیدا نشد.",
            sources=(),
            retrieved_count=retrieved_count,
            reranked_count=0,
            model=None,
            usage={},
        )

    def _trace_results(
        self,
        results: list[RetrievalResult],
    ) -> list[dict[str, Any]]:
        traced_results: list[dict[str, Any]] = []

        for result in results:
            item: dict[str, Any] = {
                "chunk_id": result.chunk_id,
                "score": float(result.score),
                "page_title": result.page_title,
                "source_url": result.source_url,
            }
            if self.capture_content:
                item["text"] = result.text[:3_000]
            traced_results.append(item)

        return traced_results

    def _set_document_attributes(
        self,
        span: Any,
        prefix: str,
        results: list[RetrievalResult],
    ) -> None:
        for index, result in enumerate(results):
            document_prefix = f"{prefix}.{index}.document"
            span.set_attribute(f"{document_prefix}.id", result.chunk_id)
            span.set_attribute(
                f"{document_prefix}.score",
                float(result.score),
            )
            if self.capture_content:
                span.set_attribute(
                    f"{document_prefix}.content",
                    result.text[:3_000],
                )
            span.set_attribute(
                f"{document_prefix}.metadata",
                json.dumps(
                    {
                        "page_title": result.page_title,
                        "section_heading": result.section_heading,
                        "source_url": result.source_url,
                    },
                    ensure_ascii=False,
                ),
            )

    @staticmethod
    def _set_usage_attributes(
        span: Any,
        usage: dict[str, Any],
    ) -> None:
        attribute_map = {
            "prompt_tokens": "llm.token_count.prompt",
            "completion_tokens": "llm.token_count.completion",
            "total_tokens": "llm.token_count.total",
        }
        for usage_key, attribute_name in attribute_map.items():
            value = usage.get(usage_key)
            if isinstance(value, int) and not isinstance(value, bool):
                span.set_attribute(attribute_name, value)

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