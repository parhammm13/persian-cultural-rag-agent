"""Build the RAG runtime and connect its Phoenix tracer."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.dense_retriever import DenseRetriever
from ..retrieval.embedder import embed_query
from ..retrieval.hybrid_retriever import HybridRetriever, ParentStore
from ..retrieval.reranker import JinaReranker
from ..retrieval.vector_store import QdrantVectorStore
from .generator import OpenRouterGenerator
from .observability import setup_phoenix_tracer
from .pipeline import RAGPipeline


logger = logging.getLogger("persian_cultural_rag.api")


def _load_project_env() -> Path | None:
    """Load the first .env found while walking up from this file."""
    script_dir = Path(__file__).resolve().parent

    for directory in (script_dir, *script_dir.parents):
        env_path = directory / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return env_path

    load_dotenv()
    return None


def _required_env(name: str) -> str:
    value = _clean_env_value(os.getenv(name, ""))
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def _clean_env_value(raw: str) -> str:
    """Strip whitespace and trailing inline `#` comments from .env values.

    A line such as ``OPENROUTER_MODEL=openrouter/free # comment`` would
    otherwise produce the invalid model slug
    ``"openrouter/free # comment"`` because python-dotenv keeps the raw
    remainder of the line.
    """
    cleaned = raw.strip()
    if " #" in cleaned:
        cleaned = cleaned.split(" #", 1)[0].rstrip()
    return cleaned


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer.") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0.")
    return value


def _nonnegative_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer.") from error
    if value < 0:
        raise RuntimeError(f"{name} must be >= 0.")
    return value


@dataclass(frozen=True, slots=True)
class RAGSettings:
    jina_api_key: str = field(repr=False)
    openrouter_api_key: str = field(repr=False)

    openrouter_base_url: str
    openrouter_model: str
    qdrant_url: str
    qdrant_collection: str

    bm25_index_dir: Path
    chunks_json: Path

    candidate_k: int = 20
    final_k: int = 5
    max_context_chars: int = 18_000
    api_timeout: int = 120
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "RAGSettings":
        env_path = _load_project_env()
        project_root = env_path.parent if env_path else Path.cwd()

        bm25_index_dir = Path(
            os.getenv("BM25_INDEX_DIR", "Data/processed/bm25")
        )
        chunks_json = Path(
            os.getenv(
                "CHUNKS_JSON",
                "Data/processed/chunked_data.json",
            )
        )

        if not bm25_index_dir.is_absolute():
            bm25_index_dir = project_root / bm25_index_dir
        if not chunks_json.is_absolute():
            chunks_json = project_root / chunks_json

        settings = cls(
            jina_api_key=_required_env("JINA_API_KEY"),
            openrouter_api_key=_required_env("OPENROUTER_API_KEY"),
            openrouter_base_url=_clean_env_value(
                os.getenv(
                    "OPENROUTER_BASE_URL",
                    "https://openrouter.ai/api/v1",
                )
            ),
            openrouter_model=_clean_env_value(
                os.getenv(
                    "OPENROUTER_MODEL",
                    "openrouter/free",
                )
            ),
            qdrant_url=os.getenv(
                "QDRANT_URL",
                "http://localhost:6333",
            ).strip(),
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION",
                "persian_cultural_chunks",
            ).strip(),
            bm25_index_dir=bm25_index_dir,
            chunks_json=chunks_json,
            candidate_k=_positive_int_env("RAG_CANDIDATE_K", 20),
            final_k=_positive_int_env("RAG_FINAL_K", 5),
            max_context_chars=_positive_int_env(
                "RAG_MAX_CONTEXT_CHARS",
                18_000,
            ),
            api_timeout=_positive_int_env("RAG_API_TIMEOUT", 120),
            max_retries=_nonnegative_int_env("RAG_MAX_RETRIES", 2),
        )

        if settings.final_k > settings.candidate_k:
            raise RuntimeError("RAG_FINAL_K must be <= RAG_CANDIDATE_K.")
        if not settings.openrouter_base_url:
            raise RuntimeError("OPENROUTER_BASE_URL must not be empty.")
        if not settings.openrouter_model:
            raise RuntimeError("OPENROUTER_MODEL must not be empty.")
        if not settings.qdrant_url:
            raise RuntimeError("QDRANT_URL must not be empty.")
        if not settings.qdrant_collection:
            raise RuntimeError("QDRANT_COLLECTION must not be empty.")

        return settings


@dataclass(frozen=True, slots=True)
class RAGRuntime:
    """Shared RAG resources used by API services for the process lifetime."""

    pipeline: RAGPipeline
    retriever: HybridRetriever
    reranker: JinaReranker
    candidate_k: int
    final_k: int


def build_rag_runtime(
    settings: RAGSettings | None = None,
) -> RAGRuntime:
    """Construct shared retrieval, reranking, generation, and tracing resources."""
    settings = settings or RAGSettings.from_env()

    # Register Phoenix before invoking any LangChain runnable. The manual stage
    # spans always work; optional LangChain auto-instrumentation is controlled
    # by PHOENIX_AUTO_INSTRUMENT_LANGCHAIN.
    tracer = setup_phoenix_tracer()
    logger.info(
        "phoenix_tracer_initialized tracer=%s endpoint=%s project=%s",
        type(tracer).__name__,
        os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "").strip() or "(unset)",
        os.getenv(
            "PHOENIX_PROJECT_NAME", "persian-cultural-rag-agent-v2"
        ).strip(),
    )

    sparse_retriever = BM25Retriever.load(
        index_dir=settings.bm25_index_dir,
    )
    parent_store = ParentStore.load(
        chunked_data_path=settings.chunks_json,
    )

    query_embedder = partial(
        embed_query,
        api_key=settings.jina_api_key,
        timeout=settings.api_timeout,
        max_retries=settings.max_retries,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    dense_retriever = DenseRetriever(
        vector_store=vector_store,
        query_embedder=query_embedder,
        top_k=settings.candidate_k,
    )
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        parent_store=parent_store,
        top_k=settings.candidate_k,
    )
    reranker = JinaReranker(
        api_key=settings.jina_api_key,
        timeout=settings.api_timeout,
        max_retries=settings.max_retries,
    )
    generator = OpenRouterGenerator(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.openrouter_model,
        timeout=settings.api_timeout,
        max_retries=settings.max_retries,
    )

    pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        candidate_k=settings.candidate_k,
        final_k=settings.final_k,
        max_context_chars=settings.max_context_chars,
        tracer=tracer,
    )

    return RAGRuntime(
        pipeline=pipeline,
        retriever=hybrid_retriever,
        reranker=reranker,
        candidate_k=settings.candidate_k,
        final_k=settings.final_k,
    )


def build_rag_pipeline(
    settings: RAGSettings | None = None,
) -> RAGPipeline:
    """Backward-compatible pipeline factory for CLI/programmatic callers."""
    return build_rag_runtime(settings).pipeline


__all__ = [
    "RAGRuntime",
    "RAGSettings",
    "build_rag_pipeline",
    "build_rag_runtime",
]
