from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from typing import Any

from .embedder import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    embed_query,
    load_jina_api_key,
)

from .retriever import (
    DEFAULT_TOP_K,
    DenseRetriever,
    RetrievalResult,
)

from .vector_store import (
    COLLECTION_NAME,
    QDRANT_URL,
    QdrantVectorStore,
)

# ============================================================
# Context result
# ============================================================

@dataclass(frozen=True, slots=True)
class RetrievedContext:
    """
    Final output produced by ContextBuilder.

    It keeps both:
    1. The formatted context for the LLM.
    2. The original retrieval results for inspection,
       citations, scores, metadata, etc.
    """

    query: str
    context: str
    results: list[RetrievalResult]

    @property
    def scores(self) -> list[float]:
        return [
            result.score
            for result in self.results
        ]

    @property
    def metadata(self) -> list[dict[str, Any]]:
        return [
            result.metadata
            for result in self.results
        ]

    @property
    def source_urls(self) -> list[str]:
        """
        Return unique source URLs while preserving order.
        """

        return list(
            dict.fromkeys(
                result.source_url
                for result in self.results
            )
        )


# ============================================================
# Context Builder
# ============================================================

class ContextBuilder:

    def __init__(
        self,
        retriever: DenseRetriever,
    ) -> None:

        self.retriever = retriever

    # ========================================================
    # Build
    # ========================================================

    def build(
        self,
        query: str,
        top_k: int | None = None,
    ) -> RetrievedContext:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query must not be empty."
            )

        # Retrieval logic remains inside DenseRetriever.
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        context = self._build_context_text(
            results
        )

        return RetrievedContext(
            query=query,
            context=context,
            results=results,
        )

    # ========================================================
    # Context formatting
    # ========================================================

    @staticmethod
    def _build_context_text(
        results: list[RetrievalResult],
    ) -> str:

        documents: list[str] = []

        for rank, result in enumerate(
            results,
            start=1,
        ):

            document = ContextBuilder._format_result(
                result=result,
                rank=rank,
            )

            documents.append(
                document
            )

        return "\n\n".join(
            documents
        )

    @staticmethod
    def _format_result(
        result: RetrievalResult,
        rank: int,
    ) -> str:

        parts = [
            f"[Document {rank}]",
            f"Title: {result.page_title}",
        ]

        if result.section_heading:
            parts.append(
                f"Section: {result.section_heading}"
            )

        parts.extend(
            [
                f"Source: {result.source_url}",
                "Content:",
                result.text.strip(),
            ]
        )

        return "\n".join(
            parts
        )


# ============================================================
# Factory
# ============================================================

def create_context_builder(
    top_k: int = DEFAULT_TOP_K,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = COLLECTION_NAME,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ContextBuilder:

    if top_k <= 0:
        raise ValueError(
            "top_k must be > 0."
        )

    if timeout <= 0:
        raise ValueError(
            "timeout must be > 0."
        )

    if max_retries < 0:
        raise ValueError(
            "max_retries must be >= 0."
        )

    # --------------------------------------------------------
    # Jina API
    # --------------------------------------------------------

    api_key, _ = load_jina_api_key()

    query_embedder = partial(
        embed_query,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )

    # --------------------------------------------------------
    # Vector store
    # --------------------------------------------------------

    vector_store = QdrantVectorStore(
        url=qdrant_url,
        collection_name=collection_name,
    )

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    retriever = DenseRetriever(
        vector_store=vector_store,
        query_embedder=query_embedder,
        top_k=top_k,
    )

    # --------------------------------------------------------
    # Context builder
    # --------------------------------------------------------

    return ContextBuilder(
        retriever=retriever
    )

# ============================================================
# Public helper
# ============================================================

def build_context(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = COLLECTION_NAME,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> RetrievedContext:

    builder = create_context_builder(
        top_k=top_k,
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        timeout=timeout,
        max_retries=max_retries,
    )

    return builder.build(
        query=query
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Retrieve relevant Persian cultural documents "
            "and build LLM-ready context."
        )
    )

    parser.add_argument(
        "query",
        type=str,
        help="User search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of documents to retrieve "
            f"(default: {DEFAULT_TOP_K})"
        ),
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=QDRANT_URL,
        help=(
            "Qdrant URL "
            f"(default: {QDRANT_URL})"
        ),
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=(
            "Qdrant collection "
            f"(default: {COLLECTION_NAME})"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=(
            "Jina API timeout "
            f"(default: {DEFAULT_TIMEOUT})"
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Jina API maximum retries "
            f"(default: {DEFAULT_MAX_RETRIES})"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    result = build_context(
        query=args.query,
        top_k=args.top_k,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    print("=" * 72)
    print("RETRIEVED CONTEXT")
    print("=" * 72)

    print()
    print(result.context)

    print()
    print("=" * 72)

    print(
        f"Retrieved documents: "
        f"{len(result.results)}"
    )

    print(
        f"Scores: "
        f"{result.scores}"
    )

    print(
        f"Sources: "
        f"{result.source_urls}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()