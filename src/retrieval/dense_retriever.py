from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import numpy as np

from .embedder import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    embed_query,
    load_jina_api_key,
)

from .vector_store import (
    COLLECTION_NAME,
    QDRANT_URL,
    QdrantVectorStore,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_TOP_K = 5


# ============================================================
# Retrieval result (canonical definition)
# ============================================================

@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """
    Clean result returned by the retriever.

    Qdrant-specific objects should not leak into the rest
    of the RAG pipeline.
    """

    score: float
    chunk_id: str
    text: str

    page_title: str
    section_heading: str | None
    source_url: str

    metadata: dict[str, Any]


# ============================================================
# Dense Retriever
# ============================================================

class DenseRetriever:

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        query_embedder: Callable[
            [str],
            np.ndarray,
        ],
        top_k: int = DEFAULT_TOP_K,
    ) -> None:

        if top_k <= 0:
            raise ValueError(
                "top_k must be > 0."
            )

        self.vector_store = vector_store
        self.query_embedder = query_embedder
        self.top_k = top_k

    # ========================================================
    # Retrieve
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:

        # ----------------------------------------------------
        # Validate query
        # ----------------------------------------------------

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query must not be empty."
            )

        # ----------------------------------------------------
        # Resolve Top-K
        # ----------------------------------------------------

        limit = (
            self.top_k
            if top_k is None
            else top_k
        )

        if limit <= 0:
            raise ValueError(
                "top_k must be > 0."
            )

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_vector = self.query_embedder(
            query
        )

        # ----------------------------------------------------
        # Vector search
        # ----------------------------------------------------

        points = (
            self.vector_store.search_by_vector(
                query_vector=query_vector,
                limit=limit,
            )
        )

        # ----------------------------------------------------
        # Format results
        # ----------------------------------------------------

        results: list[RetrievalResult] = []

        for point in points:

            result = self._format_point(
                point
            )

            results.append(
                result
            )

        return results

    # ========================================================
    # Result formatting
    # ========================================================

    @staticmethod
    def _format_point(
        point: Any,
    ) -> RetrievalResult:

        payload = point.payload

        if payload is None:
            raise RuntimeError(
                "Qdrant result has no payload."
            )

        required_fields = {
            "chunk_id",
            "text",
            "page_title",
            "page_url",
        }

        missing = (
            required_fields
            - set(payload.keys())
        )

        if missing:
            raise RuntimeError(
                "Qdrant payload is missing fields: "
                f"{sorted(missing)}"
            )

        # Keep the complete payload available as metadata.
        metadata = dict(payload)

        return RetrievalResult(
            score=float(
                point.score
            ),

            chunk_id=str(
                payload["chunk_id"]
            ),

            text=str(
                payload["text"]
            ),

            page_title=str(
                payload["page_title"]
            ),

            section_heading=(
                str(
                    payload["section_heading"]
                )
                if payload.get(
                    "section_heading"
                ) is not None
                else None
            ),

            source_url=str(
                payload["page_url"]
            ),

            metadata=metadata,
        )


# ============================================================
# CLI output
# ============================================================

def print_results(
    query: str,
    results: list[RetrievalResult],
) -> None:

    print("=" * 72)

    print(
        f"Query: {query}"
    )

    print(
        f"Results: {len(results)}"
    )

    print("=" * 72)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(
            f"[{rank}] Score: "
            f"{result.score:.6f}"
        )

        print(
            f"Title: "
            f"{result.page_title}"
        )

        if result.section_heading:
            print(
                f"Section: "
                f"{result.section_heading}"
            )

        print(
            f"Chunk ID: "
            f"{result.chunk_id}"
        )

        print(
            f"Source: "
            f"{result.source_url}"
        )

        print("-" * 72)

        print(
            result.text
        )

        print("=" * 72)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Dense retrieval over Persian cultural chunks "
            "stored in Qdrant."
        )
    )

    parser.add_argument(
        "query",
        type=str,
        help=(
            "Search query."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of results to retrieve "
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
            "Qdrant collection name "
            f"(default: {COLLECTION_NAME})"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=(
            "Jina API timeout in seconds "
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

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error(
            "--top-k must be > 0."
        )

    if args.timeout <= 0:
        parser.error(
            "--timeout must be > 0."
        )

    if args.max_retries < 0:
        parser.error(
            "--max-retries must be >= 0."
        )

    return args


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    # --------------------------------------------------------
    # Jina API key
    # --------------------------------------------------------

    api_key, _ = (
        load_jina_api_key()
    )

    # --------------------------------------------------------
    # Query embedder
    # --------------------------------------------------------

    query_embedder = partial(
        embed_query,
        api_key=api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    # --------------------------------------------------------
    # Vector store
    # --------------------------------------------------------

    vector_store = QdrantVectorStore(
        url=args.qdrant_url,
        collection_name=args.collection,
    )

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    retriever = DenseRetriever(
        vector_store=vector_store,
        query_embedder=query_embedder,
        top_k=args.top_k,
    )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    results = retriever.retrieve(
        query=args.query,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_results(
        query=args.query,
        results=results,
    )


if __name__ == "__main__":
    main()
