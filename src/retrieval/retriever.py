from __future__ import annotations

from typing import Any

from .vector_store import QdrantVectorStore


class DenseRetriever:

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        query_embedder: Any,
        top_k: int = 5,
    ) -> None:

        if top_k <= 0:
            raise ValueError(
                "top_k must be > 0."
            )

        self.vector_store = vector_store
        self.query_embedder = query_embedder
        self.top_k = top_k