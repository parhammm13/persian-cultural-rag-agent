from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    text: str
    score: float
    metadata: dict[str, Any]


class DenseRetriever:

    def __init__(
        self,
        vector_store,
        query_embedder,
    ):
        self.vector_store = vector_store
        self.query_embedder = query_embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        query_vector = self.query_embedder(
            query
        )

        return self.vector_store.search(
            vector=query_vector,
            top_k=top_k,
        )

