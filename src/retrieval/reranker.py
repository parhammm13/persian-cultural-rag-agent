"""Rerank retrieved documents with Jina's multilingual cross-encoder.

Typical pipeline::

    candidates = hybrid_retriever.retrieve(query, top_k=20)
    results = reranker.rerank(query, candidates, top_k=5)

Retrieval is fast and finds candidates. Reranking is slower but reads the
query together with every candidate, producing a more precise final order.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .types import RetrievalResult


JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
DEFAULT_MODEL = "jina-reranker-v2-base-multilingual"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 2


class JinaRerankerError(RuntimeError):
    """Raised when Jina cannot return a valid reranking response."""


class JinaReranker:
    """Rerank ``RetrievalResult`` candidates using the Jina Reranker API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        api_key = api_key.strip()
        model = model.strip()

        if not api_key:
            raise ValueError("api_key must not be empty.")
        if not model:
            raise ValueError("model must not be empty.")
        if timeout <= 0:
            raise ValueError("timeout must be > 0.")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")

        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return candidates ordered by cross-encoder relevance.

        The returned ``score`` is the Jina relevance score. The earlier
        dense/BM25/RRF score is retained in ``metadata['retrieval_score']``.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()
        if not query:
            raise ValueError("query must not be empty.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if not candidates:
            return []
        if not all(isinstance(item, RetrievalResult) for item in candidates):
            raise TypeError("candidates must contain RetrievalResult objects.")

        limit = min(top_k or len(candidates), len(candidates))
        response = self._request(
            query=query,
            documents=[candidate.text for candidate in candidates],
            top_n=limit,
        )

        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise JinaRerankerError("Jina response is missing a results list.")

        reranked: list[RetrievalResult] = []
        seen_indexes: set[int] = set()

        for item in raw_results:
            if not isinstance(item, dict):
                raise JinaRerankerError("Jina returned an invalid result item.")

            index = item.get("index")
            relevance_score = item.get("relevance_score")

            if not isinstance(index, int) or not 0 <= index < len(candidates):
                raise JinaRerankerError("Jina returned an invalid document index.")
            if index in seen_indexes:
                raise JinaRerankerError("Jina returned a duplicate document index.")
            if not isinstance(relevance_score, (int, float)):
                raise JinaRerankerError("Jina returned an invalid relevance score.")

            seen_indexes.add(index)
            candidate = candidates[index]
            score = float(relevance_score)

            reranked.append(
                replace(
                    candidate,
                    score=score,
                    metadata={
                        **candidate.metadata,
                        "retrieval_score": float(candidate.score),
                        "reranker_score": score,
                        "reranker_model": self.model,
                    },
                )
            )

        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:limit]

    def _request(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                # We already own the document text, so returning it wastes data.
                "return_documents": False,
            }
        ).encode("utf-8")

        request = Request(
            JINA_RERANK_URL,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )

        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise JinaRerankerError("Jina returned invalid JSON data.")
                return parsed
            except HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")
                # Retry only temporary server/rate-limit failures.
                if error.code not in {429, 500, 502, 503, 504}:
                    raise JinaRerankerError(
                        f"Jina request failed with HTTP {error.code}: {details}"
                    ) from error
                last_error: Exception = error
            except (URLError, TimeoutError) as error:
                last_error = error
            except json.JSONDecodeError as error:
                raise JinaRerankerError("Jina returned malformed JSON.") from error

            if attempt < self.max_retries:
                time.sleep(2**attempt)

        raise JinaRerankerError(
            f"Jina request failed after {self.max_retries + 1} attempts."
        ) from last_error