"""Compact LangChain adapter for Jina embeddings used by src_v2."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
MODEL_NAME = "jina-embeddings-v5-text-small"
DOCUMENT_TASK = "retrieval.passage"
QUERY_TASK = "retrieval.query"
EXPECTED_DIMENSION = 1024
DEFAULT_BATCH_SIZE = 32
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 6
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class JinaEmbeddingError(RuntimeError):
    """Jina request or response validation failed."""


def load_jina_api_key() -> tuple[str, Path | None]:
    """Load JINA_API_KEY from the nearest .env or the OS environment."""
    env_path: Path | None = None
    start = Path(__file__).resolve().parent
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.exists():
            load_dotenv(candidate)
            env_path = candidate
            break

    api_key = os.getenv("JINA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("JINA_API_KEY was not found in .env or environment.")
    return api_key, env_path


def _normalize(vectors: Any, dimension: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise JinaEmbeddingError(
            f"Expected embeddings with shape (N, {dimension}); got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise JinaEmbeddingError("Embedding contains NaN or Inf.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise JinaEmbeddingError("Zero-length embedding received.")
    return (array / norms).astype(np.float32, copy=False)


class JinaEmbeddings(Embeddings):
    """LangChain Embeddings implementation backed by the Jina REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = MODEL_NAME,
        dimension: int = EXPECTED_DIMENSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        session: requests.Session | None = None,
    ) -> None:
        if api_key is None:
            api_key, _ = load_jina_api_key()
        if not api_key.strip():
            raise ValueError("api_key must not be empty.")
        if dimension <= 0 or batch_size <= 0 or timeout <= 0 or max_retries < 0:
            raise ValueError("Invalid dimension, batch_size, timeout, or max_retries.")
        self.api_key = api_key.strip()
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = session or requests.Session()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages in batches, as required by LangChain."""
        if not texts:
            return []
        cleaned = [self._clean(text, "document") for text in texts]
        vectors = [
            self._request(cleaned[i : i + self.batch_size], DOCUMENT_TASK)
            for i in range(0, len(cleaned), self.batch_size)
        ]
        return np.vstack(vectors).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query, as required by LangChain."""
        return self._request([self._clean(text, "query")], QUERY_TASK)[0].tolist()

    @staticmethod
    def _clean(text: str, label: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"{label} must be a string.")
        text = text.strip()
        if not text:
            raise ValueError(f"{label} must not be empty.")
        return text

    def _request(self, texts: list[str], task: str) -> np.ndarray:
        payload = {
            "model": self.model,
            "task": task,
            "dimensions": self.dimension,
            "input": texts,
            "embedding_type": "float",
            "normalized": True,
            "truncate": False,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    JINA_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in RETRYABLE_STATUSES:
                    if attempt == self.max_retries:
                        raise JinaEmbeddingError(
                            f"Jina temporary error: HTTP {response.status_code}."
                        )
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else 2**attempt
                    except ValueError:
                        delay = 2**attempt
                    time.sleep(delay)
                    continue
                if not response.ok:
                    raise JinaEmbeddingError(
                        f"Jina error HTTP {response.status_code}: {response.text[:500]}"
                    )
                return self._parse(response, len(texts))
            except (requests.Timeout, requests.ConnectionError) as error:
                last_error = error
                if attempt == self.max_retries:
                    break
                time.sleep(2**attempt)

        raise JinaEmbeddingError("Jina request failed after retries.") from last_error

    def _parse(self, response: requests.Response, count: int) -> np.ndarray:
        try:
            data = response.json()["data"]
            items = sorted(data, key=lambda item: item["index"])
            vectors = [item["embedding"] for item in items]
        except (ValueError, KeyError, TypeError) as error:
            raise JinaEmbeddingError("Invalid response from Jina.") from error
        if len(vectors) != count:
            raise JinaEmbeddingError("Embedding count does not match input count.")
        return _normalize(vectors, self.dimension)


def embed_query(
    query: str,
    api_key: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> np.ndarray:
    """Compatibility callable used by dense_retriever and factory."""
    client = JinaEmbeddings(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )
    return np.asarray(client.embed_query(query), dtype=np.float32)


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT",
    "JinaEmbeddings",
    "embed_query",
    "load_jina_api_key",
]