from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .bm25_retriever import BM25Result, BM25Retriever
from .dense_retriever import DenseRetriever
from .embedder import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    embed_query,
    load_jina_api_key,
)
from .vector_store import COLLECTION_NAME, QDRANT_URL, QdrantVectorStore
from .types import RetrievalResult


# ============================================================
# Configuration
# ============================================================

DEFAULT_TOP_K = 5
FETCH_MULTIPLIER = 2
RRF_K = 60

logger = logging.getLogger("hybrid_retriever")


# ============================================================
# Structured logging (LangSmith-friendly)
# ============================================================

def log_event(
    event: str,
    run_id: str | None = None,
    **fields: Any,
) -> None:

    payload: dict[str, Any] = {
        "event": event,
        "ts": round(time.time(), 3),
        "component": "hybrid_retriever",
    }

    if run_id is not None:
        payload["run_id"] = run_id

    payload.update(fields)

    logger.info(
        json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
    )


# ============================================================
# Fusion (Reciprocal Rank Fusion)
# ============================================================

def fuse_rankings(
    ranked_lists: list[list[RetrievalResult]],
    k: int = RRF_K,
) -> list[RetrievalResult]:

    if k <= 0:
        raise ValueError(
            "k must be > 0."
        )

    scores: dict[str, float] = {}
    best_result: dict[str, RetrievalResult] = {}

    for ranked_list in ranked_lists:

        for rank, result in enumerate(
            ranked_list,
            start=1,
        ):

            contribution = 1.0 / (k + rank)

            scores[result.chunk_id] = (
                scores.get(result.chunk_id, 0.0)
                + contribution
            )

            if result.chunk_id not in best_result:
                best_result[result.chunk_id] = result

    fused = [
        replace(
            best_result[chunk_id],
            score=fused_score,
        )
        for chunk_id, fused_score in scores.items()
    ]

    fused.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    return fused


# ============================================================
# Duplicate merging
# ============================================================

def text_fingerprint(
    text: str,
) -> str:

    return hashlib.sha256(
        text.strip().encode("utf-8")
    ).hexdigest()


def build_duplicate_map(
    ranked_lists: list[list[RetrievalResult]],
) -> tuple[
    dict[str, str],
    int,
]:

    """
    Map every chunk_id to a canonical chunk_id.

    Chunks with byte-identical text are rewritten to the
    FIRST occurrence's chunk_id, so RRF treats them as one
    document and their evidence accumulates.
    """

    canonical_by_hash: dict[str, str] = {}
    duplicate_map: dict[str, str] = {}
    merged_count = 0

    for ranked_list in ranked_lists:

        for result in ranked_list:

            if result.chunk_id in duplicate_map:
                continue

            fingerprint = text_fingerprint(
                result.text
            )

            canonical_id = (
                canonical_by_hash.get(fingerprint)
            )

            if canonical_id is None:
                canonical_by_hash[fingerprint] = (
                    result.chunk_id
                )

                duplicate_map[result.chunk_id] = (
                    result.chunk_id
                )

            else:
                duplicate_map[result.chunk_id] = (
                    canonical_id
                )

                merged_count += 1

    return duplicate_map, merged_count


def apply_duplicate_map(
    ranked_lists: list[list[RetrievalResult]],
    duplicate_map: dict[str, str],
) -> list[list[RetrievalResult]]:

    remapped_lists: list[list[RetrievalResult]] = []

    for ranked_list in ranked_lists:

        remapped_list: list[RetrievalResult] = []
        seen_ids: set[str] = set()

        for result in ranked_list:

            canonical_id = duplicate_map[
                result.chunk_id
            ]

            if canonical_id in seen_ids:
                continue

            seen_ids.add(canonical_id)

            remapped_list.append(
                replace(
                    result,
                    chunk_id=canonical_id,
                )
            )

        remapped_lists.append(remapped_list)

    return remapped_lists


# ============================================================
# Normalization
# ============================================================

def to_retrieval_result(
    bm25_result: BM25Result,
) -> RetrievalResult:

    metadata = dict(bm25_result.metadata)

    return RetrievalResult(
        score=bm25_result.score,

        chunk_id=bm25_result.chunk_id,

        text=bm25_result.text,

        page_title=str(
            metadata.get("page_title", "")
        ),

        section_heading=(
            metadata.get("section_heading")
            or None
        ),

        source_url=str(
            metadata.get("page_url", "")
        ),

        metadata=metadata,
    )


# ============================================================
# Parent store
# ============================================================

@dataclass(frozen=True, slots=True)
class ParentRecord:

    parent_id: str
    text: str
    page_title: str
    section_heading: str | None
    source_url: str


class ParentStore:

    def __init__(
        self,
        parents: dict[str, dict[str, Any]],
    ) -> None:

        self.parents = parents

    @classmethod
    def load(
        cls,
        chunked_data_path: Path,
    ) -> ParentStore:

        with chunked_data_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        parents: dict[str, dict[str, Any]] = {}

        for parent in data["parents"]:

            parents[parent["parent_id"]] = parent

        return cls(
            parents=parents,
        )

    def get(
        self,
        parent_id: str,
    ) -> ParentRecord | None:

        parent = self.parents.get(
            parent_id
        )

        if parent is None:
            return None

        return ParentRecord(
            parent_id=parent["parent_id"],
            text=parent["text"],
            page_title=str(parent["page_title"]),
            section_heading=(
                str(parent["section_heading"])
                if parent.get("section_heading")
                else None
            ),
            source_url=str(parent["page_url"]),
        )


# ============================================================
# Hybrid retriever
# ============================================================

class HybridRetriever:

    def __init__(
        self,
        dense_retriever: DenseRetriever | None,
        sparse_retriever: BM25Retriever,
        parent_store: ParentStore,
        top_k: int = DEFAULT_TOP_K,
        fetch_multiplier: int = FETCH_MULTIPLIER,
        rrf_k: int = RRF_K,
        parent_level_fusion: bool = True,
    ) -> None:

        if top_k <= 0:
            raise ValueError(
                "top_k must be > 0."
            )

        if fetch_multiplier < 1:
            raise ValueError(
                "fetch_multiplier must be >= 1."
            )

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.parent_store = parent_store
        self.top_k = top_k
        self.fetch_multiplier = fetch_multiplier
        self.rrf_k = rrf_k
        self.parent_level_fusion = parent_level_fusion

    # ----------------------------------------------------
    # Retrieve
    # ----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query must not be empty."
            )

        limit = (
            self.top_k
            if top_k is None
            else top_k
        )

        if limit <= 0:
            raise ValueError(
                "top_k must be > 0."
            )

        run_id = uuid.uuid4().hex[:12]

        started = time.perf_counter()

        log_event(
            event="retrieve.start",
            run_id=run_id,
            query=query,
            top_k=limit,
            rrf_k=self.rrf_k,
            fetch_multiplier=self.fetch_multiplier,
            parent_level_fusion=(
                self.parent_level_fusion
            ),
            dense_enabled=(
                self.dense_retriever is not None
            ),
        )

        fetch_k = limit * self.fetch_multiplier

        ranked_lists = self._collect_ranked_lists(
            query=query,
            fetch_k=fetch_k,
            run_id=run_id,
        )

        if not ranked_lists:
            log_event(
                event="retrieve.empty",
                run_id=run_id,
            )

            return []

        duplicate_map, merged_count = (
            build_duplicate_map(
                ranked_lists=ranked_lists,
            )
        )

        ranked_lists = apply_duplicate_map(
            ranked_lists=ranked_lists,
            duplicate_map=duplicate_map,
        )

        log_event(
            event="dedupe.done",
            run_id=run_id,
            merged_duplicates=merged_count,
        )

        if self.parent_level_fusion:
            ranked_lists = [
                self._to_parent_level_results(ranked_list)
                for ranked_list in ranked_lists
            ]
            # Drop empty lists (e.g. sparse returned 0 hits)
            ranked_lists = [lst for lst in ranked_lists if lst]
            # Re-deduplicate at parent level: two children of the
            # same parent would otherwise produce duplicate parent ids
            # in a single ranked list and inflate RRF votes.
            deduped: list[list[RetrievalResult]] = []
            for lst in ranked_lists:
                seen: set[str] = set()
                uniq: list[RetrievalResult] = []
                for r in lst:
                    if r.chunk_id not in seen:
                        seen.add(r.chunk_id)
                        uniq.append(r)
                deduped.append(uniq)
            ranked_lists = deduped

        fused = fuse_rankings(
            ranked_lists=ranked_lists,
            k=self.rrf_k,
        )

        top_results = fused[:limit]

        elapsed_ms = (
            (time.perf_counter() - started)
            * 1000.0
        )

        log_event(
            event="retrieve.done",
            run_id=run_id,
            fused_candidates=len(fused),
            returned=len(top_results),
            total_latency_ms=round(elapsed_ms, 1),
            top_scores=[
                round(r.score, 5)
                for r in top_results
            ],
            top_ids=[
                r.chunk_id
                for r in top_results
            ],
        )

        return top_results

    # ----------------------------------------------------
    # Ranked list collection
    # ----------------------------------------------------

    def _collect_ranked_lists(
        self,
        query: str,
        fetch_k: int,
        run_id: str,
    ) -> list[list[RetrievalResult]]:

        ranked_lists: list[list[RetrievalResult]] = []

        if self.sparse_retriever is not None:

            stage_started = time.perf_counter()

            sparse_results = [
                to_retrieval_result(result)
                for result in self.sparse_retriever.retrieve(
                    query=query,
                    top_k=fetch_k,
                )
            ]

            log_event(
                event="sparse.done",
                run_id=run_id,
                retrieved=len(sparse_results),
                latency_ms=round(
                    (time.perf_counter() - stage_started)
                    * 1000.0,
                    1,
                ),
            )

            ranked_lists.append(sparse_results)

        if self.dense_retriever is not None:

            stage_started = time.perf_counter()

            dense_results = self.dense_retriever.retrieve(
                query=query,
                top_k=fetch_k,
            )

            log_event(
                event="dense.done",
                run_id=run_id,
                retrieved=len(dense_results),
                latency_ms=round(
                    (time.perf_counter() - stage_started)
                    * 1000.0,
                    1,
                ),
            )

            ranked_lists.append(dense_results)

        return ranked_lists

    # ----------------------------------------------------
    # Parent-level projection
    # ----------------------------------------------------

    def _to_parent_level_results(
        self,
        ranked_list: list[RetrievalResult],
    ) -> list[RetrievalResult]:

        projected: list[RetrievalResult] = []
        seen_parent_ids: set[str] = set()

        for position, result in enumerate(
            ranked_list,
            start=1,
        ):

            parent_id = result.metadata.get(
                "parent_id"
            )

            if not parent_id:
                # Keep child as-is; dedup by chunk_id already done
                if result.chunk_id not in seen_parent_ids:
                    seen_parent_ids.add(result.chunk_id)
                    projected.append(result)
                continue

            parent_id = str(parent_id)

            parent = self.parent_store.get(
                parent_id
            )

            if parent is None:
                if result.chunk_id not in seen_parent_ids:
                    seen_parent_ids.add(result.chunk_id)
                    projected.append(result)
                continue

            # Avoid duplicate parent entries from sibling children
            # in the same ranked list (first occurrence wins).
            if parent.parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent.parent_id)

            projected.append(
                RetrievalResult(
                    # Preserve original relevance score; RRF ignores
                    # score and uses rank, but logs/debugging need it.
                    score=float(result.score),
                    chunk_id=parent.parent_id,
                    text=parent.text,
                    page_title=(
                        parent.page_title
                        or result.page_title
                    ),
                    section_heading=(
                        parent.section_heading
                        or result.section_heading
                    ),
                    source_url=(
                        parent.source_url
                        or result.source_url
                    ),
                    metadata={
                        **result.metadata,
                        "expanded_parent_id": parent.parent_id,
                        "child_chunk_id": result.chunk_id,
                        "child_position_in_list": position,
                        "child_score": float(result.score),
                    },
                )
            )

        return projected


# ============================================================
# CLI output
# ============================================================

def print_results(
    query: str,
    results: list[RetrievalResult],
) -> None:

    print("=" * 72)

    print(f"Query: {query}")

    print(f"Results: {len(results)}")

    print("=" * 72)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print()
        print(f"[{rank}] Score: {result.score:.6f}")

        print(f"Title: {result.page_title}")

        if result.section_heading:
            print(f"Section: {result.section_heading}")

        print(f"Chunk ID: {result.chunk_id}")

        if result.metadata.get("expanded_parent_id"):
            print(
                f"Parent: "
                f"{result.metadata['expanded_parent_id']}"
            )

        if result.metadata.get("child_chunk_id"):
            print(
                f"Matched child: "
                f"{result.metadata['child_chunk_id']}"
            )

        print(f"Source: {result.source_url}")

        print("-" * 72)

        print(result.text)

        print("=" * 72)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Hybrid retrieval (dense + BM25 with RRF fusion, "
            "duplicate merging, and parent-level expansion) "
            "over Persian cultural chunks."
        )
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of final results (default: {DEFAULT_TOP_K})",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=RRF_K,
        help=f"RRF smoothing constant (default: {RRF_K})",
    )

    parser.add_argument(
        "--no-parent-fusion",
        action="store_true",
        help=(
            "Fuse at child level and skip parent expansion "
            "(raw child chunks)."
        ),
    )

    parser.add_argument(
        "--no-dense",
        action="store_true",
        help="Disable dense retrieval (BM25 only).",
    )

    parser.add_argument(
        "--bm25-index-dir",
        type=Path,
        default=Path("Data/processed/bm25"),
        help="Directory containing the BM25 index.",
    )

    parser.add_argument(
        "--chunks-json",
        type=Path,
        default=Path("Data/processed/chunked_data.json"),
        help="Path to chunked_data.json (for parent lookup).",
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=QDRANT_URL,
        help=f"Qdrant URL (default: {QDRANT_URL})",
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: {COLLECTION_NAME})",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Jina API timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Jina API maximum retries (default: {DEFAULT_MAX_RETRIES})",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be > 0.")

    if args.rrf_k <= 0:
        parser.error("--rrf-k must be > 0.")

    if args.timeout <= 0:
        parser.error("--timeout must be > 0.")

    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0.")

    return args


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG
        if args.verbose
        else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    sparse_retriever = BM25Retriever.load(
        index_dir=args.bm25_index_dir,
    )

    parent_store = ParentStore.load(
        chunked_data_path=args.chunks_json,
    )

    dense_retriever = None

    if not args.no_dense:

        api_key, _ = load_jina_api_key()

        query_embedder = lambda text: embed_query(  # noqa: E731
            text,
            api_key=api_key,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )

        vector_store = QdrantVectorStore(
            url=args.qdrant_url,
            collection_name=args.collection,
        )

        dense_retriever = DenseRetriever(
            vector_store=vector_store,
            query_embedder=query_embedder,
        )

    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        parent_store=parent_store,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        parent_level_fusion=not args.no_parent_fusion,
    )

    results = retriever.retrieve(
        query=args.query,
    )

    print_results(
        query=args.query,
        results=results,
    )


if __name__ == "__main__":
    main()