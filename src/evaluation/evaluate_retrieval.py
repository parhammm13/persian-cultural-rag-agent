from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# This file is intended to live at:
#   D:\Summer code\portfolio project\persian-cultural-rag-agent\src\evaluation\
#
# Therefore parents[2] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

DEFAULT_DATA_DIR = PROJECT_ROOT / "Data" / "evaluation" / "v1"
DEFAULT_BM25_INDEX_DIR = PROJECT_ROOT / "Data" / "processed" / "bm25"
DEFAULT_CHUNKS_JSON = PROJECT_ROOT / "Data" / "processed" / "chunked_data.json"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "results2"

DEFAULT_TOP_K = 10
DEFAULT_RERANK_CANDIDATES = 30
DEFAULT_RRF_K = 60
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 2

METHOD_ORDER = (
    "Dense",
    "BM25",
    "Hybrid",
    "Hybrid + Reranker",
)

# Make both common src-layout import styles work when this file is executed
# directly, e.g.:
#   python src\evaluation\evaluate_retrieval.py
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EvalExample:
    id: str
    question: str
    difficulty: str
    gold_parent_ids: tuple[str, ...]
    gold_child_ids: tuple[str, ...]
    page_id: str | None
    page_title: str | None
    challenge_type: str | None
    reference_answer: str | None
    source_file: str


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    first_relevant_rank: int | None


@dataclass(frozen=True, slots=True)
class RetrievalStack:
    dense: Any
    bm25: Any
    hybrid: Any
    reranker: Any


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {path} at line {line_number}."
                )

            rows.append(row)

    return rows


def derive_parent_id_from_child_id(child_id: str) -> str:
    """
    Convert:
        14952#lead#parent_000#child_000
    into:
        14952#lead#parent_000
    """
    child_id = str(child_id).strip()

    if "#child_" in child_id:
        return child_id.rsplit("#child_", 1)[0]

    return child_id


def parse_eval_example(row: dict[str, Any], source_file: str) -> EvalExample:
    example_id = str(row.get("id", "")).strip()
    question = str(row.get("question", "")).strip()
    difficulty = str(row.get("difficulty", "")).strip().lower()

    if not example_id:
        raise ValueError(f"Missing id in {source_file}: {row}")

    if not question:
        raise ValueError(f"Missing question for {example_id}")

    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError(
            f"Unexpected difficulty {difficulty!r} for {example_id}"
        )

    child_ids = tuple(
        str(value).strip()
        for value in row.get("child_ids", [])
        if str(value).strip()
    )

    parent_ids = tuple(
        str(value).strip()
        for value in row.get("parent_ids", [])
        if str(value).strip()
    )

    # Parent IDs are the primary relevance labels because all four retrieval
    # methods can be compared at the parent-evidence level. If a row ever
    # lacks parent_ids, derive them from child_ids as a safe fallback.
    if not parent_ids and child_ids:
        parent_ids = tuple(
            dict.fromkeys(
                derive_parent_id_from_child_id(child_id)
                for child_id in child_ids
            )
        )

    if not parent_ids:
        raise ValueError(
            f"No parent_ids (and no derivable child_ids) for {example_id}"
        )

    return EvalExample(
        id=example_id,
        question=question,
        difficulty=difficulty,
        gold_parent_ids=parent_ids,
        gold_child_ids=child_ids,
        page_id=(
            str(row["page_id"]).strip()
            if row.get("page_id") is not None
            else None
        ),
        page_title=(
            str(row["page_title"]).strip()
            if row.get("page_title") is not None
            else None
        ),
        challenge_type=(
            str(row["challenge_type"]).strip()
            if row.get("challenge_type")
            else None
        ),
        reference_answer=(
            str(row["reference_answer"]).strip()
            if row.get("reference_answer") is not None
            else None
        ),
        source_file=source_file,
    )


def load_evaluation_set(data_dir: Path) -> list[EvalExample]:
    candidate_path = data_dir / "candidate_questions.jsonl"
    challenge_path = data_dir / "challenge_questions.jsonl"

    examples: list[EvalExample] = []

    # Intentionally do NOT filter on review_status.
    # The user confirmed that all challenge/hard examples are accepted even
    # though the JSONL currently says "pending".
    for path in (candidate_path, challenge_path):
        for row in read_jsonl(path):
            examples.append(
                parse_eval_example(
                    row=row,
                    source_file=path.name,
                )
            )

    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        duplicates = sorted(
            example_id
            for example_id in set(ids)
            if ids.count(example_id) > 1
        )
        raise ValueError(
            f"Duplicate evaluation IDs found: {duplicates}"
        )

    return examples


def validate_dataset(examples: Sequence[EvalExample]) -> None:
    """
    Enforce the intended benchmark semantics:

    - easy/medium: exactly one relevant parent chunk
    - hard: one or more relevant parent chunks
    - multi-parent questions are allowed only in hard
    """
    problems: list[str] = []

    for example in examples:
        parent_count = len(set(example.gold_parent_ids))

        if example.difficulty in {"easy", "medium"} and parent_count != 1:
            problems.append(
                f"{example.id}: {example.difficulty} has "
                f"{parent_count} parent_ids"
            )

        if example.difficulty == "hard" and parent_count < 1:
            problems.append(
                f"{example.id}: hard question has no parent_ids"
            )

    if problems:
        joined = "\n  - ".join(problems)
        raise ValueError(
            "Evaluation dataset violates the expected relevance structure:\n"
            f"  - {joined}"
        )


# ---------------------------------------------------------------------------
# Retrieval package loading
# ---------------------------------------------------------------------------

def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError:
        return False
    return True


def resolve_retrieval_package(explicit_package: str | None) -> str:
    """
    The uploaded retriever files use relative imports, so they need to be
    imported as a package. Try common layouts automatically.

    You can always override with:
        --retrieval-package your.package.path
    """
    if explicit_package:
        return explicit_package

    candidates = (
        "src.retrieval",
        "retrieval",
        "src.retrievers",
        "retrievers",
    )

    for package in candidates:
        if _can_import(f"{package}.dense_retriever"):
            return package

    raise ModuleNotFoundError(
        "Could not locate the retrieval package automatically. "
        "Pass it explicitly, for example:\n"
        "  --retrieval-package src.retrieval"
    )


def import_retrieval_components(package: str) -> dict[str, Any]:
    dense_module = importlib.import_module(
        f"{package}.dense_retriever"
    )
    bm25_module = importlib.import_module(
        f"{package}.bm25_retriever"
    )
    hybrid_module = importlib.import_module(
        f"{package}.hybrid_retriever"
    )
    reranker_module = importlib.import_module(
        f"{package}.reranker"
    )
    embedder_module = importlib.import_module(
        f"{package}.embedder"
    )
    vector_store_module = importlib.import_module(
        f"{package}.vector_store"
    )

    return {
        "DenseRetriever": dense_module.DenseRetriever,
        "BM25Retriever": bm25_module.BM25Retriever,
        "HybridRetriever": hybrid_module.HybridRetriever,
        "ParentStore": hybrid_module.ParentStore,
        "JinaReranker": reranker_module.JinaReranker,
        "embed_query": embedder_module.embed_query,
        "load_jina_api_key": embedder_module.load_jina_api_key,
        "QdrantVectorStore": vector_store_module.QdrantVectorStore,
        "QDRANT_URL": vector_store_module.QDRANT_URL,
        "COLLECTION_NAME": vector_store_module.COLLECTION_NAME,
    }


def build_retrieval_stack(
    *,
    components: dict[str, Any],
    bm25_index_dir: Path,
    chunks_json: Path,
    qdrant_url: str | None,
    collection_name: str | None,
    timeout: int,
    max_retries: int,
    rrf_k: int,
) -> RetrievalStack:
    BM25Retriever = components["BM25Retriever"]
    DenseRetriever = components["DenseRetriever"]
    HybridRetriever = components["HybridRetriever"]
    ParentStore = components["ParentStore"]
    JinaReranker = components["JinaReranker"]
    QdrantVectorStore = components["QdrantVectorStore"]
    embed_query = components["embed_query"]
    load_jina_api_key = components["load_jina_api_key"]

    api_key, _ = load_jina_api_key()

    # Cache each query embedding. The same evaluation question is used by
    # Dense, Hybrid, and Hybrid+Reranker candidate generation, so this avoids
    # paying for the same Jina embedding multiple times.
    @lru_cache(maxsize=4096)
    def cached_query_embedder(text: str):
        return embed_query(
            text,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    bm25 = BM25Retriever.load(
        index_dir=bm25_index_dir,
    )

    parent_store = ParentStore.load(
        chunked_data_path=chunks_json,
    )

    vector_store = QdrantVectorStore(
        url=qdrant_url or components["QDRANT_URL"],
        collection_name=(
            collection_name
            or components["COLLECTION_NAME"]
        ),
    )

    dense = DenseRetriever(
        vector_store=vector_store,
        query_embedder=cached_query_embedder,
    )

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=bm25,
        parent_store=parent_store,
        rrf_k=rrf_k,
        parent_level_fusion=True,
    )

    reranker = JinaReranker(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )

    return RetrievalStack(
        dense=dense,
        bm25=bm25,
        hybrid=hybrid,
        reranker=reranker,
    )


# ---------------------------------------------------------------------------
# Relevance normalization
# ---------------------------------------------------------------------------

def result_chunk_id(result: Any) -> str:
    value = getattr(result, "chunk_id", None)

    if value is None:
        raise ValueError(
            f"Retrieved result has no chunk_id: {result!r}"
        )

    return str(value)


def result_metadata(result: Any) -> dict[str, Any]:
    metadata = getattr(result, "metadata", None)

    if isinstance(metadata, dict):
        return metadata

    return {}


def result_parent_id(result: Any) -> str:
    """
    Normalize Dense, BM25, Hybrid, and reranked results to one comparable
    parent-evidence ID.

    Priority:
      1. expanded_parent_id  -> Hybrid parent expansion
      2. parent_id           -> Dense/BM25 child metadata
      3. derive from child chunk_id
      4. use chunk_id itself -> already a parent result
    """
    metadata = result_metadata(result)

    expanded_parent_id = metadata.get("expanded_parent_id")
    if expanded_parent_id:
        return str(expanded_parent_id)

    parent_id = metadata.get("parent_id")
    if parent_id:
        return str(parent_id)

    chunk_id = result_chunk_id(result)

    if "#child_" in chunk_id:
        return derive_parent_id_from_child_id(chunk_id)

    return chunk_id


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    ranked_parent_ids: Sequence[str],
    gold_parent_ids: Sequence[str],
    k: int,
) -> float:
    """
    Macro Recall@k for one query.

    Example with two gold chunks:
        gold = {A, B}
        top-5 retrieves only A
        Recall@5 = 1 / 2 = 0.5

    Duplicate retrieved children mapping to the same parent do not increase
    the numerator because the intersection is a set.
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    gold = set(gold_parent_ids)

    if not gold:
        raise ValueError("gold_parent_ids must not be empty")

    retrieved = set(ranked_parent_ids[:k])

    return len(gold & retrieved) / len(gold)


def reciprocal_rank(
    ranked_parent_ids: Sequence[str],
    gold_parent_ids: Sequence[str],
) -> tuple[float, int | None]:
    gold = set(gold_parent_ids)

    for rank, parent_id in enumerate(
        ranked_parent_ids,
        start=1,
    ):
        if parent_id in gold:
            return 1.0 / rank, rank

    return 0.0, None


def score_query(
    results: Sequence[Any],
    gold_parent_ids: Sequence[str],
    max_k: int,
) -> tuple[QueryMetrics, list[str], list[str]]:
    raw_chunk_ids = [
        result_chunk_id(result)
        for result in results[:max_k]
    ]

    ranked_parent_ids = [
        result_parent_id(result)
        for result in results[:max_k]
    ]

    rr, first_rank = reciprocal_rank(
        ranked_parent_ids=ranked_parent_ids,
        gold_parent_ids=gold_parent_ids,
    )

    return (
        QueryMetrics(
            recall_at_5=recall_at_k(
                ranked_parent_ids,
                gold_parent_ids,
                k=5,
            ),
            recall_at_10=recall_at_k(
                ranked_parent_ids,
                gold_parent_ids,
                k=10,
            ),
            reciprocal_rank=rr,
            first_relevant_rank=first_rank,
        ),
        raw_chunk_ids,
        ranked_parent_ids,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_method(
    *,
    method: str,
    question: str,
    stack: RetrievalStack,
    top_k: int,
    rerank_candidates: int,
) -> Sequence[Any]:
    if method == "Dense":
        return stack.dense.retrieve(
            query=question,
            top_k=top_k,
        )

    if method == "BM25":
        return stack.bm25.retrieve(
            query=question,
            top_k=top_k,
        )

    if method == "Hybrid":
        return stack.hybrid.retrieve(
            query=question,
            top_k=top_k,
        )

    if method == "Hybrid + Reranker":
        candidates = stack.hybrid.retrieve(
            query=question,
            top_k=rerank_candidates,
        )

        return stack.reranker.rerank(
            query=question,
            candidates=list(candidates),
            top_k=top_k,
        )

    raise ValueError(f"Unknown method: {method}")


def evaluate(
    *,
    examples: Sequence[EvalExample],
    stack: RetrievalStack,
    top_k: int,
    rerank_candidates: int,
    continue_on_error: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    if top_k < 10:
        raise ValueError(
            "top_k must be at least 10 because Recall@10 is requested."
        )

    selected_examples = (
        list(examples[:limit])
        if limit is not None
        else list(examples)
    )

    records: list[dict[str, Any]] = []
    total = len(selected_examples)

    for index, example in enumerate(
        selected_examples,
        start=1,
    ):
        print(
            f"\n[{index:03d}/{total:03d}] "
            f"{example.id} | {example.difficulty}"
        )
        print(f"Question: {example.question}")
        print(
            "Gold parents: "
            + ", ".join(example.gold_parent_ids)
        )

        for method in METHOD_ORDER:
            started = time.perf_counter()
            error: str | None = None

            try:
                results = run_method(
                    method=method,
                    question=example.question,
                    stack=stack,
                    top_k=top_k,
                    rerank_candidates=rerank_candidates,
                )

                metrics, raw_chunk_ids, ranked_parent_ids = (
                    score_query(
                        results=results,
                        gold_parent_ids=example.gold_parent_ids,
                        max_k=top_k,
                    )
                )

            except Exception as exc:
                if not continue_on_error:
                    raise RuntimeError(
                        f"{method} failed on {example.id}: {exc}"
                    ) from exc

                error = f"{type(exc).__name__}: {exc}"
                raw_chunk_ids = []
                ranked_parent_ids = []
                metrics = QueryMetrics(
                    recall_at_5=0.0,
                    recall_at_10=0.0,
                    reciprocal_rank=0.0,
                    first_relevant_rank=None,
                )

            latency_ms = (
                time.perf_counter() - started
            ) * 1000.0

            print(
                f"  {method:<20} "
                f"R@5={metrics.recall_at_5:.3f} "
                f"R@10={metrics.recall_at_10:.3f} "
                f"RR={metrics.reciprocal_rank:.3f} "
                f"rank={metrics.first_relevant_rank} "
                f"{latency_ms:.0f} ms"
            )

            if error:
                print(f"    ERROR: {error}")

            records.append(
                {
                    "id": example.id,
                    "question": example.question,
                    "difficulty": example.difficulty,
                    "challenge_type": example.challenge_type,
                    "source_file": example.source_file,
                    "page_id": example.page_id,
                    "page_title": example.page_title,
                    "reference_answer": example.reference_answer,
                    "method": method,
                    "gold_parent_ids": list(
                        example.gold_parent_ids
                    ),
                    "gold_child_ids": list(
                        example.gold_child_ids
                    ),
                    "retrieved_chunk_ids": raw_chunk_ids,
                    "retrieved_parent_ids": ranked_parent_ids,
                    "relevant_parent_count": len(
                        set(example.gold_parent_ids)
                    ),
                    "recall_at_5": metrics.recall_at_5,
                    "recall_at_10": metrics.recall_at_10,
                    "reciprocal_rank": (
                        metrics.reciprocal_rank
                    ),
                    "first_relevant_rank": (
                        metrics.first_relevant_rank
                    ),
                    "latency_ms": round(latency_ms, 3),
                    "error": error,
                }
            )

    return records


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def mean(values: Iterable[float]) -> float:
    values = list(values)

    if not values:
        return 0.0

    return sum(values) / len(values)


def aggregate_records(
    records: Sequence[dict[str, Any]],
    *,
    group_key: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    group_values: list[str | None]

    if group_key is None:
        group_values = [None]
    else:
        group_values = sorted(
            {
                str(record[group_key])
                for record in records
                if record.get(group_key) is not None
            },
            key=lambda value: (
                {"easy": 0, "medium": 1, "hard": 2}.get(
                    value,
                    99,
                ),
                value,
            ),
        )

    for group_value in group_values:
        for method in METHOD_ORDER:
            subset = [
                record
                for record in records
                if record["method"] == method
                and (
                    group_key is None
                    or str(record.get(group_key))
                    == group_value
                )
            ]

            if not subset:
                continue

            row = {
                "Method": method,
                "N": len(subset),
                "Recall@5": mean(
                    record["recall_at_5"]
                    for record in subset
                ),
                "Recall@10": mean(
                    record["recall_at_10"]
                    for record in subset
                ),
                # This is MRR over the retrieved top_k list.
                "MRR": mean(
                    record["reciprocal_rank"]
                    for record in subset
                ),
                "Errors": sum(
                    1
                    for record in subset
                    if record["error"]
                ),
            }

            if group_key is not None:
                row[group_key] = group_value

            rows.append(row)

    return rows


def write_jsonl(
    path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            formatted = dict(row)

            for key in ("Recall@5", "Recall@10", "MRR"):
                if key in formatted:
                    formatted[key] = (
                        f"{float(formatted[key]):.6f}"
                    )

            writer.writerow(formatted)


def print_markdown_table(
    title: str,
    rows: Sequence[dict[str, Any]],
    group_key: str | None = None,
) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)

    if group_key is None:
        print(
            "| Method | N | Recall@5 | Recall@10 | MRR | Errors |"
        )
        print(
            "|---|---:|---:|---:|---:|---:|"
        )

        for row in rows:
            print(
                f"| {row['Method']} "
                f"| {row['N']} "
                f"| {row['Recall@5']:.4f} "
                f"| {row['Recall@10']:.4f} "
                f"| {row['MRR']:.4f} "
                f"| {row['Errors']} |"
            )
    else:
        print(
            f"| {group_key} | Method | N | "
            "Recall@5 | Recall@10 | MRR | Errors |"
        )
        print(
            "|---|---|---:|---:|---:|---:|---:|"
        )

        for row in rows:
            print(
                f"| {row[group_key]} "
                f"| {row['Method']} "
                f"| {row['N']} "
                f"| {row['Recall@5']:.4f} "
                f"| {row['Recall@10']:.4f} "
                f"| {row['MRR']:.4f} "
                f"| {row['Errors']} |"
            )


def print_dataset_summary(
    examples: Sequence[EvalExample],
) -> None:
    by_difficulty = {
        difficulty: [
            example
            for example in examples
            if example.difficulty == difficulty
        ]
        for difficulty in ("easy", "medium", "hard")
    }

    multi_parent = [
        example
        for example in examples
        if len(set(example.gold_parent_ids)) > 1
    ]

    print("=" * 88)
    print("Dataset summary")
    print("=" * 88)
    print(f"Total questions : {len(examples)}")

    for difficulty in ("easy", "medium", "hard"):
        print(
            f"{difficulty.capitalize():<15}: "
            f"{len(by_difficulty[difficulty])}"
        )

    print(
        "Multi-parent     : "
        f"{len(multi_parent)} "
        "(expected only among hard questions)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dense, BM25, Hybrid, and Hybrid + Reranker "
            "using Recall@5, Recall@10, and MRR."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Directory containing candidate_questions.jsonl and "
            "challenge_questions.jsonl. "
            f"Default: {DEFAULT_DATA_DIR}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for evaluation outputs. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    parser.add_argument(
        "--bm25-index-dir",
        type=Path,
        default=DEFAULT_BM25_INDEX_DIR,
        help=(
            "BM25 index directory. "
            f"Default: {DEFAULT_BM25_INDEX_DIR}"
        ),
    )

    parser.add_argument(
        "--chunks-json",
        type=Path,
        default=DEFAULT_CHUNKS_JSON,
        help=(
            "chunked_data.json used by ParentStore. "
            f"Default: {DEFAULT_CHUNKS_JSON}"
        ),
    )

    parser.add_argument(
        "--retrieval-package",
        type=str,
        default=None,
        help=(
            "Python package containing dense_retriever.py, "
            "bm25_retriever.py, hybrid_retriever.py, reranker.py, "
            "embedder.py, and vector_store.py. "
            "Auto-detected when omitted."
        ),
    )

    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help=(
            "Qdrant URL. Uses vector_store.QDRANT_URL when omitted."
        ),
    )

    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help=(
            "Qdrant collection. Uses vector_store.COLLECTION_NAME "
            "when omitted."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Final number of results. Must be >= 10. "
            f"Default: {DEFAULT_TOP_K}"
        ),
    )

    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=DEFAULT_RERANK_CANDIDATES,
        help=(
            "Number of Hybrid candidates sent to Jina reranker. "
            f"Default: {DEFAULT_RERANK_CANDIDATES}"
        ),
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=(
            "Hybrid Reciprocal Rank Fusion constant. "
            f"Default: {DEFAULT_RRF_K}"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=(
            "Jina request timeout in seconds. "
            f"Default: {DEFAULT_TIMEOUT}"
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Jina maximum retries. "
            f"Default: {DEFAULT_MAX_RETRIES}"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N questions. "
            "Useful for a smoke test."
        ),
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Record method failures as zero-score rows and continue. "
            "By default evaluation stops on the first error so API "
            "or configuration failures cannot silently corrupt metrics."
        ),
    )

    args = parser.parse_args()

    if args.top_k < 10:
        parser.error("--top-k must be >= 10.")

    if args.rerank_candidates < args.top_k:
        parser.error(
            "--rerank-candidates must be >= --top-k."
        )

    if args.rrf_k <= 0:
        parser.error("--rrf-k must be > 0.")

    if args.timeout <= 0:
        parser.error("--timeout must be > 0.")

    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0.")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0.")

    return args


def main() -> None:
    args = parse_args()

    examples = load_evaluation_set(
        data_dir=args.data_dir,
    )
    validate_dataset(examples)
    print_dataset_summary(examples)

    retrieval_package = resolve_retrieval_package(
        args.retrieval_package
    )

    print()
    print(f"Retrieval package: {retrieval_package}")
    print(f"Data directory   : {args.data_dir}")
    print(f"Output directory : {args.output_dir}")
    print(f"BM25 index       : {args.bm25_index_dir}")
    print(f"Chunks JSON      : {args.chunks_json}")
    print(f"Top K            : {args.top_k}")
    print(
        f"Rerank candidates: {args.rerank_candidates}"
    )

    components = import_retrieval_components(
        package=retrieval_package,
    )

    stack = build_retrieval_stack(
        components=components,
        bm25_index_dir=args.bm25_index_dir,
        chunks_json=args.chunks_json,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        timeout=args.timeout,
        max_retries=args.max_retries,
        rrf_k=args.rrf_k,
    )

    records = evaluate(
        examples=examples,
        stack=stack,
        top_k=args.top_k,
        rerank_candidates=args.rerank_candidates,
        continue_on_error=args.continue_on_error,
        limit=args.limit,
    )

    overall = aggregate_records(records)
    by_difficulty = aggregate_records(
        records,
        group_key="difficulty",
    )

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    details_path = (
        output_dir / "retrieval_per_query.jsonl"
    )
    overall_path = (
        output_dir / "retrieval_metrics_overall.csv"
    )
    difficulty_path = (
        output_dir / "retrieval_metrics_by_difficulty.csv"
    )

    write_jsonl(
        path=details_path,
        records=records,
    )
    write_csv(
        path=overall_path,
        rows=overall,
    )
    write_csv(
        path=difficulty_path,
        rows=by_difficulty,
    )

    print_markdown_table(
        title="Overall retrieval metrics",
        rows=overall,
    )

    print_markdown_table(
        title="Retrieval metrics by difficulty",
        rows=by_difficulty,
        group_key="difficulty",
    )

    print()
    print("=" * 88)
    print("Saved")
    print("=" * 88)
    print(details_path)
    print(overall_path)
    print(difficulty_path)


if __name__ == "__main__":
    main()
