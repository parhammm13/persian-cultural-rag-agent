from __future__ import annotations


import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models


# ============================================================
# Configuration
# ============================================================

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "persian_cultural_chunks"

VECTOR_SIZE = 1024
DEFAULT_BATCH_SIZE = 256


# ============================================================
# File helpers
# ============================================================

def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected JSON object: {path}"
        )

    return data


# ============================================================
# Artifact validation
# ============================================================

def load_and_validate_artifacts(
    chunked_path: Path,
    embedding_index_path: Path,
    embeddings_path: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
]:

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    chunked_data = load_json(
        chunked_path
    )

    embedding_index = load_json(
        embedding_index_path
    )

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False,
    )

    children = chunked_data["children"]
    index_items = embedding_index["items"]

    # --------------------------------------------------------
    # Validate basic structures
    # --------------------------------------------------------

    if not isinstance(children, list):
        raise TypeError(
            "'children' must be a list."
        )

    if not isinstance(index_items, list):
        raise TypeError(
            "'items' must be a list."
        )

    # --------------------------------------------------------
    # Validate embedding matrix
    # --------------------------------------------------------

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D matrix."
        )

    if embeddings.shape[1] != VECTOR_SIZE:
        raise ValueError(
            f"Invalid vector dimension: "
            f"{embeddings.shape[1]} != {VECTOR_SIZE}"
        )

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Invalid embeddings dtype: "
            f"{embeddings.dtype} != float32"
        )

    if not np.all(
        np.isfinite(embeddings)
    ):
        raise ValueError(
            "Embeddings contain NaN or Inf."
        )

    # --------------------------------------------------------
    # Validate counts
    # --------------------------------------------------------

    total_vectors = embeddings.shape[0]

    if embedding_index["total"] != len(
        index_items
    ):
        raise ValueError(
            "embedding_index total does not "
            "match items count."
        )

    if len(index_items) != total_vectors:
        raise ValueError(
            "Embedding index count does not "
            "match vector count."
        )

    if len(children) != total_vectors:
        raise ValueError(
            "Children count does not "
            "match vector count."
        )

    # --------------------------------------------------------
    # Build child lookup
    # --------------------------------------------------------

    children_by_id = {
        child["chunk_id"]: child
        for child in children
    }

    if len(children_by_id) != len(children):
        raise ValueError(
            "Duplicate chunk_id found in children."
        )

    # --------------------------------------------------------
    # Validate index mappings
    # --------------------------------------------------------

    seen_rows: set[int] = set()

    for item in index_items:

        row = item["row"]
        chunk_id = item["chunk_id"]
        expected_hash = item["text_hash"]

        # --------------------------------------------
        # Row validation
        # --------------------------------------------

        if row in seen_rows:
            raise ValueError(
                f"Duplicate embedding row: {row}"
            )

        seen_rows.add(row)

        if row < 0 or row >= total_vectors:
            raise ValueError(
                f"Invalid embedding row: {row}"
            )

        # --------------------------------------------
        # Chunk validation
        # --------------------------------------------

        child = children_by_id.get(
            chunk_id
        )

        if child is None:
            raise ValueError(
                f"Unknown chunk_id: {chunk_id}"
            )

        # --------------------------------------------
        # Text ↔ embedding validation
        # --------------------------------------------

        text = child["text"].strip()

        actual_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        if actual_hash != expected_hash:
            raise ValueError(
                "Text hash mismatch for chunk: "
                f"{chunk_id}"
            )

    return (
        children_by_id,
        index_items,
        embeddings,
    )


# ============================================================
# Qdrant Vector Store
# ============================================================

class QdrantVectorStore:

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection_name: str = COLLECTION_NAME,
    ) -> None:

        self.client = QdrantClient(
            url=url
        )

        self.collection_name = (
            collection_name
        )

    # ========================================================
    # Collection
    # ========================================================

    def create_collection(
        self,
    ) -> None:

        exists = self.client.collection_exists(
            collection_name=self.collection_name
        )

        if exists:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    # ========================================================
    # Point ID
    # ========================================================

    @staticmethod
    def make_point_id(
        chunk_id: str,
    ) -> str:

        return str(
            uuid5(
                NAMESPACE_URL,
                f"persian-cultural-rag:{chunk_id}",
            )
        )

    # ========================================================
    # Payload
    # ========================================================

    @staticmethod
    def build_payload(
        child: dict[str, Any],
        row: int,
        text_hash: str,
    ) -> dict[str, Any]:

        return {
            "row": row,

            "chunk_id":
                child["chunk_id"],

            "parent_id":
                child["parent_id"],

            "semantic_unit_id":
                child["semantic_unit_id"],

            "page_id":
                child["page_id"],

            "page_title":
                child["page_title"],

            "page_url":
                child["page_url"],

            "unit_type":
                child["unit_type"],

            "section_index":
                child["section_index"],

            "section_heading":
                child["section_heading"],

            "section_level":
                child["section_level"],

            "parent_index":
                child["parent_index"],

            "child_index":
                child["child_index"],

            "text":
                child["text"],

            "tokens":
                child["tokens"],

            "text_hash":
                text_hash,
        }

    # ========================================================
    # Build Point
    # ========================================================

    @classmethod
    def build_point(
        cls,
        child: dict[str, Any],
        vector: list[float],
        row: int,
        text_hash: str,
    ) -> models.PointStruct:

        point_id = cls.make_point_id(
            child["chunk_id"]
        )

        payload = cls.build_payload(
            child=child,
            row=row,
            text_hash=text_hash,
        )

        return models.PointStruct(
            id=point_id,
            vector=vector,
            payload=payload,
        )

    # ========================================================
    # Upsert
    # ========================================================

    def upsert_points(
        self,
        children_by_id: dict[
            str,
            dict[str, Any],
        ],
        index_items: list[
            dict[str, Any]
        ],
        embeddings: np.ndarray,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0."
            )

        batch: list[
            models.PointStruct
        ] = []

        for item in index_items:

            row = item["row"]
            chunk_id = item["chunk_id"]

            child = children_by_id[
                chunk_id
            ]

            vector = embeddings[
                row
            ]

            point = self.build_point(
                child=child,
                vector=vector.tolist(),
                row=row,
                text_hash=item["text_hash"],
            )

            batch.append(
                point
            )

            # Send full batch
            if len(batch) >= batch_size:

                self.client.upsert(
                    collection_name=(
                        self.collection_name
                    ),
                    points=batch,
                    wait=True,
                )

                batch.clear()

        # Send remaining points
        if batch:

            self.client.upsert(
                collection_name=(
                    self.collection_name
                ),
                points=batch,
                wait=True,
            )

    # ========================================================
    # Count
    # ========================================================

    def count_points(
        self,
    ) -> int:

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    # ========================================================
    # Similarity Search
    # ========================================================

    def search_by_vector(
        self,
        query_vector: list[float] | np.ndarray,
        limit: int = 5,
    ) -> list[Any]:

        if limit <= 0:
            raise ValueError(
                "limit must be > 0."
            )

        vector = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise ValueError(
                "Query vector must be 1-dimensional."
            )

        if vector.shape[0] != VECTOR_SIZE:
            raise ValueError(
                f"Invalid query vector dimension: "
                f"{vector.shape[0]} != {VECTOR_SIZE}"
            )

        if not np.all(
            np.isfinite(vector)
        ):
            raise ValueError(
                "Query vector contains NaN or Inf."
            )

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return result.points






# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Load validated embeddings and child chunks "
            "into a Qdrant collection."
        )
    )

    parser.add_argument(
        "chunked_json",
        type=Path,
        help="Path to chunked_data.json",
    )

    parser.add_argument(
        "embedding_index",
        type=Path,
        help="Path to embedding_index.json",
    )

    parser.add_argument(
        "embeddings_npy",
        type=Path,
        help="Path to embeddings.npy",
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
        help=(
            "Qdrant collection name "
            f"(default: {COLLECTION_NAME})"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of points per upsert batch "
            f"(default: {DEFAULT_BATCH_SIZE})"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    print("=" * 64)
    print("QDRANT INGESTION")
    print("=" * 64)

    print(f"Chunked JSON    : {args.chunked_json}")
    print(f"Embedding index : {args.embedding_index}")
    print(f"Embeddings      : {args.embeddings_npy}")
    print(f"Qdrant URL      : {args.qdrant_url}")
    print(f"Collection      : {args.collection}")
    print(f"Batch size      : {args.batch_size}")

    # --------------------------------------------------------
    # Load + validate
    # --------------------------------------------------------

    print("\nLoading and validating artifacts...")

    children_by_id, index_items, embeddings = (
        load_and_validate_artifacts(
            chunked_path=args.chunked_json,
            embedding_index_path=args.embedding_index,
            embeddings_path=args.embeddings_npy,
        )
    )

    print("Validation passed.")
    print(f"Children   : {len(children_by_id)}")
    print(f"Index items: {len(index_items)}")
    print(f"Embeddings : {embeddings.shape}")
    print(f"Dtype      : {embeddings.dtype}")

    # --------------------------------------------------------
    # Qdrant
    # --------------------------------------------------------

    store = QdrantVectorStore(
        url=args.qdrant_url,
        collection_name=args.collection,
    )

    print("\nCreating collection if needed...")

    store.create_collection()

    print("Collection ready.")

    # --------------------------------------------------------
    # Upsert
    # --------------------------------------------------------

    print("\nUploading points...")

    store.upsert_points(
        children_by_id=children_by_id,
        index_items=index_items,
        embeddings=embeddings,
        batch_size=args.batch_size,
    )

    print("Upload completed.")

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    count = store.count_points()

    print(f"\nQdrant point count: {count}")

    if count != len(index_items):
        raise RuntimeError(
            "Qdrant point count does not match "
            "embedding index count."
        )

    print("Point count validation passed.")

    print("=" * 64)


if __name__ == "__main__":
    main()