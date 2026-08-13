import argparse
import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
MODEL_NAME = "jina-embeddings-v5-text-small"

# For document chunks in a RAG retrieval pipeline.
# When you later embed the user's search query, use:
# task="retrieval.query"
JINA_TASK = "retrieval.passage"

EXPECTED_DIMENSION = 1024
DEFAULT_BATCH_SIZE = 32
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 6

OUTPUT_DTYPE = np.float32
SCHEMA_VERSION = 2


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ============================================================
# Environment
# ============================================================

def load_jina_api_key() -> Tuple[str, Optional[Path]]:
    """
    Find .env by walking upward from this file.

    Example project structure:

        persian-cultural-rag-agent/
        ├── .env
        └── src/
            └── data/
                └── embedder.py

    This means the .env file does not need to be copied into src/data.
    """

    script_dir = Path(__file__).resolve().parent
    env_path: Optional[Path] = None

    for directory in [script_dir, *script_dir.parents]:
        candidate = directory / ".env"

        if candidate.exists():
            load_dotenv(candidate)
            env_path = candidate
            break

    # Also supports a key already defined in the operating system.
    api_key = os.getenv("JINA_API_KEY")

    if not api_key:
        searched_from = Path(__file__).resolve()

        raise RuntimeError(
            "JINA_API_KEY was not found.\n"
            f"Searched for .env starting from: {searched_from}\n"
            "Your .env should contain:\n"
            "JINA_API_KEY=jina_your_key_here"
        )

    return api_key, env_path


# ============================================================
# General helpers
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def atomic_write_json(
    path: Path,
    data: Dict[str, Any],
) -> None:
    temp_path = path.with_name(
        path.name + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temp_path,
        path,
    )


def atomic_write_npy(
    path: Path,
    array: np.ndarray,
) -> None:
    temp_path = path.with_name(
        path.name + ".tmp"
    )

    with temp_path.open("wb") as file:
        np.save(
            file,
            array,
            allow_pickle=False,
        )

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# Input
# ============================================================

def load_dataset(
    input_path: Path,
) -> Dict[str, Any]:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input JSON not found: {input_path}"
        )

    logger.info(
        "Loading dataset: %s",
        input_path,
    )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            "Top-level JSON must be an object."
        )

    required_top_level = {
        "config",
        "stats",
        "parents",
        "children",
    }

    missing = (
        required_top_level
        - set(data.keys())
    )

    if missing:
        raise ValueError(
            "Missing top-level fields: "
            f"{sorted(missing)}"
        )

    if not isinstance(
        data["children"],
        list,
    ):
        raise TypeError(
            "'children' must be a list."
        )

    logger.info(
        "Dataset loaded | parents=%s | children=%s",
        len(data["parents"]),
        len(data["children"]),
    )

    expected_children = (
        data
        .get("stats", {})
        .get("children")
    )

    if (
        expected_children is not None
        and expected_children
        != len(data["children"])
    ):
        logger.warning(
            "Children count mismatch | "
            "stats=%s actual=%s",
            expected_children,
            len(data["children"]),
        )

    return data


# ============================================================
# Child validation
# ============================================================

def prepare_children(
    children: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    Extract only the fields required for embedding.

    Full metadata remains in chunked_data.json.
    """

    prepared: List[Dict[str, str]] = []
    seen_chunk_ids = set()

    required_fields = {
        "chunk_id",
        "parent_id",
        "text",
    }

    for index, child in enumerate(children):

        if not isinstance(child, dict):
            raise TypeError(
                f"Child #{index} must be an object."
            )

        missing = (
            required_fields
            - set(child.keys())
        )

        if missing:
            raise ValueError(
                f"Child #{index} missing fields: "
                f"{sorted(missing)}"
            )

        chunk_id = child["chunk_id"]

        if not isinstance(chunk_id, str):
            raise TypeError(
                f"Invalid chunk_id at child #{index}"
            )

        if chunk_id in seen_chunk_ids:
            raise ValueError(
                "Duplicate chunk_id detected: "
                f"{chunk_id}"
            )

        seen_chunk_ids.add(chunk_id)

        raw_text = child["text"]

        if not isinstance(raw_text, str):
            raise TypeError(
                "Text must be string for chunk: "
                f"{chunk_id}"
            )

        text = raw_text.strip()

        if not text:
            raise ValueError(
                "Empty text detected for chunk: "
                f"{chunk_id}"
            )

        prepared.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "text_hash": sha256_text(text),
            }
        )

    logger.info(
        "Child validation passed | unique chunks=%s",
        len(prepared),
    )

    return prepared


# ============================================================
# Embedding configuration fingerprint
# ============================================================

def build_embedding_config() -> Dict[str, Any]:
    return {
        "provider": "jina-ai",
        "api_url": JINA_API_URL,
        "model_name": MODEL_NAME,
        "task": JINA_TASK,
        "dimension": EXPECTED_DIMENSION,
        "embedding_type": "float",
        "normalize_embeddings": True,
        "dtype": "float32",
        "backend": "remote_api",
    }


def config_hash(
    config: Dict[str, Any],
) -> str:

    serialized = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return sha256_text(serialized)


def build_cache_key(
    text_hash: str,
    embedding_config_hash: str,
) -> str:

    return sha256_text(
        embedding_config_hash
        + ":"
        + text_hash
    )


# ============================================================
# SQLite embedding cache
# ============================================================

class EmbeddingCache:
    """
    Persistent local embedding cache.

    This is NOT Qdrant and NOT the final vector database.
    It only prevents repeated Jina API calls for text that has
    already been embedded with the same configuration.
    """

    def __init__(
        self,
        db_path: Path,
    ) -> None:

        self.db_path = db_path

        self.connection = sqlite3.connect(
            str(db_path)
        )

        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                vector BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        self.connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_embeddings_text_hash
            ON embeddings(text_hash)
            """
        )

        self.connection.commit()

    def get_many(
        self,
        cache_keys: List[str],
    ) -> Dict[str, np.ndarray]:

        if not cache_keys:
            return {}

        placeholders = ",".join(
            "?"
            for _ in cache_keys
        )

        query = f"""
            SELECT
                cache_key,
                dimension,
                dtype,
                vector
            FROM embeddings
            WHERE cache_key IN ({placeholders})
        """

        rows = self.connection.execute(
            query,
            cache_keys,
        ).fetchall()

        result: Dict[str, np.ndarray] = {}

        for (
            cache_key,
            dimension,
            dtype,
            vector_blob,
        ) in rows:

            if dimension != EXPECTED_DIMENSION:
                continue

            if dtype != "float32":
                continue

            vector = np.frombuffer(
                vector_blob,
                dtype=np.float32,
            ).copy()

            if vector.shape != (
                EXPECTED_DIMENSION,
            ):
                continue

            if not np.all(
                np.isfinite(vector)
            ):
                continue

            result[cache_key] = vector

        return result

    def put_many(
        self,
        records: List[Dict[str, Any]],
    ) -> None:

        if not records:
            return

        sql = """
            INSERT OR REPLACE INTO embeddings (
                cache_key,
                config_hash,
                text_hash,
                dimension,
                dtype,
                vector,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        values = []
        now = utc_now_iso()

        for record in records:

            vector = np.asarray(
                record["vector"],
                dtype=np.float32,
            )

            values.append(
                (
                    record["cache_key"],
                    record["config_hash"],
                    record["text_hash"],
                    EXPECTED_DIMENSION,
                    "float32",
                    vector.tobytes(),
                    now,
                )
            )

        self.connection.executemany(
            sql,
            values,
        )

        # Persist each successful API batch so reruns can resume.
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


# ============================================================
# Vector validation
# ============================================================

def validate_and_normalize_vectors(
    vectors: np.ndarray,
) -> np.ndarray:

    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if vectors.ndim == 1:
        vectors = vectors.reshape(
            1,
            -1,
        )

    if vectors.ndim != 2:
        raise RuntimeError(
            "Embedding output must be a 2D matrix."
        )

    if (
        vectors.shape[1]
        != EXPECTED_DIMENSION
    ):
        raise RuntimeError(
            "Unexpected embedding dimension: "
            f"{vectors.shape[1]} "
            f"!= {EXPECTED_DIMENSION}"
        )

    if not np.all(
        np.isfinite(vectors)
    ):
        raise RuntimeError(
            "Embedding contains NaN or Inf."
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= 0):
        raise RuntimeError(
            "Zero-length embedding detected."
        )

    # Jina is requested to normalize vectors, but normalizing
    # once more locally guarantees unit-length float32 output.
    vectors = (
        vectors / norms
    ).astype(
        np.float32,
        copy=False,
    )

    return vectors


# ============================================================
# Jina API
# ============================================================

def request_jina_embeddings(
    texts: List[str],
    api_key: str,
    timeout: int,
    max_retries: int,
) -> np.ndarray:
    """
    Send one synchronous batch to Jina Embeddings API.

    Documents use retrieval.passage.
    Later, user search queries should use retrieval.query.
    """

    payload = {
        "model": MODEL_NAME,
        "task": JINA_TASK,
        "dimensions": EXPECTED_DIMENSION,
        "input": texts,
        "embedding_type": "float",
        "normalized": True,
        "truncate": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    retryable_status_codes = {
        429,
        500,
        503,
        504,
    }

    last_error: Optional[Exception] = None

    for attempt in range(
        max_retries + 1
    ):
        try:
            response = requests.post(
                JINA_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code in retryable_status_codes:

                if attempt >= max_retries:
                    response.raise_for_status()

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:
                    try:
                        sleep_seconds = float(
                            retry_after
                        )
                    except ValueError:
                        sleep_seconds = float(
                            2 ** attempt
                        )
                else:
                    sleep_seconds = float(
                        2 ** attempt
                    )

                logger.warning(
                    "Jina temporary error HTTP %s. "
                    "Retry %s/%s after %.1fs",
                    response.status_code,
                    attempt + 1,
                    max_retries,
                    sleep_seconds,
                )

                time.sleep(
                    sleep_seconds
                )
                continue

            if not response.ok:
                raise RuntimeError(
                    "Jina API request failed | "
                    f"HTTP {response.status_code} | "
                    f"{response.text[:1000]}"
                )

            result = response.json()

            if "data" not in result:
                raise RuntimeError(
                    "Jina response does not contain 'data'."
                )

            # API results include an index. Sorting makes sure
            # the output order exactly matches the input order.
            items = sorted(
                result["data"],
                key=lambda item: item["index"],
            )

            if len(items) != len(texts):
                raise RuntimeError(
                    "Jina returned a different number of "
                    "embeddings than inputs."
                )

            vectors = np.asarray(
                [
                    item["embedding"]
                    for item in items
                ],
                dtype=np.float32,
            )

            return validate_and_normalize_vectors(
                vectors
            )

        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as error:

            last_error = error

            if attempt >= max_retries:
                break

            sleep_seconds = float(
                2 ** attempt
            )

            logger.warning(
                "Network error calling Jina. "
                "Retry %s/%s after %.1fs | %s",
                attempt + 1,
                max_retries,
                sleep_seconds,
                error,
            )

            time.sleep(
                sleep_seconds
            )

        except requests.RequestException as error:
            last_error = error
            break

    raise RuntimeError(
        "Jina API request failed after retries."
    ) from last_error


# ============================================================
# Embedding pipeline
# ============================================================

def create_embeddings(
    children: List[Dict[str, str]],
    output_dir: Path,
    api_key: str,
    batch_size: int,
    timeout: int,
    max_retries: int,
    force: bool,
) -> Dict[str, Any]:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        output_dir
        / "embedding_cache.sqlite3"
    )

    embedding_config = (
        build_embedding_config()
    )

    embedding_config_hash = (
        config_hash(
            embedding_config
        )
    )

    total = len(children)

    vectors = np.empty(
        (
            total,
            EXPECTED_DIMENSION,
        ),
        dtype=OUTPUT_DTYPE,
    )

    cache = EmbeddingCache(
        cache_path
    )

    cache_hit_rows = 0
    embedded_rows = 0
    unique_new_embeddings = 0
    api_requests = 0
    api_input_rows = 0

    total_batches = (
        total
        + batch_size
        - 1
    ) // batch_size

    logger.info("=" * 64)
    logger.info(
        "Starting Jina API embedding"
    )
    logger.info(
        "Model           : %s",
        MODEL_NAME,
    )
    logger.info(
        "Task            : %s",
        JINA_TASK,
    )
    logger.info(
        "Children        : %s",
        total,
    )
    logger.info(
        "Dimension       : %s",
        EXPECTED_DIMENSION,
    )
    logger.info(
        "Batch size      : %s",
        batch_size,
    )
    logger.info(
        "Total batches   : %s",
        total_batches,
    )
    logger.info(
        "Force re-embed  : %s",
        force,
    )
    logger.info("=" * 64)

    progress = tqdm(
        total=total,
        desc="Jina embedding",
        unit="chunk",
        dynamic_ncols=True,
    )

    try:

        for batch_number in range(
            total_batches
        ):

            start = (
                batch_number
                * batch_size
            )

            end = min(
                start + batch_size,
                total,
            )

            batch = children[
                start:end
            ]

            batch_cache_keys = []

            for item in batch:

                cache_key = build_cache_key(
                    text_hash=item[
                        "text_hash"
                    ],
                    embedding_config_hash=
                    embedding_config_hash,
                )

                item["cache_key"] = (
                    cache_key
                )

                batch_cache_keys.append(
                    cache_key
                )

            if force:
                cached_vectors = {}
            else:
                cached_vectors = (
                    cache.get_many(
                        batch_cache_keys
                    )
                )

            # Group identical texts by cache key so the same text
            # is only sent to Jina once.
            missing_by_key: Dict[
                str,
                Dict[str, Any],
            ] = {}

            batch_cache_hits = 0

            for local_index, item in enumerate(
                batch
            ):

                global_index = (
                    start
                    + local_index
                )

                cache_key = item[
                    "cache_key"
                ]

                cached = (
                    cached_vectors.get(
                        cache_key
                    )
                )

                if cached is not None:

                    vectors[
                        global_index
                    ] = cached

                    batch_cache_hits += 1
                    cache_hit_rows += 1
                    continue

                if cache_key not in missing_by_key:

                    missing_by_key[
                        cache_key
                    ] = {
                        "text":
                            item["text"],

                        "text_hash":
                            item[
                                "text_hash"
                            ],

                        "positions": [],
                    }

                missing_by_key[
                    cache_key
                ][
                    "positions"
                ].append(
                    global_index
                )

            if missing_by_key:

                missing_keys = list(
                    missing_by_key.keys()
                )

                missing_texts = [
                    missing_by_key[
                        key
                    ]["text"]
                    for key
                    in missing_keys
                ]

                new_vectors = (
                    request_jina_embeddings(
                        texts=missing_texts,
                        api_key=api_key,
                        timeout=timeout,
                        max_retries=max_retries,
                    )
                )

                api_requests += 1
                api_input_rows += len(
                    missing_texts
                )

                if (
                    len(new_vectors)
                    != len(missing_keys)
                ):
                    raise RuntimeError(
                        "Generated embedding count mismatch."
                    )

                cache_records = []

                for (
                    cache_key,
                    vector,
                ) in zip(
                    missing_keys,
                    new_vectors,
                ):

                    info = (
                        missing_by_key[
                            cache_key
                        ]
                    )

                    positions = info[
                        "positions"
                    ]

                    for position in positions:

                        vectors[
                            position
                        ] = vector

                        embedded_rows += 1

                    unique_new_embeddings += 1

                    cache_records.append(
                        {
                            "cache_key":
                                cache_key,

                            "config_hash":
                                embedding_config_hash,

                            "text_hash":
                                info[
                                    "text_hash"
                                ],

                            "vector":
                                vector,
                        }
                    )

                cache.put_many(
                    cache_records
                )

            progress.update(
                len(batch)
            )

            progress.set_postfix(
                batch=f"{batch_number + 1}/{total_batches}",
                cache=cache_hit_rows,
                api=api_requests,
                new=unique_new_embeddings,
            )

    finally:
        progress.close()
        cache.close()

    # ========================================================
    # Final validations
    # ========================================================

    if vectors.shape != (
        total,
        EXPECTED_DIMENSION,
    ):
        raise RuntimeError(
            "Final vector matrix has an invalid shape."
        )

    if not np.all(
        np.isfinite(vectors)
    ):
        raise RuntimeError(
            "Final vector matrix contains NaN or Inf."
        )

    final_norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    if not np.allclose(
        final_norms,
        1.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Final embeddings are not properly normalized."
        )

    return {
        "vectors":
            vectors,

        "embedding_config":
            embedding_config,

        "embedding_config_hash":
            embedding_config_hash,

        "cache_hit_rows":
            cache_hit_rows,

        "embedded_rows":
            embedded_rows,

        "unique_new_embeddings":
            unique_new_embeddings,

        "api_requests":
            api_requests,

        "api_input_rows":
            api_input_rows,
    }


# ============================================================
# Output metadata
# ============================================================

def build_embedding_index(
    children: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Maps each embeddings.npy row to its chunk_id.
    """

    items = []

    for row, child in enumerate(
        children
    ):

        items.append(
            {
                "row": row,

                "chunk_id":
                    child[
                        "chunk_id"
                    ],

                "text_hash":
                    child[
                        "text_hash"
                    ],
            }
        )

    return {
        "schema_version":
            SCHEMA_VERSION,

        "total":
            len(items),

        "items":
            items,
    }


def build_manifest(
    input_path: Path,
    vectors: np.ndarray,
    embedding_config: Dict[str, Any],
    embedding_config_hash: str,
    cache_hit_rows: int,
    embedded_rows: int,
    unique_new_embeddings: int,
    api_requests: int,
    api_input_rows: int,
) -> Dict[str, Any]:

    return {
        "schema_version":
            SCHEMA_VERSION,

        "created_at_utc":
            utc_now_iso(),

        "source": {
            "file":
                input_path.name,

            "sha256":
                sha256_file(
                    input_path
                ),
        },

        "embedding": {
            **embedding_config,

            "config_hash":
                embedding_config_hash,
        },

        "dataset": {
            "total_embeddings":
                int(
                    vectors.shape[0]
                ),

            "dimension":
                int(
                    vectors.shape[1]
                ),

            "cache_hit_rows":
                cache_hit_rows,

            "embedded_rows":
                embedded_rows,

            "unique_new_embeddings":
                unique_new_embeddings,

            "api_requests":
                api_requests,

            "api_input_rows":
                api_input_rows,
        },

        "output": {
            "vectors":
                "embeddings.npy",

            "index":
                "embedding_index.json",

            "cache":
                "embedding_cache.sqlite3",
        },
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate dense embeddings for child chunks "
            "using Jina Embeddings API. "
            "No local embedding model and no vector database."
        )
    )

    parser.add_argument(
        "input_json",
        type=Path,
        help=(
            "Path to chunked_data.json"
        ),
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Directory for embedding outputs"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of chunks per Jina API request "
            f"(default: {DEFAULT_BATCH_SIZE})"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=(
            "HTTP timeout in seconds "
            f"(default: {DEFAULT_TIMEOUT})"
        ),
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Retries for rate limits/server/network errors "
            f"(default: {DEFAULT_MAX_RETRIES})"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore embedding cache and call Jina again "
            "for all chunks."
        ),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0"
        )

    if args.timeout <= 0:
        raise ValueError(
            "--timeout must be > 0"
        )

    if args.max_retries < 0:
        raise ValueError(
            "--max-retries must be >= 0"
        )

    # --------------------------------------------------------
    # API key
    # --------------------------------------------------------

    api_key, env_path = (
        load_jina_api_key()
    )

    if env_path is not None:
        logger.info(
            ".env loaded from: %s",
            env_path,
        )
    else:
        logger.info(
            "JINA_API_KEY loaded from OS environment."
        )

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    data = load_dataset(
        args.input_json
    )

    children = prepare_children(
        data["children"]
    )

    # --------------------------------------------------------
    # Embed
    # --------------------------------------------------------

    result = create_embeddings(
        children=children,

        output_dir=args.output_dir,

        api_key=api_key,

        batch_size=args.batch_size,

        timeout=args.timeout,

        max_retries=args.max_retries,

        force=args.force,
    )

    vectors = result[
        "vectors"
    ]

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    vectors_path = (
        args.output_dir
        / "embeddings.npy"
    )

    index_path = (
        args.output_dir
        / "embedding_index.json"
    )

    manifest_path = (
        args.output_dir
        / "embedding_manifest.json"
    )

    embedding_index = (
        build_embedding_index(
            children
        )
    )

    manifest = build_manifest(
        input_path=args.input_json,

        vectors=vectors,

        embedding_config=result[
            "embedding_config"
        ],

        embedding_config_hash=result[
            "embedding_config_hash"
        ],

        cache_hit_rows=result[
            "cache_hit_rows"
        ],

        embedded_rows=result[
            "embedded_rows"
        ],

        unique_new_embeddings=result[
            "unique_new_embeddings"
        ],

        api_requests=result[
            "api_requests"
        ],

        api_input_rows=result[
            "api_input_rows"
        ],
    )

    logger.info(
        "Writing final embedding matrix..."
    )

    atomic_write_npy(
        vectors_path,
        vectors,
    )

    atomic_write_json(
        index_path,
        embedding_index,
    )

    atomic_write_json(
        manifest_path,
        manifest,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    logger.info("=" * 64)
    logger.info(
        "Embedding completed successfully"
    )
    logger.info(
        "Vectors shape         : %s",
        vectors.shape,
    )
    logger.info(
        "Vector dtype          : %s",
        vectors.dtype,
    )
    logger.info(
        "Cache-hit rows        : %s",
        result[
            "cache_hit_rows"
        ],
    )
    logger.info(
        "New embedded rows     : %s",
        result[
            "embedded_rows"
        ],
    )
    logger.info(
        "Unique new embeddings : %s",
        result[
            "unique_new_embeddings"
        ],
    )
    logger.info(
        "Jina API requests     : %s",
        result[
            "api_requests"
        ],
    )
    logger.info(
        "Jina API input rows   : %s",
        result[
            "api_input_rows"
        ],
    )
    logger.info(
        "Vectors               : %s",
        vectors_path,
    )
    logger.info(
        "Index                 : %s",
        index_path,
    )
    logger.info(
        "Manifest              : %s",
        manifest_path,
    )
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
