from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hazm import (
    Normalizer,
    word_tokenize,
    stopwords_list,
)
from rank_bm25 import BM25Okapi

    

normalizer = Normalizer()

STOPWORDS = set(
    stopwords_list()
)

normalizer = Normalizer()

STOPWORDS = set(
    stopwords_list()
)



def preprocess_text(text: str) -> list[str]:

    # 1. Normalize Persian text
    text = normalizer.normalize(text)


    # 2. Tokenize
    tokens = word_tokenize(text)


    # 3. Remove stopwords
    tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]


    # 4. Remove very small tokens
    tokens = [
        token
        for token in tokens
        if len(token) > 1
    ]


    return tokens


@dataclass(frozen=True)
class BM25Result:
    score: float
    chunk_id: str
    text: str
    metadata: dict[str, Any]    

class BM25Retriever:

    def __init__(
        self,
        bm25: BM25Okapi,
        documents: list[dict[str, Any]],
    ):
        self.bm25 = bm25
        self.documents = documents


    @classmethod
    def build(
        cls,
        chunked_data_path: Path,
        output_dir: Path,
    ):

        with chunked_data_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)


        children = data["children"]
        corpus = []
        documents = []
        for child in children:

            tokens = preprocess_text(
                child["text"]
            )

            corpus.append(tokens)

            documents.append(
                {
                    "chunk_id": child["chunk_id"],
                    "text": child["text"],
                    "metadata": {
                        "parent_id": child["parent_id"],
                        "page_title": child["page_title"],
                        "page_url": child["page_url"],
                        "section_heading": child["section_heading"],
                    },
                }
            )
        bm25 = BM25Okapi(corpus)
        
        output_dir.mkdir(
        parents=True,
        exist_ok=True,)
        index_path = output_dir / "index.pkl"


        with index_path.open(
            "wb"
        ) as file:

            pickle.dump(
                bm25,
                file,
            )

            metadata_path = (
        output_dir / "metadata.json"
        )


        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                documents,
                file,
                ensure_ascii=False,
                indent=2,
            )

        return cls(
        bm25=bm25,
        documents=documents,
        )


    @classmethod
    def load(cls,index_dir: Path,):

        index_path = (
            index_dir / "index.pkl"
        )

        metadata_path = (
            index_dir / "metadata.json"
        )

        with index_path.open(
        "rb"
        ) as file:

            bm25 = pickle.load(file)

        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            documents = json.load(file)

        return cls(
            bm25=bm25,
            documents=documents,
        )


    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[BM25Result]:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )


        query = query.strip()


        if not query:
            raise ValueError(
                "query must not be empty"
            )


        query_tokens = preprocess_text(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True, )[:top_k]

        results = []
        for index in ranked_indexes:

            document = self.documents[index]

            results.append(
                BM25Result(
                    score=float(
                        scores[index]
                    ),
                    chunk_id=document["chunk_id"],
                    text=document["text"],
                    metadata=document["metadata"],
                )
            )
        return results


def main():

    parser = argparse.ArgumentParser(
        description="BM25 Retriever"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # =========================
    # BUILD COMMAND
    # =========================

    build_parser = subparsers.add_parser(
        "build"
    )

    build_parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    build_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )


    # =========================
    # SEARCH COMMAND
    # =========================

    search_parser = subparsers.add_parser(
        "search"
    )

    search_parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
    )

    search_parser.add_argument(
        "query",
        type=str,
    )

    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )


    args = parser.parse_args()


    # =========================
    # BUILD
    # =========================

    if args.command == "build":

        BM25Retriever.build(
            chunked_data_path=args.input,
            output_dir=args.output,
        )

        print("BM25 index created.")


    # =========================
    # SEARCH
    # =========================

    elif args.command == "search":

        retriever = BM25Retriever.load(
            args.index_dir
        )

        results = retriever.retrieve(
            query=args.query,
            top_k=args.top_k,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print("=" * 60)
            print(f"Rank: {rank}")
            print(f"Score: {result.score:.4f}")
            print(f"Chunk ID: {result.chunk_id}")
            print(
                f"Title: "
                f"{result.metadata.get('page_title')}"
            )
            print("-" * 60)
            print(result.text)


if __name__ == "__main__":
    main()