"""Retry only JSONDecodeError rows from generation_evaluation.jsonl."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# This file is intended to live at: src/evaluation/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.rag.factory import build_rag_pipeline


RESULT_FILE = (
    PROJECT_ROOT
    / "Data"
    / "evaluation"
    / "generation"
    / "result"
    / "generation_evaluation.jsonl"
)

JUDGE_MODEL = "deepseek/deepseek-v4-flash-0731"

JUDGE_PROMPT = """You are a strict RAG evaluator.

Use ONLY the supplied question, context, and generated answer.

Do not use outside knowledge.

Score each metric from 0 to 4.

Groundedness:
0 = unsupported or contradicted by context
1 = mostly unsupported
2 = partially supported
3 = almost fully supported
4 = every factual claim is supported by context

Relevance:
0 = unrelated
1 = mostly irrelevant
2 = partially relevant
3 = mostly direct and relevant
4 = directly answers the question without meaningful irrelevant content

Citation correctness:
0 = citations are missing or unsupported
1 = most citations are wrong
2 = mixed correct/incorrect citations
3 = mostly correct citations
4 = every citation correctly points to a context block supporting its claim

Completeness:
0 = does not answer the task
1 = misses most important parts
2 = answers some important parts
3 = answers almost all important parts
4 = fully answers all parts that can be answered from the context

Return ONLY valid JSON:

{{
  "groundedness": {{"score": 0, "reason": "short reason"}},
  "relevance": {{"score": 0, "reason": "short reason"}},
  "citation_correctness": {{"score": 0, "reason": "short reason"}},
  "completeness": {{"score": 0, "reason": "short reason"}}
}}
"""


def build_judge():
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing.")

    model_name = os.getenv(
        "OPENROUTER_JUDGE_MODEL",
        JUDGE_MODEL,
    ).strip()

    base_url = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    ).strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_PROMPT),
            (
                "human",
                "Question:\n{question}\n\n"
                "Context:\n{context}\n\n"
                "Generated answer:\n{answer}",
            ),
        ]
    )

    model = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=0,
        max_completion_tokens=800,
        use_responses_api=False,
    )

    return prompt | model, model_name


def parse_judge_output(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise RuntimeError("Judge returned non-text output.")

    raw = raw.strip()

    if raw.startswith("```json") and raw.endswith("```"):
        raw = raw[7:-3].strip()
    elif raw.startswith("```") and raw.endswith("```"):
        raw = raw[3:-3].strip()

    result = json.loads(raw)

    metrics = (
        "groundedness",
        "relevance",
        "citation_correctness",
        "completeness",
    )

    for metric in metrics:
        score = result[metric]["score"]
        if not isinstance(score, int) or not 0 <= score <= 4:
            raise RuntimeError(f"Invalid {metric} score: {score}")

    result["overall"] = round(
        sum(result[metric]["score"] for metric in metrics) / len(metrics),
        2,
    )

    return result


def judge_with_retry(
    judge_chain,
    question: str,
    context: str,
    answer: str,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = judge_chain.invoke(
                {
                    "question": question,
                    "context": context or "[NO CONTEXT]",
                    "answer": answer,
                }
            )
            return parse_judge_output(response.content)

        except (json.JSONDecodeError, KeyError, TypeError, RuntimeError) as exc:
            last_error = exc
            print(
                f"  bad judge output | attempt {attempt}/{retries} | "
                f"{type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        f"Judge failed to return valid JSON after {retries} attempts: "
        f"{last_error}"
    )


def retry_one(
    old_row: dict[str, Any],
    pipeline,
    judge_chain,
    judge_model: str,
    judge_retries: int,
) -> dict[str, Any]:
    question = str(old_row["question"]).strip()

    try:
        candidates = pipeline.retriever.retrieve(
            query=question,
            top_k=pipeline.candidate_k,
        )

        reranked = pipeline.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=pipeline.final_k,
        )

        context, used_results = pipeline._build_context(reranked)

        if context:
            answer = pipeline.generator.generate(
                query=question,
                context=context,
            ).answer
        else:
            answer = "اطلاعات کافی در منابع پیدا نشد."

        evaluation = judge_with_retry(
            judge_chain=judge_chain,
            question=question,
            context=context,
            answer=answer,
            retries=judge_retries,
        )
        evaluation["judge_model"] = judge_model

        print(f"OK: {old_row['id']}")

        return {
            **old_row,
            "answer": answer,
            "retrieved_count": len(candidates),
            "reranked_count": len(used_results),
            "evaluation": evaluation,
            "error": None,
        }

    except Exception as exc:
        print(f"FAILED: {old_row['id']} | {type(exc).__name__}: {exc}")

        return {
            **old_row,
            "evaluation": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def save_results(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Retry only JSONDecodeError rows in generation evaluation."
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--judge-retries", type=int, default=3)
    args = parser.parse_args()

    old_results = load_results(RESULT_FILE)

    failed_rows = [
        row
        for row in old_results
        if str(row.get("error") or "").startswith("JSONDecodeError")
    ]

    print(f"JSON errors found: {len(failed_rows)}")

    if not failed_rows:
        print("Nothing to retry.")
        return

    pipeline = build_rag_pipeline()
    judge_chain, judge_model = build_judge()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        retried_rows = list(
            executor.map(
                lambda row: retry_one(
                    old_row=row,
                    pipeline=pipeline,
                    judge_chain=judge_chain,
                    judge_model=judge_model,
                    judge_retries=args.judge_retries,
                ),
                failed_rows,
            )
        )

    retried_by_id = {
        row["id"]: row
        for row in retried_rows
    }

    final_results = [
        retried_by_id.get(row["id"], row)
        for row in old_results
    ]

    save_results(RESULT_FILE, final_results)

    remaining_json_errors = sum(
        1
        for row in final_results
        if str(row.get("error") or "").startswith("JSONDecodeError")
    )

    print(f"Saved: {RESULT_FILE}")
    print(f"Remaining JSON errors: {remaining_json_errors}")


if __name__ == "__main__":
    main()
