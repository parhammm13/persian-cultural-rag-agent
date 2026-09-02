"""Two-stage evaluation dataset builder without question categories.

Stage 1: sample pages and find question-worthy evidence.
Stage 2: generate one natural question per supported page.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .generator import EvaluationGenerator
from .sampler import join_and_sample

DEFAULT_CHUNKED = Path("Data/processed/chunked_data.json")
DEFAULT_CATEGORY = Path("Data/iran_heritage_classification_final.csv")
DEFAULT_OUTPUT_DIR = Path("Data/evaluation/v1")


def _resolve_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    if not (root / "Data").exists():
        root = Path.cwd()
    return root / DEFAULT_CHUNKED, root / DEFAULT_CATEGORY, root / DEFAULT_OUTPUT_DIR


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_sampled_pages(sampled: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for page in sampled:
        rows.append({
            "page_id": page["page_id"],
            "page_title": page.get("page_title", ""),
            "page_url": page.get("page_url", ""),
            "final_class_code": page.get("final_class_code", ""),
            "final_class_fa": page.get("final_class_fa", ""),
            "parent_count": len(page.get("parents", [])),
            "child_count": len(page.get("children", [])),
        })
    write_jsonl(rows, path)


def plan_pages(sampled: list[dict[str, Any]], analyses: dict[str, dict[str, Any]], target_candidates: int) -> list[str]:
    """Choose up to ``target_candidates`` pages with valid evidence."""
    eligible = [
        str(page["page_id"])
        for page in sampled
        if analyses.get(str(page["page_id"]), {}).get("supported")
    ]
    return eligible[:target_candidates]


def write_review_csv(candidates: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "id", "question", "reference_answer", "difficulty",
        "page_title", "page_id", "parent_ids", "child_ids", "supporting_text",
        "review_status", "human_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            writer.writerow({
                "id": c["id"],
                "question": c["question"],
                "reference_answer": c["reference_answer"],
                "difficulty": c["difficulty"],
                "page_title": c["page_title"],
                "page_id": c["page_id"],
                "parent_ids": ";".join(c["parent_ids"]),
                "child_ids": ";".join(c["child_ids"]),
                "supporting_text": c["supporting_text"],
                "review_status": "pending",
                "human_notes": "",
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--target-candidates", type=int, default=125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunked", type=str, default=None)
    parser.add_argument("--category-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--max-pages-test", type=int, default=None)
    parser.add_argument("--cap-per-category", type=int, default=15)
    args = parser.parse_args()

    chunked_default, category_default, out_default = _resolve_paths()
    chunked_path = Path(args.chunked) if args.chunked else chunked_default
    category_path = Path(args.category_file) if args.category_file else category_default
    out_dir = Path(args.output_dir) if args.output_dir else out_default

    sampled, allocation, original_dist, total_joined = join_and_sample(
        chunked_path=chunked_path,
        category_path=category_path,
        sample_size=args.sample_size,
        seed=args.seed,
        cap_per_category=args.cap_per_category,
        exclude_excluded=True,
    )

    sampled_path = out_dir / "sampled_pages.jsonl"
    save_sampled_pages(sampled, sampled_path)
    print(f"[sample] joined={total_joined} sampled={len(sampled)}")
    print(f"[sample] allocation={allocation}")

    if args.dry_run:
        print("[dry-run] no LLM calls")
        return

    if args.max_pages_test is not None:
        sampled = sampled[:args.max_pages_test]
        print(f"[test] limited to {len(sampled)} pages")

    generator = EvaluationGenerator(cache_dir=out_dir / ".cache")
    print(f"[llm] model={generator.model_name}")

    # Stage 1: find question-worthy evidence on every selected page.
    analyses_list: list[dict[str, Any]] = []
    analyses_map: dict[str, dict[str, Any]] = {}

    for index, page in enumerate(sampled, start=1):
        analysis = generator.analyze_page(page)
        analyses_list.append(analysis)
        analyses_map[str(page["page_id"])] = analysis
        status = "supported" if analysis.get("supported") else "unsupported"
        print(f"[analyze {index}/{len(sampled)}] {page.get('page_title','')} => {status}")

    analysis_path = out_dir / "page_analysis.jsonl"
    write_jsonl(analyses_list, analysis_path)

    supported_count = sum(1 for analysis in analyses_list if analysis.get("supported"))
    print(f"[analysis] supported pages={supported_count}")

    if args.analysis_only:
        print(f"[analysis-only] saved: {analysis_path}")
        return

    # Stage 2: select supported pages without assigning question categories.
    planned_pages = plan_pages(
        sampled=sampled,
        analyses=analyses_map,
        target_candidates=args.target_candidates,
    )
    print(f"[plan] pages={len(planned_pages)}")

    page_map = {str(page["page_id"]): page for page in sampled}
    candidates: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    rejected = 0

    for index, page_id in enumerate(planned_pages, start=1):
        page = page_map[page_id]
        result = generator.generate_question(
            page=page,
            analysis=analyses_map[page_id],
        )
        if result is None:
            rejected += 1
            print(f"[generate {index}/{len(planned_pages)}] rejected")
            continue

        normalized_question = " ".join(result["question"].split())
        if normalized_question in seen_questions:
            rejected += 1
            continue
        seen_questions.add(normalized_question)

        candidate = {
            "id": f"eval_candidate_{len(candidates)+1:03d}",
            **result,
            "review_status": "pending",
        }
        candidates.append(candidate)
        print(f"[generate {index}/{len(planned_pages)}] OK")

    candidate_path = out_dir / "candidate_questions.jsonl"
    review_path = out_dir / "human_review.csv"
    write_jsonl(candidates, candidate_path)
    write_review_csv(candidates, review_path)

    print("\nFINAL")
    print(f"candidates={len(candidates)}")
    print(f"rejected={rejected}")
    print(f"page_analysis={analysis_path}")
    print(f"candidate_questions={candidate_path}")
    print(f"human_review={review_path}")


if __name__ == "__main__":
    main()
