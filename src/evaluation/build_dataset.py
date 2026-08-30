"""Build evaluation dataset: sampling + LLM candidate generation.

CLI:
  python -m src.evaluation.build_dataset --sample-size 150 --target-candidates 125 --seed 42
  python -m src.evaluation.build_dataset --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .generator import (
    EvaluationGenerator,
    VALID_DIFFICULTIES,
    VALID_QUERY_CATEGORIES,
    verify_supporting_text,
)
from .sampler import join_and_sample, DEFAULT_CATEGORY_FIELD


DEFAULT_CHUNKED = Path("Data/processed/chunked_data.json")
DEFAULT_CATEGORY = Path("Data/iran_heritage_classification_final.csv")
DEFAULT_OUTPUT_DIR = Path("Data/evaluation/v1")


def _resolve_paths(project_root: Path | None = None) -> tuple[Path, Path, Path]:
    if project_root is None:
        # build_dataset.py is at src/evaluation/build_dataset.py -> parents[2] is repo root
        project_root = Path(__file__).resolve().parents[2]
        if not (project_root / "Data").exists():
            # fallback to cwd
            if (Path.cwd() / "Data").exists():
                project_root = Path.cwd()
    chunked = project_root / DEFAULT_CHUNKED
    category = project_root / DEFAULT_CATEGORY
    out = project_root / DEFAULT_OUTPUT_DIR
    return chunked, category, out


def save_sampled_pages(sampled: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in sampled:
            # minimal + category info, keep parents/children ids but not full text for brevity?
            # spec says save selected pages – keep useful fields
            record = {
                "page_id": entry["page_id"],
                "page_title": entry["page_title"],
                "page_url": entry["page_url"],
                "final_class_code": entry.get("final_class_code", ""),
                "final_class_fa": entry.get("final_class_fa", ""),
                "final_class_en": entry.get("final_class_en", ""),
                "final_superclass_fa": entry.get("final_superclass_fa", ""),
                "final_superclass_en": entry.get("final_superclass_en", ""),
                "final_status": entry.get("final_status", ""),
                "parent_count": len(entry.get("parents", [])),
                "child_count": len(entry.get("children", [])),
                "parent_ids": [p["parent_id"] for p in entry.get("parents", [])],
                "child_ids": [c["chunk_id"] for c in entry.get("children", [])],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _current_deficits(by_cat: Counter, target_candidates: int) -> list[str]:
    """Categories currently below the uniform share of target_candidates, most deficient first."""
    avg = target_candidates / len(VALID_QUERY_CATEGORIES) if VALID_QUERY_CATEGORIES else 0
    needed = [c for c in VALID_QUERY_CATEGORIES if by_cat[c] < avg]
    needed.sort(key=lambda c: by_cat[c])  # ascending count = largest deficit first
    return needed


def build_candidates(
    sampled: list[dict[str, Any]],
    generator: EvaluationGenerator | None,
    target_candidates: int,
    seed: int,
    chunked_map: dict[str, dict[str, Any]] | None = None,
    allow_two_per_page: bool = True,
) -> tuple[list[dict[str, Any]], int, Counter, Counter]:
    """Generate candidates for sampled pages.

    Returns (candidates, rejected_count, by_category, by_difficulty)
    """
    if generator is None:
        return [], 0, Counter(), Counter()

    candidates: list[dict[str, Any]] = []
    rejected = 0
    seen_questions: set[str] = set()

    by_cat = Counter()
    by_diff = Counter()

    # Preserve the deterministic shuffled order produced by the sampler.
    # Sorting by page_id here would bias early stopping toward lower IDs.
    pages_to_process = list(sampled)

    for page in pages_to_process:
        if len(candidates) >= target_candidates:
            break
        # advisory: prefer underrepresented categories (current global deficits),
        # never force unsupported ones — the generator still validates support
        preferred = _current_deficits(by_cat, target_candidates)

        res = generator.generate_for_page(page, preferred_categories=preferred)
        if res is None:
            rejected += 1
            continue

        q = res["question"].strip()
        norm_q = " ".join(q.split())
        cat = res["selected_category"]
        diff = res["difficulty"]
        # validation already done in generator, but double-check
        if (
            norm_q in seen_questions
            or cat not in VALID_QUERY_CATEGORIES
            or diff not in VALID_DIFFICULTIES
            or not res.get("parent_ids")
            or not res.get("child_ids")
            or not q
            or not res.get("reference_answer", "").strip()
            or not res.get("supporting_text", "").strip()
        ):
            rejected += 1
            continue

        pid = page["page_id"]
        idx = len(candidates) + 1
        cand = {
            "id": f"eval_candidate_{idx:03d}",
            "question": q,
            "reference_answer": res["reference_answer"].strip(),
            "query_category": cat,
            "difficulty": diff,
            "page_id": pid,
            "page_title": page.get("page_title", ""),
            "page_categories": [
                page.get("final_class_code", ""),
                page.get("final_superclass_en", ""),
            ],
            "parent_ids": res["parent_ids"],
            "child_ids": res["child_ids"],
            "supporting_text": res["supporting_text"],
            "review_status": "pending",
        }
        candidates.append(cand)
        seen_questions.add(norm_q)
        by_cat[cat] += 1
        by_diff[diff] += 1

    # Reassign stable IDs after deduplication
    for i, c in enumerate(candidates, start=1):
        c["id"] = f"eval_candidate_{i:03d}"

    return candidates, rejected, by_cat, by_diff


def write_candidates_jsonl(candidates: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def write_candidates_csv(candidates: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "question",
        "reference_answer",
        "query_category",
        "difficulty",
        "page_title",
        "page_id",
        "parent_ids",
        "child_ids",
        "supporting_text",
        "review_status",
        "human_notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            writer.writerow(
                {
                    "id": c.get("id", ""),
                    "question": c.get("question", ""),
                    "reference_answer": c.get("reference_answer", ""),
                    "query_category": c.get("query_category", ""),
                    "difficulty": c.get("difficulty", ""),
                    "page_title": c.get("page_title", ""),
                    "page_id": c.get("page_id", ""),
                    "parent_ids": ";".join(c.get("parent_ids", [])),
                    "child_ids": ";".join(c.get("child_ids", [])),
                    "supporting_text": c.get("supporting_text", "")[:2000],
                    "review_status": c.get("review_status", ""),
                    "human_notes": "",
                }
            )


def write_insufficient(candidates: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def validate_candidate(candidate: dict[str, Any], page_map: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    """Simple deterministic validation per spec."""
    pid = candidate.get("page_id")
    if pid not in page_map:
        return False, "page not exists"
    page = page_map[pid]
    valid_parents = {p["parent_id"] for p in page.get("parents", [])}
    valid_children = {c["chunk_id"] for c in page.get("children", [])}
    child_to_parent = {c["chunk_id"]: c["parent_id"] for c in page.get("children", [])}

    for pp in candidate.get("parent_ids", []):
        if pp not in valid_parents:
            return False, f"invalid parent {pp}"
    for cc in candidate.get("child_ids", []):
        if cc not in valid_children:
            return False, f"invalid child {cc}"
        if child_to_parent.get(cc) not in candidate.get("parent_ids", []):
            return False, f"child {cc} not belong to selected parent"

    # supporting_text must actually exist inside the selected child texts
    children_by_id = {c["chunk_id"]: c for c in page.get("children", [])}
    if not verify_supporting_text(
        candidate.get("supporting_text", ""),
        [str(x) for x in candidate.get("child_ids", [])],
        children_by_id,
    ):
        return False, "supporting_text not found in selected child texts"
    if not candidate.get("question", "").strip():
        return False, "empty question"
    if not candidate.get("reference_answer", "").strip():
        return False, "empty answer"
    if not candidate.get("supporting_text", "").strip():
        return False, "empty supporting_text"
    if candidate.get("query_category") not in VALID_QUERY_CATEGORIES:
        return False, "invalid category"
    if candidate.get("difficulty") not in VALID_DIFFICULTIES:
        return False, "invalid difficulty"
    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation dataset candidates")
    parser.add_argument("--sample-size", type=int, default=150, help="Number of pages to sample")
    parser.add_argument("--target-candidates", type=int, default=125, help="Rough target candidates 120-130")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunked", type=str, default=None, help="Path to chunked_data.json")
    parser.add_argument("--category-file", type=str, default=None, help="Path to category CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir Data/evaluation/v1")
    parser.add_argument("--dry-run", action="store_true", help="Only sampling, no LLM calls")
    parser.add_argument("--max-pages-test", type=int, default=None, help="Limit LLM to N pages for testing (e.g., 5)")
    parser.add_argument("--cap-per-category", type=int, default=15, help="Max pages per category in sampling")
    parser.add_argument("--no-cache", action="store_true", help="Ignore LLM cache")
    parser.add_argument("--insufficient-count", type=int, default=18, help="Count of insufficient_information candidates")
    args = parser.parse_args()

    chunked_default, category_default, out_default = _resolve_paths()
    chunked_path = Path(args.chunked) if args.chunked else chunked_default
    category_path = Path(args.category_file) if args.category_file else category_default
    out_dir = Path(args.output_dir) if args.output_dir else out_default

    print(f"[sampler] chunked: {chunked_path}")
    print(f"[sampler] category: {category_path}")
    print(f"[sampler] output: {out_dir}")

    if not chunked_path.exists():
        raise FileNotFoundError(f"chunked_data.json not found: {chunked_path}")
    if not category_path.exists():
        raise FileNotFoundError(f"category file not found: {category_path}")

    # join & sample
    sampled, allocation, original_dist, total_joined = join_and_sample(
        chunked_path=chunked_path,
        category_path=category_path,
        sample_size=args.sample_size,
        seed=args.seed,
        cap_per_category=args.cap_per_category,
        exclude_excluded=True,
    )

    print(f"[sample] joined pages: {total_joined}")
    print(f"[sample] sampled: {len(sampled)} (target {args.sample_size}) seed={args.seed}")
    print("[sample] allocation per category:")
    for cat, n in sorted(allocation.items(), key=lambda x: -x[1]):
        orig = original_dist.get(cat, 0)
        print(f"  {cat:30s} {n:3d} / {orig:4d}")
    print(f"[sample] original categories: {len(original_dist)} total filtered pages {sum(original_dist.values())}")

    # save sampled
    sampled_path = out_dir / "sampled_pages.jsonl"
    save_sampled_pages(sampled, sampled_path)
    print(f"[output] sampled_pages: {sampled_path}")

    if args.dry_run:
        print("[dry-run] skipping LLM generation")
        print(f"  category file discovered: {category_path}")
        print(f"  sampled distribution: {dict(Counter(allocation))}")
        print("  exact command to reproduce:")
        print(f"    python -m src.evaluation.build_dataset --sample-size {args.sample_size} --target-candidates {args.target_candidates} --seed {args.seed}")
        return

    # limit for test
    if args.max_pages_test is not None:
        sampled = sampled[: args.max_pages_test]
        print(f"[test] limited to {len(sampled)} pages for LLM test")

    # LLM generation
    # cache path
    cache_path = out_dir / ".eval_llm_cache.json"
    if args.no_cache and cache_path.exists():
        cache_path.unlink()

    try:
        generator = EvaluationGenerator(cache_path=cache_path)
        print(f"[llm] model: {generator.model_name}")
    except Exception as e:
        print(f"[llm] cannot init generator (missing API key?): {e}")
        print("[llm] falling back to no candidate generation; still writing empty files for dry validation")
        generator = None  # type: ignore

    candidates, rejected, by_cat, by_diff = build_candidates(
        sampled=sampled,
        generator=generator,
        target_candidates=args.target_candidates,
        seed=args.seed,
    )

    # validate
    page_map = {p["page_id"]: p for p in sampled}
    valid_cnt = 0
    invalid = []
    for c in candidates:
        ok, reason = validate_candidate(c, page_map)
        if ok:
            valid_cnt += 1
        else:
            invalid.append((c["id"], reason))
    if invalid:
        print(f"[validate] invalid candidates: {invalid}")
    else:
        print(f"[validate] all {valid_cnt} candidates passed basic validation")

    # duplicate check already done
    print(f"[generate] candidates: {len(candidates)} rejected/skipped: {rejected}")
    print(f"[generate] by category: {dict(by_cat)}")
    print(f"[generate] by difficulty: {dict(by_diff)}")

    # write outputs
    cand_path = out_dir / "candidate_questions.jsonl"
    csv_path = out_dir / "human_review.csv"
    write_candidates_jsonl(candidates, cand_path)
    write_candidates_csv(candidates, csv_path)
    print(f"[output] candidates JSONL: {cand_path} ({len(candidates)})")
    print(f"[output] human_review CSV: {csv_path}")

    # insufficient
    insuff_path = out_dir / "insufficient_candidates.jsonl"
    if generator is not None:
        try:
            insuff = generator.generate_insufficient_candidates(count=args.insufficient_count)
            write_insufficient(insuff, insuff_path)
            print(f"[output] insufficient: {insuff_path} ({len(insuff)})")
        except Exception as e:
            print(f"[insufficient] failed: {e}")
            # fallback write deterministic
            fallback = [
                {
                    "id": f"eval_insufficient_{i+1:03d}",
                    "question": f"هزینه دقیق بازسازی این بنا در سال ۱۴۰۳ چقدر بوده است؟ {i+1}؟",
                    "reference_answer": "اطلاعات کافی در منابع یافت نشد.",
                    "query_category": "insufficient_information",
                    "difficulty": "medium",
                    "answerable": False,
                    "review_status": "pending",
                }
                for i in range(args.insufficient_count)
            ]
            write_insufficient(fallback, insuff_path)
            print(f"[output] insufficient fallback: {insuff_path}")
    else:
        fallback = [
            {
                "id": f"eval_insufficient_{i+1:03d}",
                "question": f"جزئیات محرمانه پرونده این بنا چیست؟ {i+1}؟",
                "reference_answer": "اطلاعات کافی در منابع یافت نشد.",
                "query_category": "insufficient_information",
                "difficulty": "medium",
                "answerable": False,
                "review_status": "pending",
            }
            for i in range(args.insufficient_count)
        ]
        write_insufficient(fallback, insuff_path)
        print(f"[output] insufficient (no LLM): {insuff_path}")

    # final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"files created:")
    print(f"  {sampled_path}")
    print(f"  {cand_path}")
    print(f"  {csv_path}")
    print(f"  {insuff_path}")
    if cache_path.exists():
        print(f"  {cache_path} (cache)")
    print(f"category file discovered: {category_path}")
    print(f"number of pages successfully joined: {total_joined}")
    print(f"sampled-page distribution: {dict(sorted(allocation.items(), key=lambda x: -x[1]))}")
    print(f"number of generated candidates: {len(candidates)}")
    print(f"candidate counts by query_category: {dict(by_cat)}")
    print(f"candidate counts by difficulty: {dict(by_diff)}")
    print(f"rejected candidate count: {rejected}")
    print(f"paths to JSONL and CSV outputs: {cand_path} | {csv_path}")
    print("exact command to reproduce:")
    print(f"  python -m src.evaluation.build_dataset --sample-size {args.sample_size} --target-candidates {args.target_candidates} --seed {args.seed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
