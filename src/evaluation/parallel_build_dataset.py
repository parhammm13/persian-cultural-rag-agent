"""Parallel evaluation dataset builder."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from pathlib import Path
from threading import Lock, local
from typing import Any

from .build_dataset import (
    _resolve_paths,
    save_sampled_pages,
    write_jsonl,
    write_review_csv,
)
from .generator import EvaluationGenerator
from .sampler import join_and_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=220)
    parser.add_argument("--target-candidates", type=int, default=150)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunked", type=str, default=None)
    parser.add_argument("--category-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--max-pages-test", type=int, default=None)
    parser.add_argument("--cap-per-category", type=int, default=15)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    chunked_default, category_default, out_default = _resolve_paths()
    chunked_path = Path(args.chunked) if args.chunked else chunked_default
    category_path = (
        Path(args.category_file) if args.category_file else category_default
    )
    out_dir = Path(args.output_dir) if args.output_dir else out_default
    out_dir.mkdir(parents=True, exist_ok=True)

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
        sampled = sampled[: args.max_pages_test]
        print(f"[test] limited to {len(sampled)} pages")

    # Each worker receives its own generator and cache directory.
    # This prevents concurrent cache-file writes from corrupting each other.
    worker_state = local()
    worker_numbers = count()
    worker_number_lock = Lock()

    def get_generator() -> EvaluationGenerator:
        generator = getattr(worker_state, "generator", None)

        if generator is None:
            with worker_number_lock:
                worker_number = next(worker_numbers)

            generator = EvaluationGenerator(
                cache_dir=out_dir / ".cache" / f"worker_{worker_number:02d}"
            )
            worker_state.generator = generator

        return generator

    def analyze_page(
        page: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str | None]:
        page_id = str(page["page_id"])

        try:
            analysis = get_generator().analyze_page(page)
            return page_id, analysis, None
        except Exception as exc:
            analysis = {
                "page_id": page_id,
                "page_title": page.get("page_title", ""),
                "supported": False,
                "parent_ids": [],
                "child_ids": [],
                "evidence_text": "",
                "reason": "",
            }
            return page_id, analysis, str(exc)

    def generate_question(
        page: dict[str, Any],
        analysis: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            return get_generator().generate_question(
                page=page,
                analysis=analysis,
            )
        except Exception as exc:
            print(f"[generation error] page={page['page_id']}: {exc}")
            return None

    analyses_map: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Stage 1: analyze pages concurrently.
        analysis_futures = {
            executor.submit(analyze_page, page): page
            for page in sampled
        }

        completed = 0

        for future in as_completed(analysis_futures):
            page = analysis_futures[future]
            page_id, analysis, error = future.result()
            analyses_map[page_id] = analysis
            completed += 1

            status = "supported" if analysis.get("supported") else "unsupported"
            if error:
                status = f"error: {error}"

            print(
                f"[analyze {completed}/{len(sampled)}] "
                f"{page.get('page_title', '')} => {status}"
            )

        # Preserve sampled-page order in the analysis output.
        analyses_list = [
            analyses_map[str(page["page_id"])]
            for page in sampled
        ]

        analysis_path = out_dir / "page_analysis.jsonl"
        write_jsonl(analyses_list, analysis_path)

        supported_count = sum(
            1 for analysis in analyses_list if analysis.get("supported")
        )
        print(f"[analysis] supported pages={supported_count}")

        if args.analysis_only:
            print(f"[analysis-only] saved: {analysis_path}")
            return

        page_map = {
            str(page["page_id"]): page
            for page in sampled
        }

        # Keep all supported pages available so failed generations can be
        # replaced until the requested number of questions is reached.
        eligible_page_ids = [
            str(page["page_id"])
            for page in sampled
            if analyses_map[str(page["page_id"])].get("supported")
        ]

        candidates: list[dict[str, Any]] = []
        seen_questions: set[str] = set()
        rejected = 0
        attempted = 0

        # Bounded batches avoid generating many unnecessary questions after
        # the target has already been reached.
        for batch_start in range(0, len(eligible_page_ids), args.workers):
            if len(candidates) >= args.target_candidates:
                break

            batch_ids = eligible_page_ids[
                batch_start : batch_start + args.workers
            ]

            futures = [
                executor.submit(
                    generate_question,
                    page_map[page_id],
                    analyses_map[page_id],
                )
                for page_id in batch_ids
            ]

            # Read results in sampled-page order for deterministic output.
            for page_id, future in zip(batch_ids, futures):
                if len(candidates) >= args.target_candidates:
                    break

                attempted += 1
                result = future.result()

                if result is None:
                    rejected += 1
                    print(
                        f"[generate {attempted}] page={page_id} -> rejected"
                    )
                    continue

                normalized_question = " ".join(result["question"].split())

                if normalized_question in seen_questions:
                    rejected += 1
                    print(
                        f"[generate {attempted}] page={page_id} -> duplicate"
                    )
                    continue

                seen_questions.add(normalized_question)

                candidates.append(
                    {
                        "id": f"eval_candidate_{len(candidates):03d}",
                        **result,
                        "review_status": "pending",
                    }
                )

                print(
                    f"[generate {attempted}] page={page_id} -> "
                    f"OK ({len(candidates)}/{args.target_candidates})"
                )

    candidate_path = out_dir / "candidate_questions.jsonl"
    review_path = out_dir / "human_review.csv"

    write_jsonl(candidates, candidate_path)
    write_review_csv(candidates, review_path)

    print("\nFINAL")
    print(f"requested={args.target_candidates}")
    print(f"candidates={len(candidates)}")
    print(f"attempted={attempted}")
    print(f"rejected={rejected}")
    print(f"candidate_questions={candidate_path}")
    print(f"human_review={review_path}")

    if len(candidates) < args.target_candidates:
        print(
            f"WARNING: only {len(candidates)} valid questions were generated. "
            "Increase --sample-size."
        )


if __name__ == "__main__":
    main()