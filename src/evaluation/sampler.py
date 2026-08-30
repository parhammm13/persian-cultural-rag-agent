"""Stratified sampling of pages for evaluation.

Joins Data/processed/chunked_data.json with the category file
Data/iran_heritage_classification_final.csv on page_id and performs
deterministic stratified sampling.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# discovered files
DEFAULT_CHUNKED = Path("Data/processed/chunked_data.json")
DEFAULT_CATEGORY = Path("Data/iran_heritage_classification_final.csv")

# category field used for stratification
DEFAULT_CATEGORY_FIELD = "final_class_code"

# question categories are NOT page categories; kept separate
PAGE_CATEGORY_FIELD = DEFAULT_CATEGORY_FIELD

EXCLUDED_CLASS_CODES = {"exclude_non_site", "uncertain"}


def load_chunked(path: Path) -> dict[str, dict[str, Any]]:
    """Load chunked_data.json -> page_id -> page info."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    parents = data.get("parents", [])
    children = data.get("children", [])

    page_map: dict[str, dict[str, Any]] = {}

    for p in parents:
        pid = str(p["page_id"])
        if pid not in page_map:
            page_map[pid] = {
                "page_id": pid,
                "page_title": p.get("page_title", ""),
                "page_url": p.get("page_url", ""),
                "parents": [],
                "children": [],
            }
        page_map[pid]["parents"].append(p)

    for c in children:
        pid = str(c["page_id"])
        if pid not in page_map:
            # child without parent entry (should not happen)
            page_map[pid] = {
                "page_id": pid,
                "page_title": c.get("page_title", ""),
                "page_url": c.get("page_url", ""),
                "parents": [],
                "children": [],
            }
        page_map[pid]["children"].append(c)

    # sort for determinism
    for v in page_map.values():
        v["parents"].sort(key=lambda x: x.get("parent_id", ""))
        v["children"].sort(key=lambda x: x.get("chunk_id", ""))

    return page_map


def load_category(path: Path) -> dict[str, dict[str, str]]:
    """Load category CSV -> page_id -> row dict."""
    cat_map: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "page_id" not in reader.fieldnames:
            raise ValueError(f"Category CSV missing page_id column: {reader.fieldnames}")
        for row in reader:
            pid = str(row.get("page_id", "")).strip()
            if not pid:
                continue
            cat_map[pid] = {k: (v or "").strip() for k, v in row.items()}
    return cat_map


def join_pages(
    chunked_map: dict[str, dict[str, Any]],
    cat_map: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Inner join on page_id; returns list of joined page dicts."""
    joined: list[dict[str, Any]] = []
    for pid, info in chunked_map.items():
        cat = cat_map.get(pid)
        if cat is None:
            cat = {}
        # keep original fields plus category
        entry = {
            "page_id": pid,
            "page_title": info["page_title"],
            "page_url": info["page_url"],
            "parents": info["parents"],
            "children": info["children"],
            # category fields (most relevant)
            "final_class_code": cat.get("final_class_code", ""),
            "final_class_fa": cat.get("final_class_fa", ""),
            "final_class_en": cat.get("final_class_en", ""),
            "final_superclass_fa": cat.get("final_superclass_fa", ""),
            "final_superclass_en": cat.get("final_superclass_en", ""),
            "final_status": cat.get("final_status", ""),
            "confidence": cat.get("confidence", ""),
            "title": cat.get("title", ""),
            "_raw_category": cat,
        }
        joined.append(entry)
    return joined


def stratified_sample(
    joined: list[dict[str, Any]],
    sample_size: int = 150,
    seed: int = 42,
    category_field: str = DEFAULT_CATEGORY_FIELD,
    cap_per_category: int | None = 15,
    exclude_excluded: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int], Counter]:
    """Deterministic stratified sampling.

    Uses sqrt weighting to give large categories more samples but compresses
    dominance. Applies cap_per_category to prevent single category dominance.

    Returns (sampled_pages, allocation_per_category, original_distribution)
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be >0")
    # filter
    filtered = joined
    if exclude_excluded:
        filtered = [
            j
            for j in joined
            if j.get("final_class_code", "") not in EXCLUDED_CLASS_CODES
            and j.get("final_status", "") == "ready"
            and j.get("final_class_code", "")
        ]
        # fallback: if filtered too small, keep ready only
        if len(filtered) < sample_size:
            filtered = [j for j in joined if j.get("final_status", "") == "ready"]

    if len(filtered) < sample_size:
        raise ValueError(f"Not enough pages to sample: {len(filtered)} < {sample_size}")

    # group
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in filtered:
        cat_val = entry.get(category_field, "") or "unknown"
        cat_val = str(cat_val).strip() or "unknown"
        groups[cat_val].append(entry)

    original_dist = Counter({k: len(v) for k, v in groups.items()})

    # sort groups for determinism
    sorted_cats = sorted(groups.keys())

    # compute sqrt weights
    weights: dict[str, float] = {}
    for cat in sorted_cats:
        weights[cat] = math.sqrt(len(groups[cat]))

    total_w = sum(weights.values())
    # fractional allocation
    fractional: dict[str, float] = {
        cat: weights[cat] / total_w * sample_size for cat in sorted_cats
    }

    allocation: dict[str, int] = {cat: int(math.floor(v)) for cat, v in fractional.items()}
    # at least 1 per category if possible and if category has members
    # ensure min 1 where floor was 0 but category exists, if we have enough sample_size
    # only force min 1 if num_cats <= sample_size
    if len(sorted_cats) <= sample_size:
        for cat in sorted_cats:
            if allocation[cat] == 0 and len(groups[cat]) > 0:
                allocation[cat] = 1

    allocated = sum(allocation.values())
    remainder = sample_size - allocated

    # distribute remainder by fractional part descending
    if remainder > 0:
        frac_parts = sorted(
            sorted_cats,
            key=lambda c: (fractional[c] - math.floor(fractional[c])),
            reverse=True,
        )
        idx = 0
        # iterative round-robin if remainder > categories
        while remainder > 0:
            cat = frac_parts[idx % len(frac_parts)]
            # respect cap and group size
            if cap_per_category is not None and allocation[cat] >= cap_per_category:
                idx += 1
                # check if all capped
                if all(
                    (cap_per_category is not None and allocation[c] >= cap_per_category)
                    or allocation[c] >= len(groups[c])
                    for c in sorted_cats
                ):
                    break
                if idx >= len(frac_parts) * 2:
                    break
                continue
            if allocation[cat] >= len(groups[cat]):
                idx += 1
                if idx > len(frac_parts) * 3:
                    break
                continue
            allocation[cat] += 1
            remainder -= 1
            idx += 1
    elif remainder < 0:
        # over-allocated (due to min 1 fix), reduce from largest allocation
        over = -remainder
        # sort descending allocation
        sorted_desc = sorted(sorted_cats, key=lambda c: allocation[c], reverse=True)
        for cat in sorted_desc:
            while allocation[cat] > 1 and over > 0:
                allocation[cat] -= 1
                over -= 1
            if over == 0:
                break

    # enforce cap strictly: redistribute surplus if cap exceeded
    if cap_per_category is not None:
        surplus = 0
        for cat in sorted_cats:
            if allocation[cat] > cap_per_category:
                surplus += allocation[cat] - cap_per_category
                allocation[cat] = cap_per_category
            if allocation[cat] > len(groups[cat]):
                surplus += allocation[cat] - len(groups[cat])
                allocation[cat] = len(groups[cat])
        # redistribute surplus to categories below cap
        # priority: larger groups first (more capacity)
        cats_by_capacity = sorted(
            [c for c in sorted_cats if allocation[c] < cap_per_category and allocation[c] < len(groups[c])],
            key=lambda c: len(groups[c]),
            reverse=True,
        )
        idx = 0
        while surplus > 0 and cats_by_capacity:
            cat = cats_by_capacity[idx % len(cats_by_capacity)]
            if allocation[cat] < cap_per_category and allocation[cat] < len(groups[cat]):
                allocation[cat] += 1
                surplus -= 1
            idx += 1
            # rebuild if all capped
            if idx % len(cats_by_capacity) == 0:
                cats_by_capacity = [
                    c for c in sorted_cats if allocation[c] < cap_per_category and allocation[c] < len(groups[c])
                ]
                cats_by_capacity.sort(key=lambda c: len(groups[c]), reverse=True)
                if not cats_by_capacity:
                    break

    # final adjustment to exactly sample_size
    # (relax cap_per_category if needed so we never return fewer than requested)
    final_sum = sum(allocation.values())
    if final_sum != sample_size:
        diff = sample_size - final_sum
        if diff > 0:
            # Add remaining slots fairly after relaxing the cap.
            # Round-robin avoids dumping all extra quota into one large category.
            available = [
                c for c in sorted_cats if allocation[c] < len(groups[c])
            ]
            idx = 0
            while diff > 0 and available:
                cat = available[idx % len(available)]
                if allocation[cat] < len(groups[cat]):
                    allocation[cat] += 1
                    diff -= 1
                idx += 1
                if idx % len(available) == 0:
                    available = [
                        c for c in available if allocation[c] < len(groups[c])
                    ]
                    idx = 0
        else:
            diff = -diff
            for cat in sorted(sorted_cats, key=lambda c: allocation[c], reverse=True):
                while diff > 0 and allocation[cat] > 1:
                    allocation[cat] -= 1
                    diff -= 1
                if diff == 0:
                    break

    if sum(allocation.values()) != sample_size:
        raise ValueError(
            f"Stratified sampling could not allocate exactly {sample_size} pages "
            f"(allocated {sum(allocation.values())}); total available pages: {len(filtered)}. "
            "Reduce --sample-size or --cap-per-category."
        )

    # sample within each group deterministically
    rnd = random.Random(seed)
    # shuffle category order deterministically then sample? Use sorted order sampling sequentially with same RNG
    sampled: list[dict[str, Any]] = []
    for cat in sorted_cats:
        n = allocation.get(cat, 0)
        if n <= 0:
            continue
        candidates = sorted(groups[cat], key=lambda x: x["page_id"])
        # random.sample using rnd
        # random.sample is deterministic when using Random instance
        chosen = rnd.sample(candidates, n) if n < len(candidates) else candidates
        sampled.extend(chosen)

    # final shuffle of sampled to mix categories while deterministic
    rnd2 = random.Random(seed)
    rnd2.shuffle(sampled)

    if len(sampled) != sample_size:
        raise ValueError(
            f"Stratified sampling returned {len(sampled)} pages, expected exactly {sample_size}. "
            "Allocation-vs-group mismatch — check cap_per_category and category distribution."
        )

    return sampled, allocation, original_dist


def join_and_sample(
    chunked_path: Path | str = DEFAULT_CHUNKED,
    category_path: Path | str = DEFAULT_CATEGORY,
    sample_size: int = 150,
    seed: int = 42,
    cap_per_category: int | None = 15,
    exclude_excluded: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int], Counter, int]:
    """High-level helper: load, join, sample."""
    chunked_path = Path(chunked_path)
    category_path = Path(category_path)
    c_map = load_chunked(chunked_path)
    cat_map = load_category(category_path)
    joined = join_pages(c_map, cat_map)
    sampled, alloc, dist = stratified_sample(
        joined,
        sample_size=sample_size,
        seed=seed,
        cap_per_category=cap_per_category,
        exclude_excluded=exclude_excluded,
    )
    return sampled, alloc, dist, len(joined)
