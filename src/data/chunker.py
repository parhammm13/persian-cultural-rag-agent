from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import tiktoken



# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

CHILD_TARGET_TOKENS = 400
CHILD_MAX_TOKENS = 500
CHILD_OVERLAP_TOKENS = 40

PARENT_TARGET_TOKENS = 900
PARENT_MAX_TOKENS = 1200

TOKENIZER_NAME = "cl100k_base"


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class ParentBlock:
    parent_id: str
    semantic_unit_id: str
    page_id: str
    page_title: str
    page_url: str
    unit_type: str
    section_index: int | None
    section_heading: str
    section_level: str
    parent_index: int
    text: str
    tokens: int


@dataclass
class ChildChunk:
    chunk_id: str
    parent_id: str
    semantic_unit_id: str
    page_id: str
    page_title: str
    page_url: str
    unit_type: str
    section_index: int | None
    section_heading: str
    section_level: str
    parent_index: int
    child_index: int
    text: str
    tokens: int


# ============================================================
# TOKENIZER
# ============================================================

class Tokenizer:
    def __init__(self, name: str = TOKENIZER_NAME) -> None:
        self.encoding = tiktoken.get_encoding(name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Hard token split. Use only as the last fallback."""
        token_ids = self.encoding.encode(text)
        parts: list[str] = []

        for start in range(0, len(token_ids), max_tokens):
            end = min(start + max_tokens, len(token_ids))
            part = self.encoding.decode(token_ids[start:end]).strip()
            if part:
                parts.append(part)

        return parts


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines."""
    text = clean_text(text)
    if not text:
        return []

    return [
        p.strip()
        for p in re.split(r"\n\s*\n+", text)
        if p.strip()
    ]


def split_sentences(text: str) -> list[str]:
    """Simple Persian/English sentence split."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []

    return [
        s.strip()
        for s in re.split(r"(?<=[.!?؟])\s+", text)
        if s.strip()
    ]


def make_atomic_segments(
    text: str,
    tokenizer: Tokenizer,
    max_tokens: int,
) -> list[str]:
    """
    Split in this order:
    1. Paragraph.
    2. Sentence.
    3. Hard token split.
    """
    paragraphs = split_paragraphs(text)
    segments: list[str] = []

    for paragraph in paragraphs:
        if tokenizer.count(paragraph) <= max_tokens:
            segments.append(paragraph)
            continue

        sentences = split_sentences(paragraph)

        if not sentences:
            segments.extend(
                tokenizer.split_by_tokens(paragraph, max_tokens)
            )
            continue

        for sentence in sentences:
            if tokenizer.count(sentence) <= max_tokens:
                segments.append(sentence)
            else:
                segments.extend(
                    tokenizer.split_by_tokens(sentence, max_tokens)
                )

    return segments


def tail_segments_for_overlap(
    segments: list[str],
    tokenizer: Tokenizer,
    overlap_tokens: int,
) -> list[str]:
    """Reuse complete trailing segments for overlap."""
    if overlap_tokens <= 0:
        return []

    result: list[str] = []
    token_total = 0

    for segment in reversed(segments):
        segment_tokens = tokenizer.count(segment)
        if token_total + segment_tokens > overlap_tokens:
            break

        result.insert(0, segment)
        token_total += segment_tokens

    return result


def join_segments(segments: Iterable[str]) -> str:
    return "\n\n".join(
        segment.strip()
        for segment in segments
        if segment.strip()
    ).strip()

def split_text(
    text: str,
    tokenizer: Tokenizer,
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[str]:
    """
    Split one semantic unit.

    Rules:
    - Never exceed max_tokens.
    - Prefer paragraph boundaries.
    - Use sentence boundaries when required.
    - Use hard token split only as the last fallback.
    - Use complete segments for overlap.
    """

    text = clean_text(text)

    if not text:
        return []

    # No split is required.
    if tokenizer.count(text) <= max_tokens:
        return [text]

    # Create safe semantic segments.
    segments = make_atomic_segments(
        text=text,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )

    if not segments:
        return []

    chunks: list[str] = []
    current: list[str] = []

    # ========================================================
    # HELPERS
    # ========================================================

    def current_token_count() -> int:
        """
        Count tokens in the actual joined text.

        This includes separators such as "\\n\\n".
        """
        if not current:
            return 0

        return tokenizer.count(
            join_segments(current)
        )

    def flush_current() -> None:
        """
        Save the current chunk and keep a small semantic
        overlap for the next chunk.
        """

        nonlocal current

        if not current:
            return

        chunk_text = join_segments(current)

        if not chunk_text:
            return

        chunk_tokens = tokenizer.count(
            chunk_text
        )

        # Safety check before saving.
        if chunk_tokens > max_tokens:
            raise ValueError(
                f"Chunk exceeds max_tokens before save: "
                f"{chunk_tokens} > {max_tokens}"
            )

        chunks.append(
            chunk_text
        )

        # Keep complete trailing segments as overlap.
        overlap = tail_segments_for_overlap(
            current,
            tokenizer,
            overlap_tokens,
        )

        current = overlap

    # ========================================================
    # PACK SEGMENTS INTO CHUNKS
    # ========================================================

    for segment in segments:

        segment_tokens = tokenizer.count(
            segment
        )

        # This should already be guaranteed by
        # make_atomic_segments().
        if segment_tokens > max_tokens:
            raise ValueError(
                "Atomic segment is larger than max_tokens: "
                f"{segment_tokens} > {max_tokens}"
            )

        # ----------------------------------------------------
        # Empty current chunk
        # ----------------------------------------------------

        if not current:
            current = [segment]
            continue

        # ----------------------------------------------------
        # Calculate the REAL candidate size.
        #
        # Do not use:
        #
        # current_tokens + segment_tokens
        #
        # because join_segments() adds separators.
        # ----------------------------------------------------

        candidate = current + [segment]

        candidate_text = join_segments(
            candidate
        )

        candidate_tokens = tokenizer.count(
            candidate_text
        )

        existing_tokens = current_token_count()

        # ----------------------------------------------------
        # Flush if:
        #
        # 1. Adding this segment exceeds the hard limit.
        # 2. The current chunk already reached the target.
        # ----------------------------------------------------

        if (
            candidate_tokens > max_tokens
            or existing_tokens >= target_tokens
        ):

            flush_current()

            # ------------------------------------------------
            # The overlap may make the new candidate too large.
            #
            # Remove the oldest overlap segments until the new
            # segment fits.
            # ------------------------------------------------

            while current:

                candidate_text = join_segments(
                    current + [segment]
                )

                candidate_tokens = tokenizer.count(
                    candidate_text
                )

                if candidate_tokens <= max_tokens:
                    break

                current.pop(0)

            current.append(
                segment
            )

        else:

            current.append(
                segment
            )

    # ========================================================
    # SAVE FINAL CHUNK
    # ========================================================

    if current:

        final_text = join_segments(
            current
        )

        if final_text:

            final_tokens = tokenizer.count(
                final_text
            )

            if final_tokens > max_tokens:
                raise ValueError(
                    f"Final chunk exceeds max_tokens: "
                    f"{final_tokens} > {max_tokens}"
                )

            if (
                not chunks
                or final_text != chunks[-1]
            ):
                chunks.append(
                    final_text
                )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    for index, chunk in enumerate(chunks):

        n_tokens = tokenizer.count(
            chunk
        )

        if n_tokens > max_tokens:
            raise ValueError(
                f"Chunk {index} exceeds max_tokens: "
                f"{n_tokens} > {max_tokens}"
            )

    return chunks

# ============================================================
# ID HELPERS
# ============================================================

def safe_page_id(page: dict[str, Any]) -> str:
    value = str(
        page.get("id", page.get("unique_id", ""))
    ).strip()

    if value:
        return value

    title = str(page.get("title", ""))
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:12]
    return f"generated_{digest}"


def make_semantic_unit_id(
    page_id: str,
    unit_type: str,
    section_index: int | None,
) -> str:
    if unit_type == "lead":
        return f"{page_id}#lead"

    return f"{page_id}#sec_{section_index}"


# ============================================================
# SEMANTIC UNIT ITERATOR
# ============================================================

def iter_semantic_units(
    page: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    """
    Yield the lead and each section.

    The top-level page['text'] is not chunked because it is the
    flattened copy of the page content.
    """
    page_id = safe_page_id(page)
    title = str(page.get("title", "")).strip()
    page_url = str(page.get("page_url", "")).strip()

    page_content = page.get("page_content", {})
    if not isinstance(page_content, dict):
        return

    lead = clean_text(page_content.get("lead", ""))

    if lead:
        yield {
            "page_id": page_id,
            "page_title": title,
            "page_url": page_url,
            "unit_type": "lead",
            "section_index": None,
            "section_heading": "",
            "section_level": "lead",
            "text": lead,
        }

    sections = page_content.get("sections", [])
    if not isinstance(sections, list):
        return

    for section_index, section in enumerate(sections):
        if not isinstance(section, dict):
            continue

        section_text = clean_text(section.get("text", ""))
        if not section_text:
            continue

        yield {
            "page_id": page_id,
            "page_title": title,
            "page_url": page_url,
            "unit_type": "section",
            "section_index": section_index,
            "section_heading": str(section.get("heading", "")).strip(),
            "section_level": str(section.get("level", "")).strip(),
            "text": section_text,
        }


# ============================================================
# CHUNK ONE SEMANTIC UNIT
# ============================================================
def chunk_semantic_unit(
    unit: dict[str, Any],
    tokenizer: Tokenizer,
    child_target_tokens: int,
    child_max_tokens: int,
    child_overlap_tokens: int,
    parent_target_tokens: int,
    parent_max_tokens: int,
) -> tuple[list[ParentBlock], list[ChildChunk]]:
    """Create parent blocks and child chunks for one lead or section."""

    text = unit["text"]
    total_tokens = tokenizer.count(text)

    semantic_unit_id = make_semantic_unit_id(
        page_id=unit["page_id"],
        unit_type=unit["unit_type"],
        section_index=unit["section_index"],
    )

    # Keep the semantic unit as one parent when it is small enough.
    if total_tokens <= parent_max_tokens:
        parent_texts = [text]

    else:
        parent_texts = split_text(
            text=text,
            tokenizer=tokenizer,
            target_tokens=parent_target_tokens,
            max_tokens=parent_max_tokens,
            overlap_tokens=0,
        )

    parents: list[ParentBlock] = []
    children: list[ChildChunk] = []

    # Process each parent block.
    for parent_index, parent_text in enumerate(parent_texts):

        parent_tokens = tokenizer.count(parent_text)

        parent_id = (
            f"{semantic_unit_id}"
            f"#parent_{parent_index:03d}"
        )

        # Create the parent record.
        parent = ParentBlock(
            parent_id=parent_id,
            semantic_unit_id=semantic_unit_id,
            page_id=unit["page_id"],
            page_title=unit["page_title"],
            page_url=unit["page_url"],
            unit_type=unit["unit_type"],
            section_index=unit["section_index"],
            section_heading=unit["section_heading"],
            section_level=unit["section_level"],
            parent_index=parent_index,
            text=parent_text,
            tokens=parent_tokens,
        )

        parents.append(parent)

        # Keep the parent as one child when it is small enough.
        if parent_tokens <= child_max_tokens:
            child_texts = [parent_text]

        else:
            child_texts = split_text(
                text=parent_text,
                tokenizer=tokenizer,
                target_tokens=child_target_tokens,
                max_tokens=child_max_tokens,
                overlap_tokens=child_overlap_tokens,
            )

        # Create child records.
        for child_index, child_text in enumerate(child_texts):

            child_tokens = tokenizer.count(child_text)

            chunk_id = (
                f"{parent_id}"
                f"#child_{child_index:03d}"
            )

            child = ChildChunk(
                chunk_id=chunk_id,
                parent_id=parent_id,
                semantic_unit_id=semantic_unit_id,
                page_id=unit["page_id"],
                page_title=unit["page_title"],
                page_url=unit["page_url"],
                unit_type=unit["unit_type"],
                section_index=unit["section_index"],
                section_heading=unit["section_heading"],
                section_level=unit["section_level"],
                parent_index=parent_index,
                child_index=child_index,
                text=child_text,
                tokens=child_tokens,
            )

            children.append(child)

    return parents, children
# ============================================================
# CHUNK CORPUS
# ============================================================

def chunk_pages(
    pages: list[dict[str, Any]],
    tokenizer: Tokenizer,
    child_target_tokens: int = CHILD_TARGET_TOKENS,
    child_max_tokens: int = CHILD_MAX_TOKENS,
    child_overlap_tokens: int = CHILD_OVERLAP_TOKENS,
    parent_target_tokens: int = PARENT_TARGET_TOKENS,
    parent_max_tokens: int = PARENT_MAX_TOKENS,
) -> dict[str, Any]:
    all_parents: list[ParentBlock] = []
    all_children: list[ChildChunk] = []
    semantic_unit_count = 0

    for page in pages:
        for unit in iter_semantic_units(page):
            semantic_unit_count += 1

            parents, children = chunk_semantic_unit(
                unit=unit,
                tokenizer=tokenizer,
                child_target_tokens=child_target_tokens,
                child_max_tokens=child_max_tokens,
                child_overlap_tokens=child_overlap_tokens,
                parent_target_tokens=parent_target_tokens,
                parent_max_tokens=parent_max_tokens,
            )

            all_parents.extend(parents)
            all_children.extend(children)

    parent_token_counts = [item.tokens for item in all_parents]
    child_token_counts = [item.tokens for item in all_children]

    stats = {
        "pages": len(pages),
        "semantic_units": semantic_unit_count,
        "parents": len(all_parents),
        "children": len(all_children),
        "max_parent_tokens": max(parent_token_counts, default=0),
        "max_child_tokens": max(child_token_counts, default=0),
        "mean_parent_tokens": round(
            sum(parent_token_counts) / len(parent_token_counts), 2
        ) if parent_token_counts else 0,
        "mean_child_tokens": round(
            sum(child_token_counts) / len(child_token_counts), 2
        ) if child_token_counts else 0,
    }

    return {
        "config": {
            "tokenizer": TOKENIZER_NAME,
            "child_target_tokens": child_target_tokens,
            "child_max_tokens": child_max_tokens,
            "child_overlap_tokens": child_overlap_tokens,
            "parent_target_tokens": parent_target_tokens,
            "parent_max_tokens": parent_max_tokens,
        },
        "stats": stats,
        "parents": [asdict(item) for item in all_parents],
        "children": [asdict(item) for item in all_children],
    }


# ============================================================
# VALIDATION
# ============================================================

def validate_output(result: dict[str, Any]) -> None:
    config = result["config"]
    parent_max = config["parent_max_tokens"]
    child_max = config["child_max_tokens"]

    parent_ids: set[str] = set()
    child_ids: set[str] = set()

    for parent in result["parents"]:
        parent_id = parent["parent_id"]

        if parent_id in parent_ids:
            raise ValueError(f"Duplicate parent_id: {parent_id}")

        parent_ids.add(parent_id)

        if parent["tokens"] > parent_max:
            raise ValueError(
                f"Parent exceeds maximum: {parent_id} -> {parent['tokens']}"
            )

    for child in result["children"]:
        chunk_id = child["chunk_id"]

        if chunk_id in child_ids:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")

        child_ids.add(chunk_id)

        if child["parent_id"] not in parent_ids:
            raise ValueError(f"Unknown parent_id for child: {chunk_id}")

        if child["tokens"] > child_max:
            raise ValueError(
                f"Child exceeds maximum: {chunk_id} -> {child['tokens']}"
            )


# ============================================================
# FILE I/O
# ============================================================

def load_pages(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of pages.")

    return data


def save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create section-aware parent/child chunks from "
            "cleaned Persian Wikipedia JSON."
        )
    )

    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to cleaned_data.json",
    )

    parser.add_argument(
        "output_json",
        type=Path,
        help="Path for chunked_data.json",
    )

    parser.add_argument("--child-target", type=int, default=CHILD_TARGET_TOKENS)
    parser.add_argument("--child-max", type=int, default=CHILD_MAX_TOKENS)
    parser.add_argument("--child-overlap", type=int, default=CHILD_OVERLAP_TOKENS)
    parser.add_argument("--parent-target", type=int, default=PARENT_TARGET_TOKENS)
    parser.add_argument("--parent-max", type=int, default=PARENT_MAX_TOKENS)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.child_target > args.child_max:
        raise ValueError(
            "--child-target cannot be larger than --child-max."
        )

    if args.parent_target > args.parent_max:
        raise ValueError(
            "--parent-target cannot be larger than --parent-max."
        )

    if args.child_max > args.parent_max:
        raise ValueError(
            "--child-max cannot be larger than --parent-max."
        )

    tokenizer = Tokenizer()
    pages = load_pages(args.input_json)

    result = chunk_pages(
        pages=pages,
        tokenizer=tokenizer,
        child_target_tokens=args.child_target,
        child_max_tokens=args.child_max,
        child_overlap_tokens=args.child_overlap,
        parent_target_tokens=args.parent_target,
        parent_max_tokens=args.parent_max,
    )

    validate_output(result)
    save_json(result, args.output_json)

    print("=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)

    for key, value in result["stats"].items():
        print(f"{key:24}: {value}")

    print()
    print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
