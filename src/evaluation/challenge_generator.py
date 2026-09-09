"""Generate hard/challenge evaluation questions for the Persian RAG benchmark.

Design
------
- Uses the already human-reviewed answerable questions as anchors.
- Generates 20 single-evidence hard questions:
    * 8 paraphrase / lexical mismatch
    * 6 indirect / semantic wording
    * 6 entity ambiguity / distractor-aware
- Generates 10 genuine multi-evidence questions that require 2+ child chunks.
- Keeps gold evidence grounded in the existing chunked corpus.
- Writes JSONL plus a CSV for human review.

Expected project layout is compatible with the existing evaluation package:
    Data/processed/chunked_data.json
    Data/evaluation/v1/...
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .generator import (
    _build_chat_model,
    _extract_json,
    _get_openrouter_config,
    _normalize,
)

DEFAULT_CHUNKED = Path("Data/processed/chunked_data.json")
DEFAULT_OUTPUT_DIR = Path("Data/evaluation/v1")

CHALLENGE_GENERATOR_VERSION = "2.0-full-corpus-multi"

SINGLE_CHALLENGE_QUOTAS = {
    "paraphrase_lexical_mismatch": 8,
    "indirect_semantic": 6,
    "entity_ambiguity_distractor": 6,
}
MULTI_EVIDENCE_QUOTA = 10

VALID_CHALLENGE_TYPES = set(SINGLE_CHALLENGE_QUOTAS) | {"multi_evidence"}

SINGLE_SYSTEM = """تو برای ارزیابی Retrieval در یک سیستم RAG فارسی، سوال سخت تولید می‌کنی.
فقط از سوال اصلی و evidence داده‌شده استفاده کن.
از دانش بیرونی استفاده نکن.
جواب factual را تغییر نده.
شناسه‌ها را جعل نکن.
خروجی فقط JSON معتبر باشد."""

SINGLE_USER = """نوع چالش: {challenge_type}

عنوان صفحه:
{page_title}

سوال اصلی:
{original_question}

جواب مرجع:
{reference_answer}

Evidence:
{supporting_text}

یک نسخه سخت‌تر از همان سوال بساز.

هدف، سخت‌تر کردن Retrieval است؛ نه مبهم یا غیرطبیعی کردن سوال.

قوانین عمومی:
- جواب سوال جدید باید دقیقاً همان reference_answer باشد.
- سوال باید فقط با Evidence قابل پاسخ باشد.
- هیچ fact جدیدی وارد سوال نکن.
- جواب را در خود سوال لو نده.
- سوال طبیعی و روشن فارسی باشد.
- از عبارت‌بندی سوال اصلی تا حد ممکن فاصله بگیر.
- سوال چندبخشی نساز.
- سوال را به دانشی خارج از Evidence وابسته نکن.

قانون مخصوص challenge_type:

paraphrase_lexical_mismatch:
- واژگان و ساختار سوال را جدی بازنویسی کن.
- overlap واژگانی غیرضروری با Evidence را کم کن.
- نام entity لازم برای تشخیص موضوع را حفظ کن.

indirect_semantic:
- همان fact را به شکل غیرمستقیم‌تر و معنایی بپرس.
- همچنان باید یک سوال واضح و تک‌جوابی باشد.

entity_ambiguity_distractor:
- سوال را طوری بنویس که entity درست برای بازیابی مهم باشد.
- مشخصه‌های موجود در سوال اصلی یا Evidence را حفظ کن.
- مشخصه یا اطلاعات جدید اختراع نکن.

خروجی:
{{
  "question": "...",
  "reference_answer": "{reference_answer}",
  "challenge_type": "{challenge_type}"
}}
"""

MULTI_SYSTEM = """تو برای ارزیابی Retrieval و RAG فارسی، سوال multi-evidence تولید می‌کنی.
فقط از child chunkهای داده‌شده استفاده کن.
از دانش بیرونی استفاده نکن.
سوال باید برای پاسخ کامل واقعاً به حداقل دو child مستقل نیاز داشته باشد.
شناسه‌ها را فقط از ورودی انتخاب کن.
خروجی فقط JSON معتبر باشد."""

MULTI_USER = """عنوان صفحه:
{page_title}

Child chunks:
{blocks}

یک سوال طبیعی فارسی بساز که پاسخ کامل آن به اطلاعات حداقل دو child chunk متفاوت نیاز داشته باشد.

تعریف multi-evidence واقعی:
- پاسخ باید حداقل دو بخش factual داشته باشد.
- هر بخش پاسخ باید توسط child متفاوتی پشتیبانی شود.
- یک child به تنهایی نباید برای پاسخ کامل کافی باشد.
- صرف اضافه کردن یک توضیح غیرضروری به سوال، multi-evidence محسوب نمی‌شود.

قوانین:
- از دانش بیرونی استفاده نکن.
- facts را ترکیب یا اختراع نکن.
- سوال واضح و قابل فهم باشد.
- جواب را در سوال لو نده.
- ترجیحاً 2 child استفاده کن؛ فقط اگر لازم است 3 child.
- supporting_text هر evidence باید نقل کوتاه و مستقیم از همان child باشد.

خروجی:
{{
  "question": "...",
  "reference_answer": "...",
  "child_ids": ["...", "..."],
  "supporting_texts": [
    {{"child_id": "...", "text": "..."}},
    {{"child_id": "...", "text": "..."}}
  ],
  "answer_parts": ["...", "..."],
  "challenge_type": "multi_evidence"
}}
"""

MULTI_VALIDATE_SYSTEM = """تو اعتبارسنج داده ارزیابی RAG هستی.
فقط از متن داده‌شده استفاده کن.
باید سخت‌گیر باشی.
خروجی فقط JSON معتبر باشد."""

MULTI_VALIDATE_USER = """سوال:
{question}

جواب مرجع:
{reference_answer}

answer_parts:
{answer_parts}

Evidenceهای انتخاب‌شده:
{evidence_blocks}

بررسی کن آیا این واقعا multi-evidence است.

فقط وقتی valid=true بده که:
1) پاسخ کامل حداقل دو بخش factual مستقل داشته باشد.
2) حداقل دو child متفاوت برای پاسخ کامل لازم باشند.
3) هر answer_part از evidence مربوطه قابل اثبات باشد.
4) هیچ child به تنهایی پاسخ کامل را در خود نداشته باشد.
5) سوال طبیعی و روشن باشد و fact اختراعی نداشته باشد.

خروجی:
{{
  "valid": true,
  "reason": "..."
}}
"""


def _resolve_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    if not (root / "Data").exists():
        root = Path.cwd()
    return root / DEFAULT_CHUNKED, root / DEFAULT_OUTPUT_DIR


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {path}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object at line {line_no}: {path}")
            rows.append(item)
    return rows


def _read_reviewed(path: Path) -> list[dict[str, Any]]:
    """Load accepted answerable questions from JSONL or CSV."""
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
    elif path.suffix.lower() == ".csv":
        rows = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item: dict[str, Any] = dict(row)
                for key in ("parent_ids", "child_ids"):
                    raw = str(item.get(key) or "").strip()
                    item[key] = [x.strip() for x in raw.split(";") if x.strip()]
                rows.append(item)
    else:
        raise ValueError("Reviewed input must be .jsonl or .csv")

    accepted: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("review_status") or "").strip().lower()
        if status and status != "accepted":
            continue

        # Challenge generation is based only on answerable gold questions.
        answer = str(row.get("reference_answer") or "").strip()
        child_ids = row.get("child_ids") or []
        if isinstance(child_ids, str):
            child_ids = [x.strip() for x in child_ids.split(";") if x.strip()]

        if not str(row.get("question") or "").strip():
            continue
        if not answer:
            continue
        if not child_ids:
            continue

        clean = dict(row)
        clean["child_ids"] = [str(x) for x in child_ids]

        parent_ids = clean.get("parent_ids") or []
        if isinstance(parent_ids, str):
            parent_ids = [x.strip() for x in parent_ids.split(";") if x.strip()]
        clean["parent_ids"] = [str(x) for x in parent_ids]

        accepted.append(clean)

    if not accepted:
        raise ValueError(f"No accepted answerable questions found in {path}")

    return accepted


def _load_chunked(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return page_id -> page and child_id -> child."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    page_map: dict[str, dict[str, Any]] = {}
    child_map: dict[str, dict[str, Any]] = {}

    for p in data.get("parents", []):
        page_id = str(p.get("page_id", ""))
        if not page_id:
            continue
        page = page_map.setdefault(
            page_id,
            {
                "page_id": page_id,
                "page_title": p.get("page_title", ""),
                "parents": [],
                "children": [],
            },
        )
        page["parents"].append(p)

    for c in data.get("children", []):
        page_id = str(c.get("page_id", ""))
        child_id = str(c.get("chunk_id", ""))
        if not page_id or not child_id:
            continue
        page = page_map.setdefault(
            page_id,
            {
                "page_id": page_id,
                "page_title": c.get("page_title", ""),
                "parents": [],
                "children": [],
            },
        )
        page["children"].append(c)
        child_map[child_id] = c

    for page in page_map.values():
        page["parents"].sort(key=lambda x: str(x.get("parent_id", "")))
        page["children"].sort(key=lambda x: str(x.get("chunk_id", "")))

    return page_map, child_map


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_review_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "id",
        "question",
        "original_question",
        "reference_answer",
        "difficulty",
        "challenge_type",
        "page_title",
        "page_id",
        "parent_ids",
        "child_ids",
        "supporting_text",
        "answer_parts",
        "review_status",
        "human_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            supporting = row.get("supporting_text", "")
            if not supporting and row.get("supporting_texts"):
                supporting = " || ".join(
                    f"{x.get('child_id', '')}: {x.get('text', '')}"
                    for x in row["supporting_texts"]
                )

            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "original_question": row.get("original_question", ""),
                    "reference_answer": row.get("reference_answer", ""),
                    "difficulty": row.get("difficulty", "hard"),
                    "challenge_type": row.get("challenge_type", ""),
                    "page_title": row.get("page_title", ""),
                    "page_id": row.get("page_id", ""),
                    "parent_ids": ";".join(row.get("parent_ids", [])),
                    "child_ids": ";".join(row.get("child_ids", [])),
                    "supporting_text": supporting,
                    "answer_parts": " || ".join(row.get("answer_parts", [])),
                    "review_status": row.get("review_status", "pending"),
                    "human_notes": "",
                }
            )


def _normalize_question(text: str) -> str:
    return _normalize(text)


def _contains_answer(question: str, answer: str) -> bool:
    q = _normalize(question)
    a = _normalize(answer)
    return bool(a) and a in q


def _substantial_rewrite(original: str, generated: str) -> bool:
    """Light heuristic: reject near-identical rewrites."""
    a = set(_normalize_question(original).split())
    b = set(_normalize_question(generated).split())
    if not a or not b:
        return False
    jaccard = len(a & b) / max(1, len(a | b))
    return jaccard < 0.82


def _quote_exists(text: str, child_text: str) -> bool:
    quote = _normalize(text)
    source = _normalize(child_text)
    return bool(quote) and quote in source


def _parent_ids_for_children(
    child_ids: list[str],
    child_map: dict[str, dict[str, Any]],
) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for cid in child_ids:
        child = child_map.get(cid)
        if not child:
            continue
        pid = str(child.get("parent_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            result.append(pid)
    return result


def _format_children(children: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for child in children:
        blocks.append(
            "\n".join(
                [
                    f"[child_id={child.get('chunk_id', '')} parent_id={child.get('parent_id', '')}]",
                    str(child.get("text") or "").strip(),
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


class ChallengeGenerator:
    def __init__(
        self,
        cache_dir: Path,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        chat_model: Any | None = None,
    ) -> None:
        if chat_model is None:
            if api_key is None or base_url is None or model is None:
                ak, bu, mo = _get_openrouter_config()
                api_key = api_key or ak
                base_url = base_url or bu
                model = model or mo
            chat_model = _build_chat_model(api_key, base_url, model)

        self.chat_model = chat_model
        self.model_name = getattr(chat_model, "model_name", None) or model or "unknown"

        from langchain_core.prompts import ChatPromptTemplate

        self.single_chain = (
            ChatPromptTemplate.from_messages(
                [("system", SINGLE_SYSTEM), ("human", SINGLE_USER)]
            )
            | self.chat_model
        )
        self.multi_chain = (
            ChatPromptTemplate.from_messages(
                [("system", MULTI_SYSTEM), ("human", MULTI_USER)]
            )
            | self.chat_model
        )
        self.multi_validate_chain = (
            ChatPromptTemplate.from_messages(
                [("system", MULTI_VALIDATE_SYSTEM), ("human", MULTI_VALIDATE_USER)]
            )
            | self.chat_model
        )

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "challenge_cache.json"
        self.cache = self._load_cache(self.cache_path)

    @staticmethod
    def _load_cache(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_cache(self) -> None:
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.cache_path)

    def generate_single(
        self,
        original: dict[str, Any],
        challenge_type: str,
        force: bool = False,
    ) -> dict[str, Any] | None:
        if challenge_type not in SINGLE_CHALLENGE_QUOTAS:
            raise ValueError(f"Invalid single challenge type: {challenge_type}")

        original_id = str(original.get("id") or original.get("page_id") or "")
        cache_key = f"single_v1::{challenge_type}::{original_id}"

        if not force and cache_key in self.cache:
            cached = self.cache[cache_key]
            return None if cached.get("_skip") else cached

        try:
            response = self.single_chain.invoke(
                {
                    "challenge_type": challenge_type,
                    "page_title": str(original.get("page_title") or ""),
                    "original_question": str(original.get("question") or ""),
                    "reference_answer": str(original.get("reference_answer") or ""),
                    "supporting_text": str(original.get("supporting_text") or ""),
                }
            )
            data = _extract_json(
                str(response.content if hasattr(response, "content") else response)
            )
        except Exception:
            self.cache[cache_key] = {"_skip": True}
            self._save_cache()
            return None

        if not isinstance(data, dict):
            self.cache[cache_key] = {"_skip": True}
            self._save_cache()
            return None

        question = str(data.get("question") or "").strip()
        original_question = str(original.get("question") or "").strip()
        reference_answer = str(original.get("reference_answer") or "").strip()

        if not question:
            return self._reject(cache_key)
        if _normalize(data.get("reference_answer", "")) != _normalize(reference_answer):
            return self._reject(cache_key)
        if _normalize(question) == _normalize(original_question):
            return self._reject(cache_key)
        if not _substantial_rewrite(original_question, question):
            return self._reject(cache_key)
        if _contains_answer(question, reference_answer):
            return self._reject(cache_key)

        result = {
            "question": question,
            "original_question": original_question,
            "reference_answer": reference_answer,
            "difficulty": "hard",
            "challenge_type": challenge_type,
            "page_id": str(original.get("page_id") or ""),
            "page_title": str(original.get("page_title") or ""),
            "parent_ids": [str(x) for x in original.get("parent_ids", [])],
            "child_ids": [str(x) for x in original.get("child_ids", [])],
            "supporting_text": str(original.get("supporting_text") or ""),
            "review_status": "pending",
        }

        self.cache[cache_key] = result
        self._save_cache()
        return result

    def generate_multi(
        self,
        page: dict[str, Any],
        child_map: dict[str, dict[str, Any]],
        force: bool = False,
    ) -> dict[str, Any] | None:
        page_id = str(page["page_id"])
        cache_key = f"multi_v1::{page_id}"

        if not force and cache_key in self.cache:
            cached = self.cache[cache_key]
            return None if cached.get("_skip") else cached

        # Avoid trivially tiny/empty children and cap prompt size.
        candidates = [
            c
            for c in page.get("children", [])
            if len(str(c.get("text") or "").strip()) >= 40
        ]
        if len(candidates) < 2:
            return self._reject(cache_key)

        # Keep a bounded but diverse set. Prefer children from different parents.
        selected = self._select_multi_candidates(candidates, max_children=8)
        if len(selected) < 2:
            return self._reject(cache_key)

        blocks = _format_children(selected)

        try:
            response = self.multi_chain.invoke(
                {
                    "page_title": str(page.get("page_title") or ""),
                    "blocks": blocks,
                }
            )
            data = _extract_json(
                str(response.content if hasattr(response, "content") else response)
            )
        except Exception:
            return self._reject(cache_key)

        if not isinstance(data, dict):
            return self._reject(cache_key)

        question = str(data.get("question") or "").strip()
        reference_answer = str(data.get("reference_answer") or "").strip()
        child_ids = data.get("child_ids") or []
        supporting_texts = data.get("supporting_texts") or []
        answer_parts = data.get("answer_parts") or []

        if isinstance(child_ids, str):
            child_ids = [child_ids]
        child_ids = [str(x) for x in child_ids]

        if not question or not reference_answer:
            return self._reject(cache_key)
        if len(child_ids) < 2 or len(child_ids) > 3:
            return self._reject(cache_key)
        if len(set(child_ids)) != len(child_ids):
            return self._reject(cache_key)
        if any(cid not in child_map for cid in child_ids):
            return self._reject(cache_key)
        if any(str(child_map[cid].get("page_id", "")) != page_id for cid in child_ids):
            return self._reject(cache_key)
        if _contains_answer(question, reference_answer):
            return self._reject(cache_key)
        if not isinstance(supporting_texts, list) or len(supporting_texts) < 2:
            return self._reject(cache_key)
        if not isinstance(answer_parts, list) or len(answer_parts) < 2:
            return self._reject(cache_key)

        normalized_support: list[dict[str, str]] = []
        support_child_ids: set[str] = set()

        for item in supporting_texts:
            if not isinstance(item, dict):
                return self._reject(cache_key)
            cid = str(item.get("child_id") or "")
            text = str(item.get("text") or "").strip()
            if cid not in child_ids or cid not in child_map:
                return self._reject(cache_key)
            if not _quote_exists(text, str(child_map[cid].get("text") or "")):
                return self._reject(cache_key)
            support_child_ids.add(cid)
            normalized_support.append({"child_id": cid, "text": text})

        if len(support_child_ids) < 2:
            return self._reject(cache_key)

        # LLM validation is deliberately separate from generation.
        if not self._validate_multi(
            question=question,
            reference_answer=reference_answer,
            answer_parts=[str(x).strip() for x in answer_parts if str(x).strip()],
            supporting_texts=normalized_support,
        ):
            return self._reject(cache_key)

        parent_ids = _parent_ids_for_children(child_ids, child_map)

        result = {
            "question": question,
            "original_question": "",
            "reference_answer": reference_answer,
            "difficulty": "hard",
            "challenge_type": "multi_evidence",
            "page_id": page_id,
            "page_title": str(page.get("page_title") or ""),
            "parent_ids": parent_ids,
            "child_ids": child_ids,
            "supporting_texts": normalized_support,
            "answer_parts": [str(x).strip() for x in answer_parts if str(x).strip()],
            "review_status": "pending",
        }

        self.cache[cache_key] = result
        self._save_cache()
        return result

    def _validate_multi(
        self,
        question: str,
        reference_answer: str,
        answer_parts: list[str],
        supporting_texts: list[dict[str, str]],
    ) -> bool:
        evidence_blocks = "\n\n".join(
            f"[child_id={x['child_id']}]\n{x['text']}" for x in supporting_texts
        )
        try:
            response = self.multi_validate_chain.invoke(
                {
                    "question": question,
                    "reference_answer": reference_answer,
                    "answer_parts": json.dumps(answer_parts, ensure_ascii=False),
                    "evidence_blocks": evidence_blocks,
                }
            )
            data = _extract_json(
                str(response.content if hasattr(response, "content") else response)
            )
        except Exception:
            return False

        return isinstance(data, dict) and data.get("valid") is True

    @staticmethod
    def _select_multi_candidates(
        children: list[dict[str, Any]],
        max_children: int = 8,
    ) -> list[dict[str, Any]]:
        """Prefer coverage across parents before adding same-parent children."""
        by_parent: dict[str, list[dict[str, Any]]] = {}
        for child in children:
            pid = str(child.get("parent_id", ""))
            by_parent.setdefault(pid, []).append(child)

        selected: list[dict[str, Any]] = []

        # First pass: one child from each parent.
        for pid in sorted(by_parent):
            group = sorted(
                by_parent[pid],
                key=lambda x: (
                    -len(str(x.get("text") or "")),
                    str(x.get("chunk_id", "")),
                ),
            )
            if group:
                selected.append(group[0])
            if len(selected) >= max_children:
                return selected

        # Second pass: add remaining children if the page has few parents.
        already = {str(x.get("chunk_id", "")) for x in selected}
        for child in sorted(
            children,
            key=lambda x: (
                -len(str(x.get("text") or "")),
                str(x.get("chunk_id", "")),
            ),
        ):
            cid = str(child.get("chunk_id", ""))
            if cid in already:
                continue
            selected.append(child)
            already.add(cid)
            if len(selected) >= max_children:
                break

        return selected

    def _reject(self, cache_key: str) -> None:
        self.cache[cache_key] = {"_skip": True}
        self._save_cache()
        return None


def _assign_single_types(
    originals: list[dict[str, Any]],
    rng: random.Random,
) -> list[tuple[dict[str, Any], str]]:
    """Deterministically shuffle candidates and assign challenge quotas."""
    pool = list(originals)
    rng.shuffle(pool)

    assignments: list[tuple[dict[str, Any], str]] = []
    cursor = 0
    for challenge_type, quota in SINGLE_CHALLENGE_QUOTAS.items():
        for _ in range(quota):
            if cursor >= len(pool):
                break
            assignments.append((pool[cursor], challenge_type))
            cursor += 1
    return assignments


def _candidate_multi_pages(
    page_map: dict[str, dict[str, Any]],
    rng: random.Random,
    min_child_chars: int = 40,
) -> list[dict[str, Any]]:
    """Search the ENTIRE chunk database for genuine multi-evidence candidates.

    Multi-evidence benchmark items must not be restricted to pages that already
    appeared in the reviewed single-evidence set. A page is eligible when it
    contains at least two non-trivial child chunks. Pages spanning multiple
    parents are preferred because they are more likely to support genuinely
    complementary evidence.
    """
    pages: list[dict[str, Any]] = []

    for page in page_map.values():
        useful_children = [
            child
            for child in page.get("children", [])
            if len(str(child.get("text") or "").strip()) >= min_child_chars
        ]
        if len(useful_children) < 2:
            continue

        candidate = dict(page)
        candidate["children"] = useful_children
        pages.append(candidate)

    # Deterministic tie randomization, then structural-diversity ranking.
    rng.shuffle(pages)
    pages.sort(
        key=lambda p: (
            len({str(c.get("parent_id", "")) for c in p.get("children", [])}),
            len(p.get("children", [])),
            sum(len(str(c.get("text") or "")) for c in p.get("children", [])),
        ),
        reverse=True,
    )
    return pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviewed",
        required=True,
        help="Accepted answerable questions (.jsonl or .csv)",
    )
    parser.add_argument("--chunked", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--single-target", type=int, default=20)
    parser.add_argument("--multi-target", type=int, default=MULTI_EVIDENCE_QUOTA)
    parser.add_argument(
        "--max-multi-attempts",
        type=int,
        default=200,
        help="Maximum pages to try for genuine multi-evidence generation",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.single_target != sum(SINGLE_CHALLENGE_QUOTAS.values()):
        parser.error(
            f"--single-target must be {sum(SINGLE_CHALLENGE_QUOTAS.values())} "
            "for the fixed challenge-type quotas in this version"
        )
    if args.multi_target < 0:
        parser.error("--multi-target must be >= 0")

    chunked_default, out_default = _resolve_paths()
    reviewed_path = Path(args.reviewed)
    chunked_path = Path(args.chunked) if args.chunked else chunked_default
    out_dir = Path(args.output_dir) if args.output_dir else out_default
    out_dir.mkdir(parents=True, exist_ok=True)

    originals = _read_reviewed(reviewed_path)
    page_map, child_map = _load_chunked(chunked_path)

    rng = random.Random(args.seed)

    generator = ChallengeGenerator(cache_dir=out_dir / ".challenge_cache")
    print(f"[version] {CHALLENGE_GENERATOR_VERSION}")
    print(f"[llm] model={generator.model_name}")
    print(f"[input] accepted_answerable={len(originals)}")

    results: list[dict[str, Any]] = []
    seen_questions: set[str] = set()

    # -------------------------
    # 1) Single-evidence hard
    # -------------------------
    assignments = _assign_single_types(originals, rng)

    # Keep fallback candidates available if a generated question is rejected.
    used_original_ids: set[str] = set()
    type_counts: Counter[str] = Counter()

    for original, challenge_type in assignments:
        if type_counts[challenge_type] >= SINGLE_CHALLENGE_QUOTAS[challenge_type]:
            continue

        result = generator.generate_single(
            original=original,
            challenge_type=challenge_type,
            force=args.force,
        )
        used_original_ids.add(str(original.get("id") or original.get("page_id") or ""))

        if result is None:
            print(f"[single] {challenge_type} -> rejected")
            continue

        nq = _normalize_question(result["question"])
        if nq in seen_questions:
            print(f"[single] {challenge_type} -> duplicate")
            continue

        seen_questions.add(nq)
        type_counts[challenge_type] += 1
        results.append(result)
        print(
            f"[single] {challenge_type} -> "
            f"OK ({type_counts[challenge_type]}/{SINGLE_CHALLENGE_QUOTAS[challenge_type]})"
        )

    # Fallback pass to fill any rejected single-evidence slots.
    fallback = list(originals)
    rng.shuffle(fallback)

    for challenge_type, quota in SINGLE_CHALLENGE_QUOTAS.items():
        if type_counts[challenge_type] >= quota:
            continue

        for original in fallback:
            if type_counts[challenge_type] >= quota:
                break

            original_key = str(original.get("id") or original.get("page_id") or "")
            # Reuse across different challenge types is allowed only as a last resort,
            # but avoid reusing within the same type via cache key uniqueness.
            result = generator.generate_single(
                original=original,
                challenge_type=challenge_type,
                force=args.force,
            )
            if result is None:
                continue

            nq = _normalize_question(result["question"])
            if nq in seen_questions:
                continue

            seen_questions.add(nq)
            type_counts[challenge_type] += 1
            results.append(result)
            used_original_ids.add(original_key)

    # -------------------------
    # 2) Genuine multi-evidence
    # -------------------------
    multi_pages = _candidate_multi_pages(page_map, rng)
    print(f"[multi] eligible_pages_in_full_corpus={len(multi_pages)}")
    multi_count = 0
    multi_attempts = 0

    for page in multi_pages:
        if multi_count >= args.multi_target:
            break
        if multi_attempts >= args.max_multi_attempts:
            break

        multi_attempts += 1
        result = generator.generate_multi(
            page=page,
            child_map=child_map,
            force=args.force,
        )

        if result is None:
            print(
                f"[multi {multi_attempts}] "
                f"page={page.get('page_id')} -> rejected"
            )
            continue

        nq = _normalize_question(result["question"])
        if nq in seen_questions:
            print(
                f"[multi {multi_attempts}] "
                f"page={page.get('page_id')} -> duplicate"
            )
            continue

        seen_questions.add(nq)
        multi_count += 1
        results.append(result)
        print(
            f"[multi {multi_attempts}] page={page.get('page_id')} -> "
            f"OK ({multi_count}/{args.multi_target})"
        )

    # Stable final IDs.
    for idx, row in enumerate(results):
        row["id"] = f"eval_challenge_{idx:03d}"

    jsonl_path = out_dir / "challenge_questions.jsonl"
    csv_path = out_dir / "challenge_review.csv"

    _write_jsonl(results, jsonl_path)
    _write_review_csv(results, csv_path)

    print("\nFINAL")
    print(f"single_counts={dict(type_counts)}")
    print(f"multi={multi_count}/{args.multi_target}")
    print(f"total={len(results)}")
    print(f"jsonl={jsonl_path}")
    print(f"review_csv={csv_path}")

    expected_single = sum(SINGLE_CHALLENGE_QUOTAS.values())
    if sum(type_counts.values()) < expected_single:
        print(
            "WARNING: single-evidence target was not fully reached. "
            "Review rejected generations or rerun with --force."
        )
    if multi_count < args.multi_target:
        print(
            "WARNING: genuine multi-evidence target was not fully reached. "
            "This is preferable to forcing fake multi-chunk questions. "
            "Increase --max-multi-attempts or inspect the full corpus for richer multi-chunk evidence."
        )


if __name__ == "__main__":
    main()
