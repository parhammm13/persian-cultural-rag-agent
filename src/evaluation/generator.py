"""Two-stage evaluation candidate generation without question categories.

Stage 1: find explicit, question-worthy evidence on each sampled page.
Stage 2: generate one natural question from that evidence.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from ..rag.factory import _load_project_env  # type: ignore
except Exception:
    _load_project_env = None  # type: ignore

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
MAX_CHARS_PER_PROMPT = 16000

ANALYZE_SYSTEM = """تو یک ارزیاب داده برای یک سیستم RAG فارسی هستی.
فقط از متن داده‌شده استفاده کن و از دانش بیرونی استفاده نکن.
شناسه‌ها را جعل نکن. فقط parent_id و child_idهای موجود در ورودی را برگردان.
خروجی فقط JSON معتبر باشد."""

ANALYZE_USER = """صفحه:
page_id: {page_id}
title: {page_title}

بخش‌های صفحه:
{blocks}

مشخص کن آیا این بخش‌های صفحه برای ساختن یک سوال طبیعی و قابل پاسخ، شواهد صریح و کافی دارند یا نه.

اگر شواهد کافی است، بهترین evidence را از همین ورودی انتخاب کن:
- parent_ids
- child_ids
- evidence_text: یک نقل کوتاه و عینی از یکی از childهای انتخاب‌شده
- reason: توضیح خیلی کوتاه

اگر شواهد کافی نیست supported=false بده.

فرمت:
{{
  "supported": true,
  "parent_ids": ["..."],
  "child_ids": ["..."],
  "evidence_text": "...",
  "reason": "..."
}}

قوانین:
- اگر متن برای یک سوال روشن و مفید کافی نیست، supported را به زور true نکن.
- هر child باید متعلق به parent انتخاب‌شده و همین صفحه باشد.
- evidence_text باید عیناً از یکی از childهای انتخاب‌شده قابل استخراج باشد.
"""

QUESTION_SYSTEM = """تو سوال ارزیابی برای یک سیستم RAG فارسی تولید می‌کنی.
فقط از evidence داده‌شده استفاده کن.
از دانش بیرونی استفاده نکن.
جواب باید کاملاً از evidence قابل استخراج باشد.
خروجی فقط JSON معتبر باشد."""

QUESTION_USER = """عنوان صفحه: {page_title}

Evidence:
{evidence_blocks}

یک سوال طبیعی فارسی بساز که جوابش کاملاً از Evidence قابل استخراج باشد.

خروجی:
{{
  "question": "...",
  "reference_answer": "...",
  "supporting_text": "...",
  "difficulty": "easy|medium|hard"
}}

قوانین:
- سوال نباید جواب را لو بدهد.
- سوال باید بدون دیدن متن قابل فهم باشد.
- supporting_text باید نقل کوتاه و مستقیم از evidence باشد.
- سوال را مصنوعی سخت نکن.
- difficulty: easy=واقعیت مستقیم، medium=paraphrase/ترکیب ساده، hard=چند محدودیت یا رابطه غیرساده ولی مستند.
"""


def _load_env() -> None:
    if _load_project_env is not None:
        try:
            _load_project_env()
            return
        except Exception:
            pass
    load_dotenv()
    for parent in Path(__file__).resolve().parents:
        p = parent / ".env"
        if p.exists():
            load_dotenv(p)
            break


def _get_openrouter_config() -> tuple[str, str, str]:
    _load_env()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    model = os.getenv("EVAL_GENERATION_MODEL", "").strip() or os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    return api_key, base_url, model


def _build_chat_model(api_key: str, base_url: str, model: str):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        timeout=120,
        max_retries=2,
        temperature=0.1,
        max_completion_tokens=1400,
        use_responses_api=False,
    )


def _extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("Could not parse JSON from LLM output")


def _normalize(text: str) -> str:
    table = str.maketrans({
        "ي": "ی", "ك": "ک", "ة": "ه", "أ": "ا", "إ": "ا", "ؤ": "و",
        "‌": " ", "‏": "", "‎": "", "ً": "", "ٌ": "", "ٍ": "", "َ": "", "ُ": "", "ِ": "", "ّ": "", "ْ": "",
    })
    text = (text or "").lower().translate(table)
    text = re.sub(r"[\s\[\]()«»\"'.,،؛:;!?؟\-–—*_]+", " ", text)
    return " ".join(text.split())


def verify_supporting_text(supporting_text: str, child_ids: list[str], children_by_id: dict[str, dict[str, Any]]) -> bool:
    sup = _normalize(supporting_text)
    if not sup:
        return False
    for cid in child_ids:
        child = children_by_id.get(cid)
        if child and sup in _normalize(str(child.get("text") or "")):
            return True
    return False


def _format_group(parents: list[dict[str, Any]], children_by_parent: dict[str, list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    for parent in parents:
        pid = str(parent.get("parent_id", ""))
        heading = parent.get("section_heading") or "بدون عنوان"
        lines.append(f"[PARENT] parent_id={pid} | heading={heading}")
        for child in children_by_parent.get(pid, []):
            cid = str(child.get("chunk_id", ""))
            lines.append(f"[CHILD] child_id={cid}")
            lines.append(str(child.get("text") or "").strip())
        lines.append("---")
    return "\n".join(lines)


def _group_parents(parents: list[dict[str, Any]], children_by_parent: dict[str, list[dict[str, Any]]], max_chars: int = MAX_CHARS_PER_PROMPT) -> list[list[dict[str, Any]]]:
    groups, current, current_chars = [], [], 0
    for parent in parents:
        pid = str(parent.get("parent_id", ""))
        size = 200 + sum(len(str(c.get("text") or "")) for c in children_by_parent.get(pid, []))
        if current and current_chars + size > max_chars:
            groups.append(current)
            current, current_chars = [], 0
        current.append(parent)
        current_chars += size
    if current:
        groups.append(current)
    return groups


class EvaluationGenerator:
    def __init__(self, cache_dir: Path, api_key: str | None = None, base_url: str | None = None, model: str | None = None, chat_model: Any | None = None) -> None:
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
        self.analysis_chain = ChatPromptTemplate.from_messages([("system", ANALYZE_SYSTEM), ("human", ANALYZE_USER)]) | self.chat_model
        self.question_chain = ChatPromptTemplate.from_messages([("system", QUESTION_SYSTEM), ("human", QUESTION_USER)]) | self.chat_model

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_cache_path = cache_dir / "page_analysis_cache.json"
        self.question_cache_path = cache_dir / "question_cache.json"
        self.analysis_cache = self._load_cache(self.analysis_cache_path)
        self.question_cache = self._load_cache(self.question_cache_path)

    @staticmethod
    def _load_cache(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _save_cache(path: Path, data: dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def analyze_page(self, page: dict[str, Any], force: bool = False) -> dict[str, Any]:
        page_id = str(page["page_id"])
        cache_key = f"uncategorized_v1::{page_id}"
        if not force and cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        parents = page.get("parents", [])
        children = page.get("children", [])
        children_by_parent: dict[str, list[dict[str, Any]]] = {}
        children_by_id: dict[str, dict[str, Any]] = {}
        for child in children:
            cid, pid = str(child["chunk_id"]), str(child["parent_id"])
            children_by_parent.setdefault(pid, []).append(child)
            children_by_id[cid] = child

        valid_parents = {str(p["parent_id"]) for p in parents}
        child_to_parent = {str(c["chunk_id"]): str(c["parent_id"]) for c in children}
        selected = {"supported": False, "parent_ids": [], "child_ids": [], "evidence_text": "", "reason": ""}

        for group in _group_parents(parents, children_by_parent):
            try:
                response = self.analysis_chain.invoke({
                    "page_id": page_id,
                    "page_title": page.get("page_title", ""),
                    "blocks": _format_group(group, children_by_parent),
                })
                data = _extract_json(str(response.content if hasattr(response, "content") else response))
            except Exception:
                continue

            item = data if isinstance(data, dict) else {}
            if not item.get("supported") or selected["supported"]:
                continue
            parent_ids = item.get("parent_ids", [])
            child_ids = item.get("child_ids", [])
            if isinstance(parent_ids, str):
                parent_ids = [parent_ids]
            if isinstance(child_ids, str):
                child_ids = [child_ids]
            parent_ids = [str(x) for x in parent_ids]
            child_ids = [str(x) for x in child_ids]
            evidence_text = str(item.get("evidence_text") or "").strip()
            if not parent_ids or not child_ids:
                continue
            if not all(x in valid_parents for x in parent_ids):
                continue
            if not all(x in children_by_id for x in child_ids):
                continue
            if not all(child_to_parent.get(cid) in parent_ids for cid in child_ids):
                continue
            if not verify_supporting_text(evidence_text, child_ids, children_by_id):
                continue
            selected = {
                "supported": True,
                "parent_ids": parent_ids,
                "child_ids": child_ids,
                "evidence_text": evidence_text,
                "reason": str(item.get("reason") or "").strip(),
            }

        result = {"page_id": page_id, "page_title": page.get("page_title", ""), **selected}
        self.analysis_cache[cache_key] = result
        self._save_cache(self.analysis_cache_path, self.analysis_cache)
        return result

    def generate_question(self, page: dict[str, Any], analysis: dict[str, Any], force: bool = False) -> dict[str, Any] | None:
        page_id = str(page["page_id"])
        cache_key = f"uncategorized_v1::{page_id}"
        if not force and cache_key in self.question_cache:
            cached = self.question_cache[cache_key]
            return None if cached.get("_skip") else cached

        if not analysis.get("supported"):
            return None

        child_ids = [str(x) for x in analysis.get("child_ids", [])]
        parent_ids = [str(x) for x in analysis.get("parent_ids", [])]
        children_by_id = {str(c["chunk_id"]): c for c in page.get("children", [])}
        evidence_lines = []
        for cid in child_ids:
            child = children_by_id.get(cid)
            if child:
                evidence_lines.append(f"[child_id={cid} parent_id={child.get('parent_id','')}]")
                evidence_lines.append(str(child.get("text") or "").strip())
        if not evidence_lines:
            return None

        try:
            response = self.question_chain.invoke({
                "page_title": page.get("page_title", ""),
                "evidence_blocks": "\n".join(evidence_lines),
            })
            data = _extract_json(str(response.content if hasattr(response, "content") else response))
        except Exception:
            self.question_cache[cache_key] = {"_skip": True}
            self._save_cache(self.question_cache_path, self.question_cache)
            return None

        question = str(data.get("question") or "").strip()
        answer = str(data.get("reference_answer") or "").strip()
        supporting_text = str(data.get("supporting_text") or "").strip()
        difficulty = str(data.get("difficulty") or "").strip().lower()
        if not question or not answer or difficulty not in VALID_DIFFICULTIES or not verify_supporting_text(supporting_text, child_ids, children_by_id):
            self.question_cache[cache_key] = {"_skip": True}
            self._save_cache(self.question_cache_path, self.question_cache)
            return None

        result = {
            "question": question,
            "reference_answer": answer,
            "difficulty": difficulty,
            "page_id": page_id,
            "page_title": page.get("page_title", ""),
            "parent_ids": parent_ids,
            "child_ids": child_ids,
            "supporting_text": supporting_text,
        }
        self.question_cache[cache_key] = result
        self._save_cache(self.question_cache_path, self.question_cache)
        return result
