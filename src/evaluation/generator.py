"""LLM-based candidate generation for evaluation.

Uses OpenRouter via LangChain ChatOpenAI. Validates IDs against chunked_data.
Cache page-level results to resume interrupted runs.
Handles large pages by grouping parents.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# reuse env loading from factory if available
try:
    from ..rag.factory import _load_project_env  # type: ignore
except Exception:
    _load_project_env = None  # type: ignore

VALID_QUERY_CATEGORIES = {"factual", "architecture", "history", "entity_relation", "location"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}

# keep prompts concise per spec
SYSTEM_PROMPT_ANALYZE = """تو یک متخصص تولید سوال ارزیابی برای سیستم RAG فارسی هستی.
فقط از متن داده‌شده استفاده کن؛ از دانش بیرونی استفاده نکن.
شناسه‌ها را جعل نکن؛ فقط parent_id و child_idهای داده‌شده را برگردان.
خروجی باید JSON معتبر باشد و هیچ متن اضافی نداشته باشد."""

USER_TEMPLATE = """شناسه صفحه: {page_id}
عنوان صفحه: {page_title}
دسته صفحه (برای نمونه‌گیری است، دسته سوال نیست): {page_categories}
دسته‌های پیشنهادی برای توازن داده‌ها (فقط راهنما، غیرالزامی): {preferred_categories}

بخش‌های صفحه (هر بخش یک parent و فرزندان child آن):
{blocks}

دسته‌های سوال مجاز:
- factual: پرسش واقعی/اطلاعاتی درباره ویژگی‌ها، ثبت، کاربری
- architecture: پرسش درباره معماری، سازه، مصالح، سبک، اجزا
- history: پرسش درباره تاریخ ساخت، دوره، رویدادها، بازسازی
- entity_relation: پرسش درباره ارتباط اشخاص/سازمان‌ها/بناها
- location: پرسش درباره موقعیت مکانی، استان/شهر/آدرس

وظیفه:
1) مشخص کن کدام دسته‌ها واقعاً با شواهد متنی کافی پشتیبانی می‌شوند. دسته‌ای را اجبار نکن؛ دسته‌های پیشنهادی فقط در صورت پشتیبانی شواهد اولویت دارند.
2) اگر هیچ دسته‌ای به‌خوبی پشتیبانی نمی‌شود، برگردان:
{{"supported_categories": [], "selected_category": null, "reason": "توضیح کوتاه"}}

3) در غیر این صورت، genau یک دسته را که بهترین پشتیبانی را دارد انتخاب کن و یک سوال تولید کن:
- parent_ids: لیست شناسه parentهای واقعاً مرتبط (حداقل 1)
- child_ids: لیست شناسه childهای مرتبط (حداقل 1)
- supporting_text: نقل دقیق کوتاه از متن که جواب را پشتیبانی می‌کند
- question: یک سوال طبیعی فارسی (پایان با ؟)
- reference_answer: پاسخ کوتاه مرجع فارسی
- difficulty: easy | medium | hard

فرمت JSON مورد انتظار (نمونه):
{{
  "supported_categories": ["history", "location"],
  "selected_category": "history",
  "parent_ids": ["10001#lead#parent_000"],
  "child_ids": ["10001#lead#parent_000#child_000"],
  "supporting_text": "متن عیناً از منبع...",
  "question": "این بنا در چه دوره‌ای ساخته شده است؟",
  "reference_answer": "دوره صفویه",
  "difficulty": "medium"
}}

قوانین:
- parent_ids و child_ids باید دقیقاً از فهرست داده‌شده باشند.
- هر child باید متعلق به یکی از parentهای انتخاب‌شده باشد و هر دو متعلق به همین page_id باشند.
- اگر طول متن زیاد است، فقط بخش‌های داده‌شده را تحلیل کن.
"""

INSUFFICIENT_SYSTEM = """تو متخصص تولید سوالات ارزیابی هستی. فقط JSON برگردان."""

INSUFFICIENT_USER = """حدود 18 سوال فارسی تولید کن که درباره میراث فرهنگی ایران plausibly به نظر برسند اما پاسخ‌شان در یک مجموعه ویکی‌پدیای بناهای ایرانی به‌احتمال زیاد یافت نشود (مثلاً جزئیات بسیار خصوصی، رویدادهای بسیار اخیر، یا اطلاعات محرمانه).
هر سوال باید طبیعی و کوتاه باشد.
خروجی JSON آرایه‌ای با 18 شیء به شکل زیر:
{{"question": "...", "reference_answer": "اطلاعات کافی در منابع یافت نشد.", "query_category": "insufficient_information", "difficulty": "medium"}}
فقط JSON برگردان، بدون توضیح اضافی.
"""

MAX_CHARS_PER_PROMPT = 16000  # conservative for low-cost models
CACHE_FILENAME = ".eval_llm_cache.json"


def _load_env() -> None:
    if _load_project_env is not None:
        try:
            _load_project_env()
        except Exception:
            load_dotenv()
    else:
        load_dotenv()
        # walk up for .env
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
        raise RuntimeError("OPENROUTER_API_KEY missing for evaluation generation")
    if not model:
        model = "openrouter/free"
    return api_key, base_url, model


def _format_blocks(parents: list[dict[str, Any]], children_by_parent: dict[str, list[dict[str, Any]]]) -> str:
    lines: list[str] = []
    for p in parents:
        pid = p.get("parent_id", "")
        heading = p.get("section_heading") or "بدون عنوان"
        level = p.get("section_level", "")
        text = (p.get("text") or "").strip()
        tokens = p.get("tokens", "")
        lines.append(f"[PARENT] parent_id={pid} | heading={heading} | level={level} | tokens={tokens}")
        lines.append(f"TEXT: {text}")
        kids = children_by_parent.get(pid, [])
        for c in kids:
            cid = c.get("chunk_id", "")
            ctokens = c.get("tokens", "")
            ctext = (c.get("text") or "").strip()
            lines.append(f"  [CHILD] child_id={cid} | tokens={ctokens}")
            lines.append(f"  TEXT: {ctext}")
        lines.append("---")
    return "\n".join(lines)


def _group_parents(parents: list[dict[str, Any]], children_by_parent: dict[str, list[dict[str, Any]]], max_chars: int = MAX_CHARS_PER_PROMPT) -> list[list[dict[str, Any]]]:
    """Split parents into groups so each prompt stays under max_chars."""
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_chars = 0
    for p in parents:
        # estimate chars for this parent + its children
        t = len((p.get("text") or ""))
        for c in children_by_parent.get(p.get("parent_id", ""), []):
            t += len((c.get("text") or ""))
        t += 200  # overhead
        if cur and cur_chars + t > max_chars:
            groups.append(cur)
            cur = []
            cur_chars = 0
        cur.append(p)
        cur_chars += t
    if cur:
        groups.append(cur)
    return groups if groups else [parents]


def _extract_json(text: str) -> Any:
    """Extract first JSON object/array from text."""
    text = text.strip()
    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # find json block via regex
    # look for ```json
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # find first { .. } or [ .. ]
    # greedy first object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    raise ValueError(f"Cannot parse JSON from LLM output: {text[:500]}")


def _normalize_for_match(text: str) -> str:
    """Normalize Persian text for substring verification.

    Collapses whitespace, unifies Arabic/Persian character variants
    (ي->ی, ك->ک) and digit variants, drops ZWNJ and diacritics.
    """
    text = text.lower()
    # unify Arabic and Persian variants
    table = str.maketrans(
        {
            "ي": "ی",
            "ك": "ک",
            "ة": "ه",
            "أ": "ا",
            "إ": "ا",
            "ؤ": "و",
            "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
            "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
            "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
            "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
            "‌": " ",  # ZWNJ -> space
            "‏": "",  # RLM
            "‎": "",  # LRM
            "ً": "", "ٌ": "", "ٍ": "",  # diacritics
            "َ": "", "ُ": "", "ِ": "", "ّ": "", "ْ": "",
        }
    )
    text = text.translate(table)
    text = re.sub(r"[\s\[\]()«»\"'.,،؛:;!?؟\-–—*_]+", " ", text)
    return " ".join(text.split())


def verify_supporting_text(
    supporting_text: str,
    child_ids: list[str],
    children_by_id: dict[str, dict[str, Any]],
) -> bool:
    """Check supporting_text appears inside the selected child chunks."""
    if not supporting_text or not child_ids:
        return False
    sup_norm = _normalize_for_match(supporting_text)
    if not sup_norm:
        return False
    for cid in child_ids:
        child = children_by_id.get(cid)
        if child is None:
            continue
        child_norm = _normalize_for_match(str(child.get("text") or ""))
        if sup_norm in child_norm:
            return True
    return False


def _choose_result(
    results: list[dict[str, Any]],
    preferred_categories: list[str] | None = None,
) -> dict[str, Any] | None:
    """Pick the final candidate from per-group results.

    Prefers results whose category matches the (advisory) preferred list,
    in preference order; falls back to the first valid result.
    """
    if not results:
        return None
    preferred = [c for c in (preferred_categories or []) if c in VALID_QUERY_CATEGORIES]
    for cat in preferred:
        for r in results:
            if r.get("selected_category") == cat:
                return r
    return results[0]


def _build_chat_model(api_key: str, base_url: str, model: str, temperature: float = 0.2, max_tokens: int = 1200):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        timeout=120,
        max_retries=2,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        use_responses_api=False,
    )


class EvaluationGenerator:
    """Wraps LLM calls for candidate generation with caching & validation."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        cache_path: Path | None = None,
        chat_model: Any | None = None,
    ) -> None:
        if chat_model is not None:
            self.chat_model = chat_model
            self.model_name = getattr(chat_model, "model_name", model or "mock")
        else:
            if api_key is None or base_url is None or model is None:
                ak, bu, mo = _get_openrouter_config()
                api_key = api_key or ak
                base_url = base_url or bu
                model = model or mo
            self.model_name = model  # type: ignore
            self.chat_model = _build_chat_model(api_key, base_url, model, temperature, max_tokens)  # type: ignore

        from langchain_core.prompts import ChatPromptTemplate

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT_ANALYZE),
                ("human", USER_TEMPLATE),
            ]
        )
        self.chain = (self.prompt | self.chat_model).with_config({"run_name": "eval.candidate.generate"})

        self.cache_path = cache_path
        self._cache: dict[str, Any] = {}
        if cache_path is not None and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def _cache_get(self, page_id: str) -> Any | None:
        return self._cache.get(page_id)

    def _cache_set(self, page_id: str, value: Any) -> None:
        self._cache[page_id] = value
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            # atomic write via temp
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.cache_path)

    def generate_for_page(
        self,
        page_info: dict[str, Any],
        force: bool = False,
        preferred_categories: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Generate one candidate for a page. Returns raw validated LLM dict or None if skipped.

        preferred_categories is advisory only: it steers which supported category is
        chosen but never forces an unsupported category.
        """
        page_id = str(page_info["page_id"])
        if not force:
            cached = self._cache_get(page_id)
            if cached is not None:
                # cached may be None meaning previously determined no candidate
                # we store {"_skip": True} for skip
                if isinstance(cached, dict) and cached.get("_skip"):
                    return None
                return cached

        parents: list[dict[str, Any]] = page_info.get("parents", [])
        children: list[dict[str, Any]] = page_info.get("children", [])
        if not parents:
            self._cache_set(page_id, {"_skip": True, "reason": "no parents"})
            return None

        children_by_parent: dict[str, list[dict[str, Any]]] = {}
        for c in children:
            pid = c.get("parent_id", "")
            children_by_parent.setdefault(pid, []).append(c)
        children_by_id = {c["chunk_id"]: c for c in children}

        valid_parent_ids = {p["parent_id"] for p in parents}
        valid_child_ids = {c["chunk_id"] for c in children}
        child_to_parent = {c["chunk_id"]: c["parent_id"] for c in children}

        # grouping for large pages
        groups = _group_parents(parents, children_by_parent)

        # analyze ALL groups before choosing the final candidate
        group_results: list[dict[str, Any]] = []

        for group in groups:
            blocks = _format_blocks(group, children_by_parent)
            page_title = page_info.get("page_title", "")
            page_cats = f"{page_info.get('final_class_fa','')} / {page_info.get('final_class_en','')} ({page_info.get('final_class_code','')})"
            pref_str = ", ".join(preferred_categories) if preferred_categories else "ندارد"

            try:
                resp = self.chain.invoke(
                    {
                        "page_id": page_id,
                        "page_title": page_title,
                        "page_categories": page_cats,
                        "preferred_categories": pref_str,
                        "blocks": blocks,
                    }
                )
                # AIMessage
                content = resp.content if hasattr(resp, "content") else str(resp)
                if isinstance(content, list):
                    # langchain may return list of blocks
                    parts = []
                    for b in content:
                        if isinstance(b, str):
                            parts.append(b)
                        elif isinstance(b, dict) and "text" in b:
                            parts.append(str(b["text"]))
                    content = "\n".join(parts)
                content = str(content).strip()
                data = _extract_json(content)
            except Exception as e:
                # log and try next group
                data = {"_error": str(e)}

            # validate structure
            if not isinstance(data, dict):
                continue
            if data.get("_error"):
                continue
            if data.get("_skip"):
                continue

            sc = data.get("supported_categories", [])
            sel = data.get("selected_category")

            if not sc and sel is None:
                # no candidate for this group
                continue
            # if sel is None but sc non-empty, skip
            if sel is None:
                continue
            if sel not in VALID_QUERY_CATEGORIES:
                continue

            parent_ids = data.get("parent_ids", [])
            child_ids = data.get("child_ids", [])
            # normalize to list
            if isinstance(parent_ids, str):
                parent_ids = [parent_ids]
            if isinstance(child_ids, str):
                child_ids = [child_ids]
            if not isinstance(parent_ids, list) or not parent_ids:
                continue
            if not isinstance(child_ids, list) or not child_ids:
                continue

            # validate IDs
            if not all(pid in valid_parent_ids for pid in parent_ids):
                continue
            if not all(cid in valid_child_ids for cid in child_ids):
                continue
            # child must belong to selected parent/page
            if not all(child_to_parent.get(cid) in set(parent_ids) for cid in child_ids):
                # allow if child parent is among valid parents but not selected? spec says child belongs to selected parent/page
                # so enforce
                continue

            q = (data.get("question") or "").strip()
            a = (data.get("reference_answer") or "").strip()
            sup = (data.get("supporting_text") or "").strip()
            diff = (data.get("difficulty") or "").strip().lower()

            if not q or not a or not sup:
                continue
            if diff not in VALID_DIFFICULTIES:
                diff = "medium"
                data["difficulty"] = diff

            # supporting_text must exist inside the selected child texts
            if not verify_supporting_text(sup, [str(x) for x in child_ids], children_by_id):
                continue

            # valid result for this group (kept; final choice after all groups)
            group_results.append(
                {
                    "supported_categories": sc,
                    "selected_category": sel,
                    "parent_ids": [str(x) for x in parent_ids],
                    "child_ids": [str(x) for x in child_ids],
                    "supporting_text": sup,
                    "question": q,
                    "reference_answer": a,
                    "difficulty": diff,
                    "_raw": data,
                }
            )

        best_result = _choose_result(group_results, preferred_categories)

        if best_result is None:
            self._cache_set(page_id, {"_skip": True, "reason": "no supported category or invalid IDs"})
            return None

        self._cache_set(page_id, best_result)
        return best_result

    def generate_insufficient_candidates(
        self,
        count: int = 18,
    ) -> list[dict[str, Any]]:
        """Generate insufficient_information candidates (no gold IDs)."""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", INSUFFICIENT_SYSTEM), ("human", INSUFFICIENT_USER)]
        )
        chain = (prompt | self.chat_model).with_config({"run_name": "eval.insufficient.generate"})
        try:
            resp = chain.invoke({})
            content = resp.content if hasattr(resp, "content") else str(resp)
            if isinstance(content, list):
                parts = []
                for b in content:
                    if isinstance(b, str):
                        parts.append(b)
                    elif isinstance(b, dict) and "text" in b:
                        parts.append(str(b["text"]))
                content = "\n".join(parts)
            data = _extract_json(str(content))
            # data should be list
            if isinstance(data, dict) and "questions" in data:
                data = data["questions"]
            if not isinstance(data, list):
                raise ValueError("insufficient output not list")
            out: list[dict[str, Any]] = []
            for idx, item in enumerate(data[:count]):
                q = (item.get("question") or f"سوال ناموجود {idx+1}").strip()
                diff = (item.get("difficulty") or "medium").strip().lower()
                if diff not in VALID_DIFFICULTIES:
                    diff = "medium"
                out.append(
                    {
                        "id": f"eval_insufficient_{idx+1:03d}",
                        "question": q,
                        "reference_answer": "اطلاعات کافی در منابع یافت نشد.",
                        "query_category": "insufficient_information",
                        "difficulty": diff,
                        "answerable": False,
                        "review_status": "pending",
                    }
                )
            # pad if needed
            while len(out) < count:
                out.append(
                    {
                        "id": f"eval_insufficient_{len(out)+1:03d}",
                        "question": f"جزئیات محرمانه مالک فعلی بنای تاریخی چیست؟ {len(out)+1} ؟",
                        "reference_answer": "اطلاعات کافی در منابع یافت نشد.",
                        "query_category": "insufficient_information",
                        "difficulty": "hard",
                        "answerable": False,
                        "review_status": "pending",
                    }
                )
            return out[:count]
        except Exception:
            # fallback deterministic templates
            templates = [
                "هزینه دقیق بازسازی این بنا در سال ۱۴۰۳ چقدر بوده است؟",
                "شماره تماس دفتر مدیریت فعلی این مجموعه چیست؟",
                "نام حساب بانکی هیئت امنای این بنا چیست؟",
                "برنامه دقیق مراسم خصوصی هفته آینده در این مکان چیست؟",
                "رمز عبور وای‌فای این مجموعه چیست؟",
                "درآمد ماهانه فعلی این بنای تاریخی چقدر است؟",
                "نظر شخصی و اعلام‌نشده شهردار فعلی درباره تخریب این بنا چیست؟",
                "جزئیات پرونده حقوقی محرمانه مربوط به مالکیت این بنا چیست؟",
                "پیش‌بینی قیمت زمین اطراف این بنا در سال آینده چیست؟",
                "اطلاعات پزشکی خصوصی متولی فعلی این امامزاده چیست؟",
                "لیست کامل مهمانان خصوصی مراسم افتتاحیه چیست؟",
                "کد ملی معمار اصلی این بنا چیست؟",
                "پسورد دوربین‌های مداربسته این مجموعه چیست؟",
                "برنامه سفر محرمانه وزیر به این محوطه در هفته آینده چیست؟",
                "جزئیات قرارداد محرمانه پیمانکار مرمت چیست؟",
                "موقعیت دقیق گنج پنهان ادعاشده در این قلعه کجاست؟",
                "اطلاعات حساب بانکی موقوفه این مسجد چیست؟",
                "نظر شخصی منتشرنشده باستان‌شناس کاوشگر این تپه چیست؟",
                "تعداد دقیق بازدیدکنندگان فردا در این سایت چقدر خواهد بود؟",
                "اطلاعات محرمانه نگهبانی شبانه این کاخ چیست؟",
            ]
            out = []
            for i in range(count):
                q = templates[i % len(templates)]
                out.append(
                    {
                        "id": f"eval_insufficient_{i+1:03d}",
                        "question": q,
                        "reference_answer": "اطلاعات کافی در منابع یافت نشد.",
                        "query_category": "insufficient_information",
                        "difficulty": "medium",
                        "answerable": False,
                        "review_status": "pending",
                    }
                )
            return out
