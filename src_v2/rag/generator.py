"""Grounded answer generation through LangChain's OpenRouter integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

DEFAULT_SYSTEM_PROMPT = """شما دستیار دانش فرهنگ ایران هستید.
فقط با استفاده از منابع داده‌شده پاسخ بده.
اگر منابع برای پاسخ کافی نیستند، صریح بگو اطلاعات کافی پیدا نشد.
پاسخ را به زبان پرسش بنویس و برای ادعاها از ارجاع‌هایی مانند [1] استفاده کن.
هیچ منبع یا واقعیتی را جعل نکن."""


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter cannot return a valid answer."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    answer: str
    model: str
    usage: dict[str, Any]


class OpenRouterGenerator:
    """Generate grounded answers with a LangChain prompt/model runnable."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int = 120,
        max_retries: int = 2,
        temperature: float = 0.1,
        max_tokens: int = 700,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        chat_model: Runnable[Any, AIMessage] | None = None,
    ) -> None:
        api_key = api_key.strip()
        base_url = base_url.strip().rstrip("/")
        model = model.strip()
        system_prompt = system_prompt.strip()

        if not api_key and chat_model is None:
            raise ValueError("api_key must not be empty.")
        if not base_url and chat_model is None:
            raise ValueError("base_url must not be empty.")
        if not model:
            raise ValueError("model must not be empty.")
        if not system_prompt:
            raise ValueError("system_prompt must not be empty.")
        if timeout <= 0:
            raise ValueError("timeout must be > 0.")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0.")

        self.model_name = model

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "پرسش:\n{query}\n\n"
                    "منابع:\n{context}\n\n"
                    "پاسخ مستند:",
                ),
            ]
        )

        model_runnable = chat_model or ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            use_responses_api=False,
        )

        self.chain = (prompt | model_runnable).with_config(
            {"run_name": "rag.openrouter.generate"}
        )

    def generate(
        self,
        query: str,
        context: str,
    ) -> GenerationResult:
        query = query.strip()
        context = context.strip()

        if not query:
            raise ValueError("query must not be empty.")
        if not context:
            raise ValueError("context must not be empty.")

        response = self.chain.invoke(
            {
                "query": query,
                "context": context,
            }
        )

        if not isinstance(response, AIMessage):
            raise OpenRouterError("OpenRouter returned an invalid message type.")

        answer = self._extract_answer(response)
        model = (
            response.response_metadata.get("model_name")
            or response.response_metadata.get("model")
            or self.model_name
        )

        return GenerationResult(
            answer=answer,
            model=str(model),
            usage=self._normalize_usage(response.usage_metadata),
        )

    @staticmethod
    def _extract_answer(response: AIMessage) -> str:
        content = response.content

        if isinstance(content, str):
            answer = content.strip()
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            answer = "\n".join(parts).strip()
        else:
            answer = ""

        if not answer:
            raise OpenRouterError("OpenRouter returned an empty answer.")

        return answer

    @staticmethod
    def _normalize_usage(
        usage_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not usage_metadata:
            return {}

        usage: dict[str, Any] = {}
        key_map = {
            "input_tokens": "prompt_tokens",
            "output_tokens": "completion_tokens",
            "total_tokens": "total_tokens",
        }

        for source_key, target_key in key_map.items():
            value = usage_metadata.get(source_key)
            if isinstance(value, int) and not isinstance(value, bool):
                usage[target_key] = value

        for detail_key in (
            "input_token_details",
            "output_token_details",
        ):
            details = usage_metadata.get(detail_key)
            if isinstance(details, dict):
                usage[detail_key] = dict(details)

        return usage


__all__ = [
    "GenerationResult",
    "OpenRouterError",
    "OpenRouterGenerator",
]
