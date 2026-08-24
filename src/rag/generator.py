"""Answer generation through OpenRouter; contains no self-imports."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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
    """Small, dependency-light client for OpenRouter generation."""

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
        request_opener: Callable[..., Any] | None = None,
    ) -> None:
        api_key = api_key.strip()
        base_url = base_url.strip().rstrip("/")
        model = model.strip()
        system_prompt = system_prompt.strip()

        if not api_key:
            raise ValueError("api_key must not be empty.")
        if not base_url:
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

        self.api_key = api_key
        self.endpoint = f"{base_url}/chat/completions"
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.request_opener = request_opener or urlopen

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

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": (
                        f"پرسش:\n{query}\n\n"
                        f"منابع:\n{context}\n\n"
                        "پاسخ مستند:"
                    ),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        data = self._post(payload)
        answer = self._extract_answer(data)
        actual_model = data.get("model", self.model)
        usage = data.get("usage", {})

        return GenerationResult(
            answer=answer,
            model=str(actual_model),
            usage=usage if isinstance(usage, dict) else {},
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            request = Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers=headers,
            )

            try:
                with self.request_opener(
                    request,
                    timeout=self.timeout,
                ) as response:
                    body = response.read().decode("utf-8")

                data = json.loads(body)
                if not isinstance(data, dict):
                    raise OpenRouterError("OpenRouter returned invalid JSON data.")

                return data

            except HTTPError as error:
                last_error = error
                if error.code not in {429, 500, 502, 503, 504}:
                    raise OpenRouterError(
                        f"OpenRouter request failed with HTTP {error.code}."
                    ) from error
            except (URLError, TimeoutError) as error:
                last_error = error
            except json.JSONDecodeError as error:
                raise OpenRouterError("OpenRouter returned malformed JSON.") from error

            if attempt < self.max_retries:
                time.sleep(2**attempt)

        raise OpenRouterError(
            f"OpenRouter request failed after {self.max_retries + 1} attempts."
        ) from last_error

    @staticmethod
    def _extract_answer(data: dict[str, Any]) -> str:
        choices = data.get("choices")

        if not isinstance(choices, list) or not choices:
            raise OpenRouterError("OpenRouter response has no choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise OpenRouterError("OpenRouter returned an invalid choice.")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise OpenRouterError("OpenRouter response has no message.")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OpenRouterError("OpenRouter returned an empty answer.")

        return content.strip()


__all__ = [
    "GenerationResult",
    "OpenRouterError",
    "OpenRouterGenerator",
]