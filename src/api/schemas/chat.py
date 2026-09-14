from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictAPIModel
from .conversation import MessageResponse


class ChatRequest(StrictAPIModel):
    conversation_id: int = Field(gt=0)
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(StrictAPIModel):
    conversation_id: int = Field(gt=0)
    answer: str
    user_message: MessageResponse
    assistant_message: MessageResponse
    sources: list[Any] = Field(default_factory=list)
    route: str = Field(
        default="RAG_REQUIRED",
        description=(
            "Router decision for this turn: CHAT_ONLY, RAG_REQUIRED, "
            "or OUT_OF_DOMAIN."
        ),
    )
    standalone_query: str | None = Field(
        default=None,
        description=(
            "Rewritten standalone retrieval query when "
            "route is RAG_REQUIRED; otherwise null."
        ),
    )
    used_fallback: bool = Field(
        default=False,
        description=(
            "True when the router LLM call failed and ChatService fell back "
            "to the original message for RAG."
        ),
    )
