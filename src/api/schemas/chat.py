from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictAPIModel
from .conversation import MessageResponse


class ChatRequest(StrictAPIModel):
    # Temporary pre-auth field. Replace with current_user.id in auth phase.
    user_id: int = Field(gt=0)
    conversation_id: int = Field(gt=0)
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(StrictAPIModel):
    conversation_id: int = Field(gt=0)
    answer: str
    user_message: MessageResponse
    assistant_message: MessageResponse
    sources: list[Any] = Field(default_factory=list)
