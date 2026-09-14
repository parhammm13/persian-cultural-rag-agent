from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, Field

from .common import StrictAPIModel


class ConversationCreateRequest(StrictAPIModel):
    title: str | None = Field(default=None, max_length=255)


class ConversationResponse(StrictAPIModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )

    id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    title: str | None = None
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(StrictAPIModel):
    items: list[ConversationResponse]
    count: int = Field(ge=0)


class MessageResponse(StrictAPIModel):
    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
        use_enum_values=True,
    )

    id: int = Field(gt=0)
    conversation_id: int = Field(gt=0)
    role: str
    content: str
    created_at: datetime


class MessageListResponse(StrictAPIModel):
    items: list[MessageResponse]
    count: int = Field(ge=0)
