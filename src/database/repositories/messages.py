from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.message import Message, MessageRole


class MessageRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(
        self,
        *,
        conversation_id: int,
        role: MessageRole,
        content: str,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )
        self._db.add(message)
        self._db.flush()
        self._db.refresh(message)
        return message

    def list_for_conversation(
        self,
        *,
        conversation_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[Message, ...]:
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
            .limit(limit)
            .offset(offset)
        )
        return tuple(self._db.scalars(statement).all())

    def list_recent(
        self,
        *,
        conversation_id: int,
        limit: int = 10,
    ) -> tuple[Message, ...]:
        """Return the newest N messages in chronological order."""
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
        newest_first = tuple(self._db.scalars(statement).all())
        return tuple(reversed(newest_first))
