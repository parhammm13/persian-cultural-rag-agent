from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from src.database.models.conversation import Conversation


class ConversationRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create(self, *, user_id: int, title: str | None) -> Conversation:
        conversation = Conversation(
            user_id=user_id,
            title=title,
        )
        self._db.add(conversation)
        self._db.flush()
        self._db.refresh(conversation)
        return conversation

    def get_for_user(
        self,
        *,
        conversation_id: int,
        user_id: int,
    ) -> Conversation | None:
        statement = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        return self._db.scalar(statement)

    def list_for_user(
        self,
        *,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[Conversation, ...]:
        statement = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return tuple(self._db.scalars(statement).all())

    def delete_for_user(
        self,
        *,
        conversation_id: int,
        user_id: int,
    ) -> bool:
        statement = delete(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        result = self._db.execute(statement)
        return bool(result.rowcount)
