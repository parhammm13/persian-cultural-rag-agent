from __future__ import annotations

from sqlalchemy.orm import Session

from src.database.models.conversation import Conversation
from src.database.models.message import Message
from src.database.repositories.conversations import ConversationRepository
from src.database.repositories.messages import MessageRepository
from src.database.repositories.users import UserRepository


class UserNotFoundError(LookupError):
    pass


class ConversationNotFoundError(LookupError):
    pass


class ConversationService:
    """Application service for conversation and history persistence."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._users = UserRepository(db)
        self._conversations = ConversationRepository(db)
        self._messages = MessageRepository(db)

    def create_conversation(
        self,
        *,
        user_id: int,
        title: str | None = None,
    ) -> Conversation:
        self._validate_user(user_id)
        normalized_title = self._normalize_optional_title(title)

        conversation = self._conversations.create(
            user_id=user_id,
            title=normalized_title,
        )
        self._db.commit()
        self._db.refresh(conversation)
        return conversation

    def list_conversations(
        self,
        *,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[Conversation, ...]:
        self._validate_user(user_id)
        self._validate_pagination(limit=limit, offset=offset)
        return self._conversations.list_for_user(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

    def get_conversation(
        self,
        *,
        user_id: int,
        conversation_id: int,
    ) -> Conversation:
        conversation = self._conversations.get_for_user(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if conversation is None:
            raise ConversationNotFoundError(
                f"Conversation {conversation_id} was not found for user {user_id}."
            )
        return conversation

    def delete_conversation(
        self,
        *,
        user_id: int,
        conversation_id: int,
    ) -> None:
        deleted = self._conversations.delete_for_user(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not deleted:
            self._db.rollback()
            raise ConversationNotFoundError(
                f"Conversation {conversation_id} was not found for user {user_id}."
            )
        self._db.commit()

    def list_messages(
        self,
        *,
        user_id: int,
        conversation_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[Message, ...]:
        self.get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        self._validate_pagination(limit=limit, offset=offset)
        return self._messages.list_for_conversation(
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )

    def recent_history(
        self,
        *,
        user_id: int,
        conversation_id: int,
        limit: int = 10,
    ) -> tuple[Message, ...]:
        self.get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not 1 <= limit <= 50:
            raise ValueError("history limit must be between 1 and 50")
        return self._messages.list_recent(
            conversation_id=conversation_id,
            limit=limit,
        )

    def _validate_user(self, user_id: int) -> None:
        if user_id <= 0 or self._users.get_by_id(user_id) is None:
            raise UserNotFoundError(f"User {user_id} was not found.")

    @staticmethod
    def _normalize_optional_title(title: str | None) -> str | None:
        if title is None:
            return None
        normalized = title.strip()
        return normalized or None

    @staticmethod
    def _validate_pagination(*, limit: int, offset: int) -> None:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("offset must be >= 0")
