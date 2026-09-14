from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import func, update
from sqlalchemy.orm import Session

from src.database.models.conversation import Conversation
from src.database.models.message import Message, MessageRole
from src.database.repositories.messages import MessageRepository
from src.services.conversation_service import ConversationService
class RAGServiceProtocol(Protocol):
    def query(self, query: str) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class ChatResult:
    conversation_id: int
    user_message: Message
    assistant_message: Message
    rag_result: Any


class ChatService:
    """Persist chat turns around the existing stateless RAG pipeline.

    Current phase limitation: recent history is persisted and retrievable but is
    not yet used to contextualize the RAG query. Query contextualization belongs
    to the next routing/safeguard phase in the roadmap.
    """

    def __init__(
        self,
        db: Session,
        rag_service: RAGServiceProtocol,
    ) -> None:
        self._db = db
        self._rag_service = rag_service
        self._conversations = ConversationService(db)
        self._messages = MessageRepository(db)

    def chat(
        self,
        *,
        user_id: int,
        conversation_id: int,
        message: str,
    ) -> ChatResult:
        normalized_message = self._normalize_message(message)

        conversation = self._conversations.get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        user_message = self._messages.create(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content=normalized_message,
        )
        self._touch_conversation(conversation.id)
        self._db.commit()
        self._db.refresh(user_message)

        # Do not hold an open DB transaction while calling external RAG services.
        rag_result = self._rag_service.query(normalized_message)
        answer = getattr(rag_result, "answer", None)
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("RAG result does not contain a non-empty answer.")

        assistant_message = self._messages.create(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=answer.strip(),
        )
        self._touch_conversation(conversation.id)
        self._db.commit()
        self._db.refresh(assistant_message)

        return ChatResult(
            conversation_id=conversation.id,
            user_message=user_message,
            assistant_message=assistant_message,
            rag_result=rag_result,
        )

    def _touch_conversation(self, conversation_id: int) -> None:
        # Force updated_at to move even when only child rows are inserted.
        statement = (
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(updated_at=func.now())
        )
        self._db.execute(statement)

    @staticmethod
    def _normalize_message(message: str) -> str:
        if not isinstance(message, str):
            raise TypeError("message must be a string")
        normalized = message.strip()
        if not normalized:
            raise ValueError("message must not be empty")
        return normalized
