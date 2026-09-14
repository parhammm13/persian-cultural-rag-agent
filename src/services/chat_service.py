from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

from sqlalchemy import func, update
from sqlalchemy.orm import Session

from src.database.models.conversation import Conversation
from src.database.models.message import Message, MessageRole
from src.database.repositories.messages import MessageRepository
from src.services.conversation_router import (
    ConversationRoute,
    ConversationRouter,
    HistoryTurn,
    RouteDecision,
)
from src.services.conversation_service import ConversationService


logger = logging.getLogger("persian_cultural_rag.api")


class RAGServiceProtocol(Protocol):
    def query(
        self,
        query: str,
    ) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class DirectResponse:
    """
    Small RAG-compatible response used for CHAT_ONLY and OUT_OF_DOMAIN.

    chat.py only requires .sources from rag_result, while the persisted
    assistant message is the source of the final answer.
    """

    answer: str
    sources: tuple[Any, ...] = ()
    retrieved_count: int = 0
    reranked_count: int = 0
    model: str | None = None
    usage: dict[str, Any] | None = None


class _NullSpan:
    def set_input(self, value: Any) -> None:
        pass

    def set_output(self, value: Any) -> None:
        pass

    def set_attribute(self, name: str, value: Any) -> None:
        pass

    def record_exception(self, error: Exception) -> None:
        pass


@dataclass(frozen=True, slots=True)
class ChatResult:
    conversation_id: int
    user_message: Message
    assistant_message: Message
    rag_result: Any
    route: str
    standalone_query: str | None = None
    used_fallback: bool = False


class ChatService:
    """
    Conversational application service.

    Responsibilities:
      1. validate conversation ownership,
      2. load recent persisted history,
      3. persist the original user message,
      4. route the turn,
      5. call RAG only when required,
      6. persist the assistant answer.

    RAG itself remains stateless.
    """

    DEFAULT_HISTORY_LIMIT = 8

    def __init__(
        self,
        db: Session,
        rag_service: RAGServiceProtocol,
        router: ConversationRouter,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        tracer: Any | None = None,
    ) -> None:
        if history_limit <= 0:
            raise ValueError(
                "history_limit must be > 0"
            )

        self._db = db
        self._rag_service = rag_service
        self._router = router
        self._history_limit = history_limit
        self._tracer = tracer

        self._conversations = ConversationService(db)
        self._messages = MessageRepository(db)

    def chat(
        self,
        *,
        user_id: int,
        conversation_id: int,
        message: str,
    ) -> ChatResult:
        normalized_message = self._normalize_message(
            message
        )

        conversation = self._conversations.get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        conversation_id_value = conversation.id

        # Load previous history BEFORE inserting the current message.
        previous_messages = self._conversations.recent_history(
            user_id=user_id,
            conversation_id=conversation_id_value,
            limit=self._history_limit,
        )

        history = self._snapshot_history(
            previous_messages
        )

        # Persist exactly what the user typed.
        user_message = self._messages.create(
            conversation_id=conversation_id_value,
            role=MessageRole.USER,
            content=normalized_message,
        )

        self._touch_conversation(
            conversation_id_value
        )

        self._db.commit()
        self._db.refresh(user_message)

        try:
            with self._router_span(
                message=normalized_message,
                history_count=len(history),
            ) as router_span:
                decision = self._router.route(
                    message=normalized_message,
                    history=history,
                )
                self._annotate_router_span(
                    router_span,
                    decision=decision,
                )

            logger.info(
                "chat_route_decided route=%s fallback=%s "
                "standalone_length=%s conversation_id=%s",
                decision.route.value,
                decision.used_fallback,
                len(decision.standalone_query or ""),
                conversation_id_value,
            )

            result = self._execute_decision(
                decision=decision,
            )

            answer = getattr(
                result,
                "answer",
                None,
            )

            if (
                not isinstance(answer, str)
                or not answer.strip()
            ):
                raise RuntimeError(
                    "Chat result does not contain a non-empty answer."
                )

            assistant_message = self._messages.create(
                conversation_id=conversation_id_value,
                role=MessageRole.ASSISTANT,
                content=answer.strip(),
            )

            self._touch_conversation(
                conversation_id_value
            )

            self._db.commit()
            self._db.refresh(assistant_message)

        except Exception:
            # The user turn was committed before external calls so we did
            # not hold a DB transaction open. If the downstream operation
            # fails, remove the unmatched turn so retries do not pollute
            # future conversational context.
            self._cleanup_failed_user_turn(
                user_message=user_message,
                conversation_id=conversation_id_value,
            )
            raise

        return ChatResult(
            conversation_id=conversation_id_value,
            user_message=user_message,
            assistant_message=assistant_message,
            rag_result=result,
            route=decision.route.value,
            standalone_query=decision.standalone_query,
            used_fallback=decision.used_fallback,
        )

    def _router_span(
        self,
        *,
        message: str,
        history_count: int,
    ) -> Any:
        """Phoenix span for routing/contextualization.

        Uses the injected tracer when available; otherwise a NoOp
        nullcontext so CHAT_ONLY / OUT_OF_DOMAIN turns no longer vanish
        silently from observability.
        """
        tracer = self._tracer
        start_span = getattr(
            tracer,
            "start_as_current_span",
            None,
        )
        if start_span is None:
            return nullcontext(_NullSpan())

        try:
            return start_span(
                "chat.router",
                openinference_span_kind="chain",
            )
        except TypeError:
            return start_span("chat.router")

    def _annotate_router_span(
        self,
        span: Any,
        *,
        decision: RouteDecision,
    ) -> None:
        set_attribute = getattr(span, "set_attribute", None)
        if set_attribute is None:
            return
        try:
            set_attribute("chat.route", decision.route.value)
            set_attribute("chat.fallback", decision.used_fallback)
            set_attribute(
                "chat.standalone_length",
                len(decision.standalone_query or ""),
            )
            if bool(getattr(self._router, "capture_content", False)):
                set_input = getattr(span, "set_input", None)
                if set_input is not None:
                    try:
                        set_input(
                            {
                                "standalone_query": decision.standalone_query,
                                "answer_length": len(decision.answer or ""),
                            }
                        )
                    except Exception:
                        pass
            set_output = getattr(span, "set_output", None)
            if set_output is not None:
                try:
                    set_output(
                        {
                            "route": decision.route.value,
                            "standalone_length": len(
                                decision.standalone_query or ""
                            ),
                            "fallback": decision.used_fallback,
                        }
                    )
                except Exception:
                    pass
        except Exception:
            pass

    def _execute_decision(
        self,
        *,
        decision: RouteDecision,
    ) -> Any:
        if decision.route is ConversationRoute.RAG_REQUIRED:
            query = (
                decision.standalone_query
                or ""
            ).strip()

            if not query:
                raise RuntimeError(
                    "RAG_REQUIRED decision has no standalone query."
                )

            logger.info(
                "chat_rag_dispatch query_length=%s fallback=%s",
                len(query),
                decision.used_fallback,
            )

            return self._rag_service.query(
                query
            )

        answer = (
            decision.answer
            or ""
        ).strip()

        if not answer:
            raise RuntimeError(
                f"{decision.route.value} decision has no answer."
            )

        return DirectResponse(
            answer=answer,
            sources=(),
            retrieved_count=0,
            reranked_count=0,
            model=None,
            usage={},
        )

    def _cleanup_failed_user_turn(
        self,
        *,
        user_message: Message,
        conversation_id: int,
    ) -> None:
        try:
            self._db.delete(user_message)

            self._touch_conversation(
                conversation_id
            )

            self._db.commit()

            logger.info(
                "failed_chat_user_turn_removed "
                "conversation_id=%s message_id=%s",
                conversation_id,
                getattr(user_message, "id", None),
            )

        except Exception:
            self._db.rollback()

            logger.exception(
                "failed_chat_user_turn_cleanup_failed "
                "conversation_id=%s message_id=%s",
                conversation_id,
                getattr(user_message, "id", None),
            )

    @staticmethod
    def _snapshot_history(
        messages: tuple[Message, ...],
    ) -> tuple[HistoryTurn, ...]:
        turns: list[HistoryTurn] = []

        for message in messages:
            role_raw = getattr(
                message.role,
                "value",
                message.role,
            )
            role = str(role_raw).strip().lower()
            content = str(message.content).strip()

            if (
                role in {"user", "assistant"}
                and content
            ):
                turns.append(
                    HistoryTurn(
                        role=role,
                        content=content,
                    )
                )

        return tuple(turns)

    def _touch_conversation(
        self,
        conversation_id: int,
    ) -> None:
        statement = (
            update(Conversation)
            .where(
                Conversation.id == conversation_id
            )
            .values(
                updated_at=func.now()
            )
        )

        self._db.execute(statement)

    @staticmethod
    def _normalize_message(
        message: str,
    ) -> str:
        if not isinstance(message, str):
            raise TypeError(
                "message must be a string"
            )

        normalized = message.strip()

        if not normalized:
            raise ValueError(
                "message must not be empty"
            )

        return normalized


__all__ = [
    "ChatResult",
    "ChatService",
    "DirectResponse",
]
