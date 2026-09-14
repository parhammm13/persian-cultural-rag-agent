from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from src.database.models.message import Message
from src.services.chat_service import ChatService


@dataclass
class FakeRAGResult:
    answer: str


class FakeRAGService:
    def __init__(self) -> None:
        self.received_queries: list[str] = []

    def query(
        self,
        query: str,
    ) -> FakeRAGResult:
        self.received_queries.append(query)

        return FakeRAGResult(
            answer="پاسخ تست"
        )


class FakeContextualizer:
    def __init__(
        self,
        rewritten_query: str,
    ) -> None:
        self.rewritten_query = rewritten_query

        self.received_message: str | None = None
        self.received_history: Sequence[Message] | None = None

    def contextualize(
        self,
        *,
        message: str,
        history: Sequence[Message],
    ) -> str:
        self.received_message = message
        self.received_history = history

        return self.rewritten_query