from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.user import User


class UserRepository:
    """Read-side access needed by the pre-auth chat-history phase."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def get_by_id(self, user_id: int) -> User | None:
        if user_id <= 0:
            return None
        return self._db.get(User, user_id)

    def get_by_email(self, email: str) -> User | None:
        normalized_email = email.strip().lower()
        if not normalized_email:
            return None
        statement = select(User).where(User.email == normalized_email)
        return self._db.scalar(statement)
