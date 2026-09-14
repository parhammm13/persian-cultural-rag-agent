from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from src.core.security import (
    JWTSettings,
    create_access_token,
    hash_password,
    verify_password,
)
from src.database.models.user import User
from src.database.repositories.users import UserRepository


logger = logging.getLogger("persian_cultural_rag.api")


class EmailAlreadyRegisteredError(ValueError):
    pass


class InvalidCredentialsError(ValueError):
    pass


class AuthService:
    """User registration and credential verification.

    Ownership of persistence stays in repositories; this service owns the
    auth use cases (hashing, credential checks, token issuance).
    """

    def __init__(
        self,
        db: Session,
        settings: JWTSettings,
    ) -> None:
        self._db = db
        self._settings = settings
        self._users = UserRepository(db)

    def register(self, *, email: str, password: str) -> User:
        normalized_email = email.strip().lower()
        if self._users.get_by_email(normalized_email) is not None:
            raise EmailAlreadyRegisteredError(
                f"Email is already registered: {normalized_email}"
            )

        user = self._users.create(
            email=normalized_email,
            password_hash=hash_password(password),
        )
        self._db.commit()
        self._db.refresh(user)

        logger.info(
            "user_registered user_id=%s",
            user.id,
        )
        return user

    def login(self, *, email: str, password: str) -> tuple[User, str]:
        """Verify credentials and issue a JWT access token.

        Raises InvalidCredentialsError without revealing whether the email
        exists or the password was wrong.
        """
        normalized_email = email.strip().lower()
        user = self._users.get_by_email(normalized_email)
        if user is None or not verify_password(
            password,
            user.password_hash,
        ):
            logger.warning("login_failed_invalid_credentials")
            raise InvalidCredentialsError("Invalid email or password.")

        token = create_access_token(
            user_id=user.id,
            settings=self._settings,
        )
        logger.info("user_logged_in user_id=%s", user.id)
        return user, token


__all__ = [
    "AuthService",
    "EmailAlreadyRegisteredError",
    "InvalidCredentialsError",
]
