"""Password hashing and JWT handling for API authentication.

Only the API/service layer may use this module. RAG, retrieval, and the
conversation router must never import it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

logger = logging.getLogger("persian_cultural_rag.api")

_TOKEN_TYPE = "access"
_MIN_SECRET_LENGTH = 32


@dataclass(frozen=True, slots=True)
class JWTSettings:
    secret_key: str = field(repr=False)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    @classmethod
    def from_env(cls) -> "JWTSettings":
        secret = os.getenv("JWT_SECRET_KEY", "")
        if len(secret) < _MIN_SECRET_LENGTH:
            raise RuntimeError(
                "JWT_SECRET_KEY must be configured with at least "
                f"{_MIN_SECRET_LENGTH} characters. Generate one with: "
                "python -c \"import secrets; "
                "print(secrets.token_urlsafe(48))\""
            )

        algorithm = os.getenv("JWT_ALGORITHM", "HS256").strip() or "HS256"
        expire_minutes = _positive_int_env(
            "JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
            default=60,
        )
        return cls(
            secret_key=secret,
            algorithm=algorithm,
            access_token_expire_minutes=expire_minutes,
        )


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be > 0.")
    return value


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt. Never log the result."""
    password_bytes = password.encode("utf-8")
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Compare a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except (ValueError, TypeError):
        logger.warning("password_verification_failed_malformed_hash")
        return False


def create_access_token(
    *,
    user_id: int,
    settings: JWTSettings,
    expires_minutes: int | None = None,
) -> str:
    """Create a signed JWT access token for an authenticated user."""
    now = datetime.now(timezone.utc)
    expires_delta = timedelta(
        minutes=expires_minutes
        if expires_minutes is not None
        else settings.access_token_expire_minutes
    )
    payload = {
        "sub": str(user_id),
        "type": _TOKEN_TYPE,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(
        payload,
        settings.secret_key,
        algorithm=settings.algorithm,
    )


class InvalidTokenError(ValueError):
    """Raised when a JWT is missing, malformed, expired, or of wrong type."""


def decode_access_token(*, token: str, settings: JWTSettings) -> int:
    """Validate a JWT access token and return the embedded user id."""
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.algorithm],
            options={"require": ["sub", "exp", "iat"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise InvalidTokenError("token expired") from exc
    except jwt.PyJWTError as exc:
        raise InvalidTokenError("token invalid") from exc

    if payload.get("type") != _TOKEN_TYPE:
        raise InvalidTokenError("token invalid")

    try:
        user_id = int(str(payload["sub"]))
    except (TypeError, ValueError) as exc:
        raise InvalidTokenError("token invalid") from exc
    if user_id <= 0:
        raise InvalidTokenError("token invalid")
    return user_id


__all__ = [
    "InvalidTokenError",
    "JWTSettings",
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "verify_password",
]
