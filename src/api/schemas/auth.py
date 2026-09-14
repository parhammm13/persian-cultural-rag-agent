from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, EmailStr, Field

from .common import StrictAPIModel


class RegisterRequest(StrictAPIModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(StrictAPIModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(StrictAPIModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(StrictAPIModel):
    id: int = Field(gt=0)
    email: EmailStr
    created_at: datetime

    model_config = ConfigDict(
        extra="forbid",
        from_attributes=True,
    )
