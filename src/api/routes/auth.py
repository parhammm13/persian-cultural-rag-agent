from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import CurrentUser, get_auth_service
from src.api.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from src.database.models.user import User
from src.services.auth_service import (
    AuthService,
    EmailAlreadyRegisteredError,
    InvalidCredentialsError,
)


router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)

AuthServiceDep = Annotated[
    AuthService,
    Depends(get_auth_service),
]


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    request: RegisterRequest,
    service: AuthServiceDep,
) -> UserResponse:
    try:
        user = service.register(
            email=str(request.email),
            password=request.password,
        )
    except EmailAlreadyRegisteredError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="email_already_registered",
        ) from exc

    return UserResponse.model_validate(user)


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    request: LoginRequest,
    service: AuthServiceDep,
) -> TokenResponse:
    try:
        _, token = service.login(
            email=str(request.email),
            password=request.password,
        )
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_credentials",
        ) from exc

    return TokenResponse(access_token=token)


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(user: CurrentUser) -> UserResponse:
    return UserResponse.model_validate(user)


__all__ = ["router"]
