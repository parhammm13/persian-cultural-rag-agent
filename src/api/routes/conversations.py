from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from src.api.dependencies import get_conversation_service
from src.api.schemas.conversation import (
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationResponse,
    MessageListResponse,
)
from src.services.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
    UserNotFoundError,
)


router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
)

ConversationServiceDep = Annotated[
    ConversationService,
    Depends(get_conversation_service),
]


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    request: ConversationCreateRequest,
    service: ConversationServiceDep,
) -> ConversationResponse:
    try:
        conversation = service.create_conversation(
            user_id=request.user_id,
            title=request.title,
        )
    except UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail="user_not_found") from exc

    return ConversationResponse.model_validate(conversation)


@router.get(
    "",
    response_model=ConversationListResponse,
)
def list_conversations(
    service: ConversationServiceDep,
    user_id: Annotated[int, Query(gt=0)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ConversationListResponse:
    try:
        conversations = service.list_conversations(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
    except UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail="user_not_found") from exc

    items = [ConversationResponse.model_validate(item) for item in conversations]
    return ConversationListResponse(items=items, count=len(items))


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
)
def get_conversation(
    conversation_id: int,
    service: ConversationServiceDep,
    user_id: Annotated[int, Query(gt=0)],
) -> ConversationResponse:
    try:
        conversation = service.get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="conversation_not_found") from exc

    return ConversationResponse.model_validate(conversation)


@router.get(
    "/{conversation_id}/messages",
    response_model=MessageListResponse,
)
def list_messages(
    conversation_id: int,
    service: ConversationServiceDep,
    user_id: Annotated[int, Query(gt=0)],
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> MessageListResponse:
    try:
        messages = service.list_messages(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="conversation_not_found") from exc

    return MessageListResponse(
        items=list(messages),
        count=len(messages),
    )


@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_conversation(
    conversation_id: int,
    service: ConversationServiceDep,
    user_id: Annotated[int, Query(gt=0)],
) -> Response:
    try:
        service.delete_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="conversation_not_found") from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
