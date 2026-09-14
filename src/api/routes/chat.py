from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import CurrentUser, get_chat_service
from src.api.schemas.chat import ChatRequest, ChatResponse
from src.services.chat_service import ChatService
from src.services.conversation_service import ConversationNotFoundError


router = APIRouter(tags=["chat"])

ChatServiceDep = Annotated[
    ChatService,
    Depends(get_chat_service),
]


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    service: ChatServiceDep,
    user: CurrentUser,
) -> ChatResponse:
    try:
        result = service.chat(
            user_id=user.id,
            conversation_id=request.conversation_id,
            message=request.message,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail="conversation_not_found") from exc

    rag_result = result.rag_result
    sources = list(getattr(rag_result, "sources", ()) or ())

    return ChatResponse(
        conversation_id=result.conversation_id,
        answer=result.assistant_message.content,
        user_message=result.user_message,
        assistant_message=result.assistant_message,
        sources=sources,
        route=result.route,
        standalone_query=result.standalone_query,
        used_fallback=result.used_fallback,
    )
