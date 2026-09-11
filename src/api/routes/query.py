from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_rag_service
from src.api.schemas.query import QueryRequest, QueryResponse
from src.services.rag_service import RAGService


router = APIRouter(
    prefix="/query",
    tags=["rag"],
)


@router.post(
    "",
    response_model=QueryResponse,
    summary="Run a standalone RAG query",
)
def query(
    payload: QueryRequest,
    service: Annotated[
        RAGService,
        Depends(get_rag_service),
    ],
) -> QueryResponse:
    result = service.query(payload.query)

    return QueryResponse.model_validate(
        result,
        from_attributes=True,
    )
