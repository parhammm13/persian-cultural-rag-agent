from typing import Annotated

from fastapi import APIRouter, Depends

from src.api.dependencies import get_retrieval_service
from src.api.schemas.common import ErrorResponse
from src.api.schemas.retrieval import RetrieveRequest, RetrieveResponse
from src.services.retrieval_service import RetrievalService


router = APIRouter(
    prefix="/retrieve",
    tags=["rag"],
)


@router.post(
    "",
    response_model=RetrieveResponse,
    responses={
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Run retrieval and reranking without generation",
)
def retrieve(
    payload: RetrieveRequest,
    service: Annotated[
        RetrievalService,
        Depends(get_retrieval_service),
    ],
) -> RetrieveResponse:
    results = service.retrieve(
        payload.query,
        top_k=payload.top_k,
    )

    return RetrieveResponse(
        query=payload.query,
        results=list(results),
        result_count=len(results),
    )
