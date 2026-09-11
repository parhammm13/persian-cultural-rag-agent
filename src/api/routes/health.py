from fastapi import APIRouter

from src.api.schemas.common import HealthResponse


router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="persian-cultural-rag-api",
        version="0.2.0",
    )
