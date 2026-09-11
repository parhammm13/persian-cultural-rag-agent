from .health import router as health_router
from .query import router as query_router

__all__ = [
    "health_router",
    "query_router",
]
