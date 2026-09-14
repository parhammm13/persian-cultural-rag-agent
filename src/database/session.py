from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


load_dotenv()

DATABASE_URL_ENV = "DATABASE_URL"


def get_database_url() -> str:
    """Return the configured SQL database URL.

    The value is resolved lazily so importing API modules does not require a
    live PostgreSQL connection. Non-database endpoints/tests can therefore run
    without the database being available.
    """
    database_url = os.getenv(DATABASE_URL_ENV, "").strip()
    if not database_url:
        raise RuntimeError(
            f"{DATABASE_URL_ENV} is not configured. "
            "Example: postgresql+psycopg://user:password@localhost:5432/"
            "persian_cultural_rag"
        )
    return database_url


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and cache the process-wide SQLAlchemy engine/connection pool."""
    return create_engine(
        get_database_url(),
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide Session factory; Sessions remain per-request."""
    return sessionmaker(
        bind=get_engine(),
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


def reset_database_state_for_tests() -> None:
    """Dispose cached DB resources after tests that replace DATABASE_URL."""
    get_session_factory.cache_clear()

    if get_engine.cache_info().currsize:
        get_engine().dispose()
    get_engine.cache_clear()
