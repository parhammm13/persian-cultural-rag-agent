import pytest

from src.database import session as db_session


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db_session.reset_database_state_for_tests()

    with pytest.raises(RuntimeError, match="DATABASE_URL is not configured"):
        db_session.get_database_url()


def test_engine_and_session_factory_are_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    db_session.reset_database_state_for_tests()

    engine_1 = db_session.get_engine()
    engine_2 = db_session.get_engine()
    factory_1 = db_session.get_session_factory()
    factory_2 = db_session.get_session_factory()

    assert engine_1 is engine_2
    assert factory_1 is factory_2

    db_session.reset_database_state_for_tests()
