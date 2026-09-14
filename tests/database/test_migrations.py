from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def _alembic_config(project_root: Path) -> Config:
    return Config(str(project_root / "alembic.ini"))


def test_initial_migration_upgrade_and_downgrade(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    database_path = tmp_path / "migration_test.db"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)

    config = _alembic_config(project_root)
    command.upgrade(config, "head")

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert {
            "alembic_version",
            "users",
            "conversations",
            "messages",
        } <= set(inspector.get_table_names())

        user_indexes = {item["name"] for item in inspector.get_indexes("users")}
        conversation_indexes = {
            item["name"] for item in inspector.get_indexes("conversations")
        }
        message_indexes = {
            item["name"] for item in inspector.get_indexes("messages")
        }

        assert "ix_users_email" in user_indexes
        assert "ix_conversations_user_id" in conversation_indexes
        assert "ix_messages_conversation_id" in message_indexes
        assert "ix_messages_created_at" in message_indexes

        conversation_fks = inspector.get_foreign_keys("conversations")
        message_fks = inspector.get_foreign_keys("messages")
        assert conversation_fks[0]["referred_table"] == "users"
        assert conversation_fks[0]["options"].get("ondelete") == "CASCADE"
        assert message_fks[0]["referred_table"] == "conversations"
        assert message_fks[0]["options"].get("ondelete") == "CASCADE"
    finally:
        engine.dispose()

    # The checked-in migration must fully describe the current ORM metadata.
    command.check(config)

    command.downgrade(config, "base")

    engine = create_engine(database_url)
    try:
        assert inspect(engine).get_table_names() == ["alembic_version"]
    finally:
        engine.dispose()
