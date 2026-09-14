from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.database.base import Base
from src.database.models import Conversation, Message, MessageRole, User


def test_user_conversation_message_roundtrip() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        user = User(
            email="user@example.com",
            password_hash="not-a-real-password-hash",
        )
        conversation = Conversation(
            title="کاشان",
        )
        conversation.messages.extend(
            [
                Message(
                    role=MessageRole.USER,
                    content="مسجد آقابزرگ کجاست؟",
                ),
                Message(
                    role=MessageRole.ASSISTANT,
                    content="در کاشان قرار دارد.",
                ),
            ]
        )
        user.conversations.append(conversation)

        session.add(user)
        session.commit()

        loaded_user = session.scalar(
            select(User).where(User.email == "user@example.com")
        )

        assert loaded_user is not None
        assert loaded_user.id > 0
        assert len(loaded_user.conversations) == 1
        assert loaded_user.conversations[0].title == "کاشان"
        assert [m.role for m in loaded_user.conversations[0].messages] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]


def test_orm_delete_cascade_removes_owned_objects() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        user = User(
            email="delete@example.com",
            password_hash="hash",
            conversations=[
                Conversation(
                    messages=[
                        Message(
                            role=MessageRole.USER,
                            content="hello",
                        )
                    ]
                )
            ],
        )
        session.add(user)
        session.commit()

        session.delete(user)
        session.commit()

        assert session.scalars(select(User)).all() == []
        assert session.scalars(select(Conversation)).all() == []
        assert session.scalars(select(Message)).all() == []
