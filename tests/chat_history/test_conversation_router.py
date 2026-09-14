from __future__ import annotations

from src.services.conversation_router import (
    ConversationRoute,
    ConversationRouter,
    HistoryTurn,
    RouteDecision,
)


def _router() -> ConversationRouter:
    return ConversationRouter(
        api_key="test-key",
        model="test-model",
    )


def test_parse_rag_required_followup() -> None:
    router = _router()

    decision = router._parse_decision(
        raw_content=(
            '{"route":"RAG_REQUIRED",'
            '"standalone_query":"باغ ارم چه ویژگی‌های ظاهری و معماری دارد؟",'
            '"answer":null}'
        ),
        original_message="چه شکلیه",
    )

    assert decision.route is ConversationRoute.RAG_REQUIRED
    assert (
        decision.standalone_query
        == "باغ ارم چه ویژگی‌های ظاهری و معماری دارد؟"
    )
    assert decision.answer is None


def test_parse_chat_only_history_question() -> None:
    router = _router()

    decision = router._parse_decision(
        raw_content=(
            '{"route":"CHAT_ONLY",'
            '"standalone_query":null,'
            '"answer":"درباره باغ ارم صحبت می‌کردیم."}'
        ),
        original_message="درباره چه باغی داشتیم صحبت می‌کردیم؟",
    )

    assert decision.route is ConversationRoute.CHAT_ONLY
    assert decision.standalone_query is None
    assert decision.answer == "درباره باغ ارم صحبت می‌کردیم."


def test_rejects_safety_garbage_as_query() -> None:
    router = _router()

    try:
        router._parse_decision(
            raw_content=(
                '{"route":"RAG_REQUIRED",'
                '"standalone_query":"User Safety: safe",'
                '"answer":null}'
            ),
            original_message="اون بنا از چه ساخته شده",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected garbage query to be rejected")


def test_social_turn_does_not_need_llm() -> None:
    router = _router()

    decision = router._deterministic_social_route(
        "مرسی"
    )

    assert decision is not None
    assert decision.route is ConversationRoute.CHAT_ONLY


def test_router_failure_history_fallback_is_chat_only() -> None:
    router = _router()

    history = (
        HistoryTurn(
            role="user",
            content="باغ ارم چه سالی ساخته شده",
        ),
        HistoryTurn(
            role="assistant",
            content="سال دقیق ساخت آن مشخص نیست.",
        ),
    )

    decision = router._fallback_decision(
        message="درباره چه باغی داشتیم صحبت می‌کردیم؟",
        history=history,
    )

    assert decision.route is ConversationRoute.CHAT_ONLY
    assert decision.used_fallback is True


def _takht_history() -> tuple[HistoryTurn, ...]:
    return (
        HistoryTurn(
            role="user",
            content="تخت جمشید کجاست",
        ),
        HistoryTurn(
            role="assistant",
            content="تخت جمشید در استان فارس است.",
        ),
    )


def test_router_failure_pronoun_followup_keeps_topic() -> None:
    router = _router()

    decision = router._fallback_decision(
        message="چه کسی اونو ساخت",
        history=_takht_history(),
    )

    assert decision.route is ConversationRoute.RAG_REQUIRED
    assert decision.used_fallback is True
    assert decision.standalone_query is not None
    assert "تخت جمشید" in decision.standalone_query
    assert "اونو" in decision.standalone_query


def test_router_failure_standalone_question_untouched() -> None:
    router = _router()

    decision = router._fallback_decision(
        message="باغ ارم کجاست",
        history=_takht_history(),
    )

    assert decision.route is ConversationRoute.RAG_REQUIRED
    assert decision.standalone_query == "باغ ارم کجاست"


def test_lazy_rewrite_echoing_followup_gets_topic() -> None:
    router = _router()

    decision = router._enrich_lazy_rewrite(
        decision=RouteDecision(
            route=ConversationRoute.RAG_REQUIRED,
            standalone_query="چه کسی اونو ساخت",
        ),
        original_message="چه کسی اونو ساخت",
        history=_takht_history(),
    )

    assert decision.standalone_query is not None
    assert "تخت جمشید" in decision.standalone_query


def test_proper_rewrite_is_not_modified() -> None:
    router = _router()

    rewritten = "تخت جمشید را چه کسی ساخت؟"
    decision = router._enrich_lazy_rewrite(
        decision=RouteDecision(
            route=ConversationRoute.RAG_REQUIRED,
            standalone_query=rewritten,
        ),
        original_message="چه کسی اونو ساخت",
        history=_takht_history(),
    )

    assert decision.standalone_query == rewritten
