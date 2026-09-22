import logging
from datetime import UTC, datetime, timedelta

import anthropic
import httpx
import pytest
from fakes import call, scripted, tool_results
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from chatbot.agent import build_agent
from chatbot.chat_service import SOMETHING_WENT_WRONG, TRY_AGAIN, ChatService
from chatbot.config import get_settings
from chatbot.db.models import Conversation, Message, MessageRole, Ticket, TicketReason
from chatbot.db.seed import load_seed_data, seed
from chatbot.db.session import make_engine
from chatbot.tools import CONVERSATION_LOOKUP_LIMIT, MAX_FAILED_LOOKUPS_PER_CONVERSATION, ORDER_NOT_FOUND

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
WRONG = {"order_number": "NG-10421", "email": "nobody@example.com"}


@pytest.fixture
def sessions():
    engine = make_engine("sqlite://")
    seed(engine, load_seed_data(get_settings().seed_dir))
    return sessionmaker(engine, expire_on_commit=False)


def service(sessions, model, now=NOW):
    return ChatService(sessions, agent=build_agent(model), search=lambda q: [], clock=lambda: now)


def count(sessions, model_cls) -> int:
    with sessions() as s:
        return s.scalar(select(func.count()).select_from(model_cls))


def connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


# --- happy path -------------------------------------------------------------


def test_first_message_creates_conversation_and_saves_the_turn(sessions):
    reply = service(sessions, scripted(AIMessage("Hi! How can I help?"))).send("hello")

    assert reply.answer.answer == "Hi! How can I help?" and reply.answer.error is None
    with sessions() as s:
        convo = s.get(Conversation, reply.conversation_id)
        assert [(m.role, m.content) for m in convo.messages] == [
            (MessageRole.USER, "hello"),
            (MessageRole.ASSISTANT, "Hi! How can I help?"),
        ]
        assert convo.last_active_at == NOW


def test_history_is_sent_on_the_next_turn(sessions):
    model = scripted(AIMessage("Which order?"), AIMessage("Thanks, looking now."))
    svc = service(sessions, model)
    first = svc.send("where's my order")
    second = svc.send("NG-10415", conversation_id=first.conversation_id)

    assert second.conversation_id == first.conversation_id
    sent = [(type(m).__name__, m.text) for m in model.seen[1][1:]]  # skip the system prompt
    assert sent == [
        ("HumanMessage", "where's my order"),
        ("AIMessage", "Which order?"),
        ("HumanMessage", "NG-10415"),
    ]


def test_history_is_capped_and_starts_with_a_user_message(sessions, monkeypatch):
    monkeypatch.setattr(get_settings(), "history_turns", 2)
    replies = [AIMessage(f"answer {i}") for i in range(4)]
    model = scripted(*replies)
    svc = service(sessions, model)
    cid = None
    for i in range(4):
        cid = svc.send(f"question {i}", conversation_id=cid).conversation_id

    last_call = model.seen[-1][1:]
    assert isinstance(last_call[0], HumanMessage)
    assert [m.text for m in last_call] == ["question 1", "answer 1", "question 2", "answer 2", "question 3"]


def test_unknown_conversation_id_starts_a_new_one(sessions):
    reply = service(sessions, scripted(AIMessage("Hello again."))).send("hi", conversation_id="does-not-exist")
    assert reply.conversation_id != "does-not-exist"
    assert count(sessions, Conversation) == 1


def test_ticket_is_saved_with_the_turn(sessions):
    model = scripted(
        AIMessage("", tool_calls=[call("escalate_to_human", reason="requested_human", summary="wants a person")]),
        AIMessage("A person will follow up."),
    )
    reply = service(sessions, model).send("talk to a human")

    with sessions() as s:
        ticket = s.get(Ticket, reply.answer.ticket_id)
        assert ticket.conversation_id == reply.conversation_id and ticket.reason is TicketReason.REQUESTED_HUMAN


# --- failures roll back the whole turn --------------------------------------


def test_llm_outage_saves_nothing_even_a_ticket_created_mid_turn(sessions, caplog):
    model = scripted(
        AIMessage("", tool_calls=[call("escalate_to_human", reason="other", summary="x")]),
        connection_error(),  # the API drops after the ticket was flushed
    )
    with caplog.at_level(logging.WARNING, logger="chatbot.chat_service"):
        reply = service(sessions, model).send("help")

    assert reply.answer.answer == TRY_AGAIN and reply.answer.error == "llm_unavailable"
    assert reply.conversation_id is None
    assert (count(sessions, Conversation), count(sessions, Message), count(sessions, Ticket)) == (0, 0, 0)
    assert "APIConnectionError" in caplog.text


def test_langchain_wrapped_errors_are_still_treated_as_transient(sessions):
    # langchain-anthropic re-raises SDK errors as its own types; they must keep subclassing the SDK's.
    from langchain_anthropic.chat_models import AnthropicConnectionError

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    reply = service(sessions, scripted(AnthropicConnectionError(message="down", request=request))).send("hi")
    assert reply.answer.error == "llm_unavailable"


def test_failed_turn_keeps_the_existing_conversation_untouched(sessions):
    model = scripted(AIMessage("First answer."), connection_error())
    svc = service(sessions, model)
    first = svc.send("first")
    failed = svc.send("second", conversation_id=first.conversation_id)

    assert failed.conversation_id == first.conversation_id and failed.answer.error == "llm_unavailable"
    assert count(sessions, Message) == 2


def test_unexpected_error_is_logged_and_hidden(sessions, caplog):
    model = scripted(RuntimeError("boom: internal detail"))
    with caplog.at_level(logging.ERROR, logger="chatbot.chat_service"):
        reply = service(sessions, model).send("hi")

    assert reply.answer.answer == SOMETHING_WENT_WRONG and reply.answer.error == "internal"
    assert "boom" not in reply.answer.answer
    assert "boom: internal detail" in caplog.text


def test_agent_stuck_in_a_tool_loop(sessions):
    loop = [AIMessage("", tool_calls=[call("search_help_center", query=f"q{i}")]) for i in range(20)]
    reply = service(sessions, scripted(*loop)).send("hi")
    assert reply.answer.error == "agent_loop"
    assert count(sessions, Message) == 0


@pytest.mark.parametrize("text", ["", "   ", "x" * (get_settings().max_user_message_chars + 1)])
def test_invalid_message_never_reaches_the_llm(sessions, text):
    model = scripted()
    reply = service(sessions, model).send(text)
    assert reply.answer.error == "invalid_message"
    assert model.seen == [] and count(sessions, Conversation) == 0


# --- enumeration guard across turns -----------------------------------------


def test_failed_lookups_accumulate_across_turns_until_blocked(sessions):
    turns = []
    for _ in range(MAX_FAILED_LOOKUPS_PER_CONVERSATION):  # one failed guess per turn
        turns += [AIMessage("", tool_calls=[call("lookup_order", **WRONG)]), AIMessage("Not found.")]
    turns += [AIMessage("", tool_calls=[call("lookup_order", **WRONG)]), AIMessage("Let me get a person.")]
    model = scripted(*turns)
    svc = service(sessions, model)

    cid = None
    for _ in range(MAX_FAILED_LOOKUPS_PER_CONVERSATION):
        cid = svc.send("NG-10421, nobody@example.com", conversation_id=cid).conversation_id
        assert tool_results(model)[-1].text == ORDER_NOT_FOUND

    svc.send("try again", conversation_id=cid)
    assert tool_results(model)[-1].text == CONVERSATION_LOOKUP_LIMIT
    with sessions() as s:
        assert s.get(Conversation, cid).failed_lookups == MAX_FAILED_LOOKUPS_PER_CONVERSATION


def test_successful_lookups_dont_count_as_failures(sessions):
    model = scripted(
        AIMessage("", tool_calls=[call("lookup_order", order_number="NG-10415", email="daniel.reyes@example.com")]),
        AIMessage("It shipped."),
    )
    reply = service(sessions, model).send("NG-10415 daniel.reyes@example.com")
    with sessions() as s:
        assert s.get(Conversation, reply.conversation_id).failed_lookups == 0


# --- retention --------------------------------------------------------------


def test_purge_deletes_old_conversations_but_keeps_their_tickets(sessions):
    retention = get_settings().conversation_retention_days
    old_model = scripted(
        AIMessage("", tool_calls=[call("escalate_to_human", reason="other", summary="old issue")]),
        AIMessage("A person will follow up."),
    )
    old = service(sessions, old_model, now=NOW - timedelta(days=retention + 1)).send("old question")
    recent = service(sessions, scripted(AIMessage("Hi.")), now=NOW - timedelta(days=1)).send("recent question")

    deleted = service(sessions, scripted()).purge_expired()

    assert deleted == 1
    with sessions() as s:
        assert s.get(Conversation, old.conversation_id) is None
        assert s.get(Conversation, recent.conversation_id) is not None
        assert s.scalars(select(Message.content).where(Message.content == "old question")).all() == []
        assert s.get(Ticket, old.answer.ticket_id).conversation_id is None
