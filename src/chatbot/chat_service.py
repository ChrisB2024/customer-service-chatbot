"""One customer turn, end to end: load history → run the agent → persist, all in one transaction.

If anything fails, nothing from the turn is saved (not the messages, not a ticket) and the customer
gets a friendly message instead of a stack trace.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import anthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.orm import Session, sessionmaker

from chatbot.agent import run_turn
from chatbot.chains.rag import validate_user_message
from chatbot.config import get_settings
from chatbot.db.models import Conversation, Message, MessageRole
from chatbot.db.repositories import delete_expired_conversations, load_recent_messages
from chatbot.db.session import get_sessionmaker
from chatbot.rag.retriever import retrieve_chunks
from chatbot.schemas import ChatAnswer, SourceChunk
from chatbot.tools import MAX_FAILED_LOOKUPS_PER_CONVERSATION, AgentContext

logger = logging.getLogger(__name__)

# Worth retrying: the request was fine, the service wasn't.
TRANSIENT_ERRORS = (
    anthropic.APIConnectionError,  # includes APITimeoutError
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    anthropic.ServiceUnavailableError,
    anthropic.DeadlineExceededError,
)

SUPPORT_EMAIL = "support@nimbusgear.example"
TRY_AGAIN = f"Sorry, I'm having trouble right now. Please try again in a moment, or email {SUPPORT_EMAIL}."
SOMETHING_WENT_WRONG = f"Sorry, something went wrong on our end. Please try again, or email {SUPPORT_EMAIL}."


@dataclass(frozen=True)
class Reply:
    conversation_id: str | None  # None only if the very first turn failed (nothing was created)
    answer: ChatAnswer


def _to_langchain(messages: list[Message]) -> list[BaseMessage]:
    return [HumanMessage(m.content) if m.role is MessageRole.USER else AIMessage(m.content) for m in messages]


class ChatService:
    def __init__(
        self,
        sessions: sessionmaker[Session] | None = None,
        *,
        agent: CompiledStateGraph | None = None,
        search: Callable[[str], list[SourceChunk]] = retrieve_chunks,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._sessions = sessions or get_sessionmaker()
        self._agent = agent
        self._search = search
        self._clock = clock
        self._settings = get_settings()

    def send(self, text: str, conversation_id: str | None = None) -> Reply:
        try:
            text = validate_user_message(text)
        except ValueError:
            limit = self._settings.max_user_message_chars
            answer = ChatAnswer(
                answer=f"Please send a message between 1 and {limit} characters.", error="invalid_message"
            )
            return Reply(conversation_id, answer)

        try:
            with self._sessions.begin() as session:
                return self._turn(session, text, conversation_id)
        except TRANSIENT_ERRORS as e:
            logger.warning("LLM unavailable (%s), conversation=%s", type(e).__name__, conversation_id)
            return Reply(conversation_id, ChatAnswer(answer=TRY_AGAIN, error="llm_unavailable"))
        except GraphRecursionError:
            logger.error("Agent hit the recursion limit, conversation=%s", conversation_id)
            return Reply(conversation_id, ChatAnswer(answer=SOMETHING_WENT_WRONG, error="agent_loop"))
        except Exception:
            # Bad API key, 400 from a malformed request, a bug in our code. Log it; don't show internals.
            logger.exception("Turn failed, conversation=%s", conversation_id)
            return Reply(conversation_id, ChatAnswer(answer=SOMETHING_WENT_WRONG, error="internal"))

    def _turn(self, session: Session, text: str, conversation_id: str | None) -> Reply:
        now = self._clock()
        convo = session.get(Conversation, conversation_id) if conversation_id else None
        if convo is None:
            if conversation_id:  # expired (purged) or never existed: start fresh rather than fail
                logger.info("Unknown conversation %s, starting a new one", conversation_id)
            convo = Conversation(started_at=now, last_active_at=now)
            session.add(convo)
            session.flush()

        history = _to_langchain(load_recent_messages(session, convo.id, self._settings.history_turns))
        ctx = AgentContext(
            session=session,
            conversation_id=convo.id,
            now=now,
            search=self._search,
            failed_lookup_budget=max(0, MAX_FAILED_LOOKUPS_PER_CONVERSATION - convo.failed_lookups),
        )
        answer = run_turn(text, ctx, history=history, agent=self._agent)

        session.add_all(
            [
                Message(conversation_id=convo.id, role=MessageRole.USER, content=text, created_at=now),
                Message(conversation_id=convo.id, role=MessageRole.ASSISTANT, content=answer.answer, created_at=now),
            ]
        )
        convo.last_active_at = now
        convo.failed_lookups += ctx.failed_lookups
        return Reply(convo.id, answer)

    def purge_expired(self) -> int:
        with self._sessions.begin() as session:
            return delete_expired_conversations(session, self._clock(), self._settings.conversation_retention_days)
