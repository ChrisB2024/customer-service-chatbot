import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from chatbot.chains.rag import NO_CONTEXT_ANSWER, REFUSAL_ANSWER, answer_question, format_context, parse_reply
from chatbot.config import get_settings
from chatbot.llm import REFUSAL_FALLBACK_BETA, get_chat_model
from chatbot.rag.ingest import ingest
from chatbot.rag.retriever import KnowledgeBaseEmpty, ensure_knowledge_base, retrieve_chunks
from chatbot.rag.vector_store import make_vector_store
from chatbot.schemas import SourceChunk

RESTOCKING = SourceChunk(
    source="returns_and_refunds.md",
    section="Restocking fee",
    content="Returns & Refunds > Restocking fee\n\n10% over $150.",
    score=0.85,
    updated="2026-08-01",
)
WINDOW = SourceChunk(
    source="returns_and_refunds.md", section="Return window", content="30 days.", score=0.7, updated="2026-08-01"
)


class FakeLLM:
    """Stands in for PROMPT | ChatAnthropic. Records inputs, returns a canned AIMessage."""

    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.calls: list[dict] = []
        self._reply = AIMessage(content=text, response_metadata={"stop_reason": stop_reason})
        self.runnable = RunnableLambda(self._invoke)

    def _invoke(self, inputs: dict) -> AIMessage:
        self.calls.append(inputs)
        return self._reply


# --- parsing ----------------------------------------------------------------


def test_parse_reply_extracts_and_strips_source_tags():
    parsed = parse_reply(
        "It's 10% over $150 — none below.\n\n<source>returns_and_refunds.md#Restocking fee</source>\n"
        "<source> shipping.md#Canada shipping </source>"
    )
    assert parsed.answer == "It's 10% over $150 — none below."
    assert parsed.sources == ["returns_and_refunds.md#Restocking fee", "shipping.md#Canada shipping"]


def test_parse_reply_without_tags_has_no_sources():
    assert parse_reply("I'm not sure.").sources == []


def test_format_context_labels_each_passage():
    ctx = format_context([RESTOCKING, WINDOW])
    assert '<passage id="returns_and_refunds.md#Restocking fee" updated="2026-08-01">' in ctx
    assert ctx.count("</passage>") == 2


# --- answer_question --------------------------------------------------------


def test_no_relevant_context_skips_the_llm():
    llm = FakeLLM("should not be called")
    result = answer_question("weather?", retrieve=lambda q: [], generate=llm.runnable)
    assert result.answer == NO_CONTEXT_ANSWER and result.sources == []
    assert llm.calls == []


def test_answer_with_valid_sources():
    llm = FakeLLM("10% over $150.\n<source>returns_and_refunds.md#Restocking fee</source>")
    result = answer_question("restocking fee?", retrieve=lambda q: [RESTOCKING], generate=llm.runnable)
    assert result.answer == "10% over $150."
    assert result.sources == ["returns_and_refunds.md#Restocking fee"]
    assert llm.calls[0]["question"] == "restocking fee?"
    assert "Restocking fee" in llm.calls[0]["context"]


def test_hallucinated_and_duplicate_sources_are_dropped():
    llm = FakeLLM(
        "Answer.\n<source>returns_and_refunds.md#Restocking fee</source>"
        "<source>returns_and_refunds.md#Restocking fee</source><source>made_up.md#Secret policy</source>"
    )
    result = answer_question("q", retrieve=lambda q: [RESTOCKING], generate=llm.runnable)
    assert result.sources == ["returns_and_refunds.md#Restocking fee"]


def test_refusal_returns_safe_message():
    llm = FakeLLM("", stop_reason="refusal")
    result = answer_question("q", retrieve=lambda q: [RESTOCKING], generate=llm.runnable)
    assert result.answer == REFUSAL_ANSWER and result.sources == []


def test_truncated_reply_raises():
    llm = FakeLLM("10% over", stop_reason="max_tokens")
    with pytest.raises(RuntimeError, match="max_tokens"):
        answer_question("q", retrieve=lambda q: [RESTOCKING], generate=llm.runnable)


@pytest.mark.parametrize("question", ["", "   ", "x" * (get_settings().max_user_message_chars + 1)])
def test_question_length_is_bounded(question):
    with pytest.raises(ValueError):
        answer_question(question, retrieve=lambda q: [RESTOCKING], generate=FakeLLM("x").runnable)


# --- retriever --------------------------------------------------------------


@pytest.fixture
def fake_store(tmp_path):
    store = make_vector_store(tmp_path / "chroma", "test", DeterministicFakeEmbedding(size=16))
    ingest(store, get_settings().knowledge_base_dir, 800, 100)
    return store


def test_retriever_applies_k_and_relevance_floor(fake_store):
    assert len(retrieve_chunks("anything", k=3, min_relevance=0.0, store=fake_store)) == 3
    assert retrieve_chunks("anything", k=3, min_relevance=1.01, store=fake_store) == []


def test_retriever_returns_typed_chunks(fake_store):
    chunk = retrieve_chunks("anything", k=1, min_relevance=0.0, store=fake_store)[0]
    assert chunk.ref == f"{chunk.source}#{chunk.section}"
    assert chunk.updated


def test_empty_knowledge_base_fails_loudly(tmp_path):
    empty = make_vector_store(tmp_path / "empty", "empty", DeterministicFakeEmbedding(size=16))
    with pytest.raises(KnowledgeBaseEmpty, match="chatbot.rag.ingest"):
        ensure_knowledge_base(empty)


# --- LLM config -------------------------------------------------------------


def test_chat_model_config():
    model = get_chat_model()
    settings = get_settings()
    assert model.model == settings.llm_model
    assert model.reasoning_effort == settings.llm_effort
    if settings.llm_fallbacks:
        assert model.betas == [REFUSAL_FALLBACK_BETA]
        assert model.model_kwargs == {"fallbacks": "default"}


# --- live API (opt-in: uv run pytest -m live) -------------------------------


@pytest.mark.live
def test_live_grounded_answer():
    result = answer_question("Is there a restocking fee?")
    assert "10%" in result.answer and "$150" in result.answer
    assert "returns_and_refunds.md#Restocking fee" in result.sources
    assert "\\u" not in result.answer


@pytest.mark.live
def test_live_unknown_topic_admits_it_and_offers_a_person():
    # Sources may legitimately be non-empty here (it may cite the product FAQ to say what we *do* carry).
    answer = answer_question("Do you sell laptops?").answer.lower()
    assert "not sure" in answer
    assert any(word in answer for word in ("person", "someone", "team"))
