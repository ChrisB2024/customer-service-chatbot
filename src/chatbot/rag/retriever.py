from langchain_chroma import Chroma

from chatbot.config import get_settings
from chatbot.rag.vector_store import get_vector_store
from chatbot.schemas import SourceChunk


class KnowledgeBaseEmpty(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Knowledge base not found. Run: uv run python -m chatbot.rag.ingest")


def ensure_knowledge_base(store: Chroma | None = None) -> None:
    """Fail loudly at startup. An empty collection would otherwise make every answer "I don't know"."""
    store = store or get_vector_store()
    if not store.get(limit=1, include=[])["ids"]:
        raise KnowledgeBaseEmpty()


def retrieve_chunks(
    query: str,
    *,
    k: int | None = None,
    min_relevance: float | None = None,
    store: Chroma | None = None,
) -> list[SourceChunk]:
    """Top-k passages for `query`, most relevant first, dropping anything below `min_relevance`."""
    settings = get_settings()
    store = store or get_vector_store()
    k = k or settings.retrieval_k
    floor = settings.min_relevance if min_relevance is None else min_relevance

    results = store.similarity_search_with_relevance_scores(query, k=k)
    return [
        SourceChunk(
            source=doc.metadata["source"],
            section=doc.metadata["section"],
            content=doc.page_content,
            score=round(score, 4),
            updated=doc.metadata.get("updated", ""),
        )
        for doc, score in results
        if score >= floor
    ]
