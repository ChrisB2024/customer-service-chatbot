from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from chatbot.config import get_settings
from chatbot.rag.embeddings import get_embeddings


def make_vector_store(path: Path, collection_name: str, embeddings: Embeddings) -> Chroma:
    client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
        # Cosine suits normalized sentence embeddings; it also makes relevance scores land in [0, 1].
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


@lru_cache
def get_vector_store() -> Chroma:
    settings = get_settings()
    return make_vector_store(settings.chroma_dir, settings.chroma_collection, get_embeddings())
