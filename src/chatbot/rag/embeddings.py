from functools import lru_cache
from pathlib import Path

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from chatbot.config import get_settings


class FastEmbedEmbeddings(Embeddings):
    """LangChain `Embeddings` backed by a local fastembed ONNX model.

    Documents and queries go through separate methods because some models embed them
    differently (asymmetric retrieval). The model is downloaded once into `cache_dir`.
    """

    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.passage_embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


@lru_cache
def get_embeddings() -> FastEmbedEmbeddings:
    settings = get_settings()
    return FastEmbedEmbeddings(settings.embedding_model, settings.embedding_cache_dir)
