from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM. If the key is unset, the Anthropic SDK falls back to other credential sources (e.g. `ant auth login`).
    anthropic_api_key: SecretStr | None = None
    llm_model: str = "claude-opus-5"
    llm_max_tokens: int = 16000

    # Embeddings (local, via fastembed)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path = STORAGE_DIR / "models"

    # Vector store
    chroma_dir: Path = STORAGE_DIR / "chroma"
    chroma_collection: str = "help_center"
    retrieval_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Relational DB
    database_url: str = f"sqlite:///{STORAGE_DIR / 'app.db'}"

    # Source data
    knowledge_base_dir: Path = DATA_DIR / "knowledge_base"
    seed_dir: Path = DATA_DIR / "seed"
    eval_file: Path = DATA_DIR / "eval" / "golden_questions.json"

    # Chat
    history_turns: int = 10
    max_user_message_chars: int = 2000


@lru_cache
def get_settings() -> Settings:
    return Settings()
