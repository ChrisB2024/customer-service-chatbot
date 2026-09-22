from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    llm_timeout_s: float = 60.0
    # Support chat rarely needs deep reasoning; the step 8 eval decides whether to change this.
    # Set to None for models without effort support (e.g. claude-haiku-4-5).
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "medium"
    # Server-side refusal fallbacks (Opus 5 / Fable only). Set False for other models.
    llm_fallbacks: bool = True

    # Embeddings (local, via fastembed)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path = STORAGE_DIR / "models"

    # Vector store
    chroma_dir: Path = STORAGE_DIR / "chroma"
    chroma_collection: str = "help_center"
    # 6, not 4: "price match" ranks "Price adjustments" 5th (bge-small bunches scores at 0.58-0.63 there).
    retrieval_k: int = 6
    # Cosine relevance floor. Off-topic questions score ~0.44-0.60 and real ones ~0.61-0.85 on this KB,
    # so this only drops obvious junk; the LLM judges the overlap zone.
    min_relevance: float = 0.5
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Relational DB
    database_url: str = f"sqlite:///{STORAGE_DIR / 'app.db'}"

    # Source data
    knowledge_base_dir: Path = DATA_DIR / "knowledge_base"
    seed_dir: Path = DATA_DIR / "seed"
    eval_file: Path = DATA_DIR / "eval" / "golden_questions.json"

    # Chat
    history_turns: int = 10  # past user/assistant pairs sent with each turn
    max_user_message_chars: int = 2000
    conversation_retention_days: int = 90  # promised in account_and_privacy.md


@lru_cache
def get_settings() -> Settings:
    return Settings()
