from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic

from chatbot.config import get_settings

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


@lru_cache
def get_chat_model() -> ChatAnthropic:
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "max_tokens": settings.llm_max_tokens,
        "timeout": settings.llm_timeout_s,
        "max_retries": 2,  # SDK retries 429 / 5xx / connection errors with backoff
    }
    # pydantic-settings reads .env without exporting it, so pass the key explicitly.
    # When unset, the SDK resolves credentials itself (env var, `ant auth login` profile).
    if settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key
    if settings.llm_effort:
        kwargs["effort"] = settings.llm_effort
    if settings.llm_fallbacks:
        # If a safety classifier declines, the API re-runs the request on a fallback model
        # in the same call instead of returning stop_reason="refusal".
        kwargs["betas"] = [REFUSAL_FALLBACK_BETA]
        kwargs["model_kwargs"] = {"fallbacks": "default"}
    return ChatAnthropic(**kwargs)
