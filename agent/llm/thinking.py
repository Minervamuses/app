"""LLM selection helpers for extended thinking roles."""

from __future__ import annotations

import os
from typing import Literal

from langchain_openai import ChatOpenAI

from agent.config import AgentConfig

ThinkingRole = Literal["reviewer", "rewrite", "repair"]

_ROLE_MODEL_ATTRS: dict[ThinkingRole, str] = {
    "reviewer": "thinking_reviewer_model",
    "rewrite": "thinking_rewrite_model",
    "repair": "thinking_repair_model",
}


class ExtendedModeNotConfigured(RuntimeError):
    """Raised when /thinking extended is enabled without required models."""


def missing_thinking_model_fields(config: AgentConfig) -> list[str]:
    """Return required extended-thinking model fields that are still empty."""
    return [
        attr
        for attr in _ROLE_MODEL_ATTRS.values()
        if not str(getattr(config, attr, "") or "").strip()
    ]


def require_thinking_models(config: AgentConfig) -> None:
    """Ensure all model slots required by /thinking extended are configured."""
    missing = missing_thinking_model_fields(config)
    if missing:
        raise ExtendedModeNotConfigured(
            "Extended mode requires these AgentConfig fields to be set in "
            f"agent/config.py: {', '.join(missing)}"
        )


def get_chat_model_for_role(
    config: AgentConfig,
    *,
    role: ThinkingRole,
) -> ChatOpenAI:
    """Return an OpenRouter chat model for one extended-thinking role."""
    attr = _ROLE_MODEL_ATTRS[role]
    model_name = str(getattr(config, attr, "") or "").strip()
    if not model_name:
        raise ExtendedModeNotConfigured(
            f"{attr} is empty; set it in agent/config.py AgentConfig before "
            "using /thinking extended."
        )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    max_tokens = (
        config.thinking_reviewer_max_tokens
        if role == "reviewer"
        else 1024
    )
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model_name,
        temperature=0.3,
        max_tokens=max_tokens,
        max_retries=config.llm_max_retries,
    )
