"""LLM selection helpers for extended thinking roles."""

from __future__ import annotations

from typing import Any, Literal

from agent.config import AgentConfig
from agent.llm.openrouter import get_openrouter_chat_model

ThinkingRole = Literal["reviewer", "rewrite", "repair"]

_ROLE_MODEL_ATTRS: dict[ThinkingRole, str] = {
    "reviewer": "thinking_reviewer_model",
    "rewrite": "thinking_rewrite_model",
    "repair": "thinking_repair_model",
}


class ExtendedModeNotConfigured(RuntimeError):
    """Raised when /thinking extended is enabled without required models."""


def _clean_model(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_models(models: list[str]) -> list[str]:
    """Order-preserving de-duplication of non-empty model ids."""
    return list(dict.fromkeys(model for model in models if model))


def resolve_fusion_proposer_models(config: AgentConfig) -> list[str]:
    """Resolve the ordered, de-duplicated fusion proposer model ids.

    An explicit ``thinking_fusion_proposer_models`` config keeps its order and is
    de-duplicated. An empty config falls back to the existing role models in
    priority order: ``llm_model`` → ``gen_llm_model`` → ``thinking_reviewer_model``.
    The proposer is NOT a :data:`ThinkingRole`; it has no dedicated model slot.
    """
    explicit = [
        _clean_model(model)
        for model in (getattr(config, "thinking_fusion_proposer_models", ()) or ())
    ]
    explicit = [model for model in explicit if model]
    if explicit:
        return _dedupe_models(explicit)
    return _dedupe_models(
        [
            _clean_model(config.llm_model),
            _clean_model(config.gen_llm_model),
            _clean_model(config.thinking_reviewer_model),
        ]
    )


def resolve_fusion_aggregator_model(config: AgentConfig) -> str:
    """Resolve the fusion aggregator model id.

    An explicit ``thinking_fusion_aggregator_model`` wins; an empty config falls
    back to ``judge_llm_model``.
    """
    explicit = _clean_model(getattr(config, "thinking_fusion_aggregator_model", ""))
    if explicit:
        return explicit
    return _clean_model(config.judge_llm_model)


def _missing_fusion_model_fields(config: AgentConfig) -> list[str]:
    missing: list[str] = []
    if not resolve_fusion_proposer_models(config):
        missing.append("thinking_fusion_proposer_models")
    if not resolve_fusion_aggregator_model(config):
        missing.append("thinking_fusion_aggregator_model")
    return missing


def missing_thinking_model_fields(config: AgentConfig) -> list[str]:
    """Return required extended-thinking model fields that are still empty.

    Covers the rewrite/reviewer/repair role slots plus the resolved fusion
    proposer and aggregator models. Only emptiness is checked; provider
    availability is never preflighted here.
    """
    missing = [
        attr
        for attr in _ROLE_MODEL_ATTRS.values()
        if not str(getattr(config, attr, "") or "").strip()
    ]
    missing.extend(_missing_fusion_model_fields(config))
    return missing


def require_thinking_models(config: AgentConfig) -> None:
    """Ensure all model slots required by /thinking extended are configured."""
    missing = missing_thinking_model_fields(config)
    if missing:
        raise ExtendedModeNotConfigured(
            "Extended mode requires these AgentConfig fields to be set in "
            f"agent/config.py: {', '.join(missing)}"
        )


def get_fusion_aggregator_model(config: AgentConfig) -> Any:
    """Return the OpenRouter chat model used to aggregate fusion candidates."""
    model_name = resolve_fusion_aggregator_model(config)
    if not model_name:
        raise ExtendedModeNotConfigured(
            "thinking_fusion_aggregator_model is empty and judge_llm_model is "
            "unset; configure one in agent/config.py AgentConfig before using "
            "/thinking extended."
        )
    return get_openrouter_chat_model(
        config,
        model_name=model_name,
        temperature=0.2,
        max_tokens=config.thinking_fusion_aggregator_max_tokens,
    )


def get_chat_model_for_role(
    config: AgentConfig,
    *,
    role: ThinkingRole,
) -> Any:
    """Return an OpenRouter chat model for one extended-thinking role."""
    attr = _ROLE_MODEL_ATTRS[role]
    model_name = str(getattr(config, attr, "") or "").strip()
    if not model_name:
        raise ExtendedModeNotConfigured(
            f"{attr} is empty; set it in agent/config.py AgentConfig before "
            "using /thinking extended."
        )

    max_tokens = (
        config.thinking_reviewer_max_tokens
        if role == "reviewer"
        else 1024
    )
    return get_openrouter_chat_model(
        config,
        model_name=model_name,
        temperature=0.3,
        max_tokens=max_tokens,
    )
