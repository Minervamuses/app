"""Tests for extended thinking model configuration."""

import pytest

from agent.config import AgentConfig
from agent.llm import thinking
from agent.llm.thinking import (
    ExtendedModeNotConfigured,
    get_chat_model_for_role,
    get_fusion_aggregator_model,
    missing_thinking_model_fields,
    require_thinking_models,
    resolve_fusion_aggregator_model,
    resolve_fusion_proposer_models,
)


def test_missing_thinking_model_fields_accepts_configured_defaults(tmp_path):
    cfg = AgentConfig(persist_dir=str(tmp_path))

    assert missing_thinking_model_fields(cfg) == []


def test_require_thinking_models_accepts_all_configured_defaults(tmp_path):
    cfg = AgentConfig(persist_dir=str(tmp_path))

    require_thinking_models(cfg)


def test_require_thinking_models_raises_with_missing_names(tmp_path):
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        thinking_reviewer_model="openai/gpt-5.2",
        thinking_rewrite_model="",
        thinking_repair_model="",
    )

    with pytest.raises(ExtendedModeNotConfigured) as excinfo:
        require_thinking_models(cfg)

    assert "thinking_rewrite_model" in str(excinfo.value)
    assert "thinking_repair_model" in str(excinfo.value)
    assert "agent/config.py" in str(excinfo.value)


def test_get_chat_model_for_role_applies_role_model_and_reviewer_tokens(
    monkeypatch,
    tmp_path,
):
    calls: list[dict] = []

    def fake_get_openrouter_chat_model(config, **kwargs):
        calls.append({"config": config, **kwargs})
        return object()

    monkeypatch.setattr(
        thinking,
        "get_openrouter_chat_model",
        fake_get_openrouter_chat_model,
    )
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        thinking_reviewer_model="openai/gpt-5.2",
        thinking_reviewer_max_tokens=8192,
        thinking_rewrite_model="anthropic/claude-haiku-5",
        thinking_repair_model="meta-llama/llama-3.1-8b-instruct",
        llm_max_retries=7,
    )

    get_chat_model_for_role(cfg, role="reviewer")
    get_chat_model_for_role(cfg, role="rewrite")

    assert calls[0]["config"] is cfg
    assert calls[0]["model_name"] == "openai/gpt-5.2"
    assert calls[0]["max_tokens"] == 8192
    assert calls[0]["temperature"] == 0.3
    assert calls[1]["model_name"] == "anthropic/claude-haiku-5"
    assert calls[1]["max_tokens"] == 1024


def test_resolve_fusion_proposer_models_defaults_to_role_models(tmp_path):
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        llm_model="primary",
        gen_llm_model="gen",
        thinking_reviewer_model="reviewer",
    )

    assert resolve_fusion_proposer_models(cfg) == ["primary", "gen", "reviewer"]


def test_resolve_fusion_proposer_models_dedupes_default_overlap(tmp_path):
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        llm_model="same",
        gen_llm_model="same",
        thinking_reviewer_model="reviewer",
    )

    assert resolve_fusion_proposer_models(cfg) == ["same", "reviewer"]


def test_resolve_fusion_proposer_models_preserves_explicit_order_and_dedupes(tmp_path):
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        thinking_fusion_proposer_models=("c", "a", "c", "b", "a"),
    )

    assert resolve_fusion_proposer_models(cfg) == ["c", "a", "b"]


def test_resolve_fusion_aggregator_model_defaults_to_judge(tmp_path):
    cfg = AgentConfig(persist_dir=str(tmp_path), judge_llm_model="judge")

    assert resolve_fusion_aggregator_model(cfg) == "judge"


def test_resolve_fusion_aggregator_model_prefers_explicit(tmp_path):
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        judge_llm_model="judge",
        thinking_fusion_aggregator_model="explicit-agg",
    )

    assert resolve_fusion_aggregator_model(cfg) == "explicit-agg"


def test_require_thinking_models_raises_for_missing_resolved_proposers(tmp_path):
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        llm_model="",
        gen_llm_model="",
        thinking_reviewer_model="reviewer",
        thinking_fusion_proposer_models=(),
    )
    # thinking_reviewer_model is still a role field, so blank it after using it
    # as a proposer source to force the proposer resolution to empty.
    cfg.thinking_reviewer_model = ""

    missing = missing_thinking_model_fields(cfg)

    assert "thinking_fusion_proposer_models" in missing
    with pytest.raises(ExtendedModeNotConfigured):
        require_thinking_models(cfg)


def test_require_thinking_models_raises_for_missing_resolved_aggregator(tmp_path):
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        judge_llm_model="",
        thinking_fusion_aggregator_model="",
    )

    assert "thinking_fusion_aggregator_model" in missing_thinking_model_fields(cfg)
    with pytest.raises(ExtendedModeNotConfigured):
        require_thinking_models(cfg)


def test_get_fusion_aggregator_model_uses_openrouter_with_low_temperature(
    monkeypatch,
    tmp_path,
):
    calls: list[dict] = []

    def fake_get_openrouter_chat_model(config, **kwargs):
        calls.append({"config": config, **kwargs})
        return object()

    monkeypatch.setattr(
        thinking,
        "get_openrouter_chat_model",
        fake_get_openrouter_chat_model,
    )
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        thinking_fusion_aggregator_model="agg-model",
        thinking_fusion_aggregator_max_tokens=2048,
    )

    get_fusion_aggregator_model(cfg)

    assert calls[0]["model_name"] == "agg-model"
    assert calls[0]["temperature"] == 0.2
    assert calls[0]["max_tokens"] == 2048
