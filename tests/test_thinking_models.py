"""Tests for extended thinking model configuration."""

import pytest

from agent.config import AgentConfig
from agent.llm import thinking
from agent.llm.thinking import (
    ExtendedModeNotConfigured,
    get_chat_model_for_role,
    missing_thinking_model_fields,
    require_thinking_models,
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
