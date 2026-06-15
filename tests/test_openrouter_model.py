"""Tests for LangChain LLM model factories."""

from agent.config import AgentConfig
from agent.llm import ollama, openrouter
from agent.llm.ollama import get_ollama_chat_model
from agent.llm.openrouter import get_chat_model, get_openrouter_chat_model
from agent.llm.text import invoke_text


def test_get_chat_model_uses_main_model_and_configured_token_limit(monkeypatch, tmp_path):
    calls: list[dict] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter, "ChatOpenAI", FakeChatOpenAI)
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        llm_model="deepseek/deepseek-v4-pro",
        llm_max_tokens=4096,
    )

    get_chat_model(cfg)

    assert calls[0]["model"] == "deepseek/deepseek-v4-pro"
    assert calls[0]["max_tokens"] == 4096


def test_get_chat_model_delegates_retries_to_client(monkeypatch, tmp_path):
    calls: list[dict] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter, "ChatOpenAI", FakeChatOpenAI)
    cfg = AgentConfig(persist_dir=str(tmp_path), llm_max_retries=7)

    get_chat_model(cfg)

    assert calls[0]["max_retries"] == 7


def test_get_openrouter_chat_model_applies_eval_overrides(monkeypatch, tmp_path):
    calls: list[dict] = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter, "ChatOpenAI", FakeChatOpenAI)
    cfg = AgentConfig(persist_dir=str(tmp_path), llm_max_retries=7)

    get_openrouter_chat_model(
        cfg,
        model_name="openai/gpt-5.2",
        max_tokens=300,
        temperature=0.0,
        extra_body={"reasoning": {"enabled": False}},
    )

    assert calls[0]["model"] == "openai/gpt-5.2"
    assert calls[0]["max_tokens"] == 300
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_retries"] == 7
    assert calls[0]["extra_body"] == {"reasoning": {"enabled": False}}


def test_get_ollama_chat_model_applies_filter_overrides(monkeypatch, tmp_path):
    calls: list[dict] = []

    class FakeChatOllama:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(ollama, "ChatOllama", FakeChatOllama)
    cfg = AgentConfig(persist_dir=str(tmp_path), filter_llm_model="llama3.1:8b")

    get_ollama_chat_model(cfg, max_tokens=8, temperature=0.0)

    assert calls[0]["model"] == "llama3.1:8b"
    assert calls[0]["num_predict"] == 8
    assert calls[0]["temperature"] == 0.0


def test_invoke_text_invokes_chat_model_with_human_message():
    seen_messages = []

    class FakeResponse:
        content = "  hello  "

    class FakeModel:
        def invoke(self, messages):
            seen_messages.extend(messages)
            return FakeResponse()

    result = invoke_text(FakeModel(), "hi")

    assert result == "hello"
    assert seen_messages[0].content == "hi"
