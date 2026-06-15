"""Tests for main OpenRouter chat model configuration."""

from agent.config import AgentConfig
from agent.llm import openrouter
from agent.llm.openrouter import OpenRouterLLM, get_chat_model


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


def test_openrouter_llm_passes_retries_to_openai_client(monkeypatch, tmp_path):
    init_kwargs: list[dict] = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            init_kwargs.append(kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter, "OpenAI", FakeOpenAI)
    cfg = AgentConfig(persist_dir=str(tmp_path), llm_max_retries=7)

    OpenRouterLLM(config=cfg)

    assert init_kwargs[0]["max_retries"] == 7


def test_openrouter_llm_invoke_calls_client_completion_directly(monkeypatch, tmp_path):
    """invoke() should hit the client completion API with no local retry wrapper."""
    create_calls: list[dict] = []

    class FakeMessage:
        content = "  hello  "

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            create_calls.append(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(openrouter, "OpenAI", FakeOpenAI)
    cfg = AgentConfig(persist_dir=str(tmp_path), llm_model="some/model")

    llm = OpenRouterLLM(config=cfg)
    result = llm.invoke("hi", max_tokens=42)

    assert result == "hello"
    assert create_calls[0]["model"] == "some/model"
    assert create_calls[0]["max_tokens"] == 42
    # The retry wrapper was removed; invoke must not depend on it.
    assert not hasattr(llm, "_call_with_retry")
