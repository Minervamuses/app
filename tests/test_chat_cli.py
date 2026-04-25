"""Tests for the interactive chat CLI wrapper."""

import argparse
import asyncio

import pytest


def test_chat_cli_flushes_recent_turns_on_quit(monkeypatch):
    from agent.cli import chat

    calls: list[str] = []

    class FakeSession:
        recursion_limit = 32

        async def turn(self, user_input: str) -> str:
            calls.append(f"turn:{user_input}")
            return "ok"

        async def flush_recent_turns(self) -> None:
            calls.append("flush")

    async def fake_create(*args, **kwargs):
        return FakeSession()

    inputs = iter(["hello", "q"])

    monkeypatch.setattr(chat.ChatSession, "create", fake_create)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    args = argparse.Namespace(max_turns=32, no_mcp=True)
    asyncio.run(chat._run(args))

    assert calls == ["turn:hello", "flush"]


@pytest.mark.parametrize(
    "quit_input",
    [
        "q\u200b",
        "q\ufeff",
        "ｑ",
        "\u200b",
    ],
)
def test_chat_cli_normalizes_quit_inputs(monkeypatch, quit_input):
    from agent.cli import chat

    calls: list[str] = []

    class FakeSession:
        recursion_limit = 32

        async def turn(self, user_input: str) -> str:
            calls.append(f"turn:{user_input!r}")
            return "ok"

        async def flush_recent_turns(self) -> None:
            calls.append("flush")

    async def fake_create(*args, **kwargs):
        return FakeSession()

    inputs = iter([quit_input])

    monkeypatch.setattr(chat.ChatSession, "create", fake_create)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    args = argparse.Namespace(max_turns=32, no_mcp=True)
    asyncio.run(chat._run(args))

    assert calls == ["flush"]


def test_chat_cli_does_not_normalize_regular_messages(monkeypatch):
    from agent.cli import chat

    calls: list[str] = []

    class FakeSession:
        recursion_limit = 32

        async def turn(self, user_input: str) -> str:
            calls.append(f"turn:{user_input!r}")
            return "ok"

        async def flush_recent_turns(self) -> None:
            calls.append("flush")

    async def fake_create(*args, **kwargs):
        return FakeSession()

    inputs = iter(["hello\u200b", "q"])

    monkeypatch.setattr(chat.ChatSession, "create", fake_create)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    args = argparse.Namespace(max_turns=32, no_mcp=True)
    asyncio.run(chat._run(args))

    assert calls == ["turn:'hello\\u200b'", "flush"]


def test_chat_cli_flushes_recent_turns_on_turn_error(monkeypatch):
    from agent.cli import chat

    calls: list[str] = []

    class FakeSession:
        recursion_limit = 32

        async def turn(self, user_input: str) -> str:
            calls.append(f"turn:{user_input}")
            raise RuntimeError("boom")

        async def flush_recent_turns(self) -> None:
            calls.append("flush")

    async def fake_create(*args, **kwargs):
        return FakeSession()

    inputs = iter(["hello", "q"])

    monkeypatch.setattr(chat.ChatSession, "create", fake_create)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    args = argparse.Namespace(max_turns=32, no_mcp=True)
    asyncio.run(chat._run(args))

    assert calls == ["turn:hello", "flush"]
