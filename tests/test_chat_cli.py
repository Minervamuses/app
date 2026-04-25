"""Tests for the interactive chat CLI wrapper."""

import argparse
import asyncio


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
