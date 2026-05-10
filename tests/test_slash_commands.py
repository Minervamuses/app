"""Tests for CLI slash command parsing and completion."""

import asyncio

import pytest
from prompt_toolkit.document import Document

from agent.cli.prompting import SlashCommandCompleter
from agent.cli.slash_commands import (
    SlashCommandContext,
    SlashCommandError,
    build_default_registry,
    execute_slash_command,
    parse_slash_command,
)


def test_parse_slash_command_returns_none_for_normal_input():
    assert parse_slash_command("hello there") is None


def test_parse_slash_command_splits_name_and_args():
    parsed = parse_slash_command('/status "with spaces" now')
    assert parsed is not None
    assert parsed.name == "status"
    assert parsed.args == ("with spaces", "now")


def test_parse_slash_command_rejects_empty_command():
    with pytest.raises(SlashCommandError):
        parse_slash_command("/")


def test_slash_command_completer_suggests_matches():
    registry = build_default_registry()
    completer = SlashCommandCompleter(registry)

    completions = list(
        completer.get_completions(Document(text="/st"), complete_event=None)
    )

    assert [completion.text for completion in completions] == ["status"]
    assert completions[0].display_text == "/status"


def test_slash_command_completer_ignores_normal_chat_text():
    registry = build_default_registry()
    completer = SlashCommandCompleter(registry)

    completions = list(
        completer.get_completions(Document(text="hello"), complete_event=None)
    )

    assert completions == []


def test_discuss_command_toggles_session(tmp_path):
    class FakeSession:
        config = object()
        discussion_mode = False
        discussion_log_path = None

        async def enter_discussion_mode(self):
            self.discussion_mode = True
            self.discussion_log_path = tmp_path / "discussion.md"
            return self.discussion_log_path

        async def exit_discussion_mode(self):
            self.discussion_mode = False
            self.discussion_log_path = None

    registry = build_default_registry()
    session = FakeSession()

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/discuss"),
            SlashCommandContext(session=session, registry=registry),
        )
    )
    assert session.discussion_mode is True
    assert "discussion mode ON" in result.message

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/discuss off"),
            SlashCommandContext(session=session, registry=registry),
        )
    )
    assert session.discussion_mode is False
    assert "discussion mode OFF" in result.message


def test_discuss_command_rejects_extra_args():
    class FakeSession:
        discussion_mode = False

    registry = build_default_registry()

    with pytest.raises(SlashCommandError, match="usage"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command("/discuss on extra"),
                SlashCommandContext(session=FakeSession(), registry=registry),
            )
        )


def test_handle_ingest_translates_value_error(monkeypatch, tmp_path):
    class FakeSession:
        config = object()

    def fail_ingest_single(*args, **kwargs):
        raise ValueError("refusing to ingest")

    target = tmp_path / "blocked.md"
    target.write_text("blocked", encoding="utf-8")
    monkeypatch.setattr("agent.cli.slash_commands.ingest_single", fail_ingest_single)

    with pytest.raises(SlashCommandError, match="refusing to ingest"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command(f"/ingest {target}"),
                SlashCommandContext(
                    session=FakeSession(),
                    registry=build_default_registry(),
                ),
            )
        )
