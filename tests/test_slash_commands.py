"""Tests for CLI slash command parsing and completion."""

import pytest
from prompt_toolkit.document import Document

from agent.cli.prompting import SlashCommandCompleter
from agent.cli.slash_commands import (
    SlashCommandError,
    build_default_registry,
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
