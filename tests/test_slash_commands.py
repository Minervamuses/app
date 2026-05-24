"""Tests for CLI slash command parsing and completion."""

import asyncio

import pytest
from prompt_toolkit.document import Document

from agent.config import AgentConfig
from agent.skills import SkillMetadata
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


class _FakeModeSession:
    def __init__(self, plan_log_path, config=None):
        self.config = config or AgentConfig()
        self.plan_mode = False
        self.thinking_mode = "normal"
        self.plan_log_path = None
        self._target_log_path = plan_log_path

    async def enter_plan_mode(self):
        self.plan_mode = True
        self.plan_log_path = self._target_log_path
        return self.plan_log_path

    async def exit_plan_mode(self):
        self.plan_mode = False
        self.plan_log_path = None

    def set_thinking_mode(self, mode):
        self.thinking_mode = mode


class _FakeSkillSession:
    def __init__(self, loaded_skills):
        self.config = object()
        self.loaded_skills = loaded_skills
        self.active_skill_runtime = None
        self.activated = []
        self.deactivated = False

    def activate_skill(self, name, task_mode=None):
        class Runtime:
            def __init__(self, name, task_mode):
                self.name = name
                self.task_mode = task_mode

        runtime = Runtime(name, task_mode)
        self.active_skill_runtime = runtime
        self.activated.append((name, task_mode))
        return runtime

    def deactivate_skill(self):
        self.active_skill_runtime = None
        self.deactivated = True


class _ValidatingSkillSession(_FakeSkillSession):
    def activate_skill(self, name, task_mode=None):
        from agent.skills import load_skill_manifest

        skill = next(skill for skill in self.loaded_skills if skill.name == name)
        manifest = load_skill_manifest(skill.path.parent)
        modes = manifest.get("task_modes")
        valid_modes = {mode for mode in modes if isinstance(mode, str)}
        if task_mode is not None and task_mode not in valid_modes:
            raise ValueError(f"unknown task mode for skill: {task_mode}")
        return super().activate_skill(name, task_mode)


def _write_skill(tmp_path, name="paper-writing"):
    root = tmp_path / "skills" / name
    root.mkdir(parents=True)
    skill_file = root / "SKILL.md"
    skill_file.write_text(
        f"""---
name: {name}
description: Use when writing papers.
---
""",
        encoding="utf-8",
    )
    (root / "manifest.yaml").write_text(
        """
task_modes:
  - revision
  - drafting
""",
        encoding="utf-8",
    )
    return SkillMetadata(
        name=name,
        description="Use when writing papers.",
        path=skill_file,
    )


def test_handle_mode_oneshot_switches_to_plan(tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/mode plan"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.plan_mode is True
    assert "mode -> plan" in result.message
    assert str(tmp_path / "plan.md") in result.message


def test_handle_mode_oneshot_switches_back_to_normal(tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    asyncio.run(session.enter_plan_mode())
    registry = build_default_registry()

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/mode normal"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.plan_mode is False
    assert "mode -> normal" in result.message


def test_handle_mode_interactive_selection(monkeypatch, tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    async def fake_to_thread(func, *args, **kwargs):
        return "2"

    monkeypatch.setattr("agent.cli.slash_commands.asyncio.to_thread", fake_to_thread)

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/mode"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.plan_mode is True
    assert "mode -> plan" in result.message


def test_handle_mode_cancel_on_empty_input(monkeypatch, tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    async def fake_to_thread(func, *args, **kwargs):
        return ""

    monkeypatch.setattr("agent.cli.slash_commands.asyncio.to_thread", fake_to_thread)

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/mode"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.plan_mode is False
    assert "cancelled" in result.message


def test_handle_mode_invalid_numeric_choice_raises(monkeypatch, tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    async def fake_to_thread(func, *args, **kwargs):
        return "9"

    monkeypatch.setattr("agent.cli.slash_commands.asyncio.to_thread", fake_to_thread)

    with pytest.raises(SlashCommandError, match="invalid choice"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command("/mode"),
                SlashCommandContext(session=session, registry=registry),
            )
        )


def test_handle_mode_unknown_name_raises(tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    with pytest.raises(SlashCommandError, match="unknown mode"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command("/mode mystery"),
                SlashCommandContext(session=session, registry=registry),
            )
        )


def test_handle_mode_same_mode_is_noop(tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/mode normal"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert "already in normal mode" in result.message


def test_handle_mode_rejects_extra_args(tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    with pytest.raises(SlashCommandError, match="usage"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command("/mode plan extra"),
                SlashCommandContext(session=session, registry=registry),
            )
        )


def test_registry_includes_thinking_command():
    registry = build_default_registry()

    assert registry.get("thinking") is not None


def test_handle_thinking_switches_to_extended(tmp_path):
    session = _FakeModeSession(
        tmp_path / "plan.md",
        config=AgentConfig(
            thinking_reviewer_model="openai/gpt-5.2",
            thinking_rewrite_model="anthropic/claude-haiku-5",
            thinking_repair_model="meta-llama/llama-3.1-8b-instruct",
        ),
    )
    registry = build_default_registry()

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/thinking extended"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.thinking_mode == "extended"
    assert result.message == "thinking -> extended"


def test_handle_thinking_extended_requires_role_models(tmp_path):
    session = _FakeModeSession(
        tmp_path / "plan.md",
        config=AgentConfig(
            thinking_reviewer_model="",
            thinking_rewrite_model="",
            thinking_repair_model="",
        ),
    )
    registry = build_default_registry()

    with pytest.raises(SlashCommandError, match="thinking_reviewer_model"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command("/thinking extended"),
                SlashCommandContext(session=session, registry=registry),
            )
        )

    assert session.thinking_mode == "normal"


def test_handle_thinking_switches_back_to_normal(tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    session.thinking_mode = "extended"
    registry = build_default_registry()

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/thinking normal"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.thinking_mode == "normal"
    assert result.message == "thinking -> normal"


def test_handle_thinking_interactive_selection(monkeypatch, tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    async def fake_to_thread(func, *args, **kwargs):
        assert "Current thinking mode: normal" in args[0]
        assert "[2] extended" in args[0]
        return "2"

    monkeypatch.setattr("agent.cli.slash_commands.asyncio.to_thread", fake_to_thread)

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/thinking"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.thinking_mode == "extended"
    assert result.message == "thinking -> extended"


def test_handle_thinking_cancel_on_empty_input(monkeypatch, tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    async def fake_to_thread(func, *args, **kwargs):
        return ""

    monkeypatch.setattr("agent.cli.slash_commands.asyncio.to_thread", fake_to_thread)

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/thinking"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.thinking_mode == "normal"
    assert result.message == "cancelled"


def test_handle_thinking_invalid_numeric_choice_raises(monkeypatch, tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    async def fake_to_thread(func, *args, **kwargs):
        return "9"

    monkeypatch.setattr("agent.cli.slash_commands.asyncio.to_thread", fake_to_thread)

    with pytest.raises(SlashCommandError, match="invalid choice"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command("/thinking"),
                SlashCommandContext(session=session, registry=registry),
            )
        )


def test_handle_thinking_unknown_name_raises(tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    with pytest.raises(SlashCommandError, match="unknown thinking mode"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command("/thinking deep"),
                SlashCommandContext(session=session, registry=registry),
            )
        )


def test_handle_thinking_rejects_extra_args(tmp_path):
    session = _FakeModeSession(tmp_path / "plan.md")
    registry = build_default_registry()

    with pytest.raises(SlashCommandError, match="usage"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command("/thinking extended now"),
                SlashCommandContext(session=session, registry=registry),
            )
        )


def test_handle_skill_oneshot_activates_skill_with_mode(tmp_path):
    skill = _write_skill(tmp_path)
    session = _FakeSkillSession([skill])
    registry = build_default_registry()

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/skill paper-writing revision"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.activated == [("paper-writing", "revision")]
    assert "skill -> paper-writing revision" in result.message


def test_handle_skill_oneshot_bad_mode_raises_slash_command_error(tmp_path):
    skill = _write_skill(tmp_path)
    session = _ValidatingSkillSession([skill])
    registry = build_default_registry()

    with pytest.raises(SlashCommandError, match="failed to activate skill"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command("/skill paper-writing wrong-mode"),
                SlashCommandContext(session=session, registry=registry),
            )
        )


def test_handle_skill_deactivates_with_none(tmp_path):
    skill = _write_skill(tmp_path)
    session = _FakeSkillSession([skill])
    registry = build_default_registry()

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/skill none"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.deactivated is True
    assert result.message == "skill -> none"


def test_handle_skill_interactive_selection_and_mode(monkeypatch, tmp_path):
    skill = _write_skill(tmp_path)
    session = _FakeSkillSession([skill])
    registry = build_default_registry()
    inputs = iter(["1", "1"])

    async def fake_to_thread(func, *args, **kwargs):
        return next(inputs)

    monkeypatch.setattr("agent.cli.slash_commands.asyncio.to_thread", fake_to_thread)

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/skill"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.activated == [("paper-writing", "revision")]
    assert "skill -> paper-writing revision" in result.message


def test_handle_skill_interactive_manifest_error_raises_slash_command_error(
    monkeypatch,
    tmp_path,
):
    skill = _write_skill(tmp_path)
    (skill.path.parent / "manifest.yaml").write_text("task_modes: [", encoding="utf-8")
    session = _FakeSkillSession([skill])
    registry = build_default_registry()

    async def fake_to_thread(func, *args, **kwargs):
        return "1"

    monkeypatch.setattr("agent.cli.slash_commands.asyncio.to_thread", fake_to_thread)

    with pytest.raises(SlashCommandError, match="failed to activate skill"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command("/skill"),
                SlashCommandContext(session=session, registry=registry),
            )
        )


def test_handle_skill_interactive_deactivate(monkeypatch, tmp_path):
    skill = _write_skill(tmp_path)
    session = _FakeSkillSession([skill])
    registry = build_default_registry()

    async def fake_to_thread(func, *args, **kwargs):
        return "0"

    monkeypatch.setattr("agent.cli.slash_commands.asyncio.to_thread", fake_to_thread)

    result = asyncio.run(
        execute_slash_command(
            parse_slash_command("/skill"),
            SlashCommandContext(session=session, registry=registry),
        )
    )

    assert session.deactivated is True
    assert result.message == "skill -> none"


def test_handle_skill_unknown_name_raises(tmp_path):
    skill = _write_skill(tmp_path)
    session = _FakeSkillSession([skill])
    registry = build_default_registry()

    with pytest.raises(SlashCommandError, match="unknown skill"):
        asyncio.run(
            execute_slash_command(
                parse_slash_command("/skill mystery"),
                SlashCommandContext(session=session, registry=registry),
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
