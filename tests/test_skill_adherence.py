"""End-to-end adherence checks for slash-command skill runtime."""

import asyncio
import json

from langchain_core.messages import AIMessage

from agent.cli.slash_commands import (
    SlashCommandContext,
    build_default_registry,
    execute_slash_command,
    parse_slash_command,
)
from agent.config import AgentConfig
from agent.session import ChatSession
from agent.skills.validator import validate_skill_output
from agent.tools.read_file import _read_file


def _write_academic_skill(tmp_path):
    skills_dir = tmp_path / "skills"
    root = skills_dir / "academic-paper-writing"
    refs = root / "references"
    refs.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        """---
name: academic-paper-writing
description: Use when writing academic papers.
---

# Academic Paper Writing
""",
        encoding="utf-8",
    )
    (refs / "section-playbooks.md").write_text("section reference", encoding="utf-8")
    (root / "manifest.yaml").write_text(
        """
capabilities:
  required:
    - file.read
resources:
  - path: references/section-playbooks.md
    pinned: true
task_modes:
  - revision
tool_policy:
  disallow:
    - bash
""",
        encoding="utf-8",
    )
    return skills_dir, root


class _CaptureGraph:
    def __init__(self, captured):
        self.captured = captured

    async def astream(self, state, config=None, stream_mode="updates"):
        self.captured["state"] = state
        yield {"agent": {"messages": [AIMessage(content="ok")]}}


def _make_session(tmp_path, monkeypatch, captured):
    skills_dir, _root = _write_academic_skill(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "agent.session.build_graph",
        lambda _cfg, extra_tools=None, history_store=None, **kwargs: _CaptureGraph(captured),
    )
    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))
    return ChatSession(cfg)


def test_slash_skill_command_activates_session_runtime(tmp_path, monkeypatch):
    captured = {}
    session = _make_session(tmp_path, monkeypatch, captured)
    registry = build_default_registry()

    result = asyncio.run(execute_slash_command(
        parse_slash_command("/skill academic-paper-writing revision"),
        SlashCommandContext(session=session, registry=registry),
    ))

    assert result.message == "skill -> academic-paper-writing revision"
    assert session.active_skill_runtime is not None
    assert session.active_skill_runtime.name == "academic-paper-writing"
    assert session.active_skill_runtime.task_mode == "revision"


def test_active_skill_state_and_prompt_are_ready_before_turn(tmp_path, monkeypatch):
    captured = {}
    session = _make_session(tmp_path, monkeypatch, captured)
    session.activate_skill("academic-paper-writing", "revision")

    answer = asyncio.run(session.turn("revise this abstract"))
    state = captured["state"]
    prompt_text = "\n".join(message.content for message in state["messages"])

    assert answer == "ok"
    # The session initial state carries the same serialized active-skill slice
    # (and fresh validation defaults) as the graph loader.
    serialized_keys = {
        "active_skill",
        "skill_root",
        "skill_instructions",
        "loaded_references",
        "task_mode",
        "allowed_tools",
        "denied_tools",
        "tool_policy_active",
        "validation_errors",
        "validation_attempts",
        "validation_retry_requested",
    }
    assert serialized_keys <= set(state)
    assert state["active_skill"] == "academic-paper-writing"
    assert state["task_mode"] == "revision"
    assert state["skill_instructions"].startswith("---")
    assert state["loaded_references"] == {
        "references/section-playbooks.md": "section reference"
    }
    assert state["tool_policy_active"] is True
    assert state["validation_errors"] == []
    assert state["validation_attempts"] == 0
    assert state["validation_retry_requested"] is False
    assert "# Academic Paper Writing" in prompt_text


def test_active_skill_relative_reference_resolves_to_skill_bundle(tmp_path, monkeypatch):
    captured = {}
    session = _make_session(tmp_path, monkeypatch, captured)
    runtime = session.activate_skill("academic-paper-writing", "revision")

    payload = json.loads(
        _read_file(
            "references/section-playbooks.md",
            skill_root=str(runtime.root),
        )
    )

    assert payload["path"].endswith(
        "skills/academic-paper-writing/references/section-playbooks.md"
    )
    assert payload["content"] == "section reference"


def test_active_skill_policy_disallows_bash(tmp_path, monkeypatch):
    captured = {}
    session = _make_session(tmp_path, monkeypatch, captured)
    runtime = session.activate_skill("academic-paper-writing", "revision")

    assert "read_file" in runtime.allowed_tools
    assert "bash" in runtime.denied_tools
    assert runtime.tool_policy_active is True


def test_validator_catches_uncited_quantitative_claim_for_academic_skill():
    violations = validate_skill_output(
        active_skill="academic-paper-writing",
        text="The intervention improved retention by 42%.",
    )

    assert violations


def test_no_skill_turn_keeps_skill_state_empty(tmp_path, monkeypatch):
    captured = {}
    session = _make_session(tmp_path, monkeypatch, captured)

    answer = asyncio.run(session.turn("hello"))

    assert answer == "ok"
    assert session.active_skill_runtime is None
    assert "active_skill" not in captured["state"]
