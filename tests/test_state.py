"""Tests for the shared active-skill state serializer."""

from pathlib import Path
from types import SimpleNamespace

from agent.state import skill_runtime_to_agent_state

_EXPECTED_KEYS = {
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


def _runtime(**overrides):
    base = dict(
        name="paper-writing",
        root=Path("/tmp/skills/paper-writing"),
        instructions="# Skill",
        pinned_references={"references/guide.md": "guide"},
        task_mode="revision",
        allowed_tools=frozenset({"read_file", "bash"}),
        denied_tools=frozenset({"rag_search", "recall_history"}),
        tool_policy_active=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_none_runtime_returns_empty_state():
    assert skill_runtime_to_agent_state(None) == {}


def test_active_runtime_returns_exact_key_set_without_messages():
    state = skill_runtime_to_agent_state(_runtime())

    assert set(state) == _EXPECTED_KEYS
    assert "messages" not in state
    assert state["active_skill"] == "paper-writing"
    assert state["skill_root"] == str(Path("/tmp/skills/paper-writing"))
    assert state["skill_instructions"] == "# Skill"
    assert state["task_mode"] == "revision"
    assert state["tool_policy_active"] is True
    assert state["validation_attempts"] == 0
    assert state["validation_retry_requested"] is False


def test_allowed_and_denied_tools_are_sorted():
    state = skill_runtime_to_agent_state(_runtime())

    assert state["allowed_tools"] == ["bash", "read_file"]
    assert state["denied_tools"] == ["rag_search", "recall_history"]


def test_loaded_references_is_a_copy():
    refs = {"references/guide.md": "guide"}
    runtime = _runtime(pinned_references=refs)

    state = skill_runtime_to_agent_state(runtime)

    assert state["loaded_references"] == refs
    assert state["loaded_references"] is not refs


def test_validation_errors_is_a_fresh_list_per_call():
    runtime = _runtime()

    first = skill_runtime_to_agent_state(runtime)
    second = skill_runtime_to_agent_state(runtime)

    assert first["validation_errors"] == []
    assert second["validation_errors"] == []
    assert first["validation_errors"] is not second["validation_errors"]
