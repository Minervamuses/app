"""Tests for active skill runtime loading."""

import pytest
from langchain_core.tools import tool

from agent.config import AgentConfig
from agent.skills.runtime import load_skill_runtime


@tool("read_file")
def _read_file(path: str) -> str:
    """Read a file."""
    return path


@tool("rag_search")
def _rag_search(query: str) -> str:
    """Search RAG."""
    return query


@tool("bash")
def _bash(command: str) -> str:
    """Run shell."""
    return command


def _write_skill(tmp_path):
    skills_dir = tmp_path / "skills"
    root = skills_dir / "paper"
    refs = root / "references"
    refs.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        """---
name: paper
description: Use when writing a paper.
---

# Paper
""",
        encoding="utf-8",
    )
    (refs / "guide.md").write_text("guide text", encoding="utf-8")
    (root / "manifest.yaml").write_text(
        """
capabilities:
  required:
    - file.read
  optional:
    - id: shell.execute
resources:
  - path: references/guide.md
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


def test_load_skill_runtime_reads_skill_and_pinned_references(tmp_path):
    skills_dir, root = _write_skill(tmp_path)
    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))
    capability_map = {
        "capabilities": {
            "file.read": {"local_tools": ["read_file"]},
            "shell.execute": {"local_tools": ["bash"]},
        }
    }

    runtime = load_skill_runtime(
        "paper",
        config=cfg,
        all_tools=[_read_file, _rag_search, _bash],
        task_mode="revision",
        capability_map=capability_map,
    )

    assert runtime.name == "paper"
    assert runtime.root == root.resolve()
    assert "# Paper" in runtime.instructions
    assert runtime.pinned_references == {"references/guide.md": "guide text"}
    assert runtime.allowed_tools == frozenset({"read_file"})
    assert runtime.denied_tools == frozenset({"bash"})
    assert runtime.task_mode == "revision"


def test_read_skill_resource_resolves_relative_to_skill_root(tmp_path):
    skills_dir, _root = _write_skill(tmp_path)
    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))
    runtime = load_skill_runtime(
        "paper",
        config=cfg,
        all_tools=[_read_file],
        capability_map={"capabilities": {"file.read": {"local_tools": ["read_file"]}}},
    )

    assert runtime.read_skill_resource("references/guide.md") == "guide text"


def test_read_skill_resource_blocks_path_escape(tmp_path):
    skills_dir, _root = _write_skill(tmp_path)
    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))
    runtime = load_skill_runtime(
        "paper",
        config=cfg,
        all_tools=[_read_file],
        capability_map={"capabilities": {"file.read": {"local_tools": ["read_file"]}}},
    )

    with pytest.raises(PermissionError, match="escapes skill root"):
        runtime.read_skill_resource("../../etc/passwd")


def test_read_skill_resource_missing_file_has_clear_error(tmp_path):
    skills_dir, _root = _write_skill(tmp_path)
    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))
    runtime = load_skill_runtime(
        "paper",
        config=cfg,
        all_tools=[_read_file],
        capability_map={"capabilities": {"file.read": {"local_tools": ["read_file"]}}},
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        runtime.read_skill_resource("references/missing.md")


def test_load_skill_runtime_rejects_unknown_task_mode(tmp_path):
    skills_dir, _root = _write_skill(tmp_path)
    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))

    with pytest.raises(ValueError, match="unknown task mode"):
        load_skill_runtime(
            "paper",
            config=cfg,
            all_tools=[_read_file],
            task_mode="drafting",
            capability_map={
                "capabilities": {"file.read": {"local_tools": ["read_file"]}}
            },
        )
