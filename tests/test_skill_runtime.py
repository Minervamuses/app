"""Tests for active skill runtime loading."""

from types import SimpleNamespace

import pytest
from langchain_core.tools import tool

from agent.config import AgentConfig
from agent.skills.runtime import load_skill_runtime, render_tool_availability_block


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
    assert runtime.tool_policy_active is True
    assert runtime.capability_resolution.requested_required == frozenset({"file.read"})
    assert runtime.capability_resolution.requested_optional == frozenset({"shell.execute"})
    assert runtime.capability_resolution.unresolved_optional == frozenset({"shell.execute"})
    assert runtime.task_mode == "revision"


def test_load_skill_runtime_rejects_unknown_required_capability(tmp_path):
    skills_dir, root = _write_skill(tmp_path)
    (root / "manifest.yaml").write_text(
        """
capabilities:
  required:
    - shell.exec
""",
        encoding="utf-8",
    )
    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))

    with pytest.raises(ValueError, match="shell.exec"):
        load_skill_runtime(
            "paper",
            config=cfg,
            all_tools=[_read_file, _bash],
            capability_map={
                "capabilities": {"shell.execute": {"local_tools": ["bash"]}}
            },
        )


def test_load_skill_runtime_rejects_manifest_typo_key(tmp_path):
    skills_dir, root = _write_skill(tmp_path)
    (root / "manifest.yaml").write_text(
        """
capabilites:
  required:
    - file.read
""",
        encoding="utf-8",
    )
    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))

    with pytest.raises(ValueError, match="capabilites"):
        load_skill_runtime(
            "paper",
            config=cfg,
            all_tools=[_read_file],
            capability_map={
                "capabilities": {"file.read": {"local_tools": ["read_file"]}}
            },
        )


def test_load_skill_runtime_rejects_non_bool_pinned(tmp_path):
    skills_dir, root = _write_skill(tmp_path)
    (root / "manifest.yaml").write_text(
        """
capabilities:
  required:
    - file.read
resources:
  - path: references/guide.md
    pinned: "yes"
""",
        encoding="utf-8",
    )
    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))

    with pytest.raises(ValueError, match="pinned"):
        load_skill_runtime(
            "paper",
            config=cfg,
            all_tools=[_read_file],
            capability_map={
                "capabilities": {"file.read": {"local_tools": ["read_file"]}}
            },
        )


def test_load_skill_runtime_rejects_empty_capabilities_without_policy(tmp_path):
    skills_dir, root = _write_skill(tmp_path)
    (root / "manifest.yaml").write_text(
        """
capabilities: {}
""",
        encoding="utf-8",
    )
    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))

    with pytest.raises(ValueError, match="capabilities must not be empty"):
        load_skill_runtime(
            "paper",
            config=cfg,
            all_tools=[_read_file],
            capability_map={"capabilities": {}},
        )


def test_load_skill_runtime_allows_unavailable_optional_capability(tmp_path):
    skills_dir, root = _write_skill(tmp_path)
    (root / "manifest.yaml").write_text(
        """
capabilities:
  required:
    - file.read
  optional:
    - id: web.search
      use_when: current venue information is needed
""",
        encoding="utf-8",
    )
    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))

    runtime = load_skill_runtime(
        "paper",
        config=cfg,
        all_tools=[_read_file],
        capability_map={
            "capabilities": {
                "file.read": {"local_tools": ["read_file"]},
                "web.search": {"mcp_families": ["web_search"]},
            }
        },
    )

    assert runtime.allowed_tools == frozenset({"read_file"})
    assert runtime.capability_resolution.unresolved_optional == frozenset({"web.search"})


def test_load_skill_runtime_rejects_oversized_pinned_reference(tmp_path):
    skills_dir, root = _write_skill(tmp_path)
    (root / "references" / "guide.md").write_text("x" * 20, encoding="utf-8")
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        skills_dir=str(skills_dir),
        skill_max_pinned_reference_chars=10,
    )

    with pytest.raises(ValueError, match="pinned skill reference too large"):
        load_skill_runtime(
            "paper",
            config=cfg,
            all_tools=[_read_file],
            capability_map={
                "capabilities": {"file.read": {"local_tools": ["read_file"]}}
            },
        )


def test_load_skill_runtime_rejects_oversized_total_skill_context(tmp_path):
    skills_dir, _root = _write_skill(tmp_path)
    cfg = AgentConfig(
        persist_dir=str(tmp_path),
        skills_dir=str(skills_dir),
        skill_max_pinned_reference_chars=1000,
        skill_max_total_skill_context_chars=20,
    )

    with pytest.raises(ValueError, match="total skill context too large"):
        load_skill_runtime(
            "paper",
            config=cfg,
            all_tools=[_read_file],
            capability_map={
                "capabilities": {"file.read": {"local_tools": ["read_file"]}}
            },
        )


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


def test_render_tool_availability_block_for_active_skill():
    runtime = SimpleNamespace(
        name="paper",
        task_mode="revision",
        allowed_tools=frozenset({"read_file", "rag_search"}),
        denied_tools=frozenset({"bash"}),
        tool_policy_active=True,
    )

    block = render_tool_availability_block(
        skill_runtime=runtime,
        base_tool_names=["rag_search", "read_file", "bash", "github_search"],
        mcp_families={"github_search": "github"},
    )

    assert "active_skill: paper" in block
    assert "task_mode: revision" in block
    assert "tool_policy_active: true" in block
    assert "available_tools: rag_search, read_file" in block
    assert "denied_tools: bash" in block
    assert "unavailable_base_tools: bash, github_search" in block
    assert 'Active skill policy overrides the base "always available" wording' in block


def test_render_tool_availability_block_without_active_skill_collapses_mcp_families():
    block = render_tool_availability_block(
        base_tool_names=[
            "rag_search",
            "web_search_one",
            "web_search_two",
            "github_issue",
        ],
        mcp_families={
            "web_search_one": "web_search",
            "web_search_two": "web_search",
            "github_issue": "github",
        },
    )

    assert "active_skill: (none)" in block
    assert "tool_policy_active: false" in block
    assert "available_tools: rag_search, MCP family: web_search, MCP family: github" in block
    assert "web_search_two" not in block
    assert "denied_tools: (none)" in block


def test_render_tool_availability_block_for_disallow_only_policy():
    runtime = SimpleNamespace(
        name="read-only",
        task_mode=None,
        allowed_tools=frozenset(),
        denied_tools=frozenset({"bash"}),
        tool_policy_active=True,
    )

    block = render_tool_availability_block(
        skill_runtime=runtime,
        base_tool_names=["read_file", "bash"],
    )

    assert "active_skill: read-only" in block
    assert "task_mode: (none)" in block
    assert "tool_policy_active: true" in block
    assert "available_tools: read_file" in block
    assert "denied_tools: bash" in block
    assert "unavailable_base_tools: bash" in block


def test_render_tool_availability_block_defaults_to_base_inventory():
    from agent.tools.inventory import base_tool_names

    block = render_tool_availability_block()

    assert "active_skill: (none)" in block
    assert "tool_policy_active: false" in block
    for name in base_tool_names():
        assert name in block


def test_render_tool_availability_block_keeps_empty_list_semantics():
    block = render_tool_availability_block(base_tool_names=[])

    assert "available_tools: (none)" in block
    assert "unavailable_base_tools: (none)" in block
