"""Tests for local skill discovery and prompt rendering."""

from agent.config import AgentConfig
from agent.state import AgentState
from agent.skills import build_skill_trace_entries, build_skills_prompt, discover_skills
from agent.session import ChatSession


def test_discover_skills_reads_name_description_and_path(tmp_path):
    skills_dir = tmp_path / "skills"
    target = skills_dir / "sample-skill"
    target.mkdir(parents=True)
    skill_file = target / "SKILL.md"
    skill_file.write_text(
        """---
name: sample-skill
description: Use when the user wants to draft a paper
  abstract or revise a manuscript introduction.
---

# Sample
""",
        encoding="utf-8",
    )

    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))
    skills = discover_skills(cfg)

    assert len(skills) == 1
    assert skills[0].name == "sample-skill"
    assert skills[0].description == (
        "Use when the user wants to draft a paper abstract or revise a manuscript introduction."
    )
    assert skills[0].path == skill_file.resolve()


def test_build_skills_prompt_lists_instructions_and_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skills_dir = tmp_path / "skills"
    target = skills_dir / "paper-writing"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        """---
name: paper-writing
description: Use when the user wants help with academic writing.
---
""",
        encoding="utf-8",
    )

    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir="skills")
    prompt = build_skills_prompt(cfg)

    assert "Local skills (metadata only at startup):" in prompt
    assert "Before following a skill, read its full `SKILL.md` with `read_file`." in prompt
    assert "paper-writing: Use when the user wants help with academic writing." in prompt
    assert "read `skills/paper-writing/SKILL.md`" in prompt


def test_build_skill_trace_entries_exposes_prompt_visible_metadata(tmp_path):
    skills_dir = tmp_path / "skills"
    target = skills_dir / "paper-writing"
    target.mkdir(parents=True)
    skill_file = target / "SKILL.md"
    skill_file.write_text(
        """---
name: paper-writing
description: Use when the user wants help with academic writing.
---
""",
        encoding="utf-8",
    )

    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))
    skills = discover_skills(cfg)
    entries = build_skill_trace_entries(skills)

    assert entries == [
        {
            "type": "skill_metadata",
            "name": "paper-writing",
            "path": str(skill_file.resolve()),
            "load": "metadata",
            "description": "Use when the user wants help with academic writing.",
        }
    ]


def test_chat_session_includes_discovered_skills_in_system_prompt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skills_dir = tmp_path / "skills"
    target = skills_dir / "paper-writing"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        """---
name: paper-writing
description: Use when the user wants help with academic writing.
---
""",
        encoding="utf-8",
    )

    class _FakeGraph:
        async def astream(self, state, config=None, stream_mode="updates"):
            if False:  # pragma: no cover
                yield None

    monkeypatch.setattr(
        "agent.session.build_graph",
        lambda _cfg, extra_tools=None, history_store=None: _FakeGraph(),
    )

    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir="skills")
    session = ChatSession(cfg)

    prompt = session.system_prompt_message.content
    assert "**read_file**" in prompt
    assert "paper-writing: Use when the user wants help with academic writing." in prompt
    assert session.startup_trace_entries() == [
        {
            "type": "skill_metadata",
            "name": "paper-writing",
            "path": "skills/paper-writing/SKILL.md",
            "load": "metadata",
            "description": "Use when the user wants help with academic writing.",
        }
    ]


def test_agent_state_skill_fields_are_optional():
    assert "messages" in AgentState.__optional_keys__
    assert "active_skill" in AgentState.__optional_keys__
    assert "loaded_references" in AgentState.__optional_keys__
