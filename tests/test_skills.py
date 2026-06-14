"""Tests for local skill discovery and prompt rendering."""

from agent.config import AgentConfig
from agent.state import AgentState
from agent.skills import discover_skills
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


def test_discover_skills_reads_yaml_block_scalar_description(tmp_path):
    skills_dir = tmp_path / "skills"
    target = skills_dir / "sample-skill"
    target.mkdir(parents=True)
    skill_file = target / "SKILL.md"
    skill_file.write_text(
        """---
name: sample-skill
description: >
  Use when the user wants to draft a paper
  abstract or revise a manuscript introduction.
---

# Sample
""",
        encoding="utf-8",
    )

    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))
    skills = discover_skills(cfg)

    assert len(skills) == 1
    assert skills[0].description == (
        "Use when the user wants to draft a paper abstract or revise a manuscript introduction."
    )


def test_discover_skills_skips_malformed_yaml_frontmatter(tmp_path):
    skills_dir = tmp_path / "skills"
    target = skills_dir / "bad-skill"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        """---
name: bad-skill
description: [unterminated
---

# Bad
""",
        encoding="utf-8",
    )

    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir=str(skills_dir))

    assert discover_skills(cfg) == []


def test_chat_session_discovers_skills_without_injecting_into_system_prompt(tmp_path, monkeypatch):
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
        lambda _cfg, extra_tools=None, history_store=None, **kwargs: _FakeGraph(),
    )

    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir="skills")
    session = ChatSession(cfg)

    prompt = session.system_prompt_message.content
    assert "**read_file**" in prompt
    assert "paper-writing" not in prompt
    assert "Use when the user wants help with academic writing." not in prompt

    assert len(session.loaded_skills) == 1
    assert session.loaded_skills[0].name == "paper-writing"


def test_agent_state_skill_fields_are_optional():
    assert "messages" in AgentState.__optional_keys__
    assert "active_skill" in AgentState.__optional_keys__
    assert "loaded_references" in AgentState.__optional_keys__
    assert "tool_policy_active" in AgentState.__optional_keys__
    assert "validation_retry_requested" in AgentState.__optional_keys__


def test_agent_config_exposes_skill_runtime_toggles(tmp_path):
    cfg = AgentConfig(persist_dir=str(tmp_path))

    assert cfg.skill_validation_enabled is True
    assert cfg.skill_max_validation_retries == 1
    assert cfg.skill_capability_map_path is None
    assert cfg.skill_max_pinned_reference_chars == 65536
    assert cfg.skill_max_total_skill_context_chars == 200000


def test_chat_session_activate_skill_sets_status_and_prompt_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    skills_dir = tmp_path / "skills"
    target = skills_dir / "paper-writing"
    refs = target / "references"
    refs.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        """---
name: paper-writing
description: Use when the user wants help with academic writing.
---

# Paper Writing
""",
        encoding="utf-8",
    )
    (refs / "guide.md").write_text("reference guide", encoding="utf-8")
    (target / "manifest.yaml").write_text(
        """
capabilities:
  required:
    - file.read
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

    class _FakeGraph:
        async def astream(self, state, config=None, stream_mode="updates"):
            if False:  # pragma: no cover
                yield state

    monkeypatch.setattr(
        "agent.session.build_graph",
        lambda _cfg, extra_tools=None, history_store=None, **kwargs: _FakeGraph(),
    )

    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir="skills")
    session = ChatSession(cfg)
    runtime = session.activate_skill("paper-writing", "revision")

    status = session.status_snapshot()
    prompt = "\n".join(message.content for message in session._prompt_history())

    assert runtime.name == "paper-writing"
    assert status["active_skill"] == "paper-writing"
    assert status["task_mode"] == "revision"
    assert "[Active skill]" in prompt
    assert "# Paper Writing" in prompt
    assert "reference guide" in prompt


def test_chat_session_deactivate_skill_clears_status(tmp_path, monkeypatch):
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
                yield state

    monkeypatch.setattr(
        "agent.session.build_graph",
        lambda _cfg, extra_tools=None, history_store=None, **kwargs: _FakeGraph(),
    )

    cfg = AgentConfig(persist_dir=str(tmp_path), skills_dir="skills")
    session = ChatSession(cfg)
    session.activate_skill("paper-writing")
    session.deactivate_skill()

    status = session.status_snapshot()
    assert status["active_skill"] == ""
    assert status["task_mode"] == ""
