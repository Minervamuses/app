"""Discovery of local Agent Skills."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from agent.config import AgentConfig

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"


@dataclass(frozen=True)
class SkillMetadata:
    """Minimal metadata surfaced to the model at startup."""

    name: str
    description: str
    path: Path


def resolve_skills_dir(config: AgentConfig | None = None) -> Path:
    """Resolve the local skills directory for this runtime."""
    raw_path = getattr(config, "skills_dir", None) if config is not None else None
    if not raw_path:
        return DEFAULT_SKILLS_DIR

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def discover_skills(config: AgentConfig | None = None) -> list[SkillMetadata]:
    """Discover local skills by scanning `*/SKILL.md` under the skills directory."""
    skills_dir = resolve_skills_dir(config)
    if not skills_dir.exists():
        return []

    skills: list[SkillMetadata] = []
    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            metadata = _read_skill_metadata(skill_file)
        except Exception as exc:  # pragma: no cover - defensive log path
            logger.warning("Skipping skill %s: %s", skill_file, exc)
            continue
        if metadata is None:
            continue
        skills.append(metadata)
    return skills


def _read_skill_metadata(skill_file: Path) -> SkillMetadata | None:
    """Extract name and description from a skill's YAML frontmatter."""
    text = skill_file.read_text(encoding="utf-8")
    frontmatter = _parse_frontmatter(text)
    if frontmatter is None:
        return None

    name = _metadata_text(frontmatter, "name", skill_file.parent.name).strip()
    description = _metadata_text(frontmatter, "description", "").strip()
    if not name or not description:
        return None

    return SkillMetadata(
        name=name,
        description=description,
        path=skill_file.resolve(),
    )


def _parse_frontmatter(text: str) -> Mapping[str, Any] | None:
    """Parse a YAML frontmatter block."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    end_idx = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return None

    raw_frontmatter = "\n".join(lines[1:end_idx])
    data = yaml.safe_load(raw_frontmatter)
    if not isinstance(data, Mapping):
        return None
    return data


def _metadata_text(
    frontmatter: Mapping[str, Any],
    key: str,
    default: str,
) -> str:
    value = frontmatter.get(key, default)
    return value if isinstance(value, str) else ""
