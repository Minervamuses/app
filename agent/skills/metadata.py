"""Discovery of local Agent Skills."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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

    name = frontmatter.get("name", skill_file.parent.name).strip()
    description = frontmatter.get("description", "").strip()
    if not name or not description:
        return None

    return SkillMetadata(
        name=name,
        description=description,
        path=skill_file.resolve(),
    )


def _parse_frontmatter(text: str) -> dict[str, str] | None:
    """Parse a minimal YAML frontmatter block.

    This intentionally supports only the subset we rely on here:
    top-level scalar keys plus indented continuation lines.
    """
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

    data: dict[str, str] = {}
    current_key: str | None = None
    current_parts: list[str] = []

    def _flush() -> None:
        nonlocal current_key, current_parts
        if current_key is None:
            return
        value = " ".join(part for part in current_parts if part).strip()
        data[current_key] = _strip_wrapping_quotes(value)
        current_key = None
        current_parts = []

    for raw_line in lines[1:end_idx]:
        if not raw_line.strip():
            continue

        if raw_line[:1].isspace():
            if current_key is not None:
                current_parts.append(raw_line.strip())
            continue

        if ":" not in raw_line:
            continue

        _flush()
        key, raw_value = raw_line.split(":", 1)
        current_key = key.strip()
        value = raw_value.strip()
        current_parts = [] if value in {">", "|"} else [value]

    _flush()
    return data


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
