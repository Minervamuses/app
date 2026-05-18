"""Runtime representation and resource loading for active skills."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from agent.config import AgentConfig
from agent.skills.broker import (
    CapabilityResolution,
    load_capability_map,
    resolve_capabilities,
)
from agent.skills.manifest_schema import validate_skill_manifest
from agent.skills.metadata import SkillMetadata, discover_skills


@dataclass(frozen=True)
class SkillRuntime:
    """Loaded runtime context for one active skill."""

    name: str
    root: Path
    instructions: str
    manifest: Mapping[str, Any]
    pinned_references: dict[str, str]
    allowed_tools: frozenset[str]
    denied_tools: frozenset[str]
    capability_resolution: CapabilityResolution
    task_mode: str | None = None

    @property
    def tool_policy_active(self) -> bool:
        """Whether this skill has an active tool policy."""
        return self.capability_resolution.policy_active

    def read_skill_resource(self, rel_path: str) -> str:
        """Read a resource path relative to this skill root."""
        path = (self.root / rel_path).expanduser().resolve()
        root = self.root.resolve()
        if not path.is_relative_to(root):
            raise PermissionError(f"skill resource escapes skill root: {rel_path}")
        if not path.exists():
            raise FileNotFoundError(f"skill resource does not exist: {rel_path}")
        if not path.is_file():
            raise IsADirectoryError(f"skill resource is not a regular file: {rel_path}")
        return path.read_text(encoding="utf-8", errors="replace")

    def context_block(self) -> str:
        """Render prompt-visible context for the active skill."""
        lines = [
            "[Active skill]",
            f"name: {self.name}",
        ]
        if self.task_mode:
            lines.append(f"task_mode: {self.task_mode}")
        lines.extend([
            "",
            "[SKILL.md]",
            self.instructions,
        ])
        if self.pinned_references:
            lines.append("")
            lines.append("[Pinned skill references]")
            for path, content in self.pinned_references.items():
                lines.extend([
                    f"--- {path} ---",
                    content,
                ])
        return "\n".join(lines)


def load_skill_runtime(
    name: str,
    *,
    config: AgentConfig,
    all_tools: Sequence[Any],
    mcp_families: Mapping[str, str] | None = None,
    task_mode: str | None = None,
    capability_map: Mapping[str, Any] | None = None,
) -> SkillRuntime:
    """Load a skill and resolve its runtime tool policy."""
    metadata = find_skill_metadata(name, config=config)
    if metadata is None:
        raise KeyError(f"unknown skill: {name}")

    root = metadata.path.parent.resolve()
    manifest = load_skill_manifest(root)
    _validate_task_mode(task_mode, manifest)

    instructions = metadata.path.read_text(encoding="utf-8", errors="replace")
    cap_map = capability_map or load_capability_map(
        getattr(config, "skill_capability_map_path", None)
    )
    capability_resolution = resolve_capabilities(
        manifest,
        all_tools,
        mcp_families or {},
        cap_map,
    )
    runtime = SkillRuntime(
        name=metadata.name,
        root=root,
        instructions=instructions,
        manifest=manifest,
        pinned_references={},
        allowed_tools=capability_resolution.allowed,
        denied_tools=capability_resolution.denied,
        capability_resolution=capability_resolution,
        task_mode=task_mode,
    )
    pinned_references = _load_pinned_references(runtime, manifest, config)
    _validate_total_skill_context(
        instructions=runtime.instructions,
        pinned_references=pinned_references,
        config=config,
    )
    return SkillRuntime(
        name=runtime.name,
        root=runtime.root,
        instructions=runtime.instructions,
        manifest=runtime.manifest,
        pinned_references=pinned_references,
        allowed_tools=runtime.allowed_tools,
        denied_tools=runtime.denied_tools,
        capability_resolution=runtime.capability_resolution,
        task_mode=runtime.task_mode,
    )


def find_skill_metadata(name: str, *, config: AgentConfig) -> SkillMetadata | None:
    """Find a discovered skill by name."""
    normalized = name.casefold()
    for skill in discover_skills(config):
        if skill.name.casefold() == normalized:
            return skill
    return None


def load_skill_manifest(root: Path) -> dict[str, Any]:
    """Load a skill manifest, if present."""
    manifest_path = root / "manifest.yaml"
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"skill manifest must be a mapping: {manifest_path}")
    return validate_skill_manifest(data, source=manifest_path)


def _load_pinned_references(
    runtime: SkillRuntime,
    manifest: Mapping[str, Any],
    config: AgentConfig,
) -> dict[str, str]:
    resources = manifest.get("resources")
    if not isinstance(resources, list):
        return {}

    loaded: dict[str, str] = {}
    for item in resources:
        if not isinstance(item, Mapping):
            continue
        if item.get("pinned") is not True:
            continue
        path = item.get("path")
        if not isinstance(path, str):
            continue
        content = runtime.read_skill_resource(path)
        _validate_pinned_reference_size(path, content, config)
        loaded[path] = content
    return loaded


def _validate_pinned_reference_size(
    path: str,
    content: str,
    config: AgentConfig,
) -> None:
    limit = config.skill_max_pinned_reference_chars
    size = len(content)
    if size > limit:
        raise ValueError(
            f"pinned skill reference too large: {path} "
            f"({size} chars, limit {limit})"
        )


def _validate_total_skill_context(
    *,
    instructions: str,
    pinned_references: Mapping[str, str],
    config: AgentConfig,
) -> None:
    limit = config.skill_max_total_skill_context_chars
    size = len(instructions) + sum(len(content) for content in pinned_references.values())
    if size > limit:
        raise ValueError(
            f"total skill context too large: {size} chars (limit {limit})"
        )


def _validate_task_mode(task_mode: str | None, manifest: Mapping[str, Any]) -> None:
    if task_mode is None:
        return
    modes = manifest.get("task_modes")
    if not isinstance(modes, list):
        return
    valid_modes = {mode for mode in modes if isinstance(mode, str)}
    if valid_modes and task_mode not in valid_modes:
        valid = ", ".join(sorted(valid_modes))
        raise ValueError(f"unknown task mode for skill: {task_mode} (available: {valid})")
