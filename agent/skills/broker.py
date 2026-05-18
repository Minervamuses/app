"""Capability resolution for skill-scoped tool policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CAPABILITY_MAP_PATH = Path(__file__).resolve().parent / "capability_map.yaml"


@dataclass(frozen=True)
class CapabilityResolution:
    """Resolved skill capability and tool policy state."""

    allowed: frozenset[str]
    denied: frozenset[str]
    requested_required: frozenset[str]
    requested_optional: frozenset[str]
    unresolved_required: frozenset[str]
    unresolved_optional: frozenset[str]
    policy_active: bool


def load_capability_map(path: str | Path | None = None) -> dict[str, Any]:
    """Load the runtime capability map."""
    source = Path(path).expanduser() if path else DEFAULT_CAPABILITY_MAP_PATH
    with source.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {"capabilities": {}}
    capabilities = data.get("capabilities")
    if not isinstance(capabilities, dict):
        data["capabilities"] = {}
    return data


def resolve_capabilities(
    manifest: Mapping[str, Any] | None,
    all_tools: Sequence[Any],
    mcp_families: Mapping[str, str] | None = None,
    capability_map: Mapping[str, Any] | None = None,
) -> CapabilityResolution:
    """Resolve a skill manifest into allowed and denied tool names.

    Deny rules win over capability grants. Required capabilities must resolve
    to at least one usable tool after deny rules are applied; optional
    capabilities may remain unresolved without blocking activation.
    """
    if not manifest:
        return CapabilityResolution(
            allowed=frozenset(),
            denied=frozenset(),
            requested_required=frozenset(),
            requested_optional=frozenset(),
            unresolved_required=frozenset(),
            unresolved_optional=frozenset(),
            policy_active=False,
        )

    tool_names = {getattr(tool, "name", str(tool)) for tool in all_tools}
    family_by_tool = dict(mcp_families or {})
    cap_map = capability_map or load_capability_map()
    policy_active = _policy_active(manifest)

    denied = _resolve_denied_tools(
        manifest,
        tool_names=tool_names,
        family_by_tool=family_by_tool,
    )

    requested_required = frozenset(_requested_capability_ids(manifest, "required"))
    requested_optional = frozenset(_requested_capability_ids(manifest, "optional"))

    allowed: set[str] = set()
    unresolved_required: set[str] = set()
    unresolved_optional: set[str] = set()

    for capability_id in requested_required:
        usable_tools = _usable_tools_for_capability(
            capability_id,
            tool_names=tool_names,
            family_by_tool=family_by_tool,
            capability_map=cap_map,
            denied=denied,
        )
        if not usable_tools:
            unresolved_required.add(capability_id)
            continue
        allowed.update(usable_tools)

    for capability_id in requested_optional:
        usable_tools = _usable_tools_for_capability(
            capability_id,
            tool_names=tool_names,
            family_by_tool=family_by_tool,
            capability_map=cap_map,
            denied=denied,
        )
        if not usable_tools:
            unresolved_optional.add(capability_id)
            continue
        allowed.update(usable_tools)

    resolution = CapabilityResolution(
        allowed=frozenset(allowed),
        denied=frozenset(denied),
        requested_required=requested_required,
        requested_optional=requested_optional,
        unresolved_required=frozenset(unresolved_required),
        unresolved_optional=frozenset(unresolved_optional),
        policy_active=policy_active,
    )

    if resolution.unresolved_required:
        unresolved = ", ".join(sorted(resolution.unresolved_required))
        raise ValueError(
            "required skill capabilities could not be resolved to usable tools: "
            f"{unresolved}"
        )

    return resolution


def _policy_active(manifest: Mapping[str, Any]) -> bool:
    return isinstance(manifest.get("capabilities"), Mapping) or isinstance(
        manifest.get("tool_policy"),
        Mapping,
    )


def _requested_capability_ids(manifest: Mapping[str, Any], key: str) -> list[str]:
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return []

    out: list[str] = []
    raw_items = capabilities.get(key, [])
    if not isinstance(raw_items, list):
        return out
    for item in raw_items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, Mapping):
            value = item.get("id")
            if isinstance(value, str):
                out.append(value)
    return out


def _usable_tools_for_capability(
    capability_id: str,
    *,
    tool_names: set[str],
    family_by_tool: Mapping[str, str],
    capability_map: Mapping[str, Any],
    denied: set[str],
) -> set[str]:
    return _tools_for_capability(
        capability_id,
        tool_names=tool_names,
        family_by_tool=family_by_tool,
        capability_map=capability_map,
    ) - denied


def _tools_for_capability(
    capability_id: str,
    *,
    tool_names: set[str],
    family_by_tool: Mapping[str, str],
    capability_map: Mapping[str, Any],
) -> set[str]:
    capabilities = capability_map.get("capabilities", {})
    if not isinstance(capabilities, Mapping):
        return set()
    entry = capabilities.get(capability_id, {})
    if not isinstance(entry, Mapping):
        return set()

    selected: set[str] = set()
    for name in _string_list(entry.get("local_tools")):
        if name in tool_names:
            selected.add(name)

    families = set(_string_list(entry.get("mcp_families")))
    if families:
        selected.update(
            name
            for name, family in family_by_tool.items()
            if family in families and name in tool_names
        )
    return selected


def _resolve_denied_tools(
    manifest: Mapping[str, Any],
    *,
    tool_names: set[str],
    family_by_tool: Mapping[str, str],
) -> set[str]:
    tool_policy = manifest.get("tool_policy")
    if not isinstance(tool_policy, Mapping):
        return set()

    denied: set[str] = set()
    for pattern in _string_list(tool_policy.get("disallow")):
        if pattern.endswith(".*"):
            family = pattern[:-2]
            denied.update(
                name
                for name, tool_family in family_by_tool.items()
                if tool_family == family and name in tool_names
            )
            continue
        if pattern in tool_names:
            denied.add(pattern)
    return denied


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]
