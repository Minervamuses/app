"""Tests for skill capability resolution."""

import pytest
from langchain_core.tools import tool

from agent.skills.broker import resolve_capabilities


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


@tool("web_search")
def _web_search(query: str) -> str:
    """Search web."""
    return query


@tool("github_repo")
def _github_repo(repo: str) -> str:
    """Read GitHub."""
    return repo


def test_resolve_capabilities_maps_local_and_mcp_tools():
    manifest = {
        "capabilities": {
            "required": ["file.read"],
            "optional": [{"id": "web.search"}],
        }
    }
    capability_map = {
        "capabilities": {
            "file.read": {"local_tools": ["read_file"]},
            "web.search": {"mcp_families": ["web_search"]},
        }
    }

    resolution = resolve_capabilities(
        manifest,
        [_read_file, _web_search, _github_repo],
        {"web_search": "web_search", "github_repo": "github"},
        capability_map,
    )

    assert resolution.allowed == frozenset({"read_file", "web_search"})
    assert resolution.denied == frozenset()
    assert resolution.requested_required == frozenset({"file.read"})
    assert resolution.requested_optional == frozenset({"web.search"})
    assert resolution.unresolved_required == frozenset()
    assert resolution.unresolved_optional == frozenset()
    assert resolution.policy_active is True


def test_resolve_capabilities_ignores_unavailable_mcp_family():
    manifest = {"capabilities": {"optional": [{"id": "web.search"}]}}
    capability_map = {"capabilities": {"web.search": {"mcp_families": ["web_search"]}}}

    resolution = resolve_capabilities(
        manifest,
        [_read_file],
        {},
        capability_map,
    )

    assert resolution.allowed == frozenset()
    assert resolution.denied == frozenset()
    assert resolution.unresolved_optional == frozenset({"web.search"})
    assert resolution.policy_active is True


def test_resolve_capabilities_required_denied_to_zero_raises():
    manifest = {
        "capabilities": {"required": ["file.read", "shell.execute"]},
        "tool_policy": {"disallow": ["bash"]},
    }
    capability_map = {
        "capabilities": {
            "file.read": {"local_tools": ["read_file"]},
            "shell.execute": {"local_tools": ["bash"]},
        }
    }

    with pytest.raises(ValueError, match="shell.execute"):
        resolve_capabilities(
            manifest,
            [_read_file, _bash],
            {},
            capability_map,
        )


def test_resolve_capabilities_denies_win_over_optional_grants():
    manifest = {
        "capabilities": {"required": ["file.read"], "optional": ["shell.execute"]},
        "tool_policy": {"disallow": ["bash"]},
    }
    capability_map = {
        "capabilities": {
            "file.read": {"local_tools": ["read_file"]},
            "shell.execute": {"local_tools": ["bash"]},
        }
    }

    resolution = resolve_capabilities(
        manifest,
        [_read_file, _bash],
        {},
        capability_map,
    )

    assert resolution.allowed == frozenset({"read_file"})
    assert resolution.denied == frozenset({"bash"})
    assert resolution.unresolved_optional == frozenset({"shell.execute"})


def test_resolve_capabilities_denies_mcp_family_pattern():
    manifest = {
        "capabilities": {"optional": [{"id": "github.repo.read"}]},
        "tool_policy": {"disallow": ["github.*"]},
    }
    capability_map = {
        "capabilities": {"github.repo.read": {"mcp_families": ["github"]}}
    }

    resolution = resolve_capabilities(
        manifest,
        [_github_repo],
        {"github_repo": "github"},
        capability_map,
    )

    assert resolution.allowed == frozenset()
    assert resolution.denied == frozenset({"github_repo"})
    assert resolution.unresolved_optional == frozenset({"github.repo.read"})


def test_resolve_capabilities_no_policy_is_inactive():
    resolution = resolve_capabilities({}, [_read_file], {}, {"capabilities": {}})

    assert resolution.allowed == frozenset()
    assert resolution.denied == frozenset()
    assert resolution.policy_active is False


def test_resolve_capabilities_disallow_only_policy_is_active():
    manifest = {"tool_policy": {"disallow": ["bash"]}}

    resolution = resolve_capabilities(
        manifest,
        [_read_file, _bash],
        {},
        {"capabilities": {}},
    )

    assert resolution.allowed == frozenset()
    assert resolution.denied == frozenset({"bash"})
    assert resolution.policy_active is True


def test_resolve_capabilities_unknown_required_raises():
    manifest = {"capabilities": {"required": ["shell.exec"]}}

    with pytest.raises(ValueError, match="shell.exec"):
        resolve_capabilities(
            manifest,
            [_bash],
            {},
            {"capabilities": {"shell.execute": {"local_tools": ["bash"]}}},
        )
