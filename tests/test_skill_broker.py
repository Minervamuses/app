"""Tests for skill capability resolution."""

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

    allowed, denied = resolve_capabilities(
        manifest,
        [_read_file, _web_search, _github_repo],
        {"web_search": "web_search", "github_repo": "github"},
        capability_map,
    )

    assert allowed == {"read_file", "web_search"}
    assert denied == set()


def test_resolve_capabilities_ignores_unavailable_mcp_family():
    manifest = {"capabilities": {"optional": [{"id": "web.search"}]}}
    capability_map = {"capabilities": {"web.search": {"mcp_families": ["web_search"]}}}

    allowed, denied = resolve_capabilities(
        manifest,
        [_read_file],
        {},
        capability_map,
    )

    assert allowed == set()
    assert denied == set()


def test_resolve_capabilities_denies_win_over_grants():
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

    allowed, denied = resolve_capabilities(
        manifest,
        [_read_file, _bash],
        {},
        capability_map,
    )

    assert allowed == {"read_file"}
    assert denied == {"bash"}


def test_resolve_capabilities_denies_mcp_family_pattern():
    manifest = {
        "capabilities": {"optional": [{"id": "github.repo.read"}]},
        "tool_policy": {"disallow": ["github.*"]},
    }
    capability_map = {
        "capabilities": {"github.repo.read": {"mcp_families": ["github"]}}
    }

    allowed, denied = resolve_capabilities(
        manifest,
        [_github_repo],
        {"github_repo": "github"},
        capability_map,
    )

    assert allowed == set()
    assert denied == {"github_repo"}
