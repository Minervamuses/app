"""Compatibility package for legacy `kms.adapters` imports."""

from kms_agent.adapters import create_context_tool, create_explore_tool, create_search_tool

__all__ = [
    "create_context_tool",
    "create_explore_tool",
    "create_search_tool",
]
