"""Path helpers shared by the agent core, CLI, and tests."""

from pathlib import Path


def find_app_root() -> Path:
    """Walk up from this file to the app project root."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("could not locate app project root (no pyproject.toml found)")
