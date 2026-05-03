"""prompt_toolkit-backed terminal input helpers."""

from typing import Awaitable, Callable

from prompt_toolkit import PromptSession

LineReader = Callable[[str], Awaitable[str]]


def build_prompt_session() -> PromptSession[str]:
    """Create the shared interactive prompt session for the chat CLI."""
    return PromptSession()


def build_line_reader(
    session: PromptSession[str] | None = None,
) -> LineReader:
    """Build an async line reader backed by a persistent prompt session."""
    prompt_session = session or build_prompt_session()

    async def _read_line(prompt: str) -> str:
        return await prompt_session.prompt_async(prompt)

    return _read_line
