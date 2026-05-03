"""prompt_toolkit-backed terminal input helpers."""

from typing import Awaitable, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from agent.cli.slash_commands import SlashCommandRegistry

LineReader = Callable[[str], Awaitable[str]]


class SlashCommandCompleter(Completer):
    """Complete slash commands when the prompt starts with `/`."""

    def __init__(self, registry: SlashCommandRegistry):
        self._registry = registry

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        body = text[1:]
        if " " in body:
            return

        for command in self._registry.matching_commands(body):
            alias_suffix = ""
            if command.aliases:
                alias_suffix = " | " + ", ".join(f"/{alias}" for alias in command.aliases)
            yield Completion(
                command.name,
                start_position=-len(body),
                display=f"/{command.name}",
                display_meta=f"{command.description}{alias_suffix}",
            )


def build_prompt_session(
    command_registry: SlashCommandRegistry | None = None,
) -> PromptSession[str]:
    """Create the shared interactive prompt session for the chat CLI."""
    completer = None
    if command_registry is not None:
        completer = SlashCommandCompleter(command_registry)
    return PromptSession(
        completer=completer,
        complete_while_typing=bool(completer),
        reserve_space_for_menu=8 if completer else 0,
    )


def build_line_reader(
    session: PromptSession[str] | None = None,
    command_registry: SlashCommandRegistry | None = None,
) -> LineReader:
    """Build an async line reader backed by a persistent prompt session."""
    prompt_session = session or build_prompt_session(command_registry=command_registry)

    async def _read_line(prompt: str) -> str:
        return await prompt_session.prompt_async(prompt)

    return _read_line
