"""Slash command parsing, registry, and local command handlers."""

from dataclasses import dataclass
import shlex
from typing import Awaitable, Callable


class SlashCommandError(ValueError):
    """Raised when CLI slash command input is invalid."""


@dataclass(frozen=True)
class ParsedSlashCommand:
    """A slash command parsed from raw CLI input."""

    raw_text: str
    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class SlashCommandResult:
    """Outcome from executing a slash command locally."""

    message: str = ""
    should_exit: bool = False
    clear_screen: bool = False


@dataclass(frozen=True)
class SlashCommand:
    """Definition for one registered slash command."""

    name: str
    description: str
    handler: Callable[["SlashCommandContext", ParsedSlashCommand], Awaitable[SlashCommandResult]]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlashCommandContext:
    """Runtime context passed into slash command handlers."""

    session: object
    registry: "SlashCommandRegistry"


class SlashCommandRegistry:
    """Lookup and completion support for CLI slash commands."""

    def __init__(self, commands: list[SlashCommand]):
        self._commands = tuple(commands)
        self._by_name: dict[str, SlashCommand] = {}

        for command in self._commands:
            self._register_name(command.name, command)
            for alias in command.aliases:
                self._register_name(alias, command)

    def _register_name(self, name: str, command: SlashCommand) -> None:
        normalized = name.casefold()
        if normalized in self._by_name:
            raise ValueError(f"duplicate slash command name: {name}")
        self._by_name[normalized] = command

    def all_commands(self) -> tuple[SlashCommand, ...]:
        return self._commands

    def get(self, name: str) -> SlashCommand | None:
        return self._by_name.get(name.casefold())

    def matching_commands(self, prefix: str) -> tuple[SlashCommand, ...]:
        normalized = prefix.casefold()
        return tuple(
            command
            for command in self._commands
            if command.name.casefold().startswith(normalized)
        )


def parse_slash_command(raw_input: str) -> ParsedSlashCommand | None:
    """Parse a leading slash command, or return None for normal chat input."""
    text = raw_input.strip()
    if not text.startswith("/"):
        return None
    if text == "/":
        raise SlashCommandError("slash command cannot be empty")

    try:
        parts = shlex.split(text[1:])
    except ValueError as exc:
        raise SlashCommandError(f"invalid slash command: {exc}") from exc

    if not parts:
        raise SlashCommandError("slash command cannot be empty")

    return ParsedSlashCommand(
        raw_text=text,
        name=parts[0],
        args=tuple(parts[1:]),
    )


async def execute_slash_command(
    parsed: ParsedSlashCommand,
    context: SlashCommandContext,
) -> SlashCommandResult:
    """Resolve and run a slash command against the local CLI context."""
    command = context.registry.get(parsed.name)
    if command is None:
        raise SlashCommandError(f"unknown slash command: /{parsed.name}")
    return await command.handler(context, parsed)


def build_default_registry() -> SlashCommandRegistry:
    """Create the built-in local slash command set for the chat CLI."""
    return SlashCommandRegistry(
        [
            SlashCommand(
                name="help",
                description="Show available slash commands.",
                handler=_handle_help,
            ),
            SlashCommand(
                name="status",
                description="Show local session status.",
                handler=_handle_status,
            ),
            SlashCommand(
                name="clear",
                description="Clear the terminal screen.",
                handler=_handle_clear,
            ),
            SlashCommand(
                name="quit",
                description="Exit the chat CLI.",
                aliases=("exit",),
                handler=_handle_quit,
            ),
        ]
    )


async def _handle_help(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    del parsed

    lines = ["Available slash commands:"]
    for command in context.registry.all_commands():
        alias_suffix = ""
        if command.aliases:
            alias_list = ", ".join(f"/{alias}" for alias in command.aliases)
            alias_suffix = f" (aliases: {alias_list})"
        lines.append(f"/{command.name} - {command.description}{alias_suffix}")
    return SlashCommandResult(message="\n".join(lines))


async def _handle_status(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    del parsed

    status = context.session.status_snapshot()
    lines = [
        "Session status:",
        f"session_id: {status['session_id']}",
        f"turn_count: {status['turn_count']}",
        f"recent_turn_count: {status['recent_turn_count']}",
        f"recursion_limit: {status['recursion_limit']}",
        f"last_tool_calls: {status['last_tool_counts']}",
    ]
    return SlashCommandResult(message="\n".join(lines))


async def _handle_clear(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    del context, parsed
    return SlashCommandResult(clear_screen=True)


async def _handle_quit(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    del context, parsed
    return SlashCommandResult(should_exit=True)
