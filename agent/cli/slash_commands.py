"""Slash command parsing, registry, and local command handlers."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Awaitable, Callable

from rag import ingest_repo, ingest_single, list_diff, prune_orphans


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
                name="init",
                description="Ingest the parent repo (excluding this app project).",
                handler=_handle_init,
            ),
            SlashCommand(
                name="ingest",
                description="Upsert a file or folder into the rag store.",
                handler=_handle_ingest,
            ),
            SlashCommand(
                name="sync",
                description="Show files on disk vs in the rag store (dry run).",
                handler=_handle_sync,
            ),
            SlashCommand(
                name="prune",
                description="Remove store entries whose source file is gone (add --yes to apply).",
                handler=_handle_prune,
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


def _resolve_target(arg: str | None) -> Path:
    """Expand `~` and resolve the target path argument."""
    raw = arg if arg else "."
    return Path(raw).expanduser().resolve()


def _find_app_root() -> Path:
    """Walk up from this file to the app project root (its pyproject.toml)."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("could not locate app project root (no pyproject.toml found)")


async def _handle_init(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    if parsed.args:
        raise SlashCommandError("/init takes no arguments")

    app_root = _find_app_root()
    parent_repo = app_root.parent
    skip = {app_root.name}

    files, chunks = await asyncio.to_thread(
        ingest_repo,
        str(parent_repo),
        config=context.session.config,
        skip_rel_paths=skip,
    )
    return SlashCommandResult(
        message=(
            f"initialized: {files} files, {chunks} chunks "
            f"(root={parent_repo}, excluded {app_root.name}/)"
        )
    )


async def _handle_ingest(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    if not parsed.args:
        return SlashCommandResult(message="usage: /ingest <file-or-folder>")
    if len(parsed.args) > 1:
        raise SlashCommandError("/ingest takes exactly one path argument")

    target = _resolve_target(parsed.args[0])
    if not target.exists():
        raise SlashCommandError(f"path does not exist: {target}")

    config = context.session.config

    if target.is_file():
        pid, count = await asyncio.to_thread(
            ingest_single, str(target), config=config
        )
        return SlashCommandResult(
            message=f"ingested {pid} ({count} chunks)"
        )

    if target.is_dir():
        files, chunks = await asyncio.to_thread(
            ingest_repo, str(target), config=config
        )
        return SlashCommandResult(
            message=f"ingested {files} files ({chunks} chunks) under {target}"
        )

    raise SlashCommandError(f"unsupported path type: {target}")


async def _handle_sync(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    if len(parsed.args) > 1:
        raise SlashCommandError("/sync takes at most one path argument")

    target = _resolve_target(parsed.args[0] if parsed.args else None)
    if not target.is_dir():
        raise SlashCommandError(f"not a directory: {target}")

    diff = await asyncio.to_thread(
        list_diff, str(target), context.session.config
    )

    lines = [f"Diff against {target}:"]
    missing_store = diff["missing_from_store"]
    missing_disk = diff["missing_from_disk"]

    lines.append(f"  on disk, not in store ({len(missing_store)}):")
    if missing_store:
        lines.extend(f"    + {path}" for path in missing_store)
    else:
        lines.append("    (none)")

    lines.append(f"  in store, not on disk ({len(missing_disk)}):")
    if missing_disk:
        lines.extend(f"    - {path}" for path in missing_disk)
    else:
        lines.append("    (none)")

    return SlashCommandResult(message="\n".join(lines))


async def _handle_prune(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    args = list(parsed.args)
    apply = False
    if "--yes" in args:
        apply = True
        args = [a for a in args if a != "--yes"]
    if len(args) > 1:
        raise SlashCommandError("/prune takes at most one path argument")

    target = _resolve_target(args[0] if args else None)
    if not target.is_dir():
        raise SlashCommandError(f"not a directory: {target}")

    config = context.session.config

    if not apply:
        diff = await asyncio.to_thread(list_diff, str(target), config)
        orphans = diff["missing_from_disk"]
        lines = [f"Would prune {len(orphans)} orphaned pid(s) under {target}:"]
        if orphans:
            lines.extend(f"  - {path}" for path in orphans)
            lines.append("Re-run with --yes to apply.")
        else:
            lines.append("  (none)")
        return SlashCommandResult(message="\n".join(lines))

    removed = await asyncio.to_thread(
        prune_orphans, str(target), config
    )
    return SlashCommandResult(
        message=f"pruned {len(removed)} orphaned pid(s) under {target}"
    )
