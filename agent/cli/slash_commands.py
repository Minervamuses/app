"""Slash command parsing, registry, and local command handlers."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Awaitable, Callable

import yaml

from agent.llm.thinking import ExtendedModeNotConfigured, require_thinking_models
from agent.paths import find_app_root
from agent.skills import SkillMetadata, discover_skills, load_skill_manifest
from rag import ingest_repo, ingest_single, list_diff, prune_orphans


class SlashCommandError(ValueError):
    """Raised when CLI slash command input is invalid."""


_SKILL_USER_ERRORS = (KeyError, ValueError, OSError, yaml.YAMLError)


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
                name="mode",
                description="Switch session mode (interactive picker; pass a name for one-shot).",
                handler=_handle_mode,
            ),
            SlashCommand(
                name="thinking",
                description="Switch reasoning workflow depth (normal or extended).",
                handler=_handle_thinking,
            ),
            SlashCommand(
                name="skill",
                description="Activate or deactivate a local skill.",
                handler=_handle_skill,
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
        f"plan_mode: {status.get('plan_mode', False)}",
        f"plan_log_path: {status.get('plan_log_path', '') or 'none'}",
        f"thinking_mode: {status.get('thinking_mode', 'normal')}",
        f"active_skill: {status.get('active_skill', '') or 'none'}",
        f"task_mode: {status.get('task_mode', '') or 'none'}",
    ]
    return SlashCommandResult(message="\n".join(lines))


_THINKING_MODES = ("normal", "extended")
_THINKING_MODE_DESCRIPTIONS = {
    "normal": "default direct agent flow",
    "extended": "prompt rewrite + reviewer/reviser loop",
}


def _render_thinking_prompt(current: str) -> str:
    lines = [f"Current thinking mode: {current}", "Available thinking modes:"]
    for idx, mode in enumerate(_THINKING_MODES, start=1):
        lines.append(
            f"  [{idx}] {mode}  - {_THINKING_MODE_DESCRIPTIONS[mode]}"
        )
    lines.append("Select (number or name; Enter to cancel): ")
    return "\n".join(lines)


def _resolve_thinking_choice(raw: str) -> str | None:
    """Map raw user input to a thinking mode, or None for cancel."""
    cleaned = raw.strip().lower()
    if not cleaned or cleaned in {"q", "cancel"}:
        return None
    if cleaned.isdigit():
        idx = int(cleaned) - 1
        if 0 <= idx < len(_THINKING_MODES):
            return _THINKING_MODES[idx]
        raise SlashCommandError(f"invalid choice: {cleaned}")
    return cleaned


async def _handle_thinking(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    if len(parsed.args) > 1:
        raise SlashCommandError("usage: /thinking [normal|extended]")

    current = getattr(context.session, "thinking_mode", "normal")
    if parsed.args:
        target = parsed.args[0].strip().lower()
    else:
        raw = await asyncio.to_thread(input, _render_thinking_prompt(current))
        target = _resolve_thinking_choice(raw)
        if target is None:
            return SlashCommandResult(message="cancelled")

    if target not in _THINKING_MODES:
        valid = ", ".join(_THINKING_MODES)
        raise SlashCommandError(
            f"unknown thinking mode: {target} (available: {valid})"
        )

    if target == current:
        return SlashCommandResult(message=f"already in {current} thinking mode")

    if target == "extended":
        try:
            require_thinking_models(context.session.config)
        except ExtendedModeNotConfigured as exc:
            raise SlashCommandError(str(exc)) from exc

    setter = getattr(context.session, "set_thinking_mode", None)
    if setter is not None:
        setter(target)
    else:
        setattr(context.session, "thinking_mode", target)
    return SlashCommandResult(message=f"thinking -> {target}")


@dataclass(frozen=True)
class ModeSpec:
    """Definition for one selectable session mode."""

    name: str
    description: str
    enter: Callable[[object], Awaitable["Path | None"]]
    exit: Callable[[object], Awaitable[None]]


async def _enter_normal_mode(session: object) -> "Path | None":
    del session
    return None


async def _exit_normal_mode(session: object) -> None:
    del session


async def _enter_plan_mode(session: object) -> "Path | None":
    return await session.enter_plan_mode()


async def _exit_plan_mode(session: object) -> None:
    await session.exit_plan_mode()


_MODE_REGISTRY: dict[str, ModeSpec] = {
    "normal": ModeSpec(
        name="normal",
        description="turns saved to ChromaDB (default)",
        enter=_enter_normal_mode,
        exit=_exit_normal_mode,
    ),
    "plan": ModeSpec(
        name="plan",
        description="turns saved to plan_logs/, never indexed",
        enter=_enter_plan_mode,
        exit=_exit_plan_mode,
    ),
}


def _current_mode_name(session: object) -> str:
    return "plan" if getattr(session, "plan_mode", False) else "normal"


def _render_mode_prompt(current: str) -> str:
    lines = [f"Current mode: {current}", "Available modes:"]
    modes = list(_MODE_REGISTRY.values())
    for idx, spec in enumerate(modes, start=1):
        lines.append(f"  [{idx}] {spec.name}  - {spec.description}")
    lines.append("Select (number or name; Enter to cancel): ")
    return "\n".join(lines)


def _resolve_mode_choice(raw: str) -> str | None:
    """Map raw user input to a mode name, or None for cancel.

    Numeric input maps to registry order; name input is returned as-is for
    later validation by the handler. Cancel tokens: empty, ``q``, ``cancel``.
    """
    cleaned = raw.strip().lower()
    if not cleaned or cleaned in {"q", "cancel"}:
        return None
    if cleaned.isdigit():
        idx = int(cleaned) - 1
        modes = list(_MODE_REGISTRY.values())
        if 0 <= idx < len(modes):
            return modes[idx].name
        raise SlashCommandError(f"invalid choice: {cleaned}")
    return cleaned


async def _handle_mode(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    session = context.session
    if len(parsed.args) > 1:
        raise SlashCommandError("usage: /mode [name]")

    current = _current_mode_name(session)
    if parsed.args:
        target_name: str | None = parsed.args[0].strip().lower()
    else:
        raw = await asyncio.to_thread(input, _render_mode_prompt(current))
        target_name = _resolve_mode_choice(raw)
        if target_name is None:
            return SlashCommandResult(message="cancelled")

    if target_name not in _MODE_REGISTRY:
        valid = ", ".join(_MODE_REGISTRY)
        raise SlashCommandError(
            f"unknown mode: {target_name} (available: {valid})"
        )

    if target_name == current:
        return SlashCommandResult(message=f"already in {current} mode")

    await _MODE_REGISTRY[current].exit(session)
    log_path = await _MODE_REGISTRY[target_name].enter(session)

    suffix = f" -> {log_path}" if log_path else ""
    return SlashCommandResult(message=f"mode -> {target_name}{suffix}")


def _session_skills(session: object) -> list[SkillMetadata]:
    loaded = getattr(session, "loaded_skills", None)
    if loaded is not None:
        return list(loaded)
    config = getattr(session, "config", None)
    return discover_skills(config)


def _render_skill_prompt(session: object) -> str:
    current = getattr(getattr(session, "active_skill_runtime", None), "name", "")
    current_display = current or "none"
    lines = [f"Current skill: {current_display}", "Available skills:"]
    lines.append("  [0] none")
    for idx, skill in enumerate(_session_skills(session), start=1):
        lines.append(f"  [{idx}] {skill.name}")
    lines.append("Select (number or name; Enter to cancel): ")
    return "\n".join(lines)


def _resolve_skill_choice(raw: str, skills: list[SkillMetadata]) -> str | None:
    cleaned = raw.strip().lower()
    if not cleaned or cleaned in {"q", "cancel"}:
        return None
    if cleaned in {"0", "none", "off", "deactivate"}:
        return "none"
    if cleaned.isdigit():
        idx = int(cleaned) - 1
        if 0 <= idx < len(skills):
            return skills[idx].name
        raise SlashCommandError(f"invalid choice: {cleaned}")
    return cleaned


def _find_skill(skills: list[SkillMetadata], name: str) -> SkillMetadata | None:
    normalized = name.casefold()
    for skill in skills:
        if skill.name.casefold() == normalized:
            return skill
    return None


def _task_modes_for_skill(skill: SkillMetadata) -> list[str]:
    manifest = load_skill_manifest(skill.path.parent)
    modes = manifest.get("task_modes")
    if not isinstance(modes, list):
        return []
    return [mode for mode in modes if isinstance(mode, str)]


def _skill_command_error(exc: Exception) -> SlashCommandError:
    return SlashCommandError(f"failed to activate skill: {exc}")


def _render_skill_mode_prompt(skill_name: str, modes: list[str]) -> str:
    lines = [f"Task mode for {skill_name}:", "Available modes:"]
    lines.append("  [0] none  - no task mode")
    for idx, mode in enumerate(modes, start=1):
        lines.append(f"  [{idx}] {mode}")
    lines.append("Select (number or name; Enter for none): ")
    return "\n".join(lines)


def _resolve_skill_mode_choice(raw: str, modes: list[str]) -> str | None:
    cleaned = raw.strip().lower()
    if not cleaned or cleaned in {"0", "none"}:
        return None
    if cleaned.isdigit():
        idx = int(cleaned) - 1
        if 0 <= idx < len(modes):
            return modes[idx]
        raise SlashCommandError(f"invalid choice: {cleaned}")
    if cleaned not in modes:
        valid = ", ".join(modes)
        raise SlashCommandError(f"unknown task mode: {cleaned} (available: {valid})")
    return cleaned


async def _handle_skill(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    if len(parsed.args) > 2:
        raise SlashCommandError("usage: /skill [name|none] [mode]")

    session = context.session
    skills = _session_skills(session)

    if parsed.args:
        target_name = parsed.args[0].strip().lower()
    else:
        raw = await asyncio.to_thread(input, _render_skill_prompt(session))
        target_name = _resolve_skill_choice(raw, skills)
        if target_name is None:
            return SlashCommandResult(message="cancelled")

    if target_name in {"none", "off", "deactivate"}:
        session.deactivate_skill()
        return SlashCommandResult(message="skill -> none")

    skill = _find_skill(skills, target_name)
    if skill is None:
        valid = ", ".join(skill.name for skill in skills) or "none"
        raise SlashCommandError(
            f"unknown skill: {target_name} (available: {valid})"
        )

    if len(parsed.args) == 2:
        task_mode = parsed.args[1].strip().lower()
    elif parsed.args:
        task_mode = None
    else:
        try:
            modes = _task_modes_for_skill(skill)
        except _SKILL_USER_ERRORS as exc:
            raise _skill_command_error(exc) from exc
        if modes:
            raw = await asyncio.to_thread(
                input,
                _render_skill_mode_prompt(skill.name, modes),
            )
            task_mode = _resolve_skill_mode_choice(raw, modes)
        else:
            task_mode = None

    try:
        runtime = session.activate_skill(skill.name, task_mode)
    except _SKILL_USER_ERRORS as exc:
        raise _skill_command_error(exc) from exc
    suffix = f" {runtime.task_mode}" if runtime.task_mode else ""
    return SlashCommandResult(message=f"skill -> {runtime.name}{suffix}")


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


async def _handle_init(
    context: SlashCommandContext,
    parsed: ParsedSlashCommand,
) -> SlashCommandResult:
    if parsed.args:
        raise SlashCommandError("/init takes no arguments")

    app_root = find_app_root()
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

    try:
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
    except ValueError as exc:
        raise SlashCommandError(str(exc)) from exc

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

    try:
        diff = await asyncio.to_thread(
            list_diff, str(target), context.session.config
        )
    except ValueError as exc:
        raise SlashCommandError(str(exc)) from exc

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
        try:
            diff = await asyncio.to_thread(list_diff, str(target), config)
        except ValueError as exc:
            raise SlashCommandError(str(exc)) from exc
        orphans = diff["missing_from_disk"]
        lines = [f"Would prune {len(orphans)} orphaned pid(s) under {target}:"]
        if orphans:
            lines.extend(f"  - {path}" for path in orphans)
            lines.append("Re-run with --yes to apply.")
        else:
            lines.append("  (none)")
        return SlashCommandResult(message="\n".join(lines))

    try:
        removed = await asyncio.to_thread(
            prune_orphans, str(target), config
        )
    except ValueError as exc:
        raise SlashCommandError(str(exc)) from exc
    return SlashCommandResult(
        message=f"pruned {len(removed)} orphaned pid(s) under {target}"
    )
