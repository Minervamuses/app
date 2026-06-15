"""Multi-turn conversational session for the agent."""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent.config import AgentConfig
from agent.graph import build_graph
from agent.history import (
    extract_tool_calls,
    format_tool_counts,
)
from agent.history_rag import ChatHistoryStore, get_chat_history_store
from agent.llm.thinking import (
    ExtendedModeNotConfigured,
    ThinkingRole,
    get_chat_model_for_role,
    require_thinking_models,
)
from agent.skills import (
    SkillRuntime,
    discover_skills,
    load_skill_runtime,
)
from agent.skills.runtime import render_tool_availability_block
from agent.skills.validator import validate_skill_output
from agent.state import skill_runtime_to_agent_state
from agent.tools import inventory as tool_inventory
from agent.thinking import (
    Clarify,
    ThinkingOutputError,
    append_tool_trace,
    extract_draft_for_user,
    parse_reviser_output,
    render_route_message,
    review_draft,
    route_review_report,
    rewrite_prompt,
    summarize_tool_trace,
    trim_head,
    trim_tail,
)
from agent.memory import (
    TurnRecord,
    assemble_prompt_history,
)
from agent.paths import find_app_root

logger = logging.getLogger(__name__)

# The base tool inventory, its selection policy, and the base workflow are
# owned by agent.tools.inventory (single source of truth). Only the optional
# MCP families, skill activation, and language policy live here.
SYSTEM_PROMPT = f"""You are a research assistant with access to several tool families.

{tool_inventory.render_base_tool_prompt()}

Web Search MCP tools (available only when configured):
- Use for current external information, general web discovery, or topics unlikely to exist in the local KB.

GitHub MCP tools (available only when configured):
- Use for remote GitHub state: repository content not in the local KB, pull requests, issues, Actions runs, code search across GitHub.
- Do NOT use GitHub MCP as a substitute for local git shell operations (clone, pull, rebase, commit). Those belong to the user's terminal, not to you.

Local skills (user-activated):
- Skill bundles live under `skills/<name>/`. The user activates one via the `/skill` slash command; you cannot self-activate.
- When a skill is active, its instructions and tool policy arrive as an ephemeral system message — follow them.
- If the user asks what skills are available, discover the bundle names by listing `skills/` via `bash`.

Language policy:
- Respond in the same language the user is writing in.
- When the user writes in Chinese, ALWAYS use Traditional Chinese (繁體中文). Never produce Simplified Chinese characters even if the user's input contains some.
- For other languages, match the user's input language without conversion."""

DEFAULT_RECURSION_LIMIT = 32

_REVISER_INSTRUCTION = """你可以對每一個 reviewer finding 做以下其中之一：
(a) 修改 draft 以處理該 finding；
(b) 駁斥該 finding 並在 REBUTTAL 段說明你不同意的理由。

硬性禁令：
- 不要新增無法佐證的 citation / DOI / 數據 / 樣本數 / 方法細節 / 研究發現。
- 不要新增原始 user input 與可見 context 未提供的事實。

回應格式必須嚴格使用兩個區段標記：

DRAFT:
<新版 draft 全文。這段會被回給使用者，所以保持乾淨、不要包含內部審稿討論。>

REBUTTAL:
<對 reviewer findings 的反對說明；若無，寫 (none)。這段只給下輪 Reviewer 看，不會回給使用者。>"""


@dataclass(frozen=True)
class _ToolRef:
    name: str


@dataclass(frozen=True)
class _GraphTurnResult:
    answer: str
    new_messages: list
    tool_calls: list[dict]
    trace_events: list[dict]


class ChatSession:
    """Multi-turn conversational retrieval session backed by LangGraph."""

    def __init__(
        self,
        config: AgentConfig,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        system_prompt: str = SYSTEM_PROMPT,
        extra_tools: list | None = None,
        history_store: ChatHistoryStore | None = None,
        progress_cb=None,
        web_search_tool_names: set[str] | frozenset[str] | None = None,
        mcp_families: dict[str, str] | None = None,
    ):
        self.config = config
        self.recursion_limit = recursion_limit
        self.plan_mode = False
        self.thinking_mode = "normal"
        self.plan_log_path: Path | None = None
        self.active_skill_runtime: SkillRuntime | None = None
        self.extra_tools = list(extra_tools or [])
        self.mcp_families = dict(mcp_families or {})
        self.web_search_tool_names = frozenset(web_search_tool_names or ())

        self.loaded_skills = discover_skills(config)
        self.system_prompt_message = SystemMessage(content=system_prompt)
        self.recent_turns: list[TurnRecord] = []

        self.session_id = uuid.uuid4().hex
        self._turn_counter = 0
        self.history_store = history_store or get_chat_history_store(config)
        self.graph = build_graph(
            config,
            extra_tools=extra_tools,
            history_store=self.history_store,
            skill_runtime_getter=lambda: self.active_skill_runtime,
        )
        self._thinking_role_models: dict[str, object] = {}
        self._prompt_master_skill_text_cache: str | None = None

        self.turn_logs: list[dict] = []
        self.last_tool_calls: list[dict] = []
        self.last_trace_events: list[dict] = []

        self._progress_cb = progress_cb

    def _prompt_history(self) -> list:
        base = assemble_prompt_history(
            self.system_prompt_message,
            self.recent_turns,
        )
        hints = [
            hint
            for hint in (
                self._build_active_skill_hint(),
                self._build_tool_availability_hint(),
                self._build_plan_mode_hint(),
            )
            if hint is not None
        ]
        if not hints:
            return base
        return [base[0], *hints, *base[1:]]

    def _build_active_skill_hint(self) -> SystemMessage | None:
        if self.active_skill_runtime is None:
            return None
        return SystemMessage(content=self.active_skill_runtime.context_block())

    def _active_skill_context_block(self) -> str:
        if self.active_skill_runtime is None:
            return ""
        return self.active_skill_runtime.context_block()

    def _tool_availability_block(self) -> str:
        return render_tool_availability_block(
            skill_runtime=self.active_skill_runtime,
            base_tool_names=[tool.name for tool in self._all_tool_refs()],
            mcp_families=self.mcp_families,
        )

    def _build_tool_availability_hint(self) -> SystemMessage | None:
        if self.active_skill_runtime is None:
            return None
        return SystemMessage(content=self._tool_availability_block())

    def _build_plan_mode_hint(self) -> SystemMessage | None:
        """Tell the LLM that some visible turns are plan-mode (md only),
        so it does not call recall_history looking for them in ChromaDB.
        """
        has_plan_turn = any(
            getattr(turn, "persist_target", "chroma") == "plan_log"
            for turn in self.recent_turns
        )
        if not has_plan_turn:
            return None
        return SystemMessage(content=(
            "[Mode hint] Some turns in the recent context were recorded under "
            "plan mode (stored only in plan_logs/, NOT in ChromaDB). They ARE "
            "visible to you in this prompt - do NOT call recall_history to "
            "look for them."
        ))

    async def _store_turn(self, turn: TurnRecord) -> None:
        if turn.persist_target == "plan_log":
            return
        if turn.persist_target == "none":
            return
        if turn.persist_target != "chroma":
            raise ValueError(
                f"unknown persist_target={turn.persist_target!r} on turn {turn.turn_id}"
            )
        await asyncio.to_thread(
            self.history_store.add_turn,
            turn,
            session_id=self.session_id,
            turn_id=turn.turn_id,
            timestamp=turn.timestamp,
        )

    def _new_plan_log_file(self) -> Path:
        created = datetime.now(timezone.utc)
        created_at = created.isoformat()
        safe_ts = created.strftime("%Y%m%dT%H%M%SZ")
        log_dir = find_app_root() / self.config.plan_logs_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"plan-{self.session_id}-{safe_ts}.md"
        header = (
            "---\n"
            "do_not_index: true\n"
            "generated_by: agent.plan_mode\n"
            f"session_id: {self.session_id}\n"
            f"created_at: {created_at}\n"
            "---\n\n"
            "# Plan log\n\n"
        )
        path.write_text(header, encoding="utf-8")
        return path

    async def enter_plan_mode(self) -> Path:
        """Enable plan mode for newly created turns."""
        if self.plan_mode:
            if self.plan_log_path is None:
                self.plan_log_path = self._new_plan_log_file()
            return self.plan_log_path
        self.plan_log_path = self._new_plan_log_file()
        self.plan_mode = True
        return self.plan_log_path

    async def exit_plan_mode(self) -> None:
        """Disable plan mode without mutating prompt-visible turns."""
        self.plan_mode = False
        self.plan_log_path = None

    def set_thinking_mode(self, mode: str) -> None:
        """Set the per-session thinking workflow mode."""
        normalized = mode.strip().lower()
        if normalized not in {"normal", "extended"}:
            raise ValueError(f"unknown thinking mode: {mode}")
        self.thinking_mode = normalized

    def activate_skill(self, name: str, task_mode: str | None = None) -> SkillRuntime:
        """Activate a local skill for subsequent turns."""
        runtime = load_skill_runtime(
            name,
            config=self.config,
            all_tools=self._all_tool_refs(),
            mcp_families=self.mcp_families,
            task_mode=task_mode,
        )
        self.active_skill_runtime = runtime
        return runtime

    def deactivate_skill(self) -> None:
        """Deactivate the current local skill, if any."""
        self.active_skill_runtime = None

    def _all_tool_refs(self) -> list[_ToolRef]:
        return [
            _ToolRef(name)
            for name in tool_inventory.base_tool_names(extra_tools=self.extra_tools)
        ]

    def _render_plan_block(
        self,
        *,
        turn_id: int,
        timestamp: str,
        user_input: str,
        answer: str,
        new_messages: list,
        tool_calls: list[dict],
    ) -> str:
        lines = [
            f"## Turn {turn_id} - {timestamp}",
            "",
            "**User:**",
            "",
            user_input,
            "",
        ]
        lines.extend(self._render_tool_blocks(new_messages, tool_calls))
        lines.extend([
            "**Assistant:**",
            "",
            answer,
            "",
            "---",
            "",
        ])
        return "\n".join(lines)

    def _render_tool_blocks(self, new_messages: list, tool_calls: list[dict]) -> list[str]:
        if not tool_calls:
            return []

        tool_messages: dict[str, list[ToolMessage]] = {}
        for message in new_messages:
            if not isinstance(message, ToolMessage):
                continue
            call_id = getattr(message, "tool_call_id", None)
            if call_id:
                tool_messages.setdefault(call_id, []).append(message)

        lines: list[str] = []
        for call in tool_calls:
            call_id = call.get("id")
            results = tool_messages.get(call_id, []) if call_id else []
            lines.extend([
                f"### Tool: {call.get('name', 'unknown')}",
                "",
                "```json",
                json.dumps(call.get("args", {}), ensure_ascii=False, indent=2),
                "```",
                "",
                "**Result:**",
                "",
            ])
            if not results:
                lines.extend(["(no ToolMessage matched this tool_call_id)", ""])
            else:
                for result in results:
                    content = getattr(result, "content", "") or ""
                    capped = self._cap_tool_result(str(content))
                    lines.extend(["```", capped, "```", ""])
        return lines

    def _cap_tool_result(self, content: str) -> str:
        cap = self.config.plan_log_max_tool_chars
        if len(content) <= cap:
            return content
        head = content[:cap]
        return f"{head}\n\n[truncated; original {len(content)} chars]"

    def _append_block_to_md(self, log_path: str, block: str) -> None:
        with Path(log_path).open("a", encoding="utf-8") as f:
            f.write(block)

    def _visible_context_text(self) -> str:
        lines: list[str] = []
        for turn in self.recent_turns[-self.config.agent_recent_turns_window:]:
            lines.extend([
                f"User turn {turn.turn_id}:",
                turn.user_input,
                f"Assistant turn {turn.turn_id}:",
                turn.assistant_output,
                "",
            ])
        return "\n".join(lines).strip()

    def _get_thinking_role_model(self, role: ThinkingRole):
        if role not in self._thinking_role_models:
            self._thinking_role_models[role] = get_chat_model_for_role(
                self.config,
                role=role,
            )
        return self._thinking_role_models[role]

    def _prompt_master_skill_text(self) -> str:
        if self._prompt_master_skill_text_cache is None:
            path = find_app_root() / "skills" / "_prompt-master" / "SKILL.md"
            self._prompt_master_skill_text_cache = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        return self._prompt_master_skill_text_cache

    def _rewrite_hints(
        self,
        *,
        raw_user_input: str,
        rewritten_prompt: str,
    ) -> list[SystemMessage]:
        return [
            SystemMessage(content=f"[Original user input]\n{raw_user_input}"),
            SystemMessage(content=f"[Rewritten by prompt-master]\n{rewritten_prompt}"),
        ]

    def _reviser_hints(
        self,
        *,
        raw_user_input: str,
        rewritten_prompt: str,
        draft: str,
        report,
    ) -> list[SystemMessage]:
        return [
            *self._rewrite_hints(
                raw_user_input=raw_user_input,
                rewritten_prompt=rewritten_prompt,
            ),
            SystemMessage(content=f"[Previous draft]\n{draft}"),
            SystemMessage(content=(
                "[Reviewer feedback]\n"
                f"{report.model_dump_json(indent=2)}"
            )),
            SystemMessage(content=f"[Reviser instruction]\n{_REVISER_INSTRUCTION}"),
        ]

    def _extended_error_message(self, exc: Exception) -> str:
        return (
            "Extended mode 無法安全完成本 turn，已停止以避免不安全改寫。\n"
            f"- {exc}"
        )

    async def _evict_overflow(self) -> None:
        """Spill turns past the window into the long-term store. Log + keep on failure."""
        window = self.config.agent_recent_turns_window
        hard_cap = window * 3
        while len(self.recent_turns) > window:
            oldest = self.recent_turns[0]
            try:
                await self._store_turn(oldest)
            except Exception as exc:
                logger.warning(
                    "history_rag: eviction failed for turn %s (kept in recent_turns): %s",
                    oldest.turn_id, exc,
                )
                if len(self.recent_turns) > hard_cap:
                    logger.error(
                        "history_rag: hard cap %d reached; dropping oldest turn %s unrecorded",
                        hard_cap, oldest.turn_id,
                    )
                    self.recent_turns.pop(0)
                break  # don't retry within the same turn
            self.recent_turns.pop(0)

    async def flush_recent_turns(self) -> None:
        """Persist all prompt-visible turns before the session is discarded."""
        while self.recent_turns:
            oldest = self.recent_turns[0]
            try:
                await self._store_turn(oldest)
            except Exception as exc:
                logger.warning(
                    "history_rag: shutdown flush failed for turn %s (left in recent_turns): %s",
                    oldest.turn_id, exc,
                )
                break
            self.recent_turns.pop(0)

    async def _run_graph_turn(
        self,
        user_input: str,
        *,
        extra_system_messages: list[SystemMessage] | None = None,
    ) -> _GraphTurnResult:
        """Run the existing graph once without recording the completed turn."""
        input_messages = [
            *self._prompt_history(),
            *(extra_system_messages or []),
            HumanMessage(content=user_input),
        ]
        messages: list = list(input_messages)
        initial_state = {
            "messages": input_messages,
            **skill_runtime_to_agent_state(self.active_skill_runtime),
        }
        async for update in self.graph.astream(
            initial_state,
            config={"recursion_limit": self.recursion_limit},
            stream_mode="updates",
        ):
            for node_name, delta in update.items():
                new_msgs = delta.get("messages", []) if isinstance(delta, dict) else []
                messages.extend(new_msgs)
                if self._progress_cb is not None:
                    self._progress_cb(node_name, new_msgs)
        new_messages = messages[len(input_messages):]
        tool_calls = extract_tool_calls(new_messages)
        trace_events = [
            {
                "type": "tool",
                "name": call["name"],
                "args": call["args"],
                "id": call.get("id"),
            }
            for call in tool_calls
        ]
        answer = messages[-1].content if messages else ""
        answer = answer or ""

        return _GraphTurnResult(
            answer=str(answer),
            new_messages=new_messages,
            tool_calls=tool_calls,
            trace_events=trace_events,
        )

    async def _record_turn(
        self,
        *,
        user_input: str,
        answer: str,
        new_messages: list,
        tool_calls: list[dict],
        trace_events: list[dict],
    ) -> None:
        """Persist/log the final answer for one user-visible turn."""
        turn_id = self._turn_counter + 1
        timestamp = datetime.now(timezone.utc).isoformat()
        if self.plan_mode:
            if self.plan_log_path is None:
                raise RuntimeError("plan mode is enabled without a log path")
            target = "plan_log"
            log_path = str(self.plan_log_path)
            try:
                block = self._render_plan_block(
                    turn_id=turn_id,
                    timestamp=timestamp,
                    user_input=user_input,
                    answer=answer,
                    new_messages=new_messages,
                    tool_calls=tool_calls,
                )
                await asyncio.to_thread(self._append_block_to_md, log_path, block)
            except Exception as exc:
                logger.error("plan md write failed for turn %s: %s", turn_id, exc)
                raise
        else:
            target = "chroma"
            log_path = None

        self._turn_counter = turn_id
        self.recent_turns.append(
            TurnRecord(
                user_input=user_input,
                assistant_output=answer,
                turn_id=turn_id,
                timestamp=timestamp,
                persist_target=target,
                log_path=log_path,
            )
        )
        self.last_tool_calls = tool_calls
        self.last_trace_events = trace_events
        self.turn_logs.append({
            "user_input": user_input,
            "tool_calls": tool_calls,
            "trace_events": trace_events,
            "tool_counts": format_tool_counts(tool_calls),
        })
        await self._evict_overflow()

    async def _run_normal_turn(self, user_input: str) -> tuple[str, list[dict]]:
        result = await self._run_graph_turn(user_input)
        await self._record_turn(
            user_input=user_input,
            answer=result.answer,
            new_messages=result.new_messages,
            tool_calls=result.tool_calls,
            trace_events=result.trace_events,
        )
        return result.answer, result.tool_calls

    async def _apply_final_skill_validation(
        self,
        *,
        user_input: str,
        answer: str,
        new_messages: list,
        tool_calls: list[dict],
        trace_events: list[dict],
    ) -> _GraphTurnResult:
        if not self.active_skill_runtime or not self.config.skill_validation_enabled:
            return _GraphTurnResult(answer, new_messages, tool_calls, trace_events)

        violations = validate_skill_output(
            active_skill=self.active_skill_runtime.name,
            text=answer,
        )
        if not violations:
            return _GraphTurnResult(answer, new_messages, tool_calls, trace_events)

        validation_hint = SystemMessage(content=(
            "[Extended thinking final validation errors]\n"
            + "\n".join(f"- {violation}" for violation in violations)
            + "\nRevise the supplied draft once to satisfy the active skill policy."
        ))
        revision_input = (
            "Revise the draft below to satisfy the active skill policy while "
            "preserving the original user request.\n\n"
            f"Original user request:\n{user_input}\n\n"
            f"Draft:\n{answer}"
        )
        validation_result = await self._run_graph_turn(
            revision_input,
            extra_system_messages=[validation_hint],
        )
        return _GraphTurnResult(
            answer=extract_draft_for_user(validation_result.answer),
            new_messages=[*new_messages, *validation_result.new_messages],
            tool_calls=[*tool_calls, *validation_result.tool_calls],
            trace_events=[*trace_events, *validation_result.trace_events],
        )

    async def _run_extended_turn(self, user_input: str) -> tuple[str, list[dict]]:
        try:
            require_thinking_models(self.config)
            rewrite_model = self._get_thinking_role_model("rewrite")
            reviewer_model = self._get_thinking_role_model("reviewer")
            repair_model = self._get_thinking_role_model("repair")
            prompt_master_skill = self._prompt_master_skill_text()
        except (ExtendedModeNotConfigured, RuntimeError, OSError) as exc:
            answer = self._extended_error_message(exc)
            await self._record_turn(
                user_input=user_input,
                answer=answer,
                new_messages=[],
                tool_calls=[],
                trace_events=[],
            )
            return answer, []

        skill_context = self._active_skill_context_block()
        tool_availability = self._tool_availability_block()
        try:
            rewrite_result = rewrite_prompt(
                rewrite_model,
                skill_text=prompt_master_skill,
                user_input=user_input,
                visible_context=trim_tail(
                    self._visible_context_text(),
                    self.config.thinking_rewrite_visible_chars,
                ),
                skill_context=trim_head(
                    skill_context,
                    self.config.thinking_rewrite_skill_chars,
                ),
                tool_availability=tool_availability,
            )
        except Exception as exc:
            answer = self._extended_error_message(exc)
            await self._record_turn(
                user_input=user_input,
                answer=answer,
                new_messages=[],
                tool_calls=[],
                trace_events=[],
            )
            return answer, []

        if isinstance(rewrite_result, Clarify):
            answer = rewrite_result.text or "需要補充資訊才能安全完成這個任務。"
            await self._record_turn(
                user_input=user_input,
                answer=answer,
                new_messages=[],
                tool_calls=[],
                trace_events=[],
            )
            return answer, []

        rewritten_prompt = rewrite_result.prompt
        writer_result = await self._run_graph_turn(
            rewritten_prompt,
            extra_system_messages=self._rewrite_hints(
                raw_user_input=user_input,
                rewritten_prompt=rewritten_prompt,
            ),
        )
        draft = writer_result.answer
        new_messages = list(writer_result.new_messages)
        tool_calls = list(writer_result.tool_calls)
        trace_events = list(writer_result.trace_events)
        evidence_trace_summary = summarize_tool_trace(
            writer_result.tool_calls,
            writer_result.new_messages,
            source_label="[Writer]",
            per_result_chars=self.config.thinking_tool_trace_chars,
        )
        rebuttal_history: list[str] = []
        format_warning = ""
        attempts = 0
        final_route = None

        while True:
            try:
                report = review_draft(
                    reviewer_model,
                    raw_user_input=user_input,
                    rewritten_prompt=rewritten_prompt,
                    draft=draft,
                    skill_context=skill_context,
                    evidence_trace_summary=evidence_trace_summary,
                    previous_rebuttal=rebuttal_history[-1] if rebuttal_history else "",
                    tool_availability=tool_availability,
                )
            except ThinkingOutputError as exc:
                answer = self._extended_error_message(exc)
                final_route = None
                break

            review_route = route_review_report(report, attempts=attempts)
            if review_route in {"pass", "ask_user", "stop"}:
                final_route = review_route
                answer = render_route_message(
                    review_route,
                    draft,
                    report,
                    format_warning=format_warning,
                )
                break

            reviser_result = await self._run_graph_turn(
                rewritten_prompt,
                extra_system_messages=self._reviser_hints(
                    raw_user_input=user_input,
                    rewritten_prompt=rewritten_prompt,
                    draft=draft,
                    report=report,
                ),
            )
            parsed = parse_reviser_output(
                reviser_result.answer,
                repair_model=repair_model,
            )
            draft = parsed.draft
            format_warning = format_warning or parsed.format_warning
            evidence_trace_summary = append_tool_trace(
                evidence_trace_summary,
                reviser_result.tool_calls,
                reviser_result.new_messages,
                source_label=f"[Reviser round {attempts + 1}]",
                per_result_chars=self.config.thinking_tool_trace_chars,
                total_chars_cap=self.config.thinking_tool_trace_total_chars,
            )
            rebuttal_history.append(parsed.rebuttal)
            attempts += 1
            new_messages.extend(reviser_result.new_messages)
            tool_calls.extend(reviser_result.tool_calls)
            trace_events.extend(reviser_result.trace_events)

        current = _GraphTurnResult(
            answer=answer,
            new_messages=new_messages,
            tool_calls=tool_calls,
            trace_events=trace_events,
        )
        if final_route in {"pass", "stop"}:
            current = await self._apply_final_skill_validation(
                user_input=user_input,
                answer=current.answer,
                new_messages=current.new_messages,
                tool_calls=current.tool_calls,
                trace_events=current.trace_events,
            )
        await self._record_turn(
            user_input=user_input,
            answer=current.answer,
            new_messages=current.new_messages,
            tool_calls=current.tool_calls,
            trace_events=current.trace_events,
        )
        return current.answer, current.tool_calls

    async def _run_turn(self, user_input: str) -> tuple[str, list[dict]]:
        """Process one turn and return the final answer plus tool-call trace."""
        if self.thinking_mode == "extended":
            return await self._run_extended_turn(user_input)
        return await self._run_normal_turn(user_input)

    async def turn(self, user_input: str) -> str:
        """Process one conversation turn. Returns the final text response."""
        answer, _tool_calls = await self._run_turn(user_input)
        return answer

    def status_snapshot(self) -> dict[str, str | int]:
        """Expose lightweight session state for local CLI commands."""
        return {
            "session_id": self.session_id,
            "turn_count": self._turn_counter,
            "recent_turn_count": len(self.recent_turns),
            "recursion_limit": self.recursion_limit,
            "last_tool_counts": format_tool_counts(self.last_tool_calls) or "none",
            "plan_mode": self.plan_mode,
            "plan_log_path": str(self.plan_log_path) if self.plan_log_path else "",
            "thinking_mode": self.thinking_mode,
            "active_skill": (
                self.active_skill_runtime.name
                if self.active_skill_runtime is not None
                else ""
            ),
            "task_mode": (
                self.active_skill_runtime.task_mode
                if self.active_skill_runtime is not None and self.active_skill_runtime.task_mode
                else ""
            ),
        }

    async def turn_with_trace(self, user_input: str) -> tuple[str, list[dict]]:
        """Process one turn and return the answer plus normalized tool trace."""
        return await self._run_turn(user_input)

    @classmethod
    async def create(
        cls,
        config: AgentConfig,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        system_prompt: str = SYSTEM_PROMPT,
        history_store: ChatHistoryStore | None = None,
        load_mcp: bool = True,
        progress_cb=None,
    ) -> "ChatSession":
        """Async factory that loads MCP tools (if enabled) before graph construction.

        MCP tool loading is async; turn processing stays asynchronous via
        graph.astream once the session is built.
        """
        extra_tools: list = []
        if load_mcp:
            from agent.mcp import load_mcp_tools_with_families

            try:
                extra_tools, families = await load_mcp_tools_with_families()
            except Exception:
                extra_tools = []
                families = {}
        else:
            families = {}
        web_search_tool_names = frozenset(
            name for name, family in families.items() if family == "web_search"
        )
        return cls(
            config,
            recursion_limit=recursion_limit,
            system_prompt=system_prompt,
            extra_tools=extra_tools,
            history_store=history_store,
            progress_cb=progress_cb,
            web_search_tool_names=web_search_tool_names,
            mcp_families=families,
        )
