"""Integration tests for ChatSession extended-thinking fusion mode."""

import asyncio
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.config import AgentConfig
from agent.memory import TurnRecord
from agent.session import ChatSession


# --- Fake graph factory + scripted graphs ---------------------------------


def _answer(text):
    return [{"agent": {"messages": [AIMessage(content=text)]}}]


def _tool_then_answer(name, args, call_id, result, text):
    return [
        {"agent": {"messages": [AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": call_id}],
        )]}},
        {"tools": {"messages": [ToolMessage(
            content=result, name=name, tool_call_id=call_id,
        )]}},
        {"agent": {"messages": [AIMessage(content=text)]}},
    ]


class _QueueGraph:
    def __init__(self, factory, model_id):
        self.factory = factory
        self.model_id = model_id

    async def astream(self, state, config=None, stream_mode="updates"):
        self.factory.calls.append({"model_id": self.model_id, "state": state})
        queue = self.factory.scripts.get(self.model_id)
        updates = queue.pop(0) if queue else _answer(self.factory.default)
        for update in updates:
            yield update


class _Factory:
    """Stands in for agent.session.build_graph; one scripted graph per llm_model."""

    def __init__(self, scripts=None, default="draft answer"):
        self.scripts = {key: list(value) for key, value in (scripts or {}).items()}
        self.default = default
        self.built: list[dict] = []
        self.calls: list[dict] = []

    def __call__(self, cfg, extra_tools=None, history_store=None, skill_runtime_getter=None):
        self.built.append({
            "model_id": cfg.llm_model,
            "max_tool_interactions": cfg.agent_max_tool_interactions,
            "skill_validation_enabled": cfg.skill_validation_enabled,
            "getter_is_none": skill_runtime_getter is None,
            "extra_tool_names": [getattr(t, "name", str(t)) for t in (extra_tools or [])],
        })
        return _QueueGraph(self, cfg.llm_model)


class _FakeHistoryStore:
    def __init__(self):
        self.adds: list[dict] = []

    def add_turn(self, turn: TurnRecord, *, session_id: str, turn_id: int, timestamp: str):
        self.adds.append({
            "user_input": turn.user_input,
            "assistant_output": turn.assistant_output,
        })


class _QueuedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.outputs.pop(0))


def _review_json(decision="pass", findings=None, summary="ok"):
    return json.dumps({
        "decision": decision,
        "findings": findings or [],
        "summary_for_reviser": summary,
    })


def _finding(severity="major", needs_user_input=False):
    return {
        "severity": severity,
        "dimension": "claim-evidence alignment",
        "location": "paragraph 1",
        "problem": "claim outruns evidence",
        "evidence_from_draft": "unsupported claim",
        "revision_instruction": "Soften the claim.",
        "needs_user_input": needs_user_input,
    }


def _aggregate_json(draft="fused", selected=None, dropped=None, summary="merged", removed=None):
    return json.dumps({
        "draft": draft,
        "selected_candidate_ids": selected if selected is not None else ["candidate-1"],
        "dropped_candidate_ids": dropped or [],
        "summary_for_reviewer": summary,
        "removed_or_uncertain_points": removed or [],
    })


def _cfg(tmp_path, *, proposer_models, aggregator="aggregator-model", **overrides):
    data = dict(
        persist_dir=str(tmp_path),
        llm_model="session-writer",
        thinking_rewrite_model="rewrite-model",
        thinking_reviewer_model="reviewer-model",
        thinking_repair_model="repair-model",
        thinking_fusion_proposer_models=tuple(proposer_models),
        thinking_fusion_aggregator_model=aggregator,
    )
    data.update(overrides)
    return AgentConfig(**data)


def _make_session(monkeypatch, tmp_path, factory, *, models, cfg):
    monkeypatch.setattr("agent.session.build_graph", factory)
    monkeypatch.setattr(
        "agent.session.get_chat_model_for_role",
        lambda _cfg, *, role: models[role],
    )
    monkeypatch.setattr(
        "agent.session.get_fusion_aggregator_model",
        lambda _cfg: models["aggregator"],
    )
    monkeypatch.setattr("agent.session.find_app_root", lambda: tmp_path)
    session = ChatSession(cfg, history_store=_FakeHistoryStore())
    session._prompt_master_skill_text_cache = "prompt-master skill"
    session.set_thinking_mode("extended")
    return session


def _default_models(**overrides):
    models = {
        "rewrite": _QueuedModel(["rewritten prompt"]),
        "reviewer": _QueuedModel([_review_json("pass")]),
        "repair": _QueuedModel([]),
        "aggregator": _QueuedModel([_aggregate_json(
            draft="fused", selected=["candidate-1", "candidate-2", "candidate-3"],
        )]),
    }
    models.update(overrides)
    return models


def _fusion(session):
    return session.turn_logs[-1]["fusion"]


def _state_for(factory, model_id):
    return next(call["state"] for call in factory.calls if call["model_id"] == model_id)


# --- Flow tests ------------------------------------------------------------


def test_extended_builds_independent_proposer_graphs(monkeypatch, tmp_path):
    factory = _Factory(scripts={
        "p1": [_answer("a1")], "p2": [_answer("a2")], "p3": [_answer("a3")],
    })
    models = _default_models()
    cfg = _cfg(tmp_path, proposer_models=["p1", "p2", "p3"])
    session = _make_session(monkeypatch, tmp_path, factory, models=models, cfg=cfg)

    answer = asyncio.run(session.turn("question"))

    assert answer == "fused"
    proposer_builds = [b for b in factory.built if b["model_id"] in {"p1", "p2", "p3"}]
    assert {b["model_id"] for b in proposer_builds} == {"p1", "p2", "p3"}
    for build in proposer_builds:
        # Proposers switch model via a cloned config, not just AgentState.
        assert build["max_tool_interactions"] == cfg.thinking_fusion_proposer_tool_interactions
        assert build["skill_validation_enabled"] is False
        assert build["getter_is_none"] is True
    assert len(models["rewrite"].calls) == 1


def test_proposers_use_proposer_graphs_not_session_graph(monkeypatch, tmp_path):
    factory = _Factory(scripts={
        "p1": [_answer("a1")], "p2": [_answer("a2")], "p3": [_answer("a3")],
    })
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=_default_models(), cfg=_cfg(tmp_path, proposer_models=["p1", "p2", "p3"]),
    )

    asyncio.run(session.turn("question"))

    proposer_calls = [c for c in factory.calls if c["model_id"] in {"p1", "p2", "p3"}]
    assert len(proposer_calls) == 3
    # Reviewer passed, so the session graph is never run for a writer/reviser turn.
    assert [c for c in factory.calls if c["model_id"] == "session-writer"] == []


def test_three_of_three_success_runs_full_panel(monkeypatch, tmp_path):
    factory = _Factory(scripts={
        "p1": [_answer("a1")], "p2": [_answer("a2")], "p3": [_answer("a3")],
    })
    models = _default_models()
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models, cfg=_cfg(tmp_path, proposer_models=["p1", "p2", "p3"]),
    )

    answer = asyncio.run(session.turn("question"))

    assert answer == "fused"
    assert len(models["aggregator"].calls) == 1
    assert _fusion(session)["reliability_tier"] == "full_panel"


def test_two_of_three_success_runs_partial_panel(monkeypatch, tmp_path):
    factory = _Factory(scripts={
        "p1": [_answer("a1")], "p2": [_answer("")], "p3": [_answer("a3")],
    })
    models = _default_models(aggregator=_QueuedModel([
        _aggregate_json(draft="fused", selected=["candidate-1", "candidate-3"]),
    ]))
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models, cfg=_cfg(tmp_path, proposer_models=["p1", "p2", "p3"]),
    )

    answer = asyncio.run(session.turn("question"))

    assert answer == "fused"
    assert len(models["aggregator"].calls) == 1
    fusion = _fusion(session)
    assert fusion["reliability_tier"] == "partial_panel"
    assert fusion["candidate_statuses"] == {
        "candidate-1": "success", "candidate-2": "empty", "candidate-3": "success",
    }


def test_one_success_skips_aggregator_but_still_reviews(monkeypatch, tmp_path):
    factory = _Factory(scripts={
        "p1": [_answer("only answer")], "p2": [_answer("")], "p3": [_answer("")],
    })
    models = _default_models()
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models, cfg=_cfg(tmp_path, proposer_models=["p1", "p2", "p3"]),
    )

    answer = asyncio.run(session.turn("question"))

    assert answer == "only answer"
    assert len(models["aggregator"].calls) == 0
    assert len(models["reviewer"].calls) == 1
    assert _fusion(session)["reliability_tier"] == "single_candidate"


def test_zero_success_runs_base_fallback_into_reviewer(monkeypatch, tmp_path):
    factory = _Factory(scripts={
        "p1": [_answer("")], "p2": [_answer("")], "p3": [_answer("")],
        "session-writer": [_answer("base fallback draft")],
    })
    models = _default_models()
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models, cfg=_cfg(tmp_path, proposer_models=["p1", "p2", "p3"]),
    )

    answer = asyncio.run(session.turn("question"))

    assert answer == "base fallback draft"
    assert len(models["aggregator"].calls) == 0
    assert len(models["reviewer"].calls) == 1
    fusion = _fusion(session)
    assert fusion["reliability_tier"] == "fallback"
    assert fusion["aggregator_error"] == "no_successful_candidates_base_fallback"
    assert any(c["model_id"] == "session-writer" for c in factory.calls)


def test_aggregator_invalid_json_falls_back_then_reviews(monkeypatch, tmp_path):
    factory = _Factory(scripts={"p1": [_answer("a1")], "p2": [_answer("a2")]})
    models = _default_models(aggregator=_QueuedModel(["not json at all"]))
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models, cfg=_cfg(tmp_path, proposer_models=["p1", "p2"]),
    )

    answer = asyncio.run(session.turn("question"))

    assert answer == "a1"
    assert len(models["reviewer"].calls) == 1
    fusion = _fusion(session)
    assert fusion["reliability_tier"] == "fallback"
    assert "aggregator_failure" in fusion["aggregator_error"]


def test_aggregator_schema_invalid_falls_back_then_reviews(monkeypatch, tmp_path):
    factory = _Factory(scripts={"p1": [_answer("a1")], "p2": [_answer("a2")]})
    models = _default_models(aggregator=_QueuedModel([
        _aggregate_json(draft="fused", selected=["candidate-99"]),
    ]))
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models, cfg=_cfg(tmp_path, proposer_models=["p1", "p2"]),
    )

    answer = asyncio.run(session.turn("question"))

    assert answer == "a1"
    assert len(models["reviewer"].calls) == 1
    assert _fusion(session)["reliability_tier"] == "fallback"


def test_quorum_not_met_falls_back_then_reviews(monkeypatch, tmp_path):
    factory = _Factory(scripts={
        "p1": [_answer("a1")], "p2": [_answer("a2")], "p3": [_answer("")],
    })
    models = _default_models()
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models,
        cfg=_cfg(tmp_path, proposer_models=["p1", "p2", "p3"], thinking_fusion_quorum=3),
    )

    answer = asyncio.run(session.turn("question"))

    assert answer == "a1"
    assert len(models["aggregator"].calls) == 0
    assert len(models["reviewer"].calls) == 1
    fusion = _fusion(session)
    assert fusion["reliability_tier"] == "fallback"
    assert fusion["aggregator_error"] == "quorum_not_met"


def test_no_draft_returns_extended_error(monkeypatch, tmp_path):
    factory = _Factory(scripts={
        "p1": [_answer("")], "p2": [_answer("")],
        "session-writer": [_answer("")],
    })
    models = _default_models()
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models, cfg=_cfg(tmp_path, proposer_models=["p1", "p2"]),
    )

    answer = asyncio.run(session.turn("question"))

    assert "Extended mode 無法安全完成" in answer
    assert len(models["reviewer"].calls) == 0
    assert _fusion(session)["reliability_tier"] == "fallback"


def test_reviewer_revise_path_still_works(monkeypatch, tmp_path):
    factory = _Factory(scripts={
        "p1": [_answer("draft answer")],
        "session-writer": [_answer("DRAFT:\nrevised answer\n\nREBUTTAL:\n(none)")],
    })
    models = _default_models(reviewer=_QueuedModel([
        _review_json("revise", [_finding()], "needs revision"),
        _review_json("pass"),
    ]))
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models, cfg=_cfg(tmp_path, proposer_models=["p1"]),
    )

    answer = asyncio.run(session.turn("question"))

    assert answer == "revised answer"
    assert len(models["reviewer"].calls) == 2
    # exactly one session-graph run (the reviser), no aggregator
    assert sum(1 for c in factory.calls if c["model_id"] == "session-writer") == 1
    assert len(models["aggregator"].calls) == 0


def test_final_skill_validation_still_applies(monkeypatch, tmp_path):
    factory = _Factory(scripts={
        "session-writer": [_answer("DRAFT:\nvalidated [Smith, 2020]\n\nREBUTTAL:\n(none)")],
    })
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=_default_models(), cfg=_cfg(tmp_path, proposer_models=["p1"]),
    )
    session.active_skill_runtime = SimpleNamespace(
        name="academic-paper-writing",
        root=tmp_path,
        instructions="Do not invent scholarly content.",
        pinned_references={},
        task_mode=None,
        allowed_tools=frozenset(),
        denied_tools=frozenset(),
        tool_policy_active=False,
        context_block=lambda: "[Active skill]\nname: academic-paper-writing",
    )

    result = asyncio.run(session._apply_final_skill_validation(
        user_input="revise this",
        answer="This improved outcomes by 50%",
        new_messages=[],
        tool_calls=[],
        trace_events=[],
    ))

    assert result.answer == "validated [Smith, 2020]"


def test_extended_clarification_skips_candidate_panel(monkeypatch, tmp_path):
    factory = _Factory(scripts={"p1": [_answer("should not run")]})
    models = _default_models(rewrite=_QueuedModel(["<<CLARIFY>>\n- Which journal?"]))
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models, cfg=_cfg(tmp_path, proposer_models=["p1"]),
    )

    answer = asyncio.run(session.turn("question"))

    assert "Which journal?" in answer
    assert factory.calls == []
    assert session.turn_logs[-1]["fusion"] is None


# --- Interface / invariant tests ------------------------------------------


def _two_tool_candidate_session(monkeypatch, tmp_path):
    factory = _Factory(scripts={
        "p1": [_tool_then_answer("rag_search", {"q": "x"}, "call-1", "RES1", "a1")],
        "p2": [_tool_then_answer("rag_search", {"q": "y"}, "call-1", "RES2", "a2")],
    })
    models = _default_models(aggregator=_QueuedModel([
        _aggregate_json(draft="fused", selected=["candidate-1", "candidate-2"]),
    ]))
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models, cfg=_cfg(tmp_path, proposer_models=["p1", "p2"]),
    )
    return session, factory, models


def test_turn_holds_flat_calls_metadata_and_segmented_traces(monkeypatch, tmp_path):
    session, _factory, _models = _two_tool_candidate_session(monkeypatch, tmp_path)

    asyncio.run(session.turn("question"))

    tool_calls = session.turn_logs[-1]["tool_calls"]
    candidate_calls = [c for c in tool_calls if c.get("candidate_id")]
    assert {c["candidate_id"] for c in candidate_calls} == {"candidate-1", "candidate-2"}
    # aggregator LLM call is never a tool call; only the two candidate calls exist.
    assert len(tool_calls) == 2
    assert all(c["name"] == "rag_search" for c in tool_calls)

    fusion = session.turn_logs[-1]["fusion"]
    assert fusion["candidate_statuses"] == {"candidate-1": "success", "candidate-2": "success"}
    assert fusion["model_ids"] == {"candidate-1": "p1", "candidate-2": "p2"}
    assert any(ev.get("type") == "fusion" for ev in session.last_trace_events)
    candidate_events = [ev for ev in session.last_trace_events if ev.get("candidate_id")]
    assert {ev["candidate_id"] for ev in candidate_events} == {"candidate-1", "candidate-2"}


def test_turn_with_trace_preserves_eval_shape(monkeypatch, tmp_path):
    session, _factory, _models = _two_tool_candidate_session(monkeypatch, tmp_path)

    answer, tool_calls = asyncio.run(session.turn_with_trace("question"))

    assert answer == "fused"
    assert isinstance(tool_calls, list)
    for call in tool_calls:
        assert "name" in call and "args" in call


def test_recent_turns_exclude_candidate_answers(monkeypatch, tmp_path):
    session, _factory, _models = _two_tool_candidate_session(monkeypatch, tmp_path)

    asyncio.run(session.turn("question"))

    record = session.recent_turns[-1]
    assert record.user_input == "question"
    assert record.assistant_output == "fused"
    assert "a1" not in record.assistant_output
    assert "a2" not in record.assistant_output


def test_plan_log_uses_candidate_scoped_trace_without_collision(monkeypatch, tmp_path):
    session, _factory, _models = _two_tool_candidate_session(monkeypatch, tmp_path)
    asyncio.run(session.enter_plan_mode())

    asyncio.run(session.turn("question"))

    content = session.plan_log_path.read_text(encoding="utf-8")
    assert "Fusion candidate candidate-1" in content
    assert "Fusion candidate candidate-2" in content
    seg1 = content.split("Fusion candidate candidate-2")[0]
    seg2 = content.split("Fusion candidate candidate-2")[1]
    # Each candidate's call-1 result stays inside its own segment.
    assert "RES1" in seg1 and "RES2" not in seg1
    assert "RES2" in seg2
    # The final Assistant block only shows the final answer.
    assistant_block = content.split("**Assistant:**")[-1]
    assert "fused" in assistant_block
    assert "RES1" not in assistant_block and "RES2" not in assistant_block


def test_no_active_skill_proposer_is_policy_active_read_only(monkeypatch, tmp_path):
    factory = _Factory(scripts={"p1": [_answer("a1")]})
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=_default_models(), cfg=_cfg(tmp_path, proposer_models=["p1"]),
    )

    asyncio.run(session.turn("question"))

    state = _state_for(factory, "p1")
    assert state["tool_policy_active"] is True
    assert state["allowed_tools"] == sorted([
        "rag_explore", "rag_search", "rag_get_context", "recall_history", "read_file",
    ])
    assert "bash" not in state["allowed_tools"]
    assert "bash" in state["denied_tools"]
    assert state["active_skill"] is None
    assert state["skill_instructions"] is None


def test_default_proposer_state_excludes_bash_extra_mcp(monkeypatch, tmp_path):
    factory = _Factory(scripts={"p1": [_answer("a1")]})
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=_default_models(), cfg=_cfg(tmp_path, proposer_models=["p1"]),
    )
    session.extra_tools = [SimpleNamespace(name="extratool")]
    session.mcp_families = {"mcp_tool": "web_search"}

    asyncio.run(session.turn("question"))

    state = _state_for(factory, "p1")
    assert "extratool" not in state["allowed_tools"]
    assert "mcp_tool" not in state["allowed_tools"]
    assert {"bash", "extratool", "mcp_tool"} <= set(state["denied_tools"])


def test_proposer_prompt_availability_matches_bound_tools(monkeypatch, tmp_path):
    factory = _Factory(scripts={"p1": [_answer("a1")]})
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=_default_models(), cfg=_cfg(tmp_path, proposer_models=["p1"]),
    )

    asyncio.run(session.turn("question"))

    state = _state_for(factory, "p1")
    prompt_text = "\n".join(str(m.content) for m in state["messages"])
    assert "[Tool availability]" in prompt_text
    assert "tool_policy_active: true" in prompt_text
    assert "available_tools: rag_explore" in prompt_text
    assert "denied_tools: bash" in prompt_text


def test_active_skill_proposer_keeps_instructions_with_read_only_policy(monkeypatch, tmp_path):
    factory = _Factory(scripts={"p1": [_answer("a1")]})
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=_default_models(),
        cfg=_cfg(tmp_path, proposer_models=["p1"], skill_validation_enabled=False),
    )
    session.active_skill_runtime = SimpleNamespace(
        name="paper",
        root=tmp_path,
        instructions="# Paper instructions",
        pinned_references={},
        task_mode="revision",
        allowed_tools=frozenset({"read_file"}),
        denied_tools=frozenset({"bash"}),
        tool_policy_active=True,
        context_block=lambda: "[Active skill]\nname: paper",
    )

    asyncio.run(session.turn("question"))

    state = _state_for(factory, "p1")
    assert state["active_skill"] == "paper"
    assert state["skill_instructions"] == "# Paper instructions"
    # read-only allowlist intersected with the skill's own allowed tools.
    assert state["allowed_tools"] == ["read_file"]
    assert state["tool_policy_active"] is True


def test_active_skill_deny_only_intersects_read_only(monkeypatch, tmp_path):
    factory = _Factory(scripts={"p1": [_answer("a1")]})
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=_default_models(),
        cfg=_cfg(tmp_path, proposer_models=["p1"], skill_validation_enabled=False),
    )
    session.active_skill_runtime = SimpleNamespace(
        name="paper",
        root=tmp_path,
        instructions="# Paper",
        pinned_references={},
        task_mode=None,
        allowed_tools=frozenset(),
        denied_tools=frozenset({"read_file"}),
        tool_policy_active=True,
        context_block=lambda: "[Active skill]\nname: paper",
    )

    asyncio.run(session.turn("question"))

    state = _state_for(factory, "p1")
    assert state["allowed_tools"] == sorted([
        "rag_explore", "rag_search", "rag_get_context", "recall_history",
    ])
    assert "read_file" not in state["allowed_tools"]


def test_side_effect_true_uses_session_policy_graph(monkeypatch, tmp_path):
    factory = _Factory(scripts={"p1": [_answer("a1")], "p2": [_answer("a2")]})
    models = _default_models(aggregator=_QueuedModel([
        _aggregate_json(draft="fused", selected=["candidate-1", "candidate-2"]),
    ]))
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=models,
        cfg=_cfg(
            tmp_path,
            proposer_models=["p1", "p2"],
            thinking_fusion_allow_side_effect_tools=True,
        ),
    )
    session.extra_tools = [SimpleNamespace(name="extratool")]

    asyncio.run(session.turn("question"))

    proposer_builds = [b for b in factory.built if b["model_id"] in {"p1", "p2"}]
    assert proposer_builds
    for build in proposer_builds:
        assert build["getter_is_none"] is False
        assert "extratool" in build["extra_tool_names"]
    # Session policy means no injected read-only tool policy on the proposer state.
    state = _state_for(factory, "p1")
    assert not state.get("tool_policy_active")
    assert "allowed_tools" not in state


def test_extended_records_configuration_error(monkeypatch, tmp_path):
    factory = _Factory()
    session = _make_session(
        monkeypatch, tmp_path, factory,
        models=_default_models(),
        cfg=_cfg(
            tmp_path,
            proposer_models=["p1"],
            thinking_rewrite_model="",
            thinking_reviewer_model="",
            thinking_repair_model="",
        ),
    )

    answer = asyncio.run(session.turn("question"))

    assert "thinking_rewrite_model" in answer
    assert factory.calls == []
