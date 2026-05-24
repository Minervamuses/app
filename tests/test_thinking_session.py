"""Integration tests for ChatSession extended thinking mode."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from agent.config import AgentConfig
from agent.memory import TurnRecord
from agent.session import ChatSession


class _FakeGraph:
    def __init__(self, answers=None):
        self.answers = list(answers or ["draft answer"])
        self.calls: list[dict] = []

    async def astream(self, state, config=None, stream_mode="updates"):
        self.calls.append(state)
        answer = self.answers.pop(0) if self.answers else "draft answer"
        yield {"agent": {"messages": [AIMessage(content=answer)]}}


class _FakeHistoryStore:
    def __init__(self):
        self.adds: list[dict] = []

    def add_turn(self, turn: TurnRecord, *, session_id: str, turn_id: int, timestamp: str):
        self.adds.append({
            "user_input": turn.user_input,
            "assistant_output": turn.assistant_output,
            "session_id": session_id,
            "turn_id": turn_id,
            "timestamp": timestamp,
        })


class _QueuedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.outputs.pop(0))


def _task_spec_json(decision="proceed", **overrides):
    data = {
        "task": "revise abstract",
        "task_type": "academic_revision",
        "target_output": "polished abstract",
        "confidence": "high",
        "decision": decision,
        "known_from_user": ["draft supplied"],
        "known_from_context": [],
        "allowed_assumptions": [],
        "forbidden_assumptions": ["do not add data"],
        "missing_info": [],
        "constraints": ["preserve evidence"],
        "success_criteria": ["no fabricated citations"],
        "writer_instruction": "Draft the answer.",
        "reviewer_instruction": "Check evidence alignment.",
    }
    data.update(overrides)
    import json

    return json.dumps(data)


def _review_json(decision="pass", findings=None, summary="ok"):
    import json

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


def _make_session(monkeypatch, tmp_path, graph):
    monkeypatch.setattr(
        "agent.session.build_graph",
        lambda _cfg, extra_tools=None, history_store=None, **kwargs: graph,
    )
    cfg = AgentConfig(persist_dir=str(tmp_path))
    return ChatSession(cfg, history_store=_FakeHistoryStore())


def test_extended_mode_reuses_existing_graph_with_task_spec(monkeypatch, tmp_path):
    graph = _FakeGraph(["draft answer"])
    session = _make_session(monkeypatch, tmp_path, graph)
    session.set_thinking_mode("extended")
    session._thinking_model = _QueuedModel([
        _task_spec_json(),
        _review_json(),
    ])

    answer = asyncio.run(session.turn("revise this"))

    assert answer == "draft answer"
    assert len(graph.calls) == 1
    prompt_text = "\n".join(str(msg.content) for msg in graph.calls[0]["messages"])
    assert "[Extended thinking TaskSpec]" in prompt_text
    assert len(session.recent_turns) == 1
    assert session.recent_turns[0].user_input == "revise this"
    assert session.recent_turns[0].assistant_output == "draft answer"


def test_extended_mode_clarification_does_not_call_writer_graph(monkeypatch, tmp_path):
    graph = _FakeGraph(["should not run"])
    session = _make_session(monkeypatch, tmp_path, graph)
    session.set_thinking_mode("extended")
    session._thinking_model = _QueuedModel([
        _task_spec_json(
            decision="need_clarification",
            missing_info=["target journal"],
        ),
    ])

    answer = asyncio.run(session.turn("revise this"))

    assert "target journal" in answer
    assert graph.calls == []


def test_extended_mode_revises_major_findings_until_pass(monkeypatch, tmp_path):
    graph = _FakeGraph(["draft answer"])
    session = _make_session(monkeypatch, tmp_path, graph)
    session.set_thinking_mode("extended")
    session._thinking_model = _QueuedModel([
        _task_spec_json(),
        _review_json("revise", [_finding()], "needs revision"),
        "revised answer",
        _review_json(),
    ])

    answer = asyncio.run(session.turn("revise this"))

    assert answer == "revised answer"
    assert len(session._thinking_model.calls) == 4
    assert len(graph.calls) == 1


def test_extended_mode_does_not_revise_blocker(monkeypatch, tmp_path):
    graph = _FakeGraph(["draft answer"])
    session = _make_session(monkeypatch, tmp_path, graph)
    session.set_thinking_mode("extended")
    session._thinking_model = _QueuedModel([
        _task_spec_json(),
        _review_json("block", [_finding("blocker", True)], "blocked"),
    ])

    answer = asyncio.run(session.turn("revise this"))

    assert "無法安全自動修正" in answer
    assert len(session._thinking_model.calls) == 2
    assert len(graph.calls) == 1


def test_extended_final_skill_validation_uses_graph_revision(monkeypatch, tmp_path):
    graph = _FakeGraph(["validated answer [Smith, 2020]"])
    session = _make_session(monkeypatch, tmp_path, graph)
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

    assert result.answer == "validated answer [Smith, 2020]"
    assert len(graph.calls) == 1
    prompt_text = "\n".join(str(msg.content) for msg in graph.calls[0]["messages"])
    assert "[Extended thinking final validation errors]" in prompt_text
