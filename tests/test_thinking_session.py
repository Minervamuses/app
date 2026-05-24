"""Integration tests for ChatSession extended thinking mode."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from agent.config import AgentConfig
from agent.memory import TurnRecord
from agent.session import ChatSession
from agent.thinking import REVISER_FORMAT_WARNING


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


def _configured_agent_config(tmp_path, **overrides):
    data = {
        "persist_dir": str(tmp_path),
        "thinking_rewrite_model": "anthropic/claude-haiku-5",
        "thinking_reviewer_model": "openai/gpt-5.2",
        "thinking_repair_model": "meta-llama/llama-3.1-8b-instruct",
    }
    data.update(overrides)
    return AgentConfig(**data)


def _make_session(monkeypatch, tmp_path, graph, models=None, cfg=None):
    monkeypatch.setattr(
        "agent.session.build_graph",
        lambda _cfg, extra_tools=None, history_store=None, **kwargs: graph,
    )
    models = models or {
        "rewrite": _QueuedModel(["rewritten prompt"]),
        "reviewer": _QueuedModel([_review_json()]),
        "repair": _QueuedModel([]),
    }
    monkeypatch.setattr(
        "agent.session.get_chat_model_for_role",
        lambda _cfg, *, role: models[role],
    )
    session = ChatSession(
        cfg or _configured_agent_config(tmp_path),
        history_store=_FakeHistoryStore(),
    )
    session._prompt_master_skill_text_cache = "prompt-master skill"
    return session, models


def _last_human_content(state):
    humans = [message for message in state["messages"] if isinstance(message, HumanMessage)]
    return humans[-1].content


def test_extended_mode_rewrites_prompt_then_uses_existing_graph(monkeypatch, tmp_path):
    graph = _FakeGraph(["draft answer"])
    session, models = _make_session(monkeypatch, tmp_path, graph)
    session.set_thinking_mode("extended")

    answer = asyncio.run(session.turn("revise this"))

    assert answer == "draft answer"
    assert _last_human_content(graph.calls[0]) == "rewritten prompt"
    prompt_text = "\n".join(str(msg.content) for msg in graph.calls[0]["messages"])
    assert "[Original user input]" in prompt_text
    assert "revise this" in prompt_text
    assert "[Rewritten by prompt-master]" in prompt_text
    assert len(models["rewrite"].calls) == 1
    assert len(models["reviewer"].calls) == 1
    assert len(session.recent_turns) == 1
    assert session.recent_turns[0].assistant_output == "draft answer"


def test_extended_mode_clarification_does_not_call_writer_graph(monkeypatch, tmp_path):
    graph = _FakeGraph(["should not run"])
    models = {
        "rewrite": _QueuedModel(["<<CLARIFY>>\n- Which target journal?"]),
        "reviewer": _QueuedModel([]),
        "repair": _QueuedModel([]),
    }
    session, _models = _make_session(monkeypatch, tmp_path, graph, models=models)
    session.set_thinking_mode("extended")

    answer = asyncio.run(session.turn("revise this"))

    assert "Which target journal?" in answer
    assert graph.calls == []


def test_extended_mode_revises_major_findings_until_pass(monkeypatch, tmp_path):
    graph = _FakeGraph([
        "draft answer",
        "DRAFT:\nrevised answer\n\nREBUTTAL:\n(none)",
    ])
    models = {
        "rewrite": _QueuedModel(["rewritten prompt"]),
        "reviewer": _QueuedModel([
            _review_json("revise", [_finding()], "needs revision"),
            _review_json("pass"),
        ]),
        "repair": _QueuedModel([]),
    }
    session, models = _make_session(monkeypatch, tmp_path, graph, models=models)
    session.set_thinking_mode("extended")

    answer = asyncio.run(session.turn("revise this"))

    assert answer == "revised answer"
    assert len(graph.calls) == 2
    assert len(models["reviewer"].calls) == 2
    second_review = models["reviewer"].calls[1][-1].content
    assert "[Reviser round 1]" in second_review
    assert "(none)" in second_review
    reviser_prompt = "\n".join(str(msg.content) for msg in graph.calls[1]["messages"])
    assert "[Reviewer feedback]" in reviser_prompt
    assert "DRAFT:" in reviser_prompt
    assert "REBUTTAL:" in reviser_prompt


def test_extended_mode_does_not_revise_blocker(monkeypatch, tmp_path):
    graph = _FakeGraph(["draft answer"])
    models = {
        "rewrite": _QueuedModel(["rewritten prompt"]),
        "reviewer": _QueuedModel([
            _review_json("block", [_finding("blocker", True)], "blocked"),
        ]),
        "repair": _QueuedModel([]),
    }
    session, models = _make_session(monkeypatch, tmp_path, graph, models=models)
    session.set_thinking_mode("extended")

    answer = asyncio.run(session.turn("revise this"))

    assert "無法安全自動修正" in answer
    assert len(models["reviewer"].calls) == 1
    assert len(graph.calls) == 1


def test_extended_mode_runs_final_review_after_second_revision(monkeypatch, tmp_path):
    graph = _FakeGraph([
        "draft answer",
        "DRAFT:\nrevision one\n\nREBUTTAL:\nr1",
        "DRAFT:\nrevision two\n\nREBUTTAL:\nr2",
    ])
    models = {
        "rewrite": _QueuedModel(["rewritten prompt"]),
        "reviewer": _QueuedModel([
            _review_json("revise", [_finding()], "first"),
            _review_json("revise", [_finding()], "second"),
            _review_json("revise", [_finding()], "still problematic"),
        ]),
        "repair": _QueuedModel([]),
    }
    session, models = _make_session(monkeypatch, tmp_path, graph, models=models)
    session.set_thinking_mode("extended")

    answer = asyncio.run(session.turn("revise this"))

    assert "revision two" in answer
    assert "仍需確認處" in answer
    assert "still problematic" in answer
    assert len(models["reviewer"].calls) == 3
    assert len(graph.calls) == 3


def test_extended_mode_warns_when_reviser_format_cannot_be_repaired(monkeypatch, tmp_path):
    graph = _FakeGraph(["draft answer", "Clean answer without markers"])
    models = {
        "rewrite": _QueuedModel(["rewritten prompt"]),
        "reviewer": _QueuedModel([
            _review_json("revise", [_finding()], "needs revision"),
            _review_json("pass"),
        ]),
        "repair": _QueuedModel(["still unmarked"]),
    }
    session, _models = _make_session(monkeypatch, tmp_path, graph, models=models)
    session.set_thinking_mode("extended")

    answer = asyncio.run(session.turn("revise this"))

    assert answer.startswith(REVISER_FORMAT_WARNING)
    assert "Clean answer without markers" in answer


def test_extended_mode_records_configuration_error(monkeypatch, tmp_path):
    graph = _FakeGraph(["should not run"])
    session, _models = _make_session(
        monkeypatch,
        tmp_path,
        graph,
        cfg=AgentConfig(persist_dir=str(tmp_path)),
    )
    session.set_thinking_mode("extended")

    answer = asyncio.run(session.turn("revise this"))

    assert "thinking_reviewer_model" in answer
    assert graph.calls == []


def test_extended_final_skill_validation_uses_graph_revision(monkeypatch, tmp_path):
    graph = _FakeGraph(["DRAFT:\nvalidated answer [Smith, 2020]\n\nREBUTTAL:\n(none)"])
    session, _models = _make_session(monkeypatch, tmp_path, graph)
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
