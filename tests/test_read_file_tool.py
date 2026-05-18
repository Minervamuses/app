"""Tests for the read_file StructuredTool factory."""

import json

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config import AgentConfig
from agent.state import AgentState
from agent.tools.read_file import MAX_BYTES, create_read_file_tool


def _make_tool(tmp_path) -> StructuredTool:
    return create_read_file_tool(AgentConfig(persist_dir=str(tmp_path)))


def _invoke_tool_node(tool, state):
    graph = StateGraph(AgentState)
    graph.add_node("tools", ToolNode([tool]))
    graph.add_edge(START, "tools")
    return graph.compile().invoke(state)


def test_create_read_file_tool_returns_structured_tool(tmp_path):
    tool = _make_tool(tmp_path)
    assert isinstance(tool, StructuredTool)
    assert tool.name == "read_file"


def test_read_file_returns_content_and_size(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello world\n", encoding="utf-8")

    tool = _make_tool(tmp_path)
    payload = json.loads(tool.invoke({"path": str(target)}))

    assert payload["content"] == "hello world\n"
    assert payload["size"] == len("hello world\n".encode("utf-8"))
    assert payload["path"] == str(target.resolve())


def test_read_file_missing_path_returns_error(tmp_path):
    tool = _make_tool(tmp_path)
    payload = json.loads(tool.invoke({"path": str(tmp_path / "nope.txt")}))

    assert "error" in payload
    assert "does not exist" in payload["error"]


def test_read_file_oversize_returns_error(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (MAX_BYTES + 1))

    tool = _make_tool(tmp_path)
    payload = json.loads(tool.invoke({"path": str(big)}))

    assert "error" in payload
    assert "too large" in payload["error"]
    assert str(MAX_BYTES) in payload["error"]


def test_read_file_non_utf8_bytes_replaced(tmp_path):
    target = tmp_path / "bin.dat"
    target.write_bytes(b"\xff\xfe\x00hello")

    tool = _make_tool(tmp_path)
    payload = json.loads(tool.invoke({"path": str(target)}))

    assert "content" in payload
    assert "hello" in payload["content"]


def test_read_file_directory_path_returns_error(tmp_path):
    tool = _make_tool(tmp_path)
    payload = json.loads(tool.invoke({"path": str(tmp_path)}))

    assert "error" in payload
    assert "not a regular file" in payload["error"]


def test_read_file_blocks_env_file_without_leaking_path(tmp_path):
    target = tmp_path / ".env"
    target.write_text("SECRET=value", encoding="utf-8")

    tool = _make_tool(tmp_path)
    payload = json.loads(tool.invoke({"path": str(target)}))

    assert payload == {"error": "path blocked by sensitive denylist"}
    assert str(target) not in payload["error"]


def test_read_file_blocks_ssh_config_without_leaking_path(tmp_path):
    target = tmp_path / ".ssh" / "config"
    target.parent.mkdir()
    target.write_text("Host example", encoding="utf-8")

    tool = _make_tool(tmp_path)
    payload = json.loads(tool.invoke({"path": str(target)}))

    assert payload == {"error": "path blocked by sensitive denylist"}
    assert str(target) not in payload["error"]


def test_read_file_does_not_block_token_documentation_by_substring(tmp_path):
    target = tmp_path / "access_token_design.md"
    target.write_text("doc", encoding="utf-8")

    tool = _make_tool(tmp_path)
    payload = json.loads(tool.invoke({"path": str(target)}))

    assert payload["content"] == "doc"


def test_read_file_blocks_sensitive_skill_resource_path(tmp_path):
    skill_root = tmp_path / "skills" / "paper"
    refs = skill_root / "references"
    refs.mkdir(parents=True)
    target = refs / ".env"
    target.write_text("SECRET=value", encoding="utf-8")

    tool = _make_tool(tmp_path)
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"path": "references/.env"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    result = _invoke_tool_node(tool, {
        "messages": [ai_message],
        "skill_root": str(skill_root),
    })
    payload = json.loads(result["messages"][-1].content)

    assert payload == {"error": "path blocked by sensitive denylist"}


def test_read_file_resolves_relative_path_against_active_skill_root(tmp_path):
    skill_root = tmp_path / "skills" / "paper"
    refs = skill_root / "references"
    refs.mkdir(parents=True)
    target = refs / "guide.md"
    target.write_text("guide text", encoding="utf-8")

    tool = _make_tool(tmp_path)
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"path": "references/guide.md"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    result = _invoke_tool_node(tool, {
        "messages": [ai_message],
        "skill_root": str(skill_root),
    })
    payload = json.loads(result["messages"][-1].content)

    assert payload["path"] == str(target.resolve())
    assert payload["content"] == "guide text"


def test_read_file_without_skill_root_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "references" / "guide.md"
    target.parent.mkdir()
    target.write_text("cwd guide", encoding="utf-8")

    tool = _make_tool(tmp_path)
    payload = json.loads(tool.invoke({"path": "references/guide.md"}))

    assert payload["path"] == str(target.resolve())
    assert payload["content"] == "cwd guide"


def test_read_file_active_skill_reference_missing_does_not_fallback_to_cwd(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    skill_root = tmp_path / "skills" / "paper"
    skill_root.mkdir(parents=True)
    cwd_target = tmp_path / "references" / "missing.md"
    cwd_target.parent.mkdir()
    cwd_target.write_text("cwd guide", encoding="utf-8")

    tool = _make_tool(tmp_path)
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"path": "references/missing.md"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    result = _invoke_tool_node(tool, {
        "messages": [ai_message],
        "skill_root": str(skill_root),
    })
    payload = json.loads(result["messages"][-1].content)

    assert "error" in payload
    assert "does not exist" in payload["error"]
    assert str(cwd_target.resolve()) not in payload["error"]


def test_read_file_active_skill_non_resource_relative_path_uses_cwd(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    skill_root = tmp_path / "skills" / "paper"
    skill_root.mkdir(parents=True)
    (skill_root / "draft.md").write_text("skill draft", encoding="utf-8")
    cwd_target = tmp_path / "draft.md"
    cwd_target.write_text("cwd draft", encoding="utf-8")

    tool = _make_tool(tmp_path)
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"path": "draft.md"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    result = _invoke_tool_node(tool, {
        "messages": [ai_message],
        "skill_root": str(skill_root),
    })
    payload = json.loads(result["messages"][-1].content)

    assert payload["path"] == str(cwd_target.resolve())
    assert payload["content"] == "cwd draft"


def test_read_file_blocks_skill_root_escape(tmp_path):
    skill_root = tmp_path / "skills" / "paper"
    skill_root.mkdir(parents=True)

    tool = _make_tool(tmp_path)
    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"path": "../outside.md"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    result = _invoke_tool_node(tool, {
        "messages": [ai_message],
        "skill_root": str(skill_root),
    })
    payload = json.loads(result["messages"][-1].content)

    assert "error" in payload
    assert "escapes active skill root" in payload["error"]
