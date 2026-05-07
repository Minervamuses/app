"""Tests for the read_file StructuredTool factory."""

import json

from langchain_core.tools import StructuredTool

from agent.config import AgentConfig
from agent.tools.read_file import MAX_BYTES, create_read_file_tool


def _make_tool(tmp_path) -> StructuredTool:
    return create_read_file_tool(AgentConfig(persist_dir=str(tmp_path)))


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
