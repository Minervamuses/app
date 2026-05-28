"""Tests for append-only evaluation run ledger."""

import json

import pytest

from agent.evaluation.base import EvalResult
from agent.evaluation.ledger import append_result, diff_runs, read_details, read_run, read_runs


def _result(score, *, details=None):
    return EvalResult(
        name="C1Routing",
        total=1,
        scores={"routing_accuracy": score},
        details=details or [{"id": "case-1", "passed": score == 1.0}],
        metadata={"dataset_hash": "abc123"},
    )


def test_append_result_writes_two_ledger_rows_without_overwriting(tmp_path):
    first = append_result("c1", _result(1.0), root=tmp_path, run_id="run-1")
    second = append_result("c1", _result(0.0), root=tmp_path, run_id="run-2")

    ledger_path = tmp_path / "eval" / "runs" / "c1.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == first
    assert json.loads(lines[1]) == second
    assert json.loads(lines[0])["run_id"] == "run-1"


def test_append_result_writes_readable_details(tmp_path):
    append_result("c1", _result(1.0), root=tmp_path, run_id="run-1")

    run = read_run("c1", "run-1", root=tmp_path)
    details = read_details("run-1", root=tmp_path)

    assert run["details_path"] == "runs/details/run-1.json"
    assert details["run_id"] == "run-1"
    assert details["details"] == [{"id": "case-1", "passed": True}]


def test_append_result_can_omit_details_for_frozen_test_runs(tmp_path):
    run = append_result(
        "c1",
        _result(1.0),
        root=tmp_path,
        run_id="official-test-run",
        include_details=False,
    )

    assert run["details_path"] is None
    assert not (tmp_path / "eval" / "runs" / "details" / "official-test-run.json").exists()


def test_append_result_refuses_to_overwrite_existing_details(tmp_path):
    append_result("c1", _result(1.0), root=tmp_path, run_id="run-1")

    with pytest.raises(FileExistsError):
        append_result("c1", _result(0.0), root=tmp_path, run_id="run-1")


def test_read_runs_returns_empty_list_for_missing_ledger(tmp_path):
    assert read_runs("c1", root=tmp_path) == []


def test_diff_runs_reports_metric_deltas():
    base = {"run_id": "base", "scores": {"routing_accuracy": 0.75}}
    head = {"run_id": "head", "scores": {"routing_accuracy": 1.0}}

    diff = diff_runs(base, head)

    assert diff == {
        "base_run_id": "base",
        "head_run_id": "head",
        "metric_deltas": {"routing_accuracy": 0.25},
    }
