"""Append-only run ledger for evaluator results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agent.evaluation.base import EvalResult
from agent.paths import find_app_root


@dataclass(frozen=True)
class LedgerPaths:
    """Filesystem paths for one claim ledger."""

    runs_dir: Path
    ledger_path: Path
    details_dir: Path


def ledger_paths(claim: str, *, root: str | Path | None = None) -> LedgerPaths:
    """Return ledger and details paths under `eval/runs/`."""
    base = Path(root) if root is not None else find_app_root()
    runs_dir = base / "eval" / "runs"
    return LedgerPaths(
        runs_dir=runs_dir,
        ledger_path=runs_dir / f"{claim}.jsonl",
        details_dir=runs_dir / "details",
    )


def append_result(
    claim: str,
    result: EvalResult,
    *,
    root: str | Path | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
    include_details: bool = True,
    extra_metadata: dict | None = None,
) -> dict:
    """Append one result row and optionally write per-run details.

    The ledger row is aggregate-first: details are stored separately so frozen
    test runs can pass `include_details=False` and avoid recording predictions.
    """
    paths = ledger_paths(claim, root=root)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)
    paths.details_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_id or new_run_id(claim)
    timestamp = timestamp or _utc_timestamp()
    metadata = dict(result.metadata)
    if extra_metadata:
        metadata.update(extra_metadata)

    details_path: Path | None = None
    if include_details:
        details_path = paths.details_dir / f"{run_id}.json"
        if details_path.exists():
            raise FileExistsError(f"details file already exists: {details_path}")
        details_payload = {
            "run_id": run_id,
            "claim": claim,
            "name": result.name,
            "total": result.total,
            "details": result.details,
        }
        _write_json(details_path, details_payload)

    row = {
        "run_id": run_id,
        "claim": claim,
        "timestamp": timestamp,
        "name": result.name,
        "total": result.total,
        "scores": result.scores,
        "metadata": metadata,
        "details_path": (
            str(details_path.relative_to(paths.runs_dir.parent)) if details_path else None
        ),
    }
    with paths.ledger_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row


def read_runs(claim: str, *, root: str | Path | None = None) -> list[dict]:
    """Read all ledger rows for a claim."""
    path = ledger_paths(claim, root=root).ledger_path
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid ledger JSON") from exc
    return rows


def read_run(
    claim: str,
    run_id: str,
    *,
    root: str | Path | None = None,
) -> dict:
    """Read one run row by id."""
    for row in read_runs(claim, root=root):
        if row.get("run_id") == run_id:
            return row
    raise KeyError(f"run_id not found for {claim}: {run_id}")


def read_details(
    run_id: str,
    *,
    root: str | Path | None = None,
) -> dict:
    """Read a per-run details JSON file."""
    base = Path(root) if root is not None else find_app_root()
    path = base / "eval" / "runs" / "details" / f"{run_id}.json"
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def diff_runs(base: dict, head: dict) -> dict:
    """Return metric deltas and case pass/fail transitions for two runs."""
    base_scores = base.get("scores", {})
    head_scores = head.get("scores", {})
    metric_deltas = {
        metric: head_scores.get(metric, 0) - base_scores.get(metric, 0)
        for metric in sorted(set(base_scores) | set(head_scores))
    }
    return {
        "base_run_id": base.get("run_id"),
        "head_run_id": head.get("run_id"),
        "metric_deltas": metric_deltas,
    }


def new_run_id(claim: str) -> str:
    """Generate a unique, sortable run id."""
    return f"{claim}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")
