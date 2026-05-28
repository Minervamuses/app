"""JSONL dataset loader for deterministic evaluation cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from agent.paths import find_app_root

CLAIMS = frozenset({"c1", "c2", "c3", "c4"})
SPLITS = frozenset({"dev", "test"})
_PROVENANCE_FIELDS = ("source", "labeler", "date")


class DatasetSchemaError(ValueError):
    """Raised when a dataset JSONL file does not match the eval schema."""


@dataclass(frozen=True)
class EvalCase:
    """A validated evaluator dataset row."""

    id: str
    claim: str
    split: str
    inputs: dict[str, Any]
    gold: dict[str, Any]
    provenance: dict[str, Any]
    raw: dict[str, Any]
    line_number: int


def dataset_file_path(
    claim: str,
    split: str,
    *,
    root: str | Path | None = None,
) -> Path:
    """Return the canonical repo-root dataset path for a claim split."""
    if claim not in CLAIMS:
        raise DatasetSchemaError(f"unsupported claim: {claim!r}")
    if split not in SPLITS:
        raise DatasetSchemaError(f"unsupported split: {split!r}")
    base = Path(root) if root is not None else find_app_root()
    return base / "eval" / "datasets" / claim / f"{split}.jsonl"


def dataset_hash(path: str | Path) -> str:
    """Return a sha256 hash of the dataset file bytes."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_claim_dataset(
    claim: str,
    split: str,
    *,
    root: str | Path | None = None,
) -> list[EvalCase]:
    """Load `eval/datasets/<claim>/<split>.jsonl` from the repo root."""
    path = dataset_file_path(claim, split, root=root)
    return load_dataset(path, claim=claim, split=split)


def read_dataset(
    path: str | Path,
    *,
    claim: str | None = None,
    split: str | None = None,
) -> list[EvalCase]:
    """Read API reserved for future non-scoring analysis layers."""
    return load_dataset(path, claim=claim, split=split)


def load_dataset(
    path: str | Path,
    *,
    claim: str | None = None,
    split: str | None = None,
) -> list[EvalCase]:
    """Load and validate a JSONL evaluation dataset."""
    path = Path(path)
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                raise DatasetSchemaError(_where(path, line_number, "blank lines are not allowed"))
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetSchemaError(
                    _where(path, line_number, f"invalid JSON: {exc.msg}")
                ) from exc

            case = validate_case(
                payload,
                expected_claim=claim,
                expected_split=split,
                line_number=line_number,
                path=path,
            )
            if case.id in seen_ids:
                raise DatasetSchemaError(_where(path, line_number, f"duplicate id: {case.id!r}"))
            seen_ids.add(case.id)
            cases.append(case)

    return cases


def validate_case(
    payload: Mapping[str, Any],
    *,
    expected_claim: str | None = None,
    expected_split: str | None = None,
    line_number: int = 0,
    path: str | Path | None = None,
) -> EvalCase:
    """Validate one raw JSON object and return a typed case wrapper."""
    label = _label(path, line_number)
    if not isinstance(payload, Mapping):
        raise DatasetSchemaError(f"{label}: case must be a JSON object")

    raw = dict(payload)
    case_id = _required_str(raw, "id", label)
    claim = _required_str(raw, "claim", label)
    split = _required_str(raw, "split", label)

    if claim not in CLAIMS:
        raise DatasetSchemaError(f"{label}: unsupported claim {claim!r}")
    if split not in SPLITS:
        raise DatasetSchemaError(f"{label}: unsupported split {split!r}")
    if expected_claim is not None and claim != expected_claim:
        raise DatasetSchemaError(
            f"{label}: claim {claim!r} does not match expected {expected_claim!r}"
        )
    if expected_split is not None and split != expected_split:
        raise DatasetSchemaError(
            f"{label}: split {split!r} does not match expected {expected_split!r}"
        )

    inputs = _required_mapping(raw, "inputs", label)
    gold = _required_mapping(raw, "gold", label)
    provenance = _required_mapping(raw, "provenance", label)
    for key in _PROVENANCE_FIELDS:
        _required_str(provenance, key, f"{label}: provenance")
    try:
        date.fromisoformat(str(provenance["date"]))
    except ValueError as exc:
        raise DatasetSchemaError(
            f"{label}: provenance.date must be an ISO date, got {provenance['date']!r}"
        ) from exc

    return EvalCase(
        id=case_id,
        claim=claim,
        split=split,
        inputs=dict(inputs),
        gold=dict(gold),
        provenance=dict(provenance),
        raw=raw,
        line_number=line_number,
    )


def _required_str(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetSchemaError(f"{label}: {key} must be a non-empty string")
    return value


def _required_mapping(payload: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise DatasetSchemaError(f"{label}: {key} must be an object")
    return value


def _where(path: Path, line_number: int, message: str) -> str:
    return f"{path}:{line_number}: {message}"


def _label(path: str | Path | None, line_number: int) -> str:
    if path is None:
        return f"case line {line_number}" if line_number else "case"
    return f"{path}:{line_number}"
