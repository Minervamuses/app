#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SPLIT="${SPLIT:-dev}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ALLOW_SKIPS="${ALLOW_SKIPS:-0}"
NO_MCP="${NO_MCP:-0}"
RUN_LEGACY_SUITES="${RUN_LEGACY_SUITES:-0}"
LEGACY_GENERATE_N="${LEGACY_GENERATE_N:-5}"

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing to start formal eval with a dirty worktree." >&2
  echo "Commit, stash, or restore these changes first:" >&2
  git status --short >&2
  exit 1
fi

RUN_STARTED_AT="$(date -u +"%Y%m%dT%H%M%SZ")"

echo "full_eval_started_at=${RUN_STARTED_AT}"
echo "split=${SPLIT}"
echo "head=$(git rev-parse HEAD)"
echo "head_short=$(git rev-parse --short HEAD)"
echo "branch=$(git branch --show-current)"
echo "allow_skips=${ALLOW_SKIPS}"
echo "no_mcp=${NO_MCP}"
echo "run_legacy_suites=${RUN_LEGACY_SUITES}"

MCP_ARGS=()
if [[ "$NO_MCP" == "1" ]]; then
  MCP_ARGS+=(--no-mcp)
fi

C1_ARGS=()
if [[ "$ALLOW_SKIPS" == "1" ]]; then
  C1_ARGS+=(--allow-skips)
fi

run_claim() {
  local claim="$1"
  shift || true
  echo
  echo "=================================================="
  echo "Running formal eval: ${claim}/${SPLIT}"
  echo "=================================================="
  "$PYTHON_BIN" -m agent.cli.eval \
    --claim "$claim" \
    --split "$SPLIT" \
    "${MCP_ARGS[@]}" \
    "$@"
}

run_claim c1 "${C1_ARGS[@]}"
run_claim c2
run_claim c3
run_claim c4

if [[ "$RUN_LEGACY_SUITES" == "1" ]]; then
  echo
  echo "=================================================="
  echo "Running legacy eval suites"
  echo "=================================================="
  "$PYTHON_BIN" -m agent.cli.eval --suite behavior "${MCP_ARGS[@]}"
  "$PYTHON_BIN" -m agent.cli.eval --suite thinking "${MCP_ARGS[@]}"
  "$PYTHON_BIN" -m agent.cli.eval \
    --suite e2e \
    --generate "$LEGACY_GENERATE_N" \
    "${MCP_ARGS[@]}"
fi

echo
echo "Formal eval complete."
echo "Ledger: eval/runs/*.jsonl"
