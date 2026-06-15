# Agent Evaluation

A long-lived, reproducible evaluator for the LangGraph agent. Core numbers come
from **deterministic code** (set comparison / rank math / regex / classifier
P-R-F1); the LLM/agent is the *system-under-test*, never the scorer.

Results are written **append-only** with version metadata so runs can be
compared over time and regressions detected. See `EVALUATOR_PLAN.md` at the repo
root for the full design rationale.

## LLM access contract

All agent and evaluation model calls use LangChain chat models from
`agent.llm`. Runtime uses `get_chat_model()` / `get_chat_model_for_role()`;
evaluation helper roles use the same contract through OpenRouter or Ollama
chat-model factories plus `invoke_text()` when a plain string response is
needed. The old `BaseLLM` prompt-to-text provider hierarchy is intentionally
removed.

## The four claims

| Claim | What it measures | Driver | Core metrics (Tier 1, deterministic) |
|-------|------------------|--------|--------------------------------------|
| **c1** | Tool-routing correctness | `ChatSession.turn_with_trace` → tool trace | `routing_accuracy`, `first_tool_accuracy`, `tool_family_accuracy`, `forbidden_tool_accuracy`, `filter_accuracy`, … |
| **c2** | Retrieval quality | `rag.api.search()` directly (no ChatSession) | `recall@k`, `mrr`, `ndcg@k` |
| **c3** | Skill compliance + reviewer gatekeeping | three independent sub-evaluators (below) | per sub-claim, see below |
| **c4** | End-to-end task completion | `ChatSession.turn_with_trace` | `task_success_rate`, `required_tools_accuracy`, `answer_requirements_accuracy` |

**c3 is three separate paths — they are never mixed:**

- **c3a — validator** (`C3ValidatorEvaluator`): runs the deterministic
  `validate_skill_output()` against labeled outputs. Pure function, no LLM.
  Metrics: `violation_precision/recall/f1`, `exact_match`, `false_positive_rate`.
- **c3b — reviewer-as-classifier** (`C3ReviewerEvaluator`): reviewer LLM is the
  SUT, scorer is deterministic. Calls `review_draft()` + `route_review_report()`
  directly (not through ChatSession). Reports macro P/R/F1 per group:
  `decision`, `route`, `failure_mode`, `needs_user_input`, `severity`.
- **c3c — session integration** (`C3SessionEvaluator`): drives the full
  normal/extended paths to confirm skill-validation retry actually fires.
  Normal path uses `build_graph(cfg).invoke()` and inspects
  `validation_attempts`; extended path tests `_apply_final_skill_validation`.
  (Retry is not a tool call, so `turn_with_trace` cannot observe it.)
  Metrics: `retry_accuracy`, `final_clean_accuracy`.

## Running the evaluator

```bash
conda activate app
cd <repo-root>/app

python -m agent.cli.eval --claim c1            # tool routing
python -m agent.cli.eval --claim c2            # retrieval (needs Ollama)
python -m agent.cli.eval --claim c3            # validator + reviewer + session
python -m agent.cli.eval --claim c4            # end-to-end checklist
```

Inside an interactive chat session:

```
/eval <c1|c2|c3|c4> [dev|test] [--allow-skips]
```

### CLI flags

| Flag | Meaning |
|------|---------|
| `--claim c1\|c2\|c3\|c4` | Which claim to run. |
| `--split dev\|test` | Dataset split (default `dev`). Use `test` only at milestones. |
| `--allow-skips` | Dev only: allow cases whose required tools are not loaded. **Official runs omit this** so missing tools fail fast (otherwise core numbers inflate). |
| `--output DIR` | Where the ledger lives (default `eval/`). |
| `--no-mcp` | Skip MCP tool loading. |

`--suite behavior|e2e|thinking` still runs the legacy suites during migration;
their model calls use the same LangChain access contract as the main runtime.
`--all` runs those legacy suites (not the c1–c4 claims).

### Prerequisites per claim

| Claim | Requires |
|-------|----------|
| c1 | `OPENROUTER_API_KEY` (in `.env`) |
| c2 | Ollama serving `bge-m3` + the frozen `../rag/store` snapshot |
| c3 | `OPENROUTER_API_KEY` (reviewer sub-claim) |
| c4 | `OPENROUTER_API_KEY` |

## Output

Each run prints an absolute-number summary and writes:

- `eval/runs/<claim>.jsonl` — **append-only ledger**, one line per run, never
  overwritten. Carries scores + metadata (split, dataset id/hash, …).
- `eval/runs/details/<run_id>.json` — per-case prediction / gold / pass-fail.
  **Suppressed by default on `--split test`** to keep the frozen test set from
  leaking through details.

### Comparing versions

```python
from agent.evaluation.ledger import read_runs, diff_runs

rows = read_runs("c1")
print(diff_runs(rows[-2], rows[-1]))   # metric deltas between two runs
```

## Datasets

Frozen JSONL cases live at the **repo root**, versioned with the code:

```
eval/datasets/<claim>/{dev,test}.jsonl
eval/datasets/c2/fixture.json          # frozen store fingerprint for c2
```

The loader / schema / provenance validation code lives in `datasets/` (this
package) — no JSONL is stored here. Each row carries `id`, `claim`, `split`,
`inputs`, `gold`, `provenance`.

- Iterate on **dev**; freeze **test**. Never tune on test.
- `test.jsonl` committed to the repo is a **frozen test**, not a sealed test
  (per-case is visible). A truly sealed test must live outside the repo.
- New failure cases enter **dev** first, then get promoted to **test** per
  release batch; test only grows, preserving historical comparability.

### c2 store fixture

The `../rag/store` Chroma index is a fixed, read-only eval fixture. Its
fingerprint (`fixture.json`) hashes the collection's ids + documents + metadatas
+ embeddings (not just `raw.json`, since retrieval queries Chroma). On each c2
run the live store is fingerprinted and compared; **a mismatch aborts the run**.
The store is gitignored, so a fresh checkout must restore the snapshot — see the
plan for the snapshot path.

## Module layout

```
agent/evaluation/
  base.py            # EvalResult, BaseEvaluator, tool_inventory
  datasets/          # JSONL loader + schema/provenance validation
  ledger.py          # append-only run ledger + diff helper
  repro.py           # version metadata + Chroma store fingerprint
  metrics/           # Tier-1 deterministic metrics (ranking, tool-routing)
  claims/            # c1_routing, c2_retrieval, c3a/b/c, c4_endtoend
  benchmarks/        # public benchmark spike (BEIR/SciFact)
```

## Current scope / not yet wired

- The ledger row currently records split + dataset id/hash. Full version
  metadata (`repro.collect_repro_metadata`: git shas, model ids, store
  fingerprint, seed, n_samples) is implemented but not yet wired into the CLI
  ledger write.
- No n-sample multi-run / mean±std aggregation yet (single run per invocation).
- Tier 2 LLM-judge metrics (faithfulness, holistic completion) and the
  judge–human agreement gate are not implemented; only Tier 1 numbers ship.
- The BEIR spike (`benchmarks/beir.py`) computes `ndcg@10` with a deterministic
  **lexical** baseline, not the agent's `bge-m3` retriever. It validates the
  benchmark interface; it does not yet externally benchmark the real retriever.
