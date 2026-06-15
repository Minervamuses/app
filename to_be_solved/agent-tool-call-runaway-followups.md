# Agent tool-call runaway follow-ups

status: done
source:
  - to_be_solved/archive/fix_plan.md

## Problem
The original runaway bug caused simple questions to trigger many repeated tool calls. The hard budget fixes appear to exist, but follow-up work remains around give-up behavior, evaluator cases, and choosing a data-backed tool-call limit.

## Why It Matters
Repeated low-value tool calls waste time, hide earlier tool results, and make evaluation runs noisy. The agent must stop searching when repeated retrieval is irrelevant or the answer is not in the indexed corpus.

## Current Evidence
The archived fix plan says the per-turn hard cap and exhausted raw-response cleanup were completed. Current code has `_cap_tool_calls`, tool budget notes, and `prepare_messages_for_agent` no longer drops same-turn tool results.

Remaining items in the archive include give-up discipline, evaluator case repair, choosing `agent_max_tool_interactions` from data, and adding operational safeguards.

## Desired Outcome
The agent makes a bounded number of retrieval attempts, recognizes repeated or irrelevant results, and gives an honest "not found in KB" answer instead of re-querying until the budget is exhausted.

## Acceptance Criteria
- [x] Tests confirm per-turn tool calls never exceed `agent_max_tool_interactions`.
- [x] A "not in KB" evaluation case stops after a small number of searches.
- [x] The c1 embedding case is either rewritten to target indexed content or reclassified as a graceful-give-up case.
- [x] The default tool-call limit has supporting evaluation data.
- [x] Evaluation runs include enough progress output to distinguish slow model calls from hangs.

## Resolution
- **Per-turn cap held at `agent_max_tool_interactions=4`.** The graph already
  enforces it in `agent.graph.agent_node` / `_cap_tool_calls`;
  `tests/test_graph_skill_loader.py` now also proves the cap holds across
  multiple rounds even when the model keeps emitting extra parallel calls.
- **Give-up discipline added via prompt + eval, not a graph heuristic.** The
  base tool workflow in `agent/tools/inventory.py` now tells the model to stop
  after at most 1-3 searches when results are empty/repetitive/unrelated, avoid
  `rag_get_context` on irrelevant results, and state that the indexed KB lacks
  enough evidence.
- **C1 embedding case reclassified, not duplicated.** Dev case
  `rag_context_embedding_followup` is now `rag_graceful_give_up`: a single-turn
  embedding-module question that requires `rag_search` (first tool
  `rag_search`/`rag_explore`), forbids `rag_get_context`/history/web/file/bash,
  bounds tool count to 1-3, and adds `expected_answer_regex` requiring both a
  KB/indexed cue and a not-found/insufficient-evidence cue. C1 dev stays 8 cases.
- **Answer scoring is backward compatible.** Optional gold field
  `expected_answer_regex` produces the `answer_ok` metric (`answer_accuracy`);
  cases without it score unchanged.
- **C1 eval progress.** `C1RoutingEvaluator(progress_cb=...)` and the
  `--claim c1` CLI emit case/turn/tool-call/tool-result lines so a slow model
  call is distinguishable from a hang.

## Notes
Keep runtime scope explicit: current budget is per turn. Do not silently change
to a per-conversation budget without a separate design decision. No graph-level
low-quality-result detector was added in this pass.

### Why the default cap stays 4
In the C1 dev routing run
(`eval/runs/details/c1-20260531T140213Z-9e6805a9.json`), every normal eligible
case fit within 0-4 tool calls. The only runaway was this embedding case, which
made 8 RAG calls (all `rag_search`/`rag_explore`, never `rag_get_context`)
because the embedding module was absent from the indexed KB and the agent had no
give-up discipline — not because 4 was too low. Raising the cap would have let it
search longer; the fix is to stop and answer honestly instead.
