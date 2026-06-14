# Agent tool-call runaway follow-ups

status: partially_done
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
- [ ] Tests confirm per-turn tool calls never exceed `agent_max_tool_interactions`.
- [ ] A "not in KB" evaluation case stops after a small number of searches.
- [ ] The c1 embedding case is either rewritten to target indexed content or reclassified as a graceful-give-up case.
- [ ] The default tool-call limit has supporting evaluation data.
- [ ] Evaluation runs include enough progress output to distinguish slow model calls from hangs.

## Notes
Keep runtime scope explicit: current budget is per turn. Do not silently change to a per-conversation budget without a separate design decision.
