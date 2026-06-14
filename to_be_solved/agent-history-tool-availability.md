# Agent history tool availability under active skills

status: partially_done
source:
  - to_be_solved/archive/agent_history_tool_availability_spec .md

## Problem
Active skill policy can make the tools shown to the rewriter, writer, and reviewer drift from the tools actually bound in the graph. The original failure was that `academic-paper-writing` could not use `recall_history` because the skill manifest did not declare `history.search`.

## Why It Matters
When a user asks the agent to inspect prior conversation history, the agent must use the chat-history retrieval path instead of treating the request as missing academic-writing input. Tool availability drift also makes future skill policies fragile.

## Current Evidence
The archived spec describes the full failure chain: active skill filtering removed `recall_history`, `rag_search` was used against the wrong data source, and reviewer logic could not distinguish unavailable tools from empty retrieval.

Current code appears to have partial fixes: `skills/academic-paper-writing/manifest.yaml` declares `history.search`, `agent.skills.runtime.render_tool_availability_block` exists, and tests reference active-skill tool availability. This still needs a focused verification pass before closing.

## Desired Outcome
Under `academic-paper-writing`, writer, rewriter, and reviewer all share the same runtime-derived tool availability facts, and `recall_history` is available while `bash` remains denied.

## Acceptance Criteria
- [ ] `academic-paper-writing` activation resolves `recall_history` as available.
- [ ] `bash` remains denied under the same active skill.
- [ ] Rewriter, writer, and reviewer receive the same tool availability block.
- [ ] No hardcoded full tool list remains in the thinking prompts.
- [ ] Tests cover active skill, no active skill, and disallow-only policy.
- [ ] Documentation explains `rag.search` versus `history.search`.

## Notes
Do not merge RAG KB and chat history semantics. `rag_search` remains for indexed knowledge-base content; `recall_history` remains for persisted chat turns.
