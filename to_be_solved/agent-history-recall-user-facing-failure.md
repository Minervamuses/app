# History recall failure should produce a user-facing answer

status: partially_done
source:
  - to_be_solved/archive/problem.md
  - to_be_solved/archive/agent_history_tool_availability_spec .md

## Problem
When the user expected the agent to inspect prior records, the agent repeatedly asked intake questions and eventually surfaced internal reviewer-style instructions instead of explaining what it could or could not retrieve.

## Why It Matters
This creates a poor failure mode: the user sees a checklist or internal critique when the real issue is tool routing, empty history, plan-log storage, or skill policy. The agent should be transparent about retrieval attempts and limits.

## Current Evidence
The archived transcript shows the user asking for January research outcomes, then saying the agent should be able to see prior records. The agent called RAG tools rather than `recall_history` and returned a "needs user confirmation" style message.

Current tests appear to cover some reviewer routing behavior for `retrieval_not_attempted`, `retrieval_empty`, and `tool_unavailable`, but the original end-to-end conversation still needs replay.

## Desired Outcome
For this scenario, the agent first tries the correct history retrieval path when available. If retrieval is empty, it says so plainly and offers a narrow next step. It must not expose internal reviewer instructions.

## Acceptance Criteria
- [ ] Replaying the archived scenario triggers `recall_history` when the tool is available.
- [ ] Empty history results produce an honest user-facing explanation.
- [ ] The answer distinguishes chat history, indexed KB, and plan logs.
- [ ] The answer does not ask the user to restate all research content before retrieval is attempted.
- [ ] Internal reviewer or reviser instructions are not shown to the user.

## Notes
Plan-mode logs are not chat-history Chroma records. If the missing content may be in `plan_logs/`, the answer should say that explicitly without moving or indexing those logs.
