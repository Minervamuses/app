# Agent State Cleanup Experiment

Date: 2026-04-13

## Task

Reduce unnecessary LangGraph state growth in the KMS chat agent, with emphasis on:
- removing full tool outputs from long-term history
- bounding per-turn tool-context growth
- keeping each chat session self-contained and non-persistent across restarts

## User Decisions

Accepted without condition:
- A: remove full `ToolMessage` contents from long-term history
- E: apply a fixed message-count cap to prompt-visible history
- F: session ends with the process; next launch starts from a clean conversation

Rejected for now:
- B: keep only the most recent 1 turn in history
  - User explicitly rejected this and noted there is a later plan for this area.
- C: keep only the most recent 2-3 turns in history

Partially accepted:
- D: bound same-turn tool interactions, with the window set to `4`

## Implemented Changes

1. `kms/agent/graph.py`
   - Removed LangGraph checkpoint-based cross-turn memory from the graph itself.
   - Added bounded prompt preparation before each agent step.

2. `kms/agent/history.py`
   - Added deterministic history helpers.
   - Kept only the most recent 4 tool interactions visible during a single turn.
   - Added a compact truncation note when older tool activity is dropped.

3. `kms/cli/chat.py`
   - Session history is now managed explicitly in `ChatSession`.
   - Long-term prompt history stores only:
     - system prompt
     - user messages
     - final assistant answers
   - Full tool outputs are excluded from long-term history.
   - Tool-call traces are still recorded in session logs.

4. `kms/config.py`
   - Added fixed controls:
     - `agent_max_messages = 20`
     - `agent_max_tool_interactions = 4`

5. `kms/evaluation/*.py`
   - Updated evaluators to exercise the actual compact-session chat behavior instead of relying on graph checkpoint memory.

## Notes

- This experiment intentionally does **not** implement turn-count pruning (`B`, `C`).
- The current design keeps tool usage trace information in session logs, while removing tool-result payloads from long-term prompt history.
- Further follow-up should focus on whether the remaining tool-call loop still needs additional stop guards beyond the 4-interaction window and 20-message cap.
