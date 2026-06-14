# Base tool inventory should have one source of truth

status: open
source:
  - to_be_solved/archive/deep-research-report.md

## Problem
The base tool list is maintained in multiple places, including graph construction, session skill activation, and evaluation inventory.

## Why It Matters
Tool inventory drift can make prompts, graph binding, skill policy, and evaluators disagree about which tools exist. That leads to hard-to-debug behavior where the prompt says a tool exists but the runtime does not bind it, or the evaluator scores against the wrong inventory.

## Current Evidence
The archived complexity report identifies duplicated tool inventories around `agent/session.py`, `agent/graph.py`, `agent/evaluation/base.py`, and RAG tool constants.

## Desired Outcome
One helper or module owns the local base tool inventory and exposes both tool instances and names for graph, session, and evaluation use.

## Acceptance Criteria
- [ ] Session skill activation no longer hardcodes the local tool-name list.
- [ ] Evaluation inventory uses the same source as graph/session inventory.
- [ ] Adding, removing, or renaming a base tool requires changing only one inventory source.
- [ ] Existing skill policy tests still pass.

## Notes
This is a maintenance task, not a behavior redesign. Keep the public tool names unchanged unless a separate task explicitly changes them.
