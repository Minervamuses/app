# Active skill state should use a single serializer

status: open
source:
  - to_be_solved/archive/deep-research-report.md

## Problem
Active skill runtime state is converted into graph/session state in more than one place with nearly identical dictionary construction.

## Why It Matters
Duplicated state serialization can drift when new skill fields are added. Because this state controls instructions, references, tool policy, and validation flags, drift can create subtle runtime bugs.

## Current Evidence
The archived complexity report points to duplicate active-skill state assembly in graph and session code.

## Desired Outcome
There is one serializer for converting a `SkillRuntime` into agent state, used by both graph and session paths.

## Acceptance Criteria
- [ ] Graph skill loading and session turn invocation use the same serializer.
- [ ] The serializer covers active skill name, root, instructions, pinned references, task mode, allowed tools, denied tools, tool policy, and validation defaults.
- [ ] Existing graph skill-loader and skill-adherence tests pass.
- [ ] No behavior change is introduced for inactive sessions.

## Notes
Prefer a small helper over a broad graph/session refactor.
