# Skill frontmatter parser should use PyYAML

status: done
source:
  - to_be_solved/archive/deep-research-report.md

## Problem
Skill metadata parsing uses a hand-written minimal YAML frontmatter parser even though the same package already depends on and uses PyYAML.

## Why It Matters
A custom partial parser adds edge cases and maintenance cost without clear benefit. Valid YAML that exceeds the hand-written subset may parse incorrectly.

## Current Evidence
The archived complexity report identifies `_parse_frontmatter()` in `agent/skills/metadata.py` and contrasts it with existing `yaml.safe_load()` usage elsewhere in the skills package.

## Desired Outcome
Skill frontmatter is extracted as a block and parsed with `yaml.safe_load()`, keeping the existing metadata contract intact.

## Acceptance Criteria
- [x] `_parse_frontmatter()` uses `yaml.safe_load()` or is replaced by an equivalent PyYAML-backed helper.
- [x] Missing or malformed frontmatter behavior remains deliberate and tested.
- [x] Multiline YAML values are handled by the YAML parser rather than ad hoc string concatenation.
- [x] Skill discovery tests still pass.

## Notes
Do not add a new dependency for this unless PyYAML proves insufficient.
