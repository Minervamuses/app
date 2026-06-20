# Project History Record

This folder records the major turns found by scanning all 279 commits reachable from `HEAD` on the new `report` branch as of 2026-06-20.

## Files

- `project-turning-points.md`: narrative timeline of the project turns and the commits behind each turn.
- `relevant-commits.md`: broad inventory of refactor, major-change, debug/fix, plan/spec/report, and large-diff commits.
- `all-commits.tsv`: full chronological commit log used as the scan base.
- `source-documents.md`: index of all preserved plan/spec/report/debug documents.
- `source-documents/`: exact copied source documents that could be reconstructed from the current tree or git history.

## Method

1. Created and switched to branch `report`.
2. Scanned the complete branch history with `git log --reverse` (279 commits).
3. Flagged candidate commits by conventional subject (`refactor`, `fix`), debugging language (`debug`, `diagnostic`, `runaway`, `failure`), architecture language (`split`, `rewrite`, `decouple`, `single-source`, `contract`), plan/spec/report path matches, and large diffs.
4. Searched historical paths for planning records (`plan`, `spec`, `report`, `debug`, `problem`, `to_be_solved`, `note`).
5. Copied every project-history document that was still present or reconstructable from git history.

The scan intentionally excludes stash-like commits outside the reachable `HEAD` history and excludes skill reference content that matched names like `reporting-guidelines.md` but does not describe this project's engineering history.
