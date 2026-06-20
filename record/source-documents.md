# Source Documents Preserved

These files are exact text copies of project plans, specs, reports, and debug notes that explain major turns in the repository history.
The copied content lives in `record/source-documents/`; this index records where each copy came from. `HEAD:<path>` means the copy reflects the current file contents on this branch, while `<commit>:<path>` means the copy was reconstructed from that historical git object.

| Copy | Copied from | Related commit | Why included |
| --- | --- | --- | --- |
| `source-documents/20260411-evaluation-run-report.md` | `HEAD:note/20260411/report.md` | `508e196fcd109d58e2afb00acd664321e23147b1` | Evaluation run after adding Ollama chunk quality filter and switching agent LLM to GLM-5. |
| `source-documents/20260413-agent-state-cleanup.md` | `HEAD:note/20260413/agent_state_cleanup.md` | `e757dacbdcc4f429a31f7541f367aef62707f7fb` | Agent state cleanup experiment after context growth problems. |
| `source-documents/20260418-repo-split-plan.md` | `968edf0c0a486ca05ae92e45ef38657765ce4673:repo_split_plan.md` | `968edf0c0a486ca05ae92e45ef38657765ce4673` | Historical repo split plan deleted later by `0ad1922`. |
| `source-documents/20260420-mcp-setup.md` | `HEAD:note/20260420/mcp_setup.md` | `7c09d4730d80326b387f7c23b20dcfcfb21352e6` | MCP setup notes for Web Search and GitHub tools in the Python agent. |
| `source-documents/20260524-thinking-extended-plan-v3-4.md` | `55e604b7e4d8ebfa3ff4932e7a3261eaa9a1eeac^:thinking_extended_plan.md` | `cfc79fd319a52e52020187d550f4a31937a7b210`, `147e918df1fad83228e65035bad622b4cde2dfbb` | Final historical state before the plan was replaced by later evaluation/history notes. |
| `source-documents/20260524-problem.md` | `HEAD:to_be_solved/archive/problem.md` | `147e918df1fad83228e65035bad622b4cde2dfbb`, `c94604a29365e70b651e5fb8fc5666515ab51e94` | Archived problem transcript that motivated history/tool-availability work. |
| `source-documents/20260527-evaluator-plan.md` | `HEAD:EVALUATOR_PLAN.md` | `55e604b7e4d8ebfa3ff4932e7a3261eaa9a1eeac` | Long-term evaluator rebuild plan, basis of C1-C4 evaluation package. |
| `source-documents/20260527-history-tool-availability-spec.md` | `HEAD:to_be_solved/archive/agent_history_tool_availability_spec .md` | `55e604b7e4d8ebfa3ff4932e7a3261eaa9a1eeac` | Spec for active skill history recall and tool availability mismatch. |
| `source-documents/20260527-p0-5-history-query-diagnostic.md` | `HEAD:note/20260527/p0_5_history_query_diagnostic.md` | `55e604b7e4d8ebfa3ff4932e7a3261eaa9a1eeac`, `a572356d8d88c4044637485c2421fc05c2ba2079` | Diagnostic evidence that local chat_history contained relevant January-history hits. |
| `source-documents/20260530-c1-routing-findings.md` | `HEAD:note/20260530/c1_routing_findings.md` | `379e4b6897d3630cf938f57c5f71c4ada9b6c4be` | First C1 routing run findings, including runaway/tool-selection symptoms. |
| `source-documents/20260530-tool-call-runaway-fix-plan.md` | `HEAD:to_be_solved/archive/fix_plan.md` | `370d0ac58301e6793363ce5eb0496694ebb09478`, `7e8c639d8d3788f202f9ed9121907f70a2310613`, `c94604a29365e70b651e5fb8fc5666515ab51e94` | Final archived fix plan for tool-call runaway after Phase 0/Phase 3 evidence. |
| `source-documents/20260530-tool-call-runaway-debug.md` | `HEAD:note/20260530/tool_call_runaway_debug.md` | `21510ed6ed8e411bdb80c4993ba59d20de8718a1`, `4333fa4374e2b765c91c0a14bf473b27f5a79e45` | Detailed debug record for the runaway investigation. |
| `source-documents/20260531-eval-claim-run.md` | `HEAD:note/20260531/20260531-eval-claim-run.md` | `c52268174d162e44c2c04fa8665c0b3c1cf7e88e` | C1-C4 dev evaluation run note and dataset inventory. |
| `source-documents/20260531-extended-thinking-scope-debug.md` | `HEAD:note/20260531/20260531-extended-thinking-scope-debug.md` | `aff7f99c4cbbdcf58a2d975624c39baf7a7670c8` | Extended thinking scope and global-budget debug note. |
| `source-documents/20260531-extended-thinking-example.md` | `HEAD:note/20260531/example.md` | `aff7f99c4cbbdcf58a2d975624c39baf7a7670c8` | Raw CLI example backing the extended thinking scope debug note. |
| `source-documents/20260614-complexity-audit.md` | `HEAD:to_be_solved/archive/deep-research-report.md` | `c94604a29365e70b651e5fb8fc5666515ab51e94` | Complexity audit that fed the June cleanup/refactor task cards. |
| `source-documents/20260615-skill-state-single-serializer.md` | `HEAD:to_be_solved/skill-state-single-serializer.md` | `b0e675f1faa261c24ad3f4755806f8ce7c7035d9` | Task card for centralizing active skill state serialization. |
| `source-documents/20260615-recent-changes-eval-report.md` | `HEAD:note/20260615/report.md` | `e23d36fbd006d46893937cfb01050b3787c2cff6`, `fa1cd8d36cce81cdb74da2608e124b95eea9f2d5` | June 14-15 consolidation and dev evaluation report. |

## Exclusions

- `skills/academic-paper-writing/references/literature-review.md` and similar skill reference files were excluded from this archive because they are domain guidance, not project-history plans/specs/debug records.
- `tests/test_plan_mode.py` was excluded because it is a test file whose name matches `plan`, not a planning record.
- One `git log --all` match was a stash-style commit (`On main: pre-full-eval-worktree-cleanup`); the project-history scan in this folder is based on the 279 commits reachable from `HEAD` on the new `report` branch.
