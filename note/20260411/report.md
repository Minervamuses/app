# Evaluation Run — 2026-04-11 19:15

## Setup

| Component | Value |
|---|---|
| Agent LLM | `z-ai/glm-5` (via OpenRouter) |
| Generation LLM | `google/gemini-3.1-pro-preview` |
| Judge LLM | `openai/gpt-5.2` |
| Filter LLM | `llama3.1:8b` (local Ollama) |
| Embedder | `bge-m3` (local Ollama) |
| Recursion limit | 32 |

Changes vs. previous run (2026-04-05):
- Swapped agent LLM `gemini-3-flash-preview` → `z-ai/glm-5`
- Added local Ollama chunk quality filter before question generation (rejects compiled JS, lock files, SQL DDL, etc.)
- Removed `chunk_hit_rate` metric (redundant with LLM-as-judge)

## Results

### End-to-End (25 cases — 5 skipped by chunk filter from 30 attempted)

| Metric | Value |
|---|---|
| avg_score (0-1 normalized) | 25.3% |
| avg_score_raw (0-3) | **0.76** |
| score_3_pct | 24% |
| score_2_pct | 0% |
| score_1_pct | 4% |
| score_0_pct | 72% |

### Behavior (8 built-in cases)

| Metric | Value |
|---|---|
| first_tool_accuracy | 60% |
| tool_count_accuracy | 75% |
| no_tool_accuracy | 100% |
| tools_coverage | 0% |
| filter_accuracy | 100% |

## Failure Analysis (18 zero-score cases)

| Bucket | Count | % of zeros |
|---|---|---|
| Agent hit recursion limit | 10 | 56% |
| Judge parsing failed | 8 | 44% |

Note: the distribution is **bimodal**. When the agent finishes cleanly, it scores 3/3 most of the time (6/7 non-zero cases = perfect). When it fails, it fails hard.

### Problem 1 — Agent over-loops (40% of all cases)

`z-ai/glm-5` is tool-happy. Despite SYSTEM_PROMPT saying "after 1-3 searches, synthesize", the behavior suite captured:
- "How does the scoring module work?" → **18 tool calls**
- "How does the embedding module work?" → **10 tool calls**

These are simple questions; it should take 1-2 searches max. The model keeps re-searching instead of synthesizing.

### Problem 2 — Judge parsing failures silently kill good answers (32% of all cases)

All 8 "judge parsing failed" cases had substantive agent answers starting with:
- `"Based on the research notes..."`
- `"Based on my analysis of the PiDNA1 source..."`
- `"根據最新的研究筆記與 Java 原始碼分析..."`

These look like real attempts at answering. They're being force-scored as 0 because `openai/gpt-5.2` is returning something that can't be JSON-parsed. Raw response isn't being captured, so root cause is unknown.

If half of these had scored 2+, avg_score_raw would jump from 0.76 to ~1.4.

## Ranked Next Steps

1. **Fix judge parsing** — add `response_format={"type": "json_object"}` to `OpenRouterLLM` and use it for judge calls. Log raw response on parse failure for debugging. Potential impact: recover ~4 cases, avg +0.3–0.5.
2. **Tame agent over-looping** — either tighten SYSTEM_PROMPT with stronger stop conditions or raise `recursion_limit` to 48. Tighter prompt is better long-term. Potential impact: recover ~5–8 cases, avg +0.6–1.0.
3. **Investigate why `z-ai/glm-5` over-loops** — is it a prompt adherence issue, or does glm-5 just have different tool-use defaults? May need to reconsider model choice.

## Files

- Cases: `store/eval/e2e_cases_20260411_1915.json`
- Results: `store/eval/e2e_results_20260411_1915.json`
- Behavior results: `store/eval/behavior_results_20260411_1908.json`
