# LLM access contract should be explicit

status: needs_triage
source:
  - to_be_solved/archive/deep-research-report.md

## Problem
The codebase has both LangChain chat-model factories and a `BaseLLM` wrapper hierarchy. The boundary between core runtime, evaluation, and legacy prompt-to-text usage is not explicit.

## Why It Matters
Two model access contracts increase cognitive load and make provider changes harder. It is unclear whether both paths are required long-term or one is legacy/eval-only.

## Current Evidence
The archived complexity report notes that core graph/session paths use chat-model factories, while `agent/llm/base.py`, `OpenRouterLLM`, and `OllamaLLM` provide a separate prompt-to-text abstraction.

## Desired Outcome
The project has a documented decision: either standardize on LangChain chat-model factories, or clearly isolate the `BaseLLM` path for legacy/evaluation use.

## Acceptance Criteria
- [ ] All references to `BaseLLM`, `OpenRouterLLM`, and `OllamaLLM` are inventoried.
- [ ] The intended model access boundary is documented.
- [ ] Dead or legacy-only code is moved, renamed, or documented accordingly.
- [ ] No runtime or evaluation path changes behavior unintentionally.

## Notes
Do not delete the wrapper hierarchy until its usage in CLI and evaluation code is confirmed.
