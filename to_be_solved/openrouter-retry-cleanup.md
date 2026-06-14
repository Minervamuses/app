# OpenRouter retry logic should rely on the official client

status: open
source:
  - to_be_solved/archive/deep-research-report.md

## Problem
`OpenRouterLLM` has a custom retry/backoff loop while the OpenAI client and LangChain chat model path already support retry configuration.

## Why It Matters
Hand-maintained retry code can diverge from upstream behavior and adds a separate policy for rate-limit handling. It also makes provider behavior harder to reason about.

## Current Evidence
The archived complexity report identifies `_call_with_retry()` in `agent/llm/openrouter.py`; current code also configures `max_retries=10` on the `ChatOpenAI` path.

## Desired Outcome
Retry behavior is delegated to the official client configuration, with any local wrapper kept minimal and justified.

## Acceptance Criteria
- [ ] Custom sleep/backoff retry loop is removed or reduced to a documented minimum.
- [ ] Retry count remains configurable through the official client path.
- [ ] Existing OpenRouter model tests pass.
- [ ] Rate-limit behavior is covered by a focused unit test or documented as delegated to upstream.

## Notes
Check whether `OpenRouterLLM` is still used by evaluation or legacy code before changing its public constructor.
