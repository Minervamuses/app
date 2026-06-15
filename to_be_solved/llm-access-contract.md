# LLM access contract is explicit

status: implemented
source:
  - to_be_solved/archive/deep-research-report.md

## Decision
The agent standardizes on LangChain chat models as its only LLM access
contract. Core runtime already used `ChatOpenAI` factories, so evaluation was
migrated to the same contract instead of preserving the parallel `BaseLLM`
prompt-to-text provider hierarchy.

## Implementation Summary
- `agent/graph.py` continues to use `get_chat_model(config)` for the LangGraph
  runtime model.
- Extended thinking continues to use `get_chat_model_for_role(...)`.
- Legacy e2e evaluation now uses LangChain chat models:
  - OpenRouter generator/judge via `get_openrouter_chat_model(...)`
  - local Ollama filter via `get_ollama_chat_model(...)`
  - plain-text eval calls via `agent.llm.invoke_text(...)`
- The old `BaseLLM`, `OpenRouterLLM`, and `OllamaLLM` classes were removed.

## Reference Inventory
- Runtime OpenRouter factory: `agent/llm/openrouter.py`
- Local Ollama factory: `agent/llm/ollama.py`
- Prompt-to-text helper: `agent/llm/text.py`
- Runtime graph usage: `agent/graph.py`
- Extended thinking usage: `agent/session.py`, `agent/llm/thinking.py`
- Evaluation usage: `agent/evaluation/endtoend.py`,
  `agent/evaluation/claims/c3b_reviewer.py`, `agent/evaluation/thinking.py`

## Acceptance Criteria
- [x] All references to `BaseLLM`, `OpenRouterLLM`, and `OllamaLLM` are inventoried.
- [x] The intended model access boundary is documented.
- [x] Dead or legacy-only code is removed.
- [x] No runtime or evaluation path changes behavior unintentionally.
