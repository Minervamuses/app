"""Text helpers for LangChain chat models."""

from langchain_core.messages import HumanMessage


def invoke_text(model, prompt: str) -> str:
    """Invoke a LangChain chat model with one user prompt and return text."""
    response = model.invoke([HumanMessage(content=prompt)])
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(_content_part_to_text(part) for part in content).strip()
    return str(content or "").strip()


def _content_part_to_text(part) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        return str(part.get("text") or part.get("content") or "")
    return str(part)
