"""Isolated citation-capture prototype.

This package is a *standalone experiment*. It reuses the host project's
configuration, chat model, and Web Search MCP loader (read-only imports from
``agent.*``) but is intentionally NOT wired into the LangGraph agent, the skill
system, the capability map, or any slash command.

Flow:
    topic -> Google Scholar-oriented discovery (Web Search MCP)
          -> user picks a candidate
          -> DOI / Crossref metadata route for BibTeX (with title/author/year
             verification)
          -> write to ``citation/cite/<normalized_title>.bib``

Nothing here fabricates BibTeX: if no DOI/Crossref route returns real BibTeX,
the run reports the failure and writes nothing.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
