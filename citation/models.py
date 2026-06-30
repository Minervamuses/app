"""Plain data containers shared across the citation prototype."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PaperCandidate:
    """A single discovery result the user may choose to cite.

    ``authors`` and ``year`` are best-effort: discovery sources rarely expose
    structured metadata, so empty/None means "unknown", never "absent".
    """

    title: str
    url: str | None = None
    doi: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    snippet: str = ""
    reason: str = ""

    def short_label(self) -> str:
        bits = [self.title or "(untitled)"]
        meta: list[str] = []
        if self.authors:
            head = self.authors[0]
            meta.append(head + (" et al." if len(self.authors) > 1 else ""))
        if self.year:
            meta.append(str(self.year))
        if meta:
            bits.append("(" + ", ".join(meta) + ")")
        return " ".join(bits)


@dataclass
class CrossrefMatch:
    """A Crossref ``works`` record scored against a :class:`PaperCandidate`."""

    doi: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    crossref_score: float = 0.0  # Crossref's own relevance score

    # Scores we compute locally to avoid blindly trusting Crossref ordering.
    title_similarity: float = 0.0
    year_matches: bool | None = None  # None = could not compare
    author_overlap: bool | None = None  # None = could not compare
    confidence: float = 0.0  # combined 0..1


@dataclass
class CaptureResult:
    """Outcome of a citation-capture attempt for one chosen paper."""

    ok: bool
    bibtex: str | None = None
    doi: str | None = None
    route: str = ""  # "crossref" | "scholar" | ""
    out_path: str | None = None
    # Ordered, human-readable trace of what was tried and why it failed.
    notes: list[str] = field(default_factory=list)
