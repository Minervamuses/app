"""Filename derivation and collision-safe writing for ``.bib`` files."""

from __future__ import annotations

import re
from pathlib import Path

# Characters that are unsafe across common filesystems; replaced with "_".
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Anything that is not a word char or hyphen becomes a separator.
_SEPARATORS = re.compile(r"[^\w-]+", re.UNICODE)
_MULTI_UNDERSCORE = re.compile(r"_+")
_MAX_FILENAME_STEM_CHARS = 160


def normalize_title_to_filename(title: str) -> str:
    """Turn a paper title into a safe ``.bib`` filename stem + extension.

    Rules (per spec):
      * lowercase
      * strip surrounding whitespace
      * whitespace -> underscore
      * remove/replace filesystem-unsafe characters
      * collapse consecutive underscores into one
      * ``.bib`` extension

    "Attention Is All You Need" -> "attention_is_all_you_need.bib"
    """
    text = (title or "").strip().lower()
    text = _UNSAFE.sub(" ", text)
    text = _SEPARATORS.sub("_", text)
    text = _MULTI_UNDERSCORE.sub("_", text).strip("_")
    if not text:
        text = "untitled"
    if len(text) > _MAX_FILENAME_STEM_CHARS:
        text = text[:_MAX_FILENAME_STEM_CHARS].rstrip("_") or "untitled"
    return f"{text}.bib"


def resolve_collision_free_path(cite_dir: Path, filename: str) -> Path:
    """Return a path under ``cite_dir`` that does not overwrite an existing file.

    "x.bib" -> "x.bib" if free, else "x_2.bib", "x_3.bib", ...
    """
    base = filename[:-4] if filename.endswith(".bib") else filename
    candidate = cite_dir / f"{base}.bib"
    n = 2
    while candidate.exists():
        candidate = cite_dir / f"{base}_{n}.bib"
        n += 1
    return candidate


def write_bibtex(cite_dir: Path, title: str, bibtex: str) -> Path:
    """Create ``cite_dir`` if needed and write ``bibtex`` without overwriting.

    Returns the path actually written.
    """
    cite_dir.mkdir(parents=True, exist_ok=True)
    filename = normalize_title_to_filename(title)
    out_path = resolve_collision_free_path(cite_dir, filename)
    payload = bibtex if bibtex.endswith("\n") else bibtex + "\n"
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def looks_like_bibtex(text: str | None) -> bool:
    """Cheap sanity check that ``text`` is a BibTeX entry, not an error page."""
    if not text:
        return False
    stripped = text.lstrip("﻿ \t\r\n")
    return stripped.startswith("@") and "{" in stripped and "}" in stripped


_BIBTEX_TITLE_START = re.compile(r"\btitle\s*=\s*", re.IGNORECASE)


def extract_bibtex_title(bibtex: str) -> str | None:
    """Return the ``title`` field of a BibTeX entry, if present.

    Used to name the output file from the *authoritative* title rather than a
    messy search-result label. Brace groups are flattened to plain text.
    """
    m = _BIBTEX_TITLE_START.search(bibtex or "")
    if not m:
        return None
    raw_title = _read_bibtex_field_value(bibtex, m.end())
    if raw_title is None:
        return None
    title = re.sub(r"[{}]", "", raw_title).strip().rstrip(",").strip()
    return title or None


def _read_bibtex_field_value(text: str, start: int) -> str | None:
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text):
        return None

    opener = text[i]
    if opener == "{":
        value, _end = _read_balanced_braces(text, i)
        return value
    if opener == '"':
        return _read_quoted_string(text, i)

    j = i
    while j < len(text) and text[j] not in ",\n\r":
        j += 1
    return text[i:j].strip() or None


def _read_balanced_braces(text: str, start: int) -> tuple[str | None, int]:
    depth = 0
    chars: list[str] = []
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip(), i + 1
            chars.append(ch)
        else:
            chars.append(ch)
        i += 1
    return None, i


def _read_quoted_string(text: str, start: int) -> str | None:
    chars: list[str] = []
    escaped = False
    for ch in text[start + 1:]:
        if escaped:
            chars.append(ch)
            escaped = False
            continue
        if ch == "\\":
            chars.append(ch)
            escaped = True
            continue
        if ch == '"':
            return "".join(chars).strip()
        chars.append(ch)
    return None
