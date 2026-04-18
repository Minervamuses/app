"""Path utilities for the KMS."""

import re
from pathlib import Path

_DATE_RE = re.compile(r"(\d{4})(\d{2})(\d{2})")


def extract_date(rel_path: str) -> int:
    """Extract date as YYYYMMDD integer from path if a date folder exists."""
    for part in Path(rel_path).parts:
        match = _DATE_RE.fullmatch(part)
        if match:
            return int(f"{match.group(1)}{match.group(2)}{match.group(3)}")
    return 0
