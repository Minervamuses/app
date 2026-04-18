"""ChromaDB where-clause builder."""


def date_to_int(date_str: str) -> int:
    """Convert YYYY-MM-DD to a YYYYMMDD integer."""
    return int(date_str.replace("-", ""))


def build_where(
    category: str | None = None,
    file_type: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    folder_prefix: str | None = None,
) -> dict | None:
    """Build a ChromaDB where clause from filter arguments."""
    conditions: list[dict] = []
    if folder_prefix:
        prefix = folder_prefix.rstrip("/")
        conditions.append({"folder": {"$contains": prefix}})
    if category:
        conditions.append({"category": {"$eq": category}})
    if file_type:
        conditions.append({"file_type": {"$eq": file_type}})
    if date_from:
        conditions.append({"date": {"$gte": date_to_int(date_from)}})
    if date_to:
        conditions.append({"date": {"$lte": date_to_int(date_to)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
