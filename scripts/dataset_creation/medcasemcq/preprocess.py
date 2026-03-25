def preprocess(row):
    """Convert numeric id to string for MCQSample schema."""
    if isinstance(row.get("id"), (int, float)):
        row["id"] = str(int(row["id"]))
    return row
