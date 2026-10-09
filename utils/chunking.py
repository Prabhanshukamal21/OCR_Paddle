def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
):
    """
    Split text into overlapping chunks.

    Works with both English and Hindi Unicode text.
    """

    if not text or not text.strip():
        return []

    text = text.strip()

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks