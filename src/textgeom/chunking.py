from pathlib import Path


def load_text(path: str | Path) -> str:
    """Read a text file, stripping Project Gutenberg header/footer if present."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    start = text.find("*** START OF")
    if start != -1:
        text = text[text.find("\n", start) + 1 :]
    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text.strip()


def chunk_tokens(token_ids: list[int], size: int = 256, min_size: int = 128) -> list[list[int]]:
    """Split a token id sequence into non-overlapping chunks; drop a short tail."""
    chunks = [token_ids[i : i + size] for i in range(0, len(token_ids), size)]
    return [c for c in chunks if len(c) >= min_size]
