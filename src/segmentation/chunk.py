from typing import List, Dict

def chunk_text(text: str, max_tokens: int = 1024, overlap: int = 128) -> List[str]:
    """
    Simple whitespace token chunker with overlap. Keeps chunks near max_tokens
    (by words, not model tokens) to avoid heavy dependencies here.
    """
    toks = text.split()
    if not toks:
        return []
    out = []
    i = 0
    while i < len(toks):
        j = min(len(toks), i + max_tokens)
        out.append(" ".join(toks[i:j]))
        if j == len(toks):
            break
        i = max(0, j - overlap)
    return out

def build_thread_text(posts: List[Dict]) -> str:
    """
    Concatenate a list of post dicts (with 'timestamp' and 'text') into a single thread string.
    """
    lines = []
    for p in sorted(posts, key=lambda x: x.get("timestamp", 0)):
        ts = p.get("timestamp", "")
        lines.append(f"[{ts}]\n{p.get('text','').strip()}\n")
    return "\n".join(lines)
