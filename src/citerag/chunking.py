import re
from itertools import groupby

from citerag.models import Chunk, Document
from citerag.parsing import Passage

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")


def _tokens(text: str) -> int:
    # ponytail: whitespace words approximate tokens; swap in a real tokenizer if limits get tight
    return len(text.split())


def _units(text: str, size: int) -> list[str]:
    """Split text into sentences; sentences longer than `size` are cut into word windows."""
    units: list[str] = []
    for sentence in SENTENCE_END.split(text.strip()):
        words = sentence.split()
        if len(words) <= size:
            units.append(sentence)
        else:
            units += [" ".join(words[i : i + size]) for i in range(0, len(words), size)]
    return [u for u in units if u.strip()]


def chunk_passages(
    doc: Document, passages: list[Passage], size: int = 500, overlap: int = 50
) -> list[Chunk]:
    """Pack passages into ~`size` token chunks, repeating up to `overlap` tokens between them.

    Chunks break on sentence boundaries where possible and never span two pages or
    sections, so every chunk keeps the page, section and start line it came from.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    chunks: list[Chunk] = []
    for (page, section), group in groupby(passages, key=lambda p: (p.page, p.section)):
        # (text, start line, separator before it)
        units = [
            (u, p.line, " " if i else "\n\n")
            for p in group
            for i, u in enumerate(_units(p.text, size))
        ]
        start = 0
        while start < len(units):
            end, count = start, 0
            while end < len(units) and (end == start or count + _tokens(units[end][0]) <= size):
                count += _tokens(units[end][0])
                end += 1
            window = units[start:end]
            text = window[0][0] + "".join(sep + u for u, _, sep in window[1:])
            metadata = {k: v for k, v in (("section", section), ("line", window[0][1])) if v}
            chunks.append(
                Chunk(
                    document_id=doc.id,
                    filename=doc.filename,
                    text=text,
                    page=page,
                    metadata=metadata,
                )
            )
            if end == len(units):
                break
            # step back over trailing units that fit in the overlap budget
            back, kept = end, 0
            while back - 1 > start and kept + _tokens(units[back - 1][0]) <= overlap:
                back -= 1
                kept += _tokens(units[back][0])
            start = back
    return chunks
