import re

from citerag.llm import Message

# [1], [2][3] and the common model variant [1, 2]; leading whitespace is captured so a dropped
# marker doesn't leave a stray space before punctuation.
MARKER = re.compile(r"(\s*)\[(\d+(?:\s*,\s*\d+)*)\]")

REFUSAL = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You answer questions using only the numbered sources below.
Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- Cite every sentence with the number of the source(s) it comes from, like [1] or [2][3].
- If the sources do not contain the answer, reply exactly: {REFUSAL}"""


def build_messages(question: str, sources: list[dict]) -> list[Message]:
    """Prompt with sources numbered from 1; each source has text, filename and optional page."""
    blocks = []
    for n, s in enumerate(sources, 1):
        where = s["filename"] + (f", page {s['page']}" if s.get("page") is not None else "")
        blocks.append(f"[{n}] ({where})\n{s['text']}")
    context = "\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{context}\n\nQuestion: {question}"},
    ]


def is_refusal(answer: str) -> bool:
    # Models sometimes drop the period or change case, so match loosely.
    return REFUSAL.rstrip(".").lower() in answer.lower()


def validate_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    """Drop [n] markers that point at no source; return the cleaned answer and the
    distinct source numbers it cites, in ascending order."""
    cited: set[int] = set()

    def keep_valid(m: re.Match) -> str:
        valid = [n for n in (int(x) for x in m[2].split(",")) if 1 <= n <= n_sources]
        cited.update(valid)
        return m[1] + "".join(f"[{n}]" for n in valid) if valid else ""

    return MARKER.sub(keep_valid, answer), sorted(cited)
