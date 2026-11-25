import re
from pathlib import PurePath

from pydantic import BaseModel

HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")


class Passage(BaseModel):
    """A block of source text plus where it came from, for citations."""

    text: str
    line: int | None = None
    section: str | None = None
    page: int | None = None


def parse_text(text: str, markdown: bool = False) -> list[Passage]:
    """Split text into blank-line separated paragraphs with their start line.

    For Markdown, headings start a new section and are not emitted as passages;
    fenced code blocks are kept whole.
    """
    passages: list[Passage] = []
    buf: list[str] = []
    start = 0
    section: str | None = None
    in_fence = False

    def flush() -> None:
        if buf:
            passages.append(Passage(text="\n".join(buf), line=start, section=section))
            buf.clear()

    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if markdown and line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        elif markdown and not in_fence and (m := HEADING.match(line)):
            flush()
            section = m.group(1)
            continue
        if not line and not in_fence:
            flush()
            continue
        if not buf:
            start = n
        buf.append(line)
    flush()
    return passages


def parse(filename: str, data: bytes) -> list[Passage]:
    """Parse an uploaded file by extension."""
    ext = PurePath(filename).suffix.lower()
    if ext in (".txt", ".md", ".markdown"):
        return parse_text(data.decode("utf-8-sig"), markdown=ext != ".txt")
    raise ValueError(f"Unsupported file type: {ext or filename}")
