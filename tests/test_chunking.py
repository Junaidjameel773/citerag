import pytest

from citerag.chunking import chunk_passages
from citerag.models import Document
from citerag.parsing import Passage, parse

DOC = Document(filename="a.md")


def sentences(n: int, words: int = 10) -> str:
    return " ".join(
        " ".join([f"s{i}w{j}" for j in range(words - 1)] + [f"end{i}."]) for i in range(n)
    )


def test_small_passages_merge_into_one_chunk():
    passages = parse("a.md", b"# Intro\n\nOne.\n\nTwo.\n")
    [chunk] = chunk_passages(DOC, passages)
    assert chunk.text == "One.\n\nTwo."
    assert chunk.metadata == {"section": "Intro", "line": 3}
    assert chunk.document_id == DOC.id and chunk.filename == "a.md"


def test_size_and_overlap_on_sentence_boundaries():
    chunks = chunk_passages(DOC, [Passage(text=sentences(10), line=1)], size=30, overlap=10)
    for c in chunks:
        assert len(c.text.split()) <= 30
        assert c.text.endswith(".")  # never cut mid-sentence
    # last sentence of each chunk starts the next one
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.text.split(". ")[0] + "." == a.text.split(". ")[-1]
    assert "end9." in chunks[-1].text
    assert len(chunks) == 5


def test_no_overlap():
    chunks = chunk_passages(DOC, [Passage(text=sentences(6))], size=20, overlap=0)
    assert len(chunks) == 3
    assert " ".join(c.text for c in chunks) == sentences(6)


def test_long_sentence_split_by_words():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_passages(DOC, [Passage(text=text)], size=10, overlap=0)
    assert [len(c.text.split()) for c in chunks] == [10, 10, 5]


def test_never_spans_pages():
    passages = [Passage(text="Page one.", page=1), Passage(text="Page two.", page=2)]
    chunks = chunk_passages(DOC, passages)
    assert [(c.text, c.page) for c in chunks] == [("Page one.", 1), ("Page two.", 2)]
    assert chunks[0].metadata == {}


def test_bad_overlap():
    with pytest.raises(ValueError):
        chunk_passages(DOC, [], size=10, overlap=10)
