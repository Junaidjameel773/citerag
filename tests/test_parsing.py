from pathlib import Path

import pytest

from citerag.parsing import parse

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    return parse(name, (FIXTURES / name).read_bytes())


def test_plain_text_paragraphs_with_lines():
    passages = load("sample.txt")
    assert [(p.text, p.line, p.section) for p in passages] == [
        ("First paragraph line one.\nline two.", 1, None),
        ("Second paragraph.", 5, None),
    ]


def test_markdown_sections():
    passages = load("sample.md")
    assert [(p.line, p.section) for p in passages] == [
        (1, None),
        (5, "Install"),
        (10, "Options"),
        (12, "Options"),
    ]
    assert passages[1].text == "Run the installer.\nThen restart."


def test_markdown_code_fence_kept_whole():
    code = load("sample.md")[-1].text
    assert code.startswith("```bash\n# not a heading\n\necho hi")
    assert code.endswith("```")


def test_txt_does_not_treat_hash_as_heading():
    assert parse("a.txt", b"# hello\n")[0].text == "# hello"


def test_utf8_bom_stripped():
    assert parse("a.md", "﻿hi".encode())[0].text == "hi"


def test_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported"):
        parse("a.docx", b"")
