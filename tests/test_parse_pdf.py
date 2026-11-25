import pytest

from citerag.parsing import parse


def make_pdf(pages: list[str | None]) -> bytes:
    """Build a tiny valid PDF; a None page has no text (like a scanned page)."""
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for text in pages:
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET" if text else ""
        objs.append(f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
        objs.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {len(objs)} 0 R >>"
        )
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"

    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{off:010d} 00000 n \n" for off in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_pdf_pages_keep_page_numbers_and_skip_empty():
    passages = parse("doc.pdf", make_pdf(["Hello from page one.", None, "Page three answer."]))
    assert [(p.page, p.text) for p in passages] == [
        (1, "Hello from page one."),
        (3, "Page three answer."),
    ]


def test_pdf_with_no_text_returns_nothing():
    assert parse("scan.PDF", make_pdf([None, None])) == []


def test_invalid_pdf_raises_value_error():
    with pytest.raises(ValueError, match="Invalid PDF"):
        parse("broken.pdf", b"not a pdf")
