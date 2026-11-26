from fastapi.testclient import TestClient
from test_parse_pdf import make_pdf

from citerag.config import settings
from citerag.main import app, registry

client = TestClient(app)


def upload(name: str, data: bytes, content_type: str = "application/octet-stream"):
    return client.post("/documents", files={"file": (name, data, content_type)})


def test_upload_markdown():
    r = upload("notes.md", b"# Title\n\nHello world.\n", "text/markdown")
    assert r.status_code == 201
    body = r.json()
    assert body["filename"] == "notes.md" and body["chunks"] == 1
    assert registry.get(body["id"]).metadata == {"chunk_count": 1}


def test_upload_pdf(store):
    r = upload("a.pdf", make_pdf(["Page one", "Page two"]), "application/pdf")
    assert r.status_code == 201
    assert r.json()["chunks"] == 2
    stored = store.collection.get(where={"document_id": r.json()["id"]})
    assert sorted(m["page"] for m in stored["metadatas"]) == [1, 2]


def test_unsupported_extension():
    assert upload("a.docx", b"x").status_code == 415


def test_content_type_mismatch():
    assert upload("a.pdf", b"x", "image/png").status_code == 415


def test_too_large(monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    assert upload("a.txt", b"a" * (1024 * 1024 + 1), "text/plain").status_code == 413


def test_invalid_pdf():
    assert upload("a.pdf", b"not a pdf", "application/pdf").status_code == 400


def test_invalid_utf8():
    assert upload("a.txt", b"\xff\xfe\xfa", "text/plain").status_code == 400


def test_empty_file():
    assert upload("a.txt", b"\n\n", "text/plain").status_code == 422
