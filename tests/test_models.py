from citerag.models import Chunk, Document, DocumentRegistry


def test_ids_are_unique():
    assert Document(filename="a.txt").id != Document(filename="a.txt").id


def test_chunk_fields():
    doc = Document(filename="a.pdf")
    chunk = Chunk(document_id=doc.id, filename=doc.filename, text="hi", page=3)
    assert chunk.page == 3
    assert chunk.metadata == {}


def test_registry_add_get_list():
    reg = DocumentRegistry()
    a = reg.add(Document(filename="a.txt"))
    b = reg.add(Document(filename="b.md", metadata={"lang": "en"}))
    assert reg.get(a.id) == a
    assert reg.get("missing") is None
    assert [d.filename for d in reg.list()] == ["a.txt", "b.md"]
    assert reg.get(b.id).metadata == {"lang": "en"}
