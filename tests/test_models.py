from citerag.models import Chunk, Document


def test_ids_are_unique():
    assert Document(filename="a.txt").id != Document(filename="a.txt").id


def test_chunk_fields():
    doc = Document(filename="a.pdf")
    chunk = Chunk(document_id=doc.id, filename=doc.filename, text="hi", page=3)
    assert chunk.page == 3
    assert chunk.metadata == {}
