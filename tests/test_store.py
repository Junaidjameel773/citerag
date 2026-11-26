import pytest

from citerag.models import Chunk
from citerag.store import ChunkStore


def make_chunk(text: str, **kw) -> Chunk:
    return Chunk(document_id="d1", filename="a.pdf", text=text, **kw)


def test_upsert_stores_text_embedding_and_metadata(store):
    chunk = make_chunk("hello", page=2, metadata={"section": "Intro", "line": 3})
    store.upsert([chunk, make_chunk("no page")], [[1.0, 0.0], [0.0, 1.0]])
    got = store.collection.get(ids=[chunk.id], include=["documents", "metadatas", "embeddings"])
    assert got["documents"] == ["hello"]
    assert got["metadatas"][0] == {
        "document_id": "d1",
        "filename": "a.pdf",
        "page": 2,
        "section": "Intro",
        "line": 3,
    }
    assert list(got["embeddings"][0]) == [1.0, 0.0]
    assert store.collection.count() == 2


def test_upsert_by_id_replaces(store):
    chunk = make_chunk("v1")
    store.upsert([chunk], [[1.0, 0.0]])
    store.upsert([chunk.model_copy(update={"text": "v2"})], [[0.0, 1.0]])
    assert store.collection.count() == 1
    assert store.collection.get(ids=[chunk.id])["documents"] == ["v2"]


def test_persists_across_clients(tmp_path):
    path = str(tmp_path / "db")
    ChunkStore(path).upsert([make_chunk("kept")], [[1.0, 0.0]])
    assert ChunkStore(path).collection.get()["documents"] == ["kept"]


def test_length_mismatch(store):
    with pytest.raises(ValueError):
        store.upsert([make_chunk("x")], [])
