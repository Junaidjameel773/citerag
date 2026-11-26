from fastapi.testclient import TestClient

from citerag.main import app, get_cached_embedder
from citerag.models import Chunk

client = TestClient(app)

VECTORS = {"cats": [1.0, 0.0], "dogs": [0.0, 1.0], "pets": [0.6, 0.8]}


class KeywordEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [VECTORS[t] for t in texts]


def seed(store):
    chunks = [
        Chunk(document_id="d1", filename="a.pdf", text="cats", page=3, metadata={"line": 7}),
        Chunk(document_id="d2", filename="b.md", text="dogs", metadata={"section": "Dogs"}),
    ]
    store.upsert(chunks, [VECTORS["cats"], VECTORS["dogs"]])
    app.dependency_overrides[get_cached_embedder] = KeywordEmbedder
    return chunks


def test_store_search_orders_by_score(store):
    cats, dogs = seed(store)
    hits = store.search(VECTORS["cats"], k=2)
    assert [h["id"] for h in hits] == [cats.id, dogs.id]
    assert abs(hits[0]["score"] - 1.0) < 1e-6
    assert abs(hits[1]["score"]) < 1e-6


def test_search_empty_store(store):
    assert store.search([1.0, 0.0]) == []
    app.dependency_overrides[get_cached_embedder] = KeywordEmbedder
    assert client.post("/search", json={"query": "cats"}).json() == []


def test_search_endpoint_returns_hits_with_metadata(store):
    cats, dogs = seed(store)
    r = client.post("/search", json={"query": "dogs", "k": 1})
    assert r.status_code == 200
    [hit] = r.json()
    assert hit["chunk_id"] == dogs.id
    assert hit == {
        "chunk_id": dogs.id,
        "document_id": "d2",
        "filename": "b.md",
        "page": None,
        "text": "dogs",
        "score": hit["score"],
        "metadata": {"section": "Dogs"},
    }
    assert abs(hit["score"] - 1.0) < 1e-6


def test_search_k_larger_than_collection(store):
    seed(store)
    hits = client.post("/search", json={"query": "pets", "k": 10}).json()
    assert [h["text"] for h in hits] == ["dogs", "cats"]
    assert hits[0]["score"] > hits[1]["score"]
    assert hits[1]["page"] == 3 and hits[1]["metadata"] == {"line": 7}


def test_search_validation():
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "   "}).status_code == 422
    assert client.post("/search", json={"query": "x", "k": 0}).status_code == 422
    assert client.post("/search", json={"query": "x", "k": 51}).status_code == 422
