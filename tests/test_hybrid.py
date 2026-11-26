from fastapi.testclient import TestClient

from citerag.config import settings
from citerag.main import app, get_cached_embedder
from citerag.models import Chunk
from citerag.store import rrf

client = TestClient(app)


def test_rrf_sums_weighted_reciprocal_ranks():
    a, b, c = ({"id": x, "text": x, "score": 9.0, "metadata": {}} for x in "abc")
    fused = rrf([[a, b], [b, c]], [1.0, 2.0], c=0)
    # a: 1/1; b: 1/2 + 2/1; c: 2/2
    assert [(h["id"], h["score"]) for h in fused] == [("b", 2.5), ("a", 1.0), ("c", 1.0)]


def test_rrf_zero_weight_ignores_ranking():
    a, b = ({"id": x, "text": x, "score": 0.0, "metadata": {}} for x in "ab")
    assert [h["id"] for h in rrf([[a], [b]], [0.0, 1.0])][0] == "b"


VECTORS = {"apple pie": [1.0, 0.0], "banana bread": [0.0, 1.0], "zzz": [1.0, 0.0]}


class FakeEmbedder:
    def embed(self, texts):
        return [VECTORS[t] for t in texts]


def seed(store):
    # "zzz" sits on the apple vector but shares no words with anything.
    chunks = [Chunk(document_id="d", filename="f.md", text=t) for t in VECTORS]
    store.upsert(chunks, [VECTORS[t] for t in VECTORS])
    app.dependency_overrides[get_cached_embedder] = FakeEmbedder


def test_hybrid_combines_vector_and_keyword(store):
    seed(store)
    # Vector ties apple pie and zzz; BM25 matches only zzz, which breaks the tie.
    hits = client.post("/search", json={"query": "zzz", "k": 3, "mode": "hybrid"}).json()
    assert hits[0]["text"] == "zzz"  # top in both rankings
    assert {h["text"] for h in hits} == set(VECTORS)
    assert hits[0]["score"] > hits[1]["score"] > 0


def test_hybrid_respects_weights(store, monkeypatch):
    seed(store)
    monkeypatch.setattr(settings, "hybrid_vector_weight", 0.0)
    hits = client.post("/search", json={"query": "zzz", "k": 1, "mode": "hybrid"}).json()
    assert [h["text"] for h in hits] == ["zzz"]
    monkeypatch.setattr(settings, "hybrid_vector_weight", 1.0)
    monkeypatch.setattr(settings, "hybrid_keyword_weight", 0.0)
    # Only vector counts now; "banana bread" is last in vector order.
    hits = client.post("/search", json={"query": "zzz", "k": 3, "mode": "hybrid"}).json()
    assert hits[-1]["text"] == "banana bread"


def test_hybrid_empty_collection(store):
    app.dependency_overrides[get_cached_embedder] = FakeEmbedder
    r = client.post("/search", json={"query": "zzz", "mode": "hybrid", "collection": "nope"})
    assert r.json() == []
