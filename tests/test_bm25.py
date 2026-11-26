from fastapi.testclient import TestClient

from citerag.bm25 import bm25_scores, tokenize
from citerag.main import app, get_cached_embedder
from citerag.models import Chunk

client = TestClient(app)


class NoEmbedder:
    def embed(self, texts):
        raise AssertionError("keyword search must not embed")


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Cats, DOGS & café_2!") == ["cats", "dogs", "café_2"]


def test_scores_rank_matching_docs_and_zero_otherwise():
    docs = ["cats eat fish", "dogs bark", "cats nap all day long"]
    scores = bm25_scores("Cats fish", docs)
    assert scores[1] == 0.0
    assert scores[0] > scores[2] > 0  # two matching terms beat one


def test_term_in_half_of_tiny_corpus_still_scores():
    # Plain Okapi IDF is log(1) = 0 here; the Lucene variant keeps it positive.
    assert bm25_scores("cats", ["cats", "dogs"])[0] > 0


def test_shorter_doc_wins_for_same_term_frequency():
    short, long = bm25_scores("rare", ["rare word", "rare word plus many other filler words"])
    assert short > long


def test_rarer_term_weighs_more():
    docs = ["common rare", "common", "common", "common"]
    assert bm25_scores("rare", docs)[0] > bm25_scores("common", docs)[0]


def test_empty_inputs():
    assert bm25_scores("x", []) == []
    assert bm25_scores("", ["some text"]) == [0.0]


def seed(store, collection=None):
    chunks = [
        Chunk(document_id="d1", filename="a.md", text="Invoice INV-2041 is overdue"),
        Chunk(document_id="d2", filename="b.md", text="Shipping policy for parcels", page=4),
        Chunk(document_id="d3", filename="c.md", text="Overdue invoices get a reminder"),
    ]
    store.upsert(chunks, [[1.0, 0.0]] * 3, collection)
    return chunks


def test_store_keyword_search_orders_and_filters(store):
    inv, _, reminder = seed(store)
    hits = store.keyword_search("overdue invoice INV-2041", k=5)
    assert [h["id"] for h in hits] == [inv.id, reminder.id]
    assert hits[0]["score"] > hits[1]["score"] > 0
    assert store.keyword_search("overdue", k=1)[0]["id"] in {inv.id, reminder.id}
    assert store.keyword_search("nothing matches", k=5) == []
    assert store.keyword_search("overdue", collection="missing") == []


def test_search_endpoint_keyword_mode(store):
    seed(store, "ops")
    app.dependency_overrides[get_cached_embedder] = NoEmbedder
    r = client.post("/search", json={"query": "parcels", "mode": "keyword", "collection": "ops"})
    assert r.status_code == 200
    [hit] = r.json()
    assert (hit["document_id"], hit["filename"], hit["page"]) == ("d2", "b.md", 4)
    assert hit["score"] > 0


def test_search_mode_validation():
    assert client.post("/search", json={"query": "x", "mode": "fuzzy"}).status_code == 422
