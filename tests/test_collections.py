from fastapi.testclient import TestClient

from citerag.main import app, get_cached_llm

client = TestClient(app)


def upload(text: str, collection: str | None = None):
    params = {"collection": collection} if collection else {}
    files = {"file": ("notes.txt", text.encode(), "text/plain")}
    r = client.post("/documents", files=files, params=params)
    assert r.status_code == 201, r.text
    return r.json()


def test_collections_are_isolated_for_listing_and_search():
    default_doc = upload("default stuff")
    hr_doc = upload("hr policy", "hr")
    eng_doc = upload("eng runbook", "eng-team")

    assert [d["id"] for d in client.get("/documents").json()] == [default_doc["id"]]
    listed = client.get("/documents", params={"collection": "hr"}).json()
    assert [d["id"] for d in listed] == [hr_doc["id"]]

    hits = client.post("/search", json={"query": "q", "k": 10, "collection": "eng-team"}).json()
    assert {h["document_id"] for h in hits} == {eng_doc["id"]}
    hits = client.post("/search", json={"query": "q", "k": 10}).json()
    assert {h["document_id"] for h in hits} == {default_doc["id"]}


def test_unknown_collection_is_empty_and_not_created(store):
    assert client.get("/documents", params={"collection": "nope"}).json() == []
    assert client.post("/search", json={"query": "q", "collection": "nope"}).json() == []
    assert client.delete("/documents/x", params={"collection": "nope"}).status_code == 404
    assert [c.name for c in store.client.list_collections()] == ["chunks"]  # only the default


def test_edge_names_accepted_and_chunks_is_not_the_default():
    upload("default stuff")
    for name in ["a", "x" * 61, "v1.2_docs", "chunks"]:
        doc = upload(f"in {name}", name)
        assert client.get("/documents", params={"collection": name}).json() == [doc]
    assert len(client.get("/documents").json()) == 1


def test_delete_only_affects_its_collection():
    doc = upload("hr policy", "hr")
    assert client.delete(f"/documents/{doc['id']}").status_code == 404  # not in default
    assert client.delete(f"/documents/{doc['id']}", params={"collection": "hr"}).status_code == 204
    assert client.get("/documents", params={"collection": "hr"}).json() == []


def test_ask_uses_collection():
    class EchoLLM:
        def __init__(self):
            self.prompts = []

        def complete(self, messages):
            self.prompts.append(messages[-1]["content"])
            return "From HR [1]."

    llm = EchoLLM()
    app.dependency_overrides[get_cached_llm] = lambda: llm
    upload("default stuff")
    hr_doc = upload("hr policy", "hr")
    body = client.post("/ask", json={"question": "q", "collection": "hr"}).json()
    assert [c["document_id"] for c in body["citations"]] == [hr_doc["id"]]
    assert "hr policy" in llm.prompts[0] and "default stuff" not in llm.prompts[0]


def test_invalid_collection_names_rejected():
    for bad in ["-hr", "hr-", ".", "has space", "x" * 62, "a/b"]:
        assert client.get("/documents", params={"collection": bad}).status_code == 422, bad
        r = client.post("/search", json={"query": "q", "collection": bad})
        assert r.status_code == 422, bad
    files = {"file": ("n.txt", b"x", "text/plain")}
    assert client.post("/documents?collection=a!", files=files).status_code == 422
