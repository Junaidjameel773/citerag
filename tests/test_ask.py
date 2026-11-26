from fastapi.testclient import TestClient

from citerag.answer import REFUSAL, build_messages, is_refusal
from citerag.main import app, get_cached_embedder, get_cached_llm
from citerag.models import Chunk

client = TestClient(app)

VECTORS = {"cats": [1.0, 0.0], "dogs": [0.0, 1.0], "What do cats eat?": [0.9, 0.1]}


class KeywordEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [VECTORS[t] for t in texts]


class FakeLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.answer


def use_llm(answer: str) -> FakeLLM:
    llm = FakeLLM(answer)
    app.dependency_overrides[get_cached_llm] = lambda: llm
    app.dependency_overrides[get_cached_embedder] = KeywordEmbedder
    return llm


def seed(store):
    chunks = [
        Chunk(document_id="d1", filename="cats.pdf", text="cats", page=2),
        Chunk(document_id="d2", filename="dogs.md", text="dogs"),
    ]
    store.upsert(chunks, [VECTORS["cats"], VECTORS["dogs"]])
    return chunks


def test_build_messages_numbers_sources_with_file_and_page():
    sources = [
        {"text": "Cats eat fish.", "filename": "cats.pdf", "page": 2},
        {"text": "Dogs bark.", "filename": "dogs.md", "page": None},
    ]
    system, user = build_messages("What do cats eat?", sources)
    assert system["role"] == "system" and "[1]" in system["content"]
    assert REFUSAL in system["content"]
    assert "[1] (cats.pdf, page 2)\nCats eat fish." in user["content"]
    assert "[2] (dogs.md)\nDogs bark." in user["content"]
    assert user["content"].endswith("Question: What do cats eat?")


def test_is_refusal_is_lenient():
    assert is_refusal(REFUSAL)
    assert is_refusal("i don't know based on the provided documents")
    assert not is_refusal("Cats eat fish [1].")


def test_ask_returns_answer_with_numbered_citations(store):
    cats, dogs = seed(store)
    llm = use_llm("Cats eat fish [1].")
    r = client.post("/ask", json={"question": "What do cats eat?", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Cats eat fish [1]."
    assert body["refused"] is False
    assert [(c["n"], c["chunk_id"]) for c in body["citations"]] == [(1, cats.id), (2, dogs.id)]
    first = body["citations"][0]
    assert (first["filename"], first["page"], first["text"]) == ("cats.pdf", 2, "cats")
    # Sources are passed to the LLM in the same order as the citation numbers.
    assert "[1] (cats.pdf, page 2)\ncats" in llm.calls[0][1]["content"]


def test_ask_refusal_from_llm_has_no_citations(store):
    seed(store)
    use_llm(REFUSAL)
    body = client.post("/ask", json={"question": "What do cats eat?"}).json()
    assert body == {"answer": REFUSAL, "refused": True, "citations": []}


def test_ask_empty_store_refuses_without_calling_llm():
    llm = use_llm("should not be used")
    body = client.post("/ask", json={"question": "What do cats eat?"}).json()
    assert body == {"answer": REFUSAL, "refused": True, "citations": []}
    assert llm.calls == []


def test_ask_validation():
    use_llm("x")
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "  "}).status_code == 422
    assert client.post("/ask", json={"question": "x", "k": 0}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
