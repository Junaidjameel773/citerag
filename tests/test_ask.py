import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from citerag.answer import REFUSAL, build_messages, is_refusal, validate_citations
from citerag.llm import OpenAILLM
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

    def stream(self, messages):
        self.calls.append(messages)
        # Split into word-ish tokens like a real model would.
        yield from (w + " " for w in self.answer.split(" ")[:-1])
        yield self.answer.split(" ")[-1]


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
    llm = use_llm("Cats eat fish [2]. Dogs bark [1].")
    r = client.post("/ask", json={"question": "What do cats eat?", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Cats eat fish [2]. Dogs bark [1]."
    assert body["refused"] is False and body["uncited"] is False
    assert [(c["n"], c["chunk_id"]) for c in body["citations"]] == [(1, cats.id), (2, dogs.id)]
    first = body["citations"][0]
    assert (first["filename"], first["page"], first["text"]) == ("cats.pdf", 2, "cats")
    # Sources are passed to the LLM in the same order as the citation numbers.
    assert "[1] (cats.pdf, page 2)\ncats" in llm.calls[0][1]["content"]


def test_ask_returns_only_cited_sources_and_drops_invalid_markers(store):
    cats, _ = seed(store)
    use_llm("Cats eat fish [1][7]. They nap [9].")
    body = client.post("/ask", json={"question": "What do cats eat?", "k": 2}).json()
    assert body["answer"] == "Cats eat fish [1]. They nap."
    assert [(c["n"], c["chunk_id"]) for c in body["citations"]] == [(1, cats.id)]


def test_ask_flags_answer_without_citations(store):
    seed(store)
    use_llm("Cats eat fish [5].")
    body = client.post("/ask", json={"question": "What do cats eat?"}).json()
    assert body == {"answer": "Cats eat fish.", "refused": False, "uncited": True, "citations": []}


REFUSED = {"answer": REFUSAL, "refused": True, "uncited": False, "citations": []}


def test_ask_refusal_from_llm_has_no_citations(store):
    seed(store)
    use_llm(REFUSAL)
    assert client.post("/ask", json={"question": "What do cats eat?"}).json() == REFUSED


def test_ask_empty_store_refuses_without_calling_llm():
    llm = use_llm("should not be used")
    assert client.post("/ask", json={"question": "What do cats eat?"}).json() == REFUSED
    assert llm.calls == []


@pytest.mark.parametrize(
    ("answer", "expected", "cited"),
    [
        ("A [1]. B [2][3].", "A [1]. B [2][3].", [1, 2, 3]),
        ("A [1, 3].", "A [1][3].", [1, 3]),
        ("A [0][4] [3].", "A [3].", [3]),
        ("A [2]. B [2].", "A [2]. B [2].", [2]),
        ("No markers.", "No markers.", []),
        ("Index [x] and list [] stay.", "Index [x] and list [] stay.", []),
    ],
)
def test_validate_citations(answer, expected, cited):
    assert validate_citations(answer, 3) == (expected, cited)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        event, data = block.split("\n")
        events.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_ask_stream_sends_tokens_then_done_with_citations(store):
    cats, _ = seed(store)
    use_llm("Cats eat fish [1][9].")
    r = client.post("/ask?stream=true", json={"question": "What do cats eat?", "k": 2})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    tokens = [data["text"] for name, data in events if name == "token"]
    assert "".join(tokens) == "Cats eat fish [1][9]."  # raw model output
    assert len(tokens) == 4
    name, done = events[-1]
    assert name == "done" and [n for n, _ in events].count("done") == 1
    assert done["answer"] == "Cats eat fish [1]."  # cleaned in the final event
    assert done["refused"] is False and done["uncited"] is False
    assert [(c["n"], c["chunk_id"]) for c in done["citations"]] == [(1, cats.id)]


def test_ask_stream_refusal(store):
    seed(store)
    use_llm(REFUSAL)
    events = parse_sse(client.post("/ask?stream=true", json={"question": "cats"}).text)
    assert events[-1] == ("done", REFUSED)


def test_ask_stream_empty_store_only_sends_done():
    llm = use_llm("unused")
    events = parse_sse(client.post("/ask?stream=true", json={"question": "cats"}).text)
    assert events == [("done", REFUSED)]
    assert llm.calls == []


def test_openai_stream_skips_empty_chunks():
    def chunk(content, choices=True):
        delta = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)] if choices else [])

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return iter([chunk("Hel"), chunk(None), chunk("lo"), chunk(None, choices=False)])

    client_ = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = OpenAILLM(model="m", client=client_)
    assert list(llm.stream([{"role": "user", "content": "hi"}])) == ["Hel", "lo"]
    assert calls[0]["stream"] is True


def test_ask_validation():
    use_llm("x")
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "  "}).status_code == 422
    assert client.post("/ask", json={"question": "x", "k": 0}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
