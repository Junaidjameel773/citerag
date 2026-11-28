from fastapi.testclient import TestClient

from citerag.answer import build_rewrite_messages
from citerag.main import app, get_cached_embedder, get_cached_llm
from citerag.models import Chunk

client = TestClient(app)

VECTORS = {"cats": [1.0, 0.0], "dogs": [0.0, 1.0], "What do dogs eat?": [0.1, 0.9]}

HISTORY = [
    {"role": "user", "content": "What do cats eat?"},
    {"role": "assistant", "content": "Cats eat fish [1]."},
]


class KeywordEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [VECTORS.get(t, [0.0, 1.0]) for t in texts]


class ScriptedLLM:
    """Returns the given replies in order: first the rewritten query, then the answer."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)

    def stream(self, messages):
        yield self.complete(messages)


def setup(store, *replies: str) -> tuple[ScriptedLLM, KeywordEmbedder]:
    store.upsert(
        [
            Chunk(document_id="d1", filename="cats.md", text="cats"),
            Chunk(document_id="d2", filename="dogs.md", text="dogs"),
        ],
        [VECTORS["cats"], VECTORS["dogs"]],
    )
    llm, embedder = ScriptedLLM(*replies), KeywordEmbedder()
    app.dependency_overrides[get_cached_llm] = lambda: llm
    app.dependency_overrides[get_cached_embedder] = lambda: embedder
    return llm, embedder


def test_build_rewrite_messages_includes_history_and_question():
    system, user = build_rewrite_messages("And dogs?", HISTORY)
    assert system["role"] == "system" and "standalone" in system["content"]
    assert "user: What do cats eat?\nassistant: Cats eat fish [1]." in user["content"]
    assert user["content"].endswith("Latest question: And dogs?")


def test_follow_up_is_rewritten_before_retrieval(store):
    llm, embedder = setup(store, " What do dogs eat?\n", "Dogs eat meat [1].")
    r = client.post("/ask", json={"question": "And dogs?", "k": 1, "history": HISTORY})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Dogs eat meat [1]."
    assert body["citations"][0]["filename"] == "dogs.md"
    assert embedder.queries == ["What do dogs eat?"]
    assert "Latest question: And dogs?" in llm.calls[0][1]["content"]
    assert llm.calls[1][1]["content"].endswith("Question: What do dogs eat?")


def test_empty_rewrite_falls_back_to_original_question(store):
    _, embedder = setup(store, "  ", "Dogs [1].")
    r = client.post("/ask", json={"question": "And dogs?", "k": 1, "history": HISTORY})
    assert r.status_code == 200
    assert embedder.queries == ["And dogs?"]


def test_follow_up_rewritten_when_streaming(store):
    _, embedder = setup(store, "What do dogs eat?", "Dogs eat meat [1].")
    r = client.post("/ask?stream=true", json={"question": "And dogs?", "history": HISTORY})
    assert r.status_code == 200
    assert embedder.queries == ["What do dogs eat?"]


def test_no_history_skips_rewrite(store):
    llm, embedder = setup(store, "Dogs eat meat [1].")
    r = client.post("/ask", json={"question": "What do dogs eat?", "k": 1})
    assert r.json()["answer"] == "Dogs eat meat [1]."
    assert len(llm.calls) == 1 and embedder.queries == ["What do dogs eat?"]


def test_history_validation(store):
    llm, _ = setup(store)
    bad_role = [{"role": "system", "content": "x"}]
    assert client.post("/ask", json={"question": "x", "history": bad_role}).status_code == 422
    too_long = [{"role": "user", "content": "x"}] * 51
    assert client.post("/ask", json={"question": "x", "history": too_long}).status_code == 422
    # A blank question is rejected without spending an LLM call on the rewrite.
    assert client.post("/ask", json={"question": " ", "history": HISTORY}).status_code == 422
    assert llm.calls == []
