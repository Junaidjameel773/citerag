import pytest
from fastapi.testclient import TestClient

from citerag import rerank
from citerag.main import app, get_cached_embedder, get_cached_llm, get_cached_reranker
from citerag.models import Chunk

client = TestClient(app)


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return "Answer [1]."


class LengthReranker:
    """Prefers longer passages, the opposite of the vector ranking set up below."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        return [float(len(t)) for t in texts]


class QueryEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def seed(store, n: int) -> list[Chunk]:
    # Chunk i is closest to the query for small i but has the shortest text.
    chunks = [Chunk(document_id="d", filename="f.txt", text="x" * (i + 1)) for i in range(n)]
    store.upsert(chunks, [[1.0, i / n] for i in range(n)])
    return chunks


def use(reranker) -> FakeLLM:
    llm = FakeLLM()
    app.dependency_overrides[get_cached_llm] = lambda: llm
    app.dependency_overrides[get_cached_embedder] = QueryEmbedder
    app.dependency_overrides[get_cached_reranker] = lambda: reranker
    return llm


def test_ask_reranks_candidate_pool_and_keeps_top_k(store):
    chunks = seed(store, 10)
    reranker = LengthReranker()
    llm = use(reranker)
    body = client.post("/ask", json={"question": "q", "k": 2}).json()
    # The reranker sees 4k = 8 vector candidates, not the whole store.
    query, texts = reranker.calls[0]
    assert query == "q" and len(texts) == 8
    # Longest of those 8 wins and becomes source [1] with the reranker's score.
    assert body["citations"][0]["chunk_id"] == chunks[7].id
    assert body["citations"][0]["score"] == 8.0
    prompt = llm.calls[0][1]["content"]
    assert "[1] (f.txt)\n" + "x" * 8 in prompt and "[2] (f.txt)\n" + "x" * 7 in prompt
    assert "[3]" not in prompt


def test_ask_without_reranker_uses_vector_order(store):
    chunks = seed(store, 3)
    use(None)
    body = client.post("/ask", json={"question": "q", "k": 1}).json()
    assert body["citations"][0]["chunk_id"] == chunks[0].id


def test_ask_rerank_empty_store_refuses():
    reranker = LengthReranker()
    llm = use(reranker)
    assert client.post("/ask", json={"question": "q"}).json()["refused"] is True
    assert llm.calls == []


def test_get_reranker_off_by_default(monkeypatch):
    monkeypatch.setattr(rerank.settings, "rerank", False)
    assert rerank.get_reranker() is None


def test_cross_encoder_without_local_extra_explains(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="uv sync --extra local"):
        rerank.CrossEncoderReranker()
