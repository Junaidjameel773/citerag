import sys
from types import SimpleNamespace

import pytest

from citerag.config import settings
from citerag.embeddings import LocalEmbedder, OpenAIEmbedder, get_embedder


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def create(self, model: str, input: list[str]):
        self.calls.append((model, input))
        # Return out of order to check results are sorted by index.
        data = [SimpleNamespace(index=i, embedding=[float(len(t))]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))


def test_embeds_in_batches_preserving_order():
    fake = FakeEmbeddings()
    embedder = OpenAIEmbedder(model="m", batch_size=2, client=SimpleNamespace(embeddings=fake))
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    assert embedder.embed(texts) == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert [c[1] for c in fake.calls] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert all(c[0] == "m" for c in fake.calls)


def test_empty_input_makes_no_calls():
    fake = FakeEmbeddings()
    assert OpenAIEmbedder(client=SimpleNamespace(embeddings=fake)).embed([]) == []
    assert fake.calls == []


def test_default_client_uses_configured_model_and_retries(monkeypatch):
    monkeypatch.setattr(settings, "embedding_model", "my-model")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    embedder = OpenAIEmbedder()
    assert embedder.model == "my-model"
    assert embedder.client.max_retries == settings.embedding_retries


class FakeArray(list):
    def tolist(self):
        return list(self)


def fake_sentence_transformers(monkeypatch):
    class SentenceTransformer:
        def __init__(self, name: str) -> None:
            self.name = name

        def encode(self, texts, batch_size):
            return FakeArray([[float(len(t)), 0.0] for t in texts])

    module = SimpleNamespace(SentenceTransformer=SentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_local_embedder(monkeypatch):
    fake_sentence_transformers(monkeypatch)
    embedder = LocalEmbedder()
    assert embedder.model.name == "all-MiniLM-L6-v2"
    assert embedder.embed(["ab", "c"]) == [[2.0, 0.0], [1.0, 0.0]]
    assert embedder.embed([]) == []


def test_local_embedder_without_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="local"):
        LocalEmbedder()


def test_get_embedder(monkeypatch):
    fake_sentence_transformers(monkeypatch)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "embeddings", "local")
    assert isinstance(get_embedder(), LocalEmbedder)
    monkeypatch.setattr(settings, "embeddings", "openai")
    assert isinstance(get_embedder(), OpenAIEmbedder)
    monkeypatch.setattr(settings, "embeddings", "nope")
    with pytest.raises(ValueError):
        get_embedder()
