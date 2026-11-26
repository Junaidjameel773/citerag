import pytest

from citerag.main import app, get_cached_embedder, get_store
from citerag.store import ChunkStore


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]


@pytest.fixture(autouse=True)
def store(tmp_path):
    """Every test gets an empty Chroma store in a temp dir and offline embeddings."""
    chunk_store = ChunkStore(str(tmp_path / "chroma"))
    app.dependency_overrides[get_store] = lambda: chunk_store
    app.dependency_overrides[get_cached_embedder] = FakeEmbedder
    yield chunk_store
    app.dependency_overrides.clear()
