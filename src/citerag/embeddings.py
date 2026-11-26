from typing import Protocol

from citerag.config import settings


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Embeds texts in batches with the OpenAI embeddings API."""

    def __init__(self, model: str | None = None, batch_size: int = 100, client=None) -> None:
        if client is None:
            from openai import OpenAI

            # The SDK retries 429 rate limits (and 5xx) with exponential backoff.
            client = OpenAI(
                api_key=settings.openai_api_key or None, max_retries=settings.embedding_retries
            )
        self.client = client
        self.model = model or settings.embedding_model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(
                model=self.model, input=texts[i : i + self.batch_size]
            )
            vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return vectors


class LocalEmbedder:
    """Offline embeddings with sentence-transformers (install the `local` extra)."""

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Local embeddings need the 'local' extra: uv sync --extra local"
            ) from e
        self.model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(texts, batch_size=64).tolist()


def get_embedder() -> Embedder:
    if settings.embeddings == "openai":
        return OpenAIEmbedder()
    if settings.embeddings == "local":
        return LocalEmbedder()
    raise ValueError(f"Unknown CITERAG_EMBEDDINGS '{settings.embeddings}'; use openai or local")
