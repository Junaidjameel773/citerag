from citerag.config import settings

# How many candidates per requested source are retrieved for the reranker to reorder.
# ponytail: fixed 4x pool; make it a setting if recall or latency needs tuning.
POOL = 4


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a sentence-transformers cross-encoder
    (install the `local` extra). Higher score means more relevant."""

    def __init__(self, model: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError("Reranking needs the 'local' extra: uv sync --extra local") from e
        self.model = CrossEncoder(model or settings.rerank_model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


def get_reranker() -> CrossEncoderReranker | None:
    """The reranker when CITERAG_RERANK is on, else None."""
    return CrossEncoderReranker() if settings.rerank else None
