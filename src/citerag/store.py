import chromadb
from chromadb.errors import NotFoundError

from citerag.bm25 import bm25_scores
from citerag.config import settings
from citerag.models import Chunk

# Chroma name of the collection used when no collection is given (kept from v0.1 so
# existing data stays visible).
DEFAULT_COLLECTION = "chunks"


class ChunkStore:
    """Chunks, their embeddings and metadata in persistent Chroma collections, one per
    document set. `collection=None` means the default collection."""

    def __init__(self, path: str | None = None) -> None:
        self.client = chromadb.PersistentClient(path=path or settings.chroma_path)
        self.collection = self._collection(None, create=True)

    def _collection(self, name: str | None, create: bool = False):
        """The named collection; None if it doesn't exist and create is False."""
        # Prefix named sets: allows names shorter than Chroma's 3-char minimum and keeps a
        # set called "chunks" from aliasing the default collection.
        name = f"c_{name}" if name else DEFAULT_COLLECTION
        if create:
            return self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        try:
            return self.client.get_collection(name)
        except NotFoundError:
            return None

    def upsert(
        self, chunks: list[Chunk], embeddings: list[list[float]], collection: str | None = None
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._collection(collection, create=True).upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_metadata(c) for c in chunks],
        )

    def search(
        self, embedding: list[float], k: int = 5, collection: str | None = None
    ) -> list[dict]:
        """Top-k chunks by cosine similarity (1 = identical), best first."""
        coll = self._collection(collection)
        if coll is None or not coll.count():
            return []
        res = coll.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": id_, "text": text, "score": 1 - dist, "metadata": meta}
            for id_, text, meta, dist in zip(
                res["ids"][0],
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
                strict=True,
            )
        ]

    def keyword_search(self, query: str, k: int = 5, collection: str | None = None) -> list[dict]:
        """Top-k chunks by BM25 score, best first; chunks sharing no term are left out."""
        coll = self._collection(collection)
        if coll is None:
            return []
        # ponytail: rebuilds the BM25 index from every chunk per query; cache it per
        # collection (invalidated on upsert/delete) if collections get large.
        res = coll.get(include=["documents", "metadatas"])
        scores = bm25_scores(query, res["documents"])
        ranked = sorted(
            (
                {"id": id_, "text": text, "score": score, "metadata": meta}
                for id_, text, meta, score in zip(
                    res["ids"], res["documents"], res["metadatas"], scores, strict=True
                )
                if score > 0
            ),
            key=lambda h: h["score"],
            reverse=True,
        )
        return ranked[:k]

    def hybrid_search(
        self, query: str, embedding: list[float], k: int = 5, collection: str | None = None
    ) -> list[dict]:
        """Top-k chunks by reciprocal rank fusion of vector and BM25 results, best first."""
        # ponytail: fuses the top 4k of each list; a chunk ranked lower in both is missed.
        n = k * 4
        fused = rrf(
            [self.search(embedding, n, collection), self.keyword_search(query, n, collection)],
            [settings.hybrid_vector_weight, settings.hybrid_keyword_weight],
        )
        return fused[:k]

    def documents(self, collection: str | None = None) -> list[dict]:
        """One entry per document: id, filename and chunk count, in insertion order."""
        coll = self._collection(collection)
        if coll is None:
            return []
        # ponytail: scans all chunk metadata; keep a documents table if this gets slow
        docs: dict[str, dict] = {}
        for m in coll.get(include=["metadatas"])["metadatas"]:
            doc = docs.setdefault(
                m["document_id"], {"id": m["document_id"], "filename": m["filename"], "chunks": 0}
            )
            doc["chunks"] += 1
        return list(docs.values())

    def delete_document(self, document_id: str, collection: str | None = None) -> int:
        """Delete a document's chunks and vectors; returns how many were removed."""
        coll = self._collection(collection)
        if coll is None:
            return 0
        ids = coll.get(where={"document_id": document_id}, include=[])["ids"]
        if ids:
            coll.delete(ids=ids)
        return len(ids)


def rrf(rankings: list[list[dict]], weights: list[float], c: int = 60) -> list[dict]:
    """Reciprocal rank fusion: a hit scores sum(weight / (c + rank)) over the rankings it
    appears in (rank starts at 1). Returns the hits, best first, with `score` set to that sum."""
    fused: dict[str, dict] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, hit in enumerate(ranking, 1):
            entry = fused.setdefault(hit["id"], {**hit, "score": 0.0})
            entry["score"] += weight / (c + rank)
    return sorted(fused.values(), key=lambda h: h["score"], reverse=True)


def _metadata(chunk: Chunk) -> dict:
    # Chroma metadata values must be scalars and not None.
    meta = {"document_id": chunk.document_id, "filename": chunk.filename, **chunk.metadata}
    if chunk.page is not None:
        meta["page"] = chunk.page
    return {k: v for k, v in meta.items() if v is not None}
