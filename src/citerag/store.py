import chromadb

from citerag.config import settings
from citerag.models import Chunk


class ChunkStore:
    """Chunks, their embeddings and metadata in a persistent Chroma collection."""

    def __init__(self, path: str | None = None) -> None:
        client = chromadb.PersistentClient(path=path or settings.chroma_path)
        self.collection = client.get_or_create_collection(
            "chunks", metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[_metadata(c) for c in chunks],
        )

    def search(self, embedding: list[float], k: int = 5) -> list[dict]:
        """Top-k chunks by cosine similarity (1 = identical), best first."""
        if not self.collection.count():
            return []
        res = self.collection.query(
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

    def documents(self) -> list[dict]:
        """One entry per document: id, filename and chunk count, in insertion order."""
        # ponytail: scans all chunk metadata; keep a documents table if this gets slow
        docs: dict[str, dict] = {}
        for m in self.collection.get(include=["metadatas"])["metadatas"]:
            doc = docs.setdefault(
                m["document_id"], {"id": m["document_id"], "filename": m["filename"], "chunks": 0}
            )
            doc["chunks"] += 1
        return list(docs.values())

    def delete_document(self, document_id: str) -> int:
        """Delete a document's chunks and vectors; returns how many were removed."""
        ids = self.collection.get(where={"document_id": document_id}, include=[])["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)


def _metadata(chunk: Chunk) -> dict:
    # Chroma metadata values must be scalars and not None.
    meta = {"document_id": chunk.document_id, "filename": chunk.filename, **chunk.metadata}
    if chunk.page is not None:
        meta["page"] = chunk.page
    return {k: v for k, v in meta.items() if v is not None}
