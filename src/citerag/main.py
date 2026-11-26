from functools import lru_cache
from pathlib import PurePath
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from citerag import __version__
from citerag.chunking import chunk_passages
from citerag.config import settings
from citerag.embeddings import Embedder, get_embedder
from citerag.models import Document
from citerag.parsing import parse
from citerag.store import ChunkStore

# Accepted content types per extension; octet-stream is what many clients send for unknown types.
CONTENT_TYPES = {
    ".pdf": {"application/pdf"},
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/x-markdown", "text/plain"},
    ".markdown": {"text/markdown", "text/x-markdown", "text/plain"},
}

app = FastAPI(
    title="citerag",
    version=__version__,
    description="Document Q&A API with grounded, cited answers.",
)


@lru_cache
def get_store() -> ChunkStore:
    return ChunkStore()


@lru_cache
def get_cached_embedder() -> Embedder:
    return get_embedder()


class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunks: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/documents", status_code=201)
async def upload_document(
    file: UploadFile,
    store: Annotated[ChunkStore, Depends(get_store)],
    embedder: Annotated[Embedder, Depends(get_cached_embedder)],
) -> DocumentInfo:
    filename = file.filename or ""
    ext = PurePath(filename).suffix.lower()
    if ext not in CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Use .pdf, .md or .txt")
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in CONTENT_TYPES[ext] | {"application/octet-stream", ""}:
        raise HTTPException(415, f"Content type '{content_type}' does not match '{ext}'")

    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")

    try:
        passages = parse(filename, data)
    except ValueError as e:  # includes UnicodeDecodeError
        raise HTTPException(400, f"Could not parse file: {e}") from e
    if not passages:
        raise HTTPException(422, "No text could be extracted from the file")

    doc = Document(filename=filename)
    chunks = chunk_passages(doc, passages)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    return DocumentInfo(id=doc.id, filename=filename, chunks=len(chunks))


@app.get("/documents")
def list_documents(store: Annotated[ChunkStore, Depends(get_store)]) -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in store.documents()]


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, store: Annotated[ChunkStore, Depends(get_store)]) -> None:
    if not store.delete_document(document_id):
        raise HTTPException(404, "Document not found")
