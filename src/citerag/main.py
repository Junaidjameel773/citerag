from functools import lru_cache
from pathlib import PurePath
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from citerag import __version__
from citerag.answer import REFUSAL, build_messages, is_refusal, validate_citations
from citerag.chunking import chunk_passages
from citerag.config import settings
from citerag.embeddings import Embedder, get_embedder
from citerag.llm import LLM, get_llm
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


@lru_cache
def get_cached_llm() -> LLM:
    return get_llm()


class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=50)


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    text: str
    score: float
    metadata: dict[str, Any]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=50)


class Citation(SearchHit):
    n: int  # the [n] marker used in the answer


class AskResponse(BaseModel):
    answer: str
    refused: bool  # true when the documents do not contain the answer
    uncited: bool = False  # true when an answer cites no valid source; treat it as unverified
    citations: list[Citation]  # only the sources the answer actually cites


def _search(store: ChunkStore, embedder: Embedder, query: str, k: int) -> list[SearchHit]:
    if not query.strip():
        raise HTTPException(422, "Query must not be blank")
    hits = []
    for h in store.search(embedder.embed([query])[0], k):
        meta = dict(h["metadata"])
        hits.append(
            SearchHit(
                chunk_id=h["id"],
                document_id=meta.pop("document_id"),
                filename=meta.pop("filename"),
                page=meta.pop("page", None),
                text=h["text"],
                score=h["score"],
                metadata=meta,
            )
        )
    return hits


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


@app.post("/search")
def search(
    req: SearchRequest,
    store: Annotated[ChunkStore, Depends(get_store)],
    embedder: Annotated[Embedder, Depends(get_cached_embedder)],
) -> list[SearchHit]:
    return _search(store, embedder, req.query, req.k)


@app.post("/ask")
def ask(
    req: AskRequest,
    store: Annotated[ChunkStore, Depends(get_store)],
    embedder: Annotated[Embedder, Depends(get_cached_embedder)],
    llm: Annotated[LLM, Depends(get_cached_llm)],
) -> AskResponse:
    hits = _search(store, embedder, req.question, req.k)
    if not hits:  # nothing to ground an answer in; don't pay for an LLM call
        return AskResponse(answer=REFUSAL, refused=True, citations=[])
    answer = llm.complete(build_messages(req.question, [h.model_dump() for h in hits]))
    if is_refusal(answer):
        return AskResponse(answer=answer, refused=True, citations=[])
    answer, cited = validate_citations(answer, len(hits))
    citations = [Citation(n=n, **hits[n - 1].model_dump()) for n in cited]
    return AskResponse(answer=answer, refused=False, uncited=not cited, citations=citations)


@app.get("/documents")
def list_documents(store: Annotated[ChunkStore, Depends(get_store)]) -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in store.documents()]


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, store: Annotated[ChunkStore, Depends(get_store)]) -> None:
    if not store.delete_document(document_id):
        raise HTTPException(404, "Document not found")
