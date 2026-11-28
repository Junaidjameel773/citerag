import json
from collections.abc import Iterator
from functools import lru_cache
from pathlib import PurePath
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from citerag import __version__
from citerag.answer import (
    REFUSAL,
    build_messages,
    build_rewrite_messages,
    is_refusal,
    validate_citations,
)
from citerag.chunking import chunk_passages
from citerag.config import settings
from citerag.embeddings import Embedder, get_embedder
from citerag.llm import LLM, get_llm
from citerag.models import Document
from citerag.parsing import parse
from citerag.rerank import POOL, CrossEncoderReranker, get_reranker
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


@lru_cache
def get_cached_reranker() -> CrossEncoderReranker | None:
    return get_reranker()


class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunks: int


# 1-61 chars of letters, digits, '.', '_', '-', alphanumeric at both ends (fits Chroma's
# naming rules once the store adds its prefix).
COLLECTION_PATTERN = r"^[a-zA-Z0-9]([a-zA-Z0-9._-]{0,59}[a-zA-Z0-9])?$"
COLLECTION_HELP = "Document set to use; omit for the default collection"
CollectionQuery = Annotated[
    str | None, Query(pattern=COLLECTION_PATTERN, description=COLLECTION_HELP)
]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=50)
    mode: Literal["vector", "keyword", "hybrid"] = Field(
        "vector",
        description="vector: cosine similarity; keyword: BM25; hybrid: both, fused by RRF",
    )
    collection: str | None = Field(None, pattern=COLLECTION_PATTERN, description=COLLECTION_HELP)


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    text: str
    score: float
    metadata: dict[str, Any]


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=50)
    collection: str | None = Field(None, pattern=COLLECTION_PATTERN, description=COLLECTION_HELP)
    history: list[Turn] = Field(
        default_factory=list,
        max_length=50,
        description="Earlier turns, oldest first; follow-ups are rewritten into standalone queries",
    )


class Citation(SearchHit):
    n: int  # the [n] marker used in the answer


class AskResponse(BaseModel):
    answer: str
    refused: bool  # true when the documents do not contain the answer
    uncited: bool = False  # true when an answer cites no valid source; treat it as unverified
    citations: list[Citation]  # only the sources the answer actually cites


def _search(
    store: ChunkStore,
    embedder: Embedder,
    query: str,
    k: int,
    collection: str | None,
    mode: str = "vector",
) -> list[SearchHit]:
    if not query.strip():
        raise HTTPException(422, "Query must not be blank")
    if mode == "keyword":
        raw = store.keyword_search(query, k, collection)
    elif mode == "hybrid":
        raw = store.hybrid_search(query, embedder.embed([query])[0], k, collection)
    else:
        raw = store.search(embedder.embed([query])[0], k, collection)
    hits = []
    for h in raw:
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
    collection: CollectionQuery = None,
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
    store.upsert(chunks, embedder.embed([c.text for c in chunks]), collection)
    return DocumentInfo(id=doc.id, filename=filename, chunks=len(chunks))


@app.post("/search")
def search(
    req: SearchRequest,
    store: Annotated[ChunkStore, Depends(get_store)],
    embedder: Annotated[Embedder, Depends(get_cached_embedder)],
) -> list[SearchHit]:
    return _search(store, embedder, req.query, req.k, req.collection, req.mode)


def _finish(answer: str, hits: list[SearchHit]) -> AskResponse:
    if is_refusal(answer):
        return AskResponse(answer=answer, refused=True, citations=[])
    answer, cited = validate_citations(answer, len(hits))
    citations = [Citation(n=n, **hits[n - 1].model_dump()) for n in cited]
    return AskResponse(answer=answer, refused=False, uncited=not cited, citations=citations)


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    store: Annotated[ChunkStore, Depends(get_store)],
    embedder: Annotated[Embedder, Depends(get_cached_embedder)],
    llm: Annotated[LLM, Depends(get_cached_llm)],
    reranker: Annotated[CrossEncoderReranker | None, Depends(get_cached_reranker)],
    stream: bool = False,
):
    """With `?stream=true`, responds with Server-Sent Events: `token` events carrying
    `{"text": ...}` as the model writes, then one `done` event with the full AskResponse
    (cleaned answer and citations)."""
    question = req.question
    if req.history and question.strip():
        history = [t.model_dump() for t in req.history]
        # Fall back to the original question if the model returns nothing usable.
        question = llm.complete(build_rewrite_messages(question, history)).strip() or question
    if reranker is None:
        hits = _search(store, embedder, question, req.k, req.collection)
    else:
        hits = _search(store, embedder, question, req.k * POOL, req.collection)
        for hit, score in zip(hits, reranker.score(question, [h.text for h in hits]), strict=True):
            hit.score = score
        hits = sorted(hits, key=lambda h: h.score, reverse=True)[: req.k]
    messages = build_messages(question, [h.model_dump() for h in hits])
    if not stream:
        if not hits:  # nothing to ground an answer in; don't pay for an LLM call
            return AskResponse(answer=REFUSAL, refused=True, citations=[])
        return _finish(llm.complete(messages), hits)

    def events() -> Iterator[str]:
        if not hits:
            done = AskResponse(answer=REFUSAL, refused=True, citations=[])
            yield _sse("done", done.model_dump_json())
            return
        parts = []
        for token in llm.stream(messages):
            parts.append(token)
            yield _sse("token", json.dumps({"text": token}))
        yield _sse("done", _finish("".join(parts), hits).model_dump_json())

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/documents")
def list_documents(
    store: Annotated[ChunkStore, Depends(get_store)], collection: CollectionQuery = None
) -> list[DocumentInfo]:
    return [DocumentInfo(**d) for d in store.documents(collection)]


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    store: Annotated[ChunkStore, Depends(get_store)],
    collection: CollectionQuery = None,
) -> None:
    if not store.delete_document(document_id, collection):
        raise HTTPException(404, "Document not found")
