from pathlib import PurePath

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from citerag import __version__
from citerag.chunking import chunk_passages
from citerag.config import settings
from citerag.models import Document, DocumentRegistry
from citerag.parsing import parse

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
registry = DocumentRegistry()


class UploadResult(BaseModel):
    id: str
    filename: str
    chunks: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile) -> UploadResult:
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
    # ponytail: chunks are not stored yet; vector store persistence lands in #8
    doc.metadata["chunk_count"] = len(chunks)
    registry.add(doc)
    return UploadResult(id=doc.id, filename=filename, chunks=len(chunks))
