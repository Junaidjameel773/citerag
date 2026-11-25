from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid4().hex


class Chunk(BaseModel):
    id: str = Field(default_factory=_new_id)
    document_id: str
    filename: str
    text: str
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    id: str = Field(default_factory=_new_id)
    filename: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRegistry:
    """In-memory store of documents, keyed by id."""

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}

    def add(self, doc: Document) -> Document:
        self._docs[doc.id] = doc
        return doc

    def get(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def list(self) -> list[Document]:
        return list(self._docs.values())
