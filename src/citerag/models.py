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
