from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Base:
    def model_dump(self, mode: str | None = None, exclude: set[str] | None = None) -> dict[str, Any]:
        _ = mode
        data = asdict(self)
        if exclude:
            for key in exclude:
                data.pop(key, None)
        return data


@dataclass
class Document(Base):
    doc_id: str
    source_path: str
    doc_type: str
    title: str
    text: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    hash_sha256: str = ""
    access_level: str = "internal"


@dataclass
class Chunk(Base):
    chunk_id: str
    doc_id: str
    text: str
    text_hash: str
    page_range: str | None = None
    heading_path: str | None = None
    quality_flags: list[str] = field(default_factory=list)

    @classmethod
    def model_validate_json(cls, line: str) -> "Chunk":
        return cls(**json.loads(line))


@dataclass
class RetrievedChunk(Base):
    chunk: Chunk
    score: float


@dataclass
class Answer(Base):
    answer: str
    citations: list[str]
    retrieval: dict[str, Any]
    run_id: str
