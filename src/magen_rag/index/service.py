from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from magen_rag.models import Chunk


class FaissIndexService:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def build_faiss(self, vectors: np.ndarray) -> faiss.Index:
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        return self.index

    def save(self, index_path: Path, metadata_path: Path, chunks: list[Chunk]) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text("\n".join(json.dumps(c.model_dump(), ensure_ascii=False) for c in chunks), encoding="utf-8")

    @staticmethod
    def load(index_path: Path, metadata_path: Path) -> tuple[faiss.Index, list[Chunk]]:
        index = faiss.read_index(str(index_path))
        chunks = [Chunk.model_validate_json(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return index, chunks
