from __future__ import annotations

import numpy as np

from magen_rag.models import Chunk, RetrievedChunk


class RetrievalService:
    def __init__(self, index, chunks: list[Chunk]) -> None:
        self.index = index
        self.chunks = chunks

    def retrieve(self, query_vector: np.ndarray, top_k: int, score_threshold: float | None = None) -> list[RetrievedChunk]:
        q = query_vector.reshape(1, -1).astype("float32")
        scores, ids = self.index.search(q, top_k)
        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            if score_threshold is not None and float(score) < score_threshold:
                continue
            out.append(RetrievedChunk(chunk=self.chunks[int(idx)], score=float(score)))
        return out
