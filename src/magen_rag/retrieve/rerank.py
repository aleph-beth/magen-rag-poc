from __future__ import annotations

from magen_rag.models import RetrievedChunk


class RerankService:
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_n: int = 5) -> list[RetrievedChunk]:
        _ = query
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]
