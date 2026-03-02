from __future__ import annotations

import json
from pathlib import Path

from magen_rag.pipeline import MagenRAGPipeline


def run_eval(pipeline: MagenRAGPipeline, queries_path: Path, top_k: int = 10) -> dict[str, float]:
    rows = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recall_hits = 0
    refusal_hits = 0
    for row in rows:
        res = pipeline.query(row["question"], top_k=top_k)
        got_doc_ids = {c["chunk"]["doc_id"] for c in res.retrieval["chunks"]}
        expected = set(row.get("expected_doc_ids", []))
        if expected and (got_doc_ids & expected):
            recall_hits += 1
        must_refuse = bool(row.get("must_refuse", False))
        refused = res.answer.startswith("INSUFFICIENT_EVIDENCE")
        if must_refuse == refused:
            refusal_hits += 1
    n = max(len(rows), 1)
    return {
        "Recall@k": recall_hits / n,
        "RefusalAccuracy": refusal_hits / n,
    }
