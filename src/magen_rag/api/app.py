from __future__ import annotations

from fastapi import FastAPI

from magen_rag.config import load_settings
from magen_rag.pipeline import MagenRAGPipeline

app = FastAPI(title="magen-rag-poc")
settings = load_settings()
pipeline = MagenRAGPipeline(settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: dict):
    count = pipeline.ingest(req["path"])
    return {"chunks": count}


@app.post("/index/build")
def build_index() -> dict[str, int]:
    count = pipeline.build_index()
    return {"indexed_chunks": count}


@app.post("/query")
def query(req: dict):
    return pipeline.query(req["query"], req.get("top_k"), req.get("use_rerank", False)).model_dump()
