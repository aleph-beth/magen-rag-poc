from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass
class ChunkConfig:
    chunk_tokens_target: int = 550
    chunk_tokens_min: int = 250
    chunk_tokens_max: int = 800
    overlap_tokens: int = 100


@dataclass
class RetrievalConfig:
    top_k_retrieve: int = 10
    score_threshold: float | None = None


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model_name: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"


@dataclass
class AppSettings:
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


@lru_cache(maxsize=1)
def load_settings(config_path: str | Path = "config/settings.yaml") -> AppSettings:
    settings = AppSettings()
    if os.getenv("MAGEN_EMBEDDING_MODEL_NAME"):
        settings.embedding_model_name = os.environ["MAGEN_EMBEDDING_MODEL_NAME"]
    path = Path(config_path)
    if not path.exists() or importlib.util.find_spec("yaml") is None:
        return settings
    yaml = importlib.import_module("yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "embedding_model_name" in payload:
        settings.embedding_model_name = payload["embedding_model_name"]
    if "embedding_dim" in payload:
        settings.embedding_dim = int(payload["embedding_dim"])
    if "chunk" in payload:
        settings.chunk = ChunkConfig(**payload["chunk"])
    if "retrieval" in payload:
        settings.retrieval = RetrievalConfig(**payload["retrieval"])
    if "llm" in payload:
        settings.llm = LLMConfig(**payload["llm"])
    return settings
