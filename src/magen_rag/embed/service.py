from __future__ import annotations

import hashlib
import importlib
import importlib.util

import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str, embedding_dim: int = 384) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model = None
        if importlib.util.find_spec("sentence_transformers") is not None:
            sentence_transformers = importlib.import_module("sentence_transformers")
            self._model = sentence_transformers.SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vectors, dtype="float32")
        vectors = [self._hash_embed(t) for t in texts]
        return np.vstack(vectors).astype("float32")

    def _hash_embed(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        arr = np.frombuffer((digest * ((self.embedding_dim // len(digest)) + 1))[: self.embedding_dim], dtype=np.uint8)
        vec = arr.astype(np.float32)
        norm = np.linalg.norm(vec) + 1e-12
        return vec / norm
