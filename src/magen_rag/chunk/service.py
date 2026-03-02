from __future__ import annotations

import hashlib
import re

from magen_rag.config import ChunkConfig
from magen_rag.models import Chunk, Document


def _tokenize(text: str) -> list[str]:
    return text.split()


def _flags(text: str) -> list[str]:
    flags: list[str] = []
    if len(text) < 50:
        flags.append("too_short")
    non_alpha = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if text and (non_alpha / len(text)) > 0.25:
        flags.append("low_quality")
    if re.search(r"(.{5,})\1{3,}", text):
        flags.append("low_quality")
    return sorted(set(flags))


def chunk_documents(docs: list[Document], cfg: ChunkConfig) -> list[Chunk]:
    output: list[Chunk] = []
    seen_hashes: set[str] = set()
    step = max(1, cfg.chunk_tokens_target - cfg.overlap_tokens)
    for doc in docs:
        tokens = _tokenize(doc.text)
        if not tokens:
            continue
        chunks_text: list[str] = []
        for i in range(0, len(tokens), step):
            slice_tokens = tokens[i : i + cfg.chunk_tokens_target]
            if not slice_tokens:
                continue
            chunks_text.append(" ".join(slice_tokens))
        merged: list[str] = []
        for t in chunks_text:
            toks = _tokenize(t)
            if merged and len(toks) < cfg.chunk_tokens_min:
                merged[-1] = merged[-1] + " " + t
            elif len(toks) > cfg.chunk_tokens_max:
                for j in range(0, len(toks), cfg.chunk_tokens_max):
                    merged.append(" ".join(toks[j : j + cfg.chunk_tokens_max]))
            else:
                merged.append(t)

        for idx, text in enumerate(merged, start=1):
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            flags = _flags(text)
            if text_hash in seen_hashes:
                flags.append("duplicate")
            seen_hashes.add(text_hash)
            output.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}_{idx:04d}",
                    doc_id=doc.doc_id,
                    text=text,
                    text_hash=text_hash,
                    quality_flags=sorted(set(flags)),
                )
            )
    return output
