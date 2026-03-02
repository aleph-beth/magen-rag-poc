from __future__ import annotations

import json
from pathlib import Path

from magen_rag.chunk.service import chunk_documents
from magen_rag.config import AppSettings
from magen_rag.embed.service import EmbeddingService
from magen_rag.index.service import FaissIndexService
from magen_rag.ingest.service import ingest_folder, write_manifest
from magen_rag.llm.service import LLMService, extract_citations
from magen_rag.logging.run_logger import RunLogger
from magen_rag.models import Answer
from magen_rag.prompt.builder import build_context_prompt
from magen_rag.retrieve.rerank import RerankService
from magen_rag.retrieve.service import RetrievalService


class MagenRAGPipeline:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.embedder = EmbeddingService(settings.embedding_model_name, settings.embedding_dim)
        self.logger = RunLogger(settings.runs_dir)

    def ingest(self, source_path: str) -> int:
        docs = ingest_folder(source_path)
        chunks = chunk_documents(docs, self.settings.chunk)
        manifest_path = self.settings.data_dir / "corpus_manifest.jsonl"
        chunks_path = self.settings.data_dir / "processed" / "chunks.jsonl"
        write_manifest(docs, manifest_path)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.write_text("\n".join(json.dumps(c.model_dump(), ensure_ascii=False) for c in chunks), encoding="utf-8")
        return len(chunks)

    def build_index(self) -> int:
        chunks_path = self.settings.data_dir / "processed" / "chunks.jsonl"
        chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        texts = [c["text"] for c in chunks]
        vectors = self.embedder.embed_texts(texts)
        svc = FaissIndexService(vectors.shape[1])
        svc.build_faiss(vectors)
        chunk_models = [self._to_chunk(c) for c in chunks]
        svc.save(self.settings.data_dir / "index" / "faiss.index", self.settings.data_dir / "index" / "metadata.jsonl", chunk_models)
        return len(chunk_models)

    def query(self, query: str, top_k: int | None = None, use_rerank: bool = False) -> Answer:
        index, chunks = FaissIndexService.load(self.settings.data_dir / "index" / "faiss.index", self.settings.data_dir / "index" / "metadata.jsonl")
        retriever = RetrievalService(index, chunks)
        q_vec = self.embedder.embed_texts([query])[0]
        retrieved = retriever.retrieve(q_vec, top_k or self.settings.retrieval.top_k_retrieve, self.settings.retrieval.score_threshold)
        if use_rerank:
            retrieved = RerankService().rerank(query, retrieved, top_n=min(5, len(retrieved)))

        prompt = build_context_prompt(query, retrieved)
        llm = LLMService(self.settings.llm.ollama_base_url, self.settings.llm.model_name)
        answer_text = llm.generate(prompt)
        citations = extract_citations(answer_text)

        run_id, run_dir = self.logger.start()
        self.logger.write_json(run_dir, "query.json", {"query": query, "top_k": top_k, "use_rerank": use_rerank})
        self.logger.write_json(run_dir, "retrieved_chunks.json", {"chunks": [r.model_dump(mode='json') for r in retrieved]})
        self.logger.write_text(run_dir, "prompt.txt", prompt)
        self.logger.write_json(run_dir, "answer.json", {"answer": answer_text, "citations": citations})

        return Answer(
            answer=answer_text,
            citations=citations,
            retrieval={"chunks": [r.model_dump(mode="json") for r in retrieved]},
            run_id=run_id,
        )

    @staticmethod
    def _to_chunk(payload):
        from magen_rag.models import Chunk

        return Chunk(**payload)
