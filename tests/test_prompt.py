from magen_rag.models import Chunk, RetrievedChunk
from magen_rag.prompt.builder import build_context_prompt


def test_prompt_contains_context_and_citation_shape():
    chunk = Chunk(chunk_id="c1", doc_id="d1", page_range="1-1", text="abc", text_hash="h")
    prompt = build_context_prompt("question", [RetrievedChunk(chunk=chunk, score=0.9)])
    assert "CONTEXT" in prompt
    assert "[d1:c1:1-1]" in prompt
