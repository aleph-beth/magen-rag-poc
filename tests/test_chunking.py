from magen_rag.chunk.service import chunk_documents
from magen_rag.config import ChunkConfig
from magen_rag.models import Document


def test_chunk_documents_produces_ids():
    doc = Document(
        doc_id="d1",
        source_path="x",
        doc_type="txt",
        title="t",
        text="word " * 700,
        hash_sha256="h",
    )
    chunks = chunk_documents([doc], ChunkConfig(chunk_tokens_target=200, chunk_tokens_min=50, chunk_tokens_max=300, overlap_tokens=20))
    assert chunks
    assert chunks[0].chunk_id.startswith("d1_")
