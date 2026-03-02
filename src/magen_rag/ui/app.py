from __future__ import annotations

import streamlit as st

from magen_rag.config import load_settings
from magen_rag.pipeline import MagenRAGPipeline

settings = load_settings()
pipeline = MagenRAGPipeline(settings)

st.title("Magen Assistant SOC — Local RAG")
source = st.text_input("Corpus folder", value="data/raw")
if st.button("Ingest"):
    st.success(f"Chunks created: {pipeline.ingest(source)}")
if st.button("Build index"):
    st.success(f"Indexed chunks: {pipeline.build_index()}")
query = st.text_area("Question")
use_rerank = st.checkbox("Use rerank", value=False)
if st.button("Ask") and query.strip():
    result = pipeline.query(query, use_rerank=use_rerank)
    st.subheader("Answer")
    st.write(result.answer)
    st.subheader("Citations")
    st.json(result.citations)
    st.subheader("Retrieved Chunks")
    st.json(result.retrieval)
