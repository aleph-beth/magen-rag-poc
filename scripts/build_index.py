#!/usr/bin/env python
from magen_rag.config import load_settings
from magen_rag.pipeline import MagenRAGPipeline

pipeline = MagenRAGPipeline(load_settings())
print({"indexed_chunks": pipeline.build_index()})
