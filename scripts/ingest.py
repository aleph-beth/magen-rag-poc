#!/usr/bin/env python
from magen_rag.config import load_settings
from magen_rag.pipeline import MagenRAGPipeline
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

pipeline = MagenRAGPipeline(load_settings())
print({"chunks": pipeline.ingest(args.path)})
