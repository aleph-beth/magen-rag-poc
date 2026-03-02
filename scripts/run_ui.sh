#!/usr/bin/env bash
set -euo pipefail
streamlit run src/magen_rag/ui/app.py --server.port 8501 --server.address 0.0.0.0
