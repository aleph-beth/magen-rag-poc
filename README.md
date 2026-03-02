# magen-rag-poc

PoC d'assistant SOC en RAG 100% local (ingestion -> chunking -> embeddings -> FAISS -> génération sourcée).

## Démarrage rapide

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Pipeline CLI

```bash
python scripts/ingest.py data/raw
python scripts/build_index.py
```

## API

```bash
bash scripts/run_api.sh
```

Endpoints:
- `GET /health`
- `POST /ingest`
- `POST /index/build`
- `POST /query`

## UI

```bash
bash scripts/run_ui.sh
```

## Évaluation

```bash
python -c "from pathlib import Path; from magen_rag.config import load_settings; from magen_rag.pipeline import MagenRAGPipeline; from magen_rag.eval.harness import run_eval; print(run_eval(MagenRAGPipeline(load_settings()), Path('eval/queries.jsonl')))"
```

## Structure

- `src/magen_rag/` : services coeur
- `scripts/` : launchers
- `config/settings.yaml` : configuration
- `data/` : corpus, chunks, index
- `runs/` : traces d'exécution
