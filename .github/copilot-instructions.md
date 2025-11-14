# Copilot Instructions

## Project Snapshot
- Hybrid recommender for Vietnamese cosmetics combining ALS, BPR, and PhoBERT; current repo holds design specs under `tasks/` and raw/processed datasets in `data/`.
- Treat the task markdown files (`00`-`08`) as the single source of truth for architecture, required modules, expected artifacts, and timelines.
- No production Python package exists yet; new code should mirror the directory plans documented in the tasks (e.g., `recsys/cf/*.py`, `service/recommender/*.py`).

## Data Contracts
- Raw CSVs live in `data/published_data/` with interaction logs (`data_reviews_purchase.csv`), product metadata, and attribute tables; Vietnamese text requires `encoding="utf-8"`.
- Processed outputs are expected in `data/processed/` (Parquet interactions, CSR matrices in `.npz`, mappings JSON, stats JSON); maintain the filenames and schemas spelled out in Task 01.
- Hold-out evaluation splits follow leave-one-out per user; always mask seen items and log data hashes/timestamps for reproducibility.

## Module Expectations
- `recsys/cf/data.py` should expose loaders, positive-signal filtering, contiguous ID mapping, CSR builders, and save/load helpers exactly as enumerated in Task 01.
- `recsys/cf/als.py` and `recsys/cf/bpr.py` must align with Task 02 training loops, returning embeddings plus rich metadata (params, metrics, data hash) ready for registry ingestion.
- `recsys/cf/metrics.py` underpins evaluation (Recall@K, NDCG@K, coverage, diversity metrics) and is reused by training scripts and dashboards (see Task 03).
- Serving stack mirrors Task 05: loaders + recommender core + rerank/fallback components with FastAPI entrypoints; configs live in `service/config/*.yaml`.

## Workflows & Tooling
- Python 3.10+ with `numpy`, `pandas`, `scipy`, `implicit`, `torch`, `transformers`, `fastapi`, `uvicorn`; prefer separate requirement files (`requirements_cf.txt`, etc.) as referenced in `tasks/README.md`.
- Standard flow: run data refresh (`scripts/refresh_data.py`), train (`scripts/train_cf.py` or `train_both_models.py`), evaluate (`scripts/evaluate_models.py` or `evaluate_hybrid.py`), then serve via `uvicorn service.api:app --reload`.
- Model artifacts belong in `artifacts/cf/<model_type>/`; update `artifacts/cf/registry.json` via `scripts/update_registry.py` while preserving historical entries for rollback.

## Conventions & Quality
- Maintain detailed logging (training, service, drift) feeding SQLite DBs in `logs/`; alerting thresholds defined in prospective `config/alerts_config.yaml` (Tasks 06-07).
- Every saved artifact must embed `data_hash`, config snapshot, and git commit for reproducibility as emphasized across tasks.
- Follow PEP8 with type hints, use docstrings for public APIs, and add focused unit tests per module (data preprocessing, metrics, registry, recommender logic) before merging.
- Cross-component features (hybrid rerank, BERT refresh jobs) depend on PhoBERT embeddings stored in `data/published_data/content_based_embeddings/`; cache via LRU and record versions in the registry metadata.
