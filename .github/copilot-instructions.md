# Copilot Instructions

## Project Snapshot
- **Hybrid Vietnamese cosmetics recommender** combining ALS (Alternating Least Squares), BPR (Bayesian Personalized Ranking), and PhoBERT for ~300K users, 2.2K products, 369K interactions
- Architecture specs live in `tasks/*.md` (00-08): treat these as authoritative for module structure, data contracts, and expected artifacts
- **Key challenge**: ~1.23 interactions/user (high sparsity) + 95% 5-star ratings (rating skew) → strategy uses **user segmentation** (≥2 interactions = trainable, ~8.6% users) and **content-first** approach
- Implementation status: Data layer (Task 01) partially complete with modular classes in `recsys/cf/data/`; training/serving layers (Tasks 02-05) planned but not yet implemented

## Critical Data Realities & Strategic Choices
- **Trainable users**: ≥2 interactions + ≥1 positive (rating ≥4) → ~26K users (~8.6%) get CF recommendations
- **Cold-start users**: <2 interactions → ~274K users (~91.4%) get content-based (PhoBERT item similarity + popularity)
- **Why ≥2 threshold**: Balance between data hunger and statistical viability; BERT initialization + higher regularization (λ=0.1) compensates for sparsity
- **Test sets**: Only positive interactions (rating ≥4) to measure "items user will like," not "items user will dislike"
- See `tasks/STRATEGY_UPDATE_26K_USERS.md` for full rationale on threshold choice and traffic implications

## Data Layer Architecture (Task 01 - Partially Implemented)
### Modular Class Structure (in `recsys/cf/data/processing/`)
```
DataProcessor (data.py)           # Unified orchestration interface
├─ DataValidator                  # Step 1: Validate, clean, dedupe (strict timestamp/rating checks)
├─ FeatureEngineer                # Step 2: Comment quality, is_positive/is_hard_negative labels
├─ UserFilter                     # Step 2.3: Segment users (trainable vs cold-start)
├─ IDMapper                       # Step 3: Contiguous user_id/product_id → u_idx/i_idx
├─ TemporalSplitter               # Step 4: Leave-one-out split (latest positive per user → test)
├─ MatrixBuilder                  # Step 5: CSR matrices (confidence, binary), user sets, metadata
├─ DataSaver                      # Step 6: Persist Parquet, NPZ, PKL, JSON artifacts
└─ VersionRegistry                # Step 7: Track data_hash, config, git commit for reproducibility
```
### Key Files & Usage
- **Entry point**: `scripts/run_task01_complete.py` orchestrates all 7 steps sequentially
- **Imports**: Always use `from recsys.cf.data import DataProcessor` (not submodules directly)
- **Config**: `data/processed/user_item_mappings.json` has metadata including `positive_threshold=4`, `hard_negative_threshold=3`
- **Critical outputs**: 
  - `X_train_confidence.npz` (ALS input, values = rating + comment_quality [1-6])
  - `X_train_binary.npz` (BPR optional)
  - `user_metadata.pkl` (has `is_trainable_user` flag for serving routing)
  - `data_stats.json` (includes global normalization ranges for Task 08 hybrid reranking)

## Data Contracts & File Standards
### Raw Data (in `data/published_data/`)
- `data_reviews_purchase.csv`: **Always use `encoding='utf-8'`** for Vietnamese text; columns = `user_id`, `product_id`, `rating` (1-5), `comment`, `cmt_date`
- `data_product.csv`: Metadata with `num_sold_time` (popularity), `avg_star`, `brand`, `processed_description`
- `data_product_attribute.csv`: `ingredient`, `feature`, `skin_type`, etc. for content-based filtering
### Processed Data (in `data/processed/`)
- **Parquet**: `interactions.parquet` with all engineered features (`confidence_score`, `is_positive`, `is_hard_negative`, `is_trainable_user`)
- **Sparse matrices**: `.npz` files (scipy CSR format) for ALS/BPR training
- **Mappings**: JSON with bidirectional dicts (`user_to_idx`, `idx_to_user`, `item_to_idx`, `idx_to_item`) plus metadata (data_hash, thresholds)
- **User sets**: `.pkl` files (pickled dicts) for `user_pos_train` (set of positive item indices) and `user_hard_neg_train` (dual sets: explicit + implicit negatives)
- **Stats**: `data_stats.json` includes global normalization ranges (popularity p01/p99, confidence_score min/max) needed for hybrid reranking
### Validation Rules
- **No NaT timestamps**: Drop rows with missing `cmt_date` (prevents data leakage in temporal splits)
- **Rating range**: Strict [1.0, 5.0] enforcement; drop invalids, never impute
- **Reproducibility**: Every artifact must embed `data_hash` (MD5 of raw CSVs), `timestamp`, and `git_commit`

## Training Strategy (Task 02 - Planned)
### ALS (Alternating Least Squares)
- **Input**: `X_train_confidence.npz` with **sentiment-enhanced confidence** = rating (1-5) + comment_quality (0-1) → range [1-6]
- **Library**: `implicit.als.AlternatingLeastSquares` (C++ backend, GPU support)
- **Hyperparameters** (adjusted for ≥2 threshold):
  - `factors=64`, `regularization=0.05-0.15` (higher than typical due to sparsity), `iterations=15`, `alpha=5-10` (lower due to 1-6 confidence range vs binary)
- **BERT initialization**: Project PhoBERT embeddings (768-dim) to item factors (64-dim) via SVD; prevents "random drift" for sparse items
- **Grid search**: 54 configs covering factors={32,64,128} × reg={0.05,0.1,0.15} × alpha={5,10,20}
### BPR (Bayesian Personalized Ranking)
- **Input**: `user_pos_train.pkl` (positive sets) + `user_hard_neg_train.pkl` (hard negatives)
- **Negative sampling**: 30% hard (explicit rating ≤3 + implicit popular items not bought) + 70% random unseen
- **Hyperparameters**: `factors=64`, `learning_rate=0.05`, `regularization=0.0001`, `epochs=50`, `samples_per_epoch=5×num_positives`
- **Custom implementation** (no library): SGD updates with BPR loss, early stopping on validation Recall@10
### Artifacts to Save
- `artifacts/cf/{als|bpr}/{model_type}_{U|V}.npy` (user/item embeddings)
- `{model_type}_params.json` (hyperparams, training time)
- `{model_type}_metrics.json` (Recall@K, NDCG@K vs popularity baseline)
- `{model_type}_metadata.json` (data_hash, git commit, **score_range** for normalization)

## Serving Architecture (Task 05 - Planned)
### Dual-Path Routing
```
Request (user_id) → Load user_metadata.pkl → Check is_trainable_user
├─ True (≥2 interactions, ~8.6% traffic)
│  ├─ CF Recommender: U[u_idx] @ V.T → scores
│  ├─ Filter seen items (from user_history_cache)
│  ├─ Hybrid rerank: w_cf=0.3, w_content=0.4, w_popularity=0.2, w_quality=0.1
│  └─ Return personalized top-K
└─ False (<2 interactions, ~91.4% traffic)
   ├─ Fallback: PhoBERT item-item similarity to user history (if exists)
   ├─ Mix with popularity (Top-50 items by num_sold_time)
   ├─ Rerank: w_content=0.6, w_popularity=0.3, w_quality=0.1
   └─ Return content-based top-K
```
### Key Classes (in `service/recommender/`)
- `CFModelLoader` (loader.py): Singleton loading models from registry, hot-reload on updates
- `CFRecommender` (recommender.py): Core logic with `recommend(user_id, topk, exclude_seen, filter_params)` and `batch_recommend(user_ids)`
- `PhoBERTEmbeddingLoader` (phobert_loader.py): Load/cache BERT embeddings, compute user profiles from history
- Fallback strategies (fallback.py): `_fallback_item_similarity()` (content-based) and `_fallback_popularity()` for new users
### API Endpoints (FastAPI in `service/api.py`)
- `POST /recommend`: Single user, returns `{user_id, recommendations[], count, fallback}`
- `POST /batch_recommend`: Multiple users efficiently
- `POST /reload_model`: Hot-reload from registry without downtime
- `GET /health`: Check service status, current model_id

## Standard Workflows & Commands
### Data Pipeline (Full Refresh)
```powershell
# Run complete Task 01 pipeline (7 steps: validate → engineer → filter → map → split → build → save)
python scripts/run_task01_complete.py
# Output: 11 files in data/processed/ (Parquet, NPZ, JSON, PKL)
```
### Training Models (Planned)
```powershell
# Train ALS with grid search
python scripts/train_cf.py --model als --config config/als_config.yaml --output artifacts/cf/als

# Train BPR
python scripts/train_cf.py --model bpr --config config/bpr_config.yaml --output artifacts/cf/bpr

# Parallel training (both models)
python scripts/train_both_models.py --auto-select  # Auto-registers best model to registry
```
### Serving (Planned)
```powershell
# Start FastAPI service (4 workers, auto-reload)
uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 4 --reload

# Test endpoint
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{"user_id": 12345, "topk": 10}'
```
### Evaluation (Planned)
```powershell
# Evaluate single model
python scripts/evaluate_models.py --model-id als_20250115_v1 --metrics recall ndcg --k-values 10 20

# Compare ALS vs BPR vs baselines
python scripts/evaluate_hybrid.py --output reports/cf_eval_summary.csv
```

## Vietnamese Text Handling
- **Encoding**: Always `encoding='utf-8'` for pandas `read_csv()`, file I/O
- **PhoBERT tokenizer**: Use `vinai/phobert-base` from Hugging Face; requires `sentencepiece` library
- **Text preprocessing**: Concatenate multiple fields with `[SEP]` token: `"Tên: {product_name} [SEP] Thành phần: {ingredient} [SEP] Công dụng: {feature}..."`
- **Comment quality keywords**: Positive signals include "thấm nhanh", "hiệu quả", "thơm", "mịn", "sáng da" (see Task 01 Step 2.0)

## Code Quality & Testing Conventions
- **Type hints**: All public functions must have typed parameters and return types (e.g., `def load_model(model_id: Optional[str] = None) -> Dict[str, Any]`)
- **Docstrings**: Use Google-style with Args/Returns/Raises sections
- **Logging**: Use `logging` module (not print); structured logs with `logger.info(f"user_id={uid}, latency={lat:.3f}s, fallback={is_fallback}")`
- **Testing**: Unit tests per module in `tests/`; test scripts like `test_step1_complete.py` validate end-to-end pipelines
- **Errors to avoid**:
  - Never use placeholder dates for missing timestamps (data leakage risk)
  - Never impute ratings outside [1.0, 5.0] range
  - Never save artifacts without embedding data_hash/git_commit
  - Never train CF on users with <2 interactions (waste of compute, noisy results)

## File Naming & Directory Patterns
- **Scripts**: `run_task01_*.py` (orchestration), `test_step*_*.py` (validation), `train_*.py` (ML training), `evaluate_*.py` (metrics)
- **Configs**: YAML in `config/` (e.g., `data_config.yaml`, `training_config.yaml`, `serving_config.yaml`)
- **Artifacts**: Timestamped subdirs like `artifacts/cf/als/20250115_v1/` containing `.npy`, `.json`, `.pkl` files
- **Logs**: `logs/cf/als.log`, `logs/service/api.log` with rotation (keep last 10 runs)
- **Reports**: `reports/cf_eval_summary.csv`, `reports/drift_analysis/*.png`

## Performance Benchmarks & Targets
- **Data processing**: <1 min for 369K interactions (Parquet load + CSR build)
- **ALS training**: 1-2 min for 26K users × 2.2K items (15 iterations, CPU)
- **BPR training**: 20-30 min for 50 epochs (1.8M triplets/epoch, CPU)
- **CF serving latency**: <100ms per user (scoring + filtering + top-K)
- **Content-based latency**: <200ms per user (BERT similarity + reranking) → **critical bottleneck** (91.4% traffic)
- **Expected metrics** (trainable users):
  - Popularity baseline: Recall@10 ~0.12-0.15
  - ALS: Recall@10 >0.20 (+40% vs baseline)
  - BPR: Recall@10 >0.22
  - Hybrid: Recall@10 ~0.28-0.35

## Common Pitfalls & Debugging Tips
1. **"User not in mappings"**: Cold-start user → should trigger fallback, not error
2. **Matrix shape mismatch**: Verify `num_users`/`num_items` from `user_item_mappings.json` metadata match CSR dimensions
3. **Memory errors in ALS**: Reduce `factors` or enable GPU (`use_gpu=True`)
4. **BPR loss oscillates**: Lower learning rate, add decay schedule
5. **Low Recall@10**: Check if test set is positive-only (rating ≥4); verify seen-item filtering works
6. **Slow serving**: Pre-cache user histories, pre-compute V @ V.T for item-item similarity, use batch inference
7. **Vietnamese text garbled**: Ensure UTF-8 encoding everywhere (CSV read, JSON save, API responses)

## Integration Points & Dependencies
- **PhoBERT embeddings**: Generated via `scripts/generate_bert_embeddings.py` → stored in `data/processed/content_based_embeddings/product_embeddings.pt`
- **Attribute filtering**: Uses `data_product_attribute.csv` with standardized `skin_type_standardized` (list format: `['acne', 'oily']`)
- **Model registry**: JSON at `artifacts/cf/registry.json` with `current_best` pointer and historical `models[]` array
- **Monitoring DBs**: SQLite in `logs/training_metrics.db` and `logs/service_metrics.db` (schema TBD in Task 06)
- **External dependencies**: `implicit` (ALS), `torch`+`transformers` (PhoBERT), `fastapi`+`uvicorn` (serving), `scipy` (sparse matrices), `pandas`+`pyarrow` (Parquet)

## When Adding New Features
- **Extend data layer**: Add new class in `recsys/cf/data/processing/`, expose via `DataProcessor` method
- **New model type**: Create `recsys/cf/{model_name}.py`, follow artifact structure (U.npy, V.npy, params.json, metadata.json with score_range)
- **New API endpoint**: Add to `service/api.py` with Pydantic request/response models
- **New metric**: Add function to `recsys/cf/metrics.py`, update `evaluate_model()` to compute it
- **Config changes**: Update relevant YAML in `config/`, document in task markdown files
