# ALS + BPR Recommendation System - Implementation Tasks

## Tổng Quan

Thư mục này chứa tài liệu chi tiết cho việc xây dựng hệ thống gợi ý Collaborative Filtering (ALS + BPR) cho e-commerce mỹ phẩm Việt Nam. Hệ thống được thiết kế theo kiến trúc production-ready với đầy đủ tính năng: training, evaluation, serving, monitoring, và automation.

## Cấu Trúc Tasks

### [00 - System Architecture](./00_system_architecture.md)
**Mục tiêu**: Overview tổng quan về kiến trúc hệ thống

**Nội dung**:
- Sơ đồ luồng dữ liệu end-to-end
- 7 tầng chính: Data, Training, Evaluation, Registry, Serving, Monitoring, Automation
- Cấu trúc thư mục dự án
- Integration points với PhoBERT và attributes
- Nguyên tắc thiết kế (modularity, versioning, monitoring)

**Timeline**: Không cần implement (chỉ documentation)

---

### [01 - Data Layer](./01_data_layer.md)
**Mục tiêu**: Xây dựng pipeline xử lý dữ liệu ổn định và reproducible

**Nội dung**:
- Load và validate raw CSV (reviews, products, attributes)
- Preprocessing: deduplication, type enforcement, filtering
- Positive signal definition (rating ≥ 4)
- ID mapping (user/item → contiguous indices)
- Temporal split (leave-one-out)
- CSR matrix construction
- Save processed data (Parquet, NPZ, JSON)
- Data versioning với hash tracking

**Module**: `recsys/cf/data.py`

**Outputs**:
- `data/processed/interactions.parquet`
- `data/processed/user_item_mappings.json`
- `data/processed/X_train.npz`
- `data/processed/user_pos_train.pkl`

**Timeline**: ~4 days

---

### [02 - Training Pipelines](./02_training_pipelines.md)
**Mục tiêu**: Implement ALS và BPR training pipelines

**Nội dung**:
- **ALS Pipeline**:
  - Confidence matrix construction (C = 1 + α*X)
  - Train với `implicit` library
  - Hyperparameter tuning (factors, reg, alpha, iterations)
- **BPR Pipeline**:
  - Negative sampling (uniform/popularity-biased)
  - SGD updates với BPR loss
  - Early stopping với validation
- Shared preprocessing từ Task 01
- Save artifacts (U, V, params, metrics)
- Grid search cho hyperparameters

**Modules**: `recsys/cf/als.py`, `recsys/cf/bpr.py`

**Scripts**: `scripts/train_cf.py`

**Timeline**: ~6-7 days

---

### [03 - Evaluation & Metrics](./03_evaluation_metrics.md)
**Mục tiêu**: Đánh giá models với metrics chuẩn RecSys

**Nội dung**:
- **Metrics**: Recall@K, NDCG@K, Precision@K, MRR, MAP@K, Coverage
- **Baseline comparisons**: Popularity (num_sold_time, train frequency)
- **Statistical testing**: Paired t-test, Cohen's d
- **Visualization**: Bar charts, K-value sensitivity, coverage vs accuracy
- **Error analysis**: Cold-start, niche items, per-user stratification

**Module**: `recsys/cf/metrics.py`

**Outputs**: `reports/cf_eval_summary.csv`

**Timeline**: ~3 days

---

### [04 - Model Registry](./04_model_registry.md)
**Mục tiêu**: Quản lý phiên bản models và select best cho production

**Nội dung**:
- **Registry schema**: JSON với model metadata, metrics, hyperparameters
- **Operations**: Register, select best, load, list, archive, delete
- **Versioning**: Data hash, git commit, timestamp tracking
- **A/B testing support**: Canary deployment, traffic split
- **Rollback mechanism**: Restore previous best model
- **Audit trail**: Log mọi registry changes

**Module**: `recsys/cf/registry.py`

**File**: `artifacts/cf/registry.json`

**Scripts**: `scripts/update_registry.py`, `scripts/cleanup_old_models.py`

**Timeline**: ~3 days

---

### [05 - Serving Layer](./05_serving_layer.md)
**Mục tiêu**: Production serving với API, fallback, và reranking

**Nội dung**:
- **Model Loader**: Load từ registry, hot-reload khi update
- **Core Recommender**: Scoring, filtering, top-K generation
- **Cold-start fallback**: Popularity-based hoặc PhoBERT search
- **Attribute filtering**: Brand, skin_type, ingredient filters
- **API endpoints**: FastAPI với /recommend, /batch_recommend, /reload_model
- **Performance optimization**: Batch inference, caching

**Modules**: 
- `service/recommender/loader.py`
- `service/recommender/recommender.py`
- `service/recommender/fallback.py`
- `service/api.py`

**Config**: `service/config/serving_config.yaml`

**Timeline**: ~6 days

---

### [06 - Monitoring & Logging](./06_monitoring_logging.md)
**Mục tiêu**: Theo dõi training, service health, và data drift

**Nội dung**:
- **Training monitoring**: Logs, metrics DB (SQLite), progress tracking
- **Service monitoring**: Request logs, latency, fallback rate, error rate
- **Data drift detection**: Rating distribution (KS test), popularity shift (Spearman)
- **Alerting**: Email/Slack alerts cho critical issues
- **Dashboard**: Streamlit dashboard cho metrics visualization
- **Retrain trigger**: Automatic retrain khi detect drift

**Modules**: `recsys/cf/logging_utils.py`

**Databases**:
- `logs/training_metrics.db`
- `logs/service_metrics.db`

**Scripts**: `service/dashboard.py`, `scripts/detect_drift.py`

**Timeline**: ~6 days

---

### [07 - Automation & Scheduling](./07_automation_scheduling.md)
**Mục tiêu**: Tự động hóa toàn bộ ML pipeline

**Nội dung**:
- **Orchestration scripts**:
  - Data refresh (daily)
  - Model training (weekly)
  - Deployment (automated)
  - Health checks (hourly)
- **Scheduling**: Cron jobs (Linux), Task Scheduler (Windows), Airflow DAG
- **Error handling**: Retry logic với exponential backoff
- **Pipeline monitoring**: Run tracker DB, status dashboard
- **Cleanup**: Log rotation, old model archival

**Scripts**:
- `scripts/refresh_data.py`
- `scripts/train_both_models.py`
- `scripts/deploy_model_update.py`
- `scripts/health_check.py`
- `scripts/cleanup_logs.py`

**Airflow**: `airflow/dags/cf_pipeline_dag.py` (optional)

**Timeline**: ~6.5 days

---

### [08 - Hybrid Reranking](./08_hybrid_reranking.md)
**Mục tiêu**: Kết hợp CF với PhoBERT embeddings và product attributes

**Nội dung**:
- **PhoBERT integration**: Load embeddings, compute similarity
- **Hybrid reranker**: Combine CF + content + popularity + quality
- **Weighted scoring**: Configurable weights cho mỗi signal
- **Diversity penalty**: Giảm similar items trong top-K
- **Attribute filtering**: Brand, skin_type filtering
- **Attribute boosting**: Boost specific brands/categories
- **Evaluation**: Diversity metrics, category coverage

**Modules**:
- `service/recommender/phobert_loader.py`
- `service/recommender/rerank.py`
- `service/recommender/filters.py`

**Config**: `service/config/rerank_config.yaml`

**Timeline**: ~6.5 days

---

### [09 - Smart Search](./09_smart_search.md)
**Mục tiêu**: Tích hợp tính năng tìm kiếm thông minh sử dụng PhoBERT embeddings

**Nội dung**:
- **Query Encoder**: Encode text queries thành embeddings với PhoBERT
- **Search Index**: FAISS/exact search cho semantic similarity
- **Smart Search Service**: Text-to-product discovery
- **Similar Items Search**: Item-item similarity
- **User Profile Search**: Recommendations từ browsing history
- **Reranking**: Multi-signal ranking (semantic, popularity, quality)
- **API Endpoints**: /search, /search/similar, /search/profile

**Modules**:
- `service/search/query_encoder.py`
- `service/search/search_index.py`
- `service/search/smart_search.py`

**API Endpoints**:
- `POST /search`: Text query search
- `POST /search/similar`: Item-item search
- `POST /search/profile`: User profile-based search

**Config**: `service/config/search_config.yaml`

**Timeline**: ~7 days

---

## Thứ Tự Triển Khai Khuyến Nghị

### Phase 1: Foundation (Tuần 1-2)
1. **Task 01 - Data Layer** (4 days)
   - Critical foundation cho tất cả tasks khác
   - Test với full 369K interactions dataset
2. **Task 02 - Training Pipelines** (6-7 days)
   - Implement ALS trước (nhanh hơn)
   - BPR sau (có thể parallel nếu có resources)
3. **Task 03 - Evaluation** (3 days)
   - Cần có ngay để verify training pipelines
   - Benchmark với popularity baseline

### Phase 2: Production Infrastructure (Tuần 3-4)
4. **Task 04 - Model Registry** (3 days)
   - Quản lý models từ Phase 1
   - Setup versioning từ đầu
5. **Task 05 - Serving Layer** (6 days)
   - API cho integration
   - Test với load testing tools

### Phase 3: Operations (Tuần 5)
6. **Task 06 - Monitoring & Logging** (6 days)
   - Critical cho production readiness
   - Setup alerting sớm
7. **Task 07 - Automation** (6.5 days)
   - Có thể parallel với Task 06
   - Start với manual scripts, automate sau

### Phase 4: Enhancements (Tuần 6)
8. **Task 08 - Hybrid Reranking** (6.5 days)
   - Optional enhancement
   - Có thể implement từng phần (content first, diversity sau)

### Phase 5: Smart Features (Tuần 7)
9. **Task 09 - Smart Search** (7 days)
   - Semantic search với PhoBERT embeddings
   - Text query → product discovery
   - Similar items và user profile search
   - Tích hợp với embeddings đã có sẵn

**Tổng timeline**: ~7 tuần (với 1 người full-time)

## Integration với Hệ Thống Hiện Tại

### PhoBERT Recommendation System
**Location**: `model/phobert_recommendation.py`

**Reuse**:
- Embeddings từ `data/published_data/content_based_embeddings/`
- Product metadata từ `data_product.csv`, `data_product_attribute.csv`
- Evaluation logic (`PhoBERTEvaluator` class)

**Integration points**:
- Task 08: Load PhoBERT embeddings cho content similarity
- Task 05: Fallback tới PhoBERT cho cold-start users
- Hybrid reranker: Combine CF scores với PhoBERT cosine similarity

### Research Baseline (Elasticsearch)
**Location**: `origins/face_cleanser_recommendation_dataset/src/`

**Không reuse** (different approach):
- Elasticsearch BM25 text search → replaced by PostgreSQL FTS (Task 06 trong task/06_vector_fts_infrastructure.md)
- Attribute embeddings → replaced by explicit attribute filtering (Task 08)

**Preserve**:
- Dataset structure và preprocessing logic
- Evaluation metrics (Recall, NDCG, MRR)

## Configuration Management

### Global Config Structure
```
config/
├── data_config.yaml          # Task 01: Data processing settings
├── training_config.yaml      # Task 02: ALS/BPR hyperparameters
├── serving_config.yaml       # Task 05: API và fallback settings
├── rerank_config.yaml        # Task 08: Hybrid weights
└── alerts_config.yaml        # Task 06: Monitoring thresholds
```

### Environment Variables
```bash
# Database
export CF_METRICS_DB="logs/training_metrics.db"
export CF_SERVICE_DB="logs/service_metrics.db"

# Paths
export CF_DATA_PATH="data/processed"
export CF_ARTIFACTS_PATH="artifacts/cf"
export CF_REGISTRY_PATH="artifacts/cf/registry.json"

# API
export CF_API_HOST="0.0.0.0"
export CF_API_PORT="8000"

# Alerts
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export EMAIL_PASSWORD="your_app_password"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

## Testing Strategy

### Unit Tests
- [ ] Data preprocessing functions (`recsys/cf/data.py`)
- [ ] Metrics calculations (`recsys/cf/metrics.py`)
- [ ] Registry operations (`recsys/cf/registry.py`)
- [ ] Recommender logic (`service/recommender/recommender.py`)

### Integration Tests
- [ ] End-to-end training pipeline
- [ ] API endpoints (health, recommend, batch, reload)
- [ ] Reranking với multiple signals
- [ ] Drift detection logic

### Performance Tests
- [ ] Data processing speed (target: <1 min cho 369K interactions)
- [ ] ALS training time (target: <2 min cho 12K users)
- [ ] Serving latency (target: <100ms per user)
- [ ] Batch recommendation throughput (target: >100 users/sec)

### End-to-End Test
```bash
# 1. Process data
python scripts/refresh_data.py

# 2. Train both models
python scripts/train_both_models.py --auto-select

# 3. Start service
uvicorn service.api:app --reload

# 4. Test API
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 12345, "topk": 10}'

# 5. Check metrics
python scripts/evaluate_hybrid.py

# 6. Run health check
python scripts/health_check.py
```

## Dependencies

### Core Dependencies
```
# requirements_cf.txt
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.9.0
scikit-learn>=1.2.0
implicit>=0.6.0          # ALS
torch>=1.13.0            # PhoBERT embeddings
transformers>=4.25.0     # PhoBERT model
pyarrow>=10.0.0          # Parquet
```

### Serving Dependencies
```
fastapi>=0.95.0
uvicorn>=0.21.0
pydantic>=1.10.0
```

### Monitoring Dependencies
```
streamlit>=1.20.0        # Dashboard
plotly>=5.13.0           # Plots
apscheduler>=3.10.0      # Scheduling
```

### Development Dependencies
```
pytest>=7.2.0
black>=23.0.0            # Code formatting
flake8>=6.0.0            # Linting
```

## Troubleshooting

### Common Issues

**Issue**: Data preprocessing fails với UnicodeDecodeError
- **Solution**: Ensure `encoding='utf-8'` khi read CSV
- **Check**: Vietnamese characters display correctly

**Issue**: ALS training OOM (Out of Memory)
- **Solution**: Reduce `factors` hoặc subsample users
- **Alternative**: Use GPU với `use_gpu=True`

**Issue**: BPR training very slow
- **Solution**: Reduce `samples_per_epoch` multiplier (5 → 3)
- **Alternative**: Use vectorized sampling

**Issue**: Service returns fallback for all users
- **Solution**: Check mappings loaded correctly
- **Verify**: `user_id` trong mappings matches request

**Issue**: PhoBERT embeddings not found
- **Solution**: Check path `data/published_data/content_based_embeddings/`
- **Verify**: Files exist: `product_embeddings.pt`

## Contribution Guidelines

### Code Style
- Follow PEP 8
- Use type hints
- Docstrings cho all public functions
- Comments cho complex logic

### Commit Messages
```
[Task XX] Brief description

- Detailed change 1
- Detailed change 2

Fixes #issue_number
```

### Pull Request Template
```
## Task
Task XX - [Name]

## Changes
- [ ] Implementation complete
- [ ] Tests added
- [ ] Documentation updated

## Testing
Describe testing performed

## Performance
Latency/throughput benchmarks (if applicable)
```

## Roadmap

### Completed
- [x] Task specifications documented

### In Progress
- [ ] Task 01: Data Layer
- [ ] Task 02: Training Pipelines

### Planned
- [ ] Task 03-08: See phase timeline above

### Future Enhancements
- [ ] Deep Learning models (NCF, BERT4Rec)
- [ ] Real-time stream processing
- [ ] Multi-armed bandit exploration
- [ ] Conversational recommendations
- [ ] Explainable recommendations

## References

- **Project README**: `../README_PHOBERT.md`
- **Research baseline**: `../../origins/README.md`
- **Task roadmap**: `../../task/cf_tasks.md`
- **PhoBERT paper**: VinAI Research GitHub
- **Implicit library**: https://github.com/benfred/implicit
- **BPR paper**: Rendle et al., UAI 2009

## Contact

For questions về implementation:
- Check task markdown files trước
- Review code examples trong task specs
- Open issue với tag [Task XX]

---

**Last Updated**: 2025-01-27  
**Version**: 1.0  
**Status**: Documentation Complete, Implementation Pending
