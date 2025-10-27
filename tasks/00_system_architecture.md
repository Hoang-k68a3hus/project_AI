# Kiến Trúc Hệ Thống Gợi Ý (ALS + BPR)

## Tổng Quan

Hệ thống gợi ý hybrid kết hợp Collaborative Filtering (ALS, BPR) với khả năng mở rộng rerank bằng PhoBERT embeddings và thuộc tính sản phẩm. Kiến trúc được thiết kế theo nguyên tắc modular, hỗ trợ versioning, monitoring và tự động hóa.

## Sơ Đồ Luồng Dữ Liệu

```
Raw Data (CSV)
    ↓
Data Processing Layer (tiền xử lý, normalization)
    ↓
Training Pipelines (ALS / BPR)
    ↓
Model Registry (versioning, tracking)
    ↓
Serving Layer (API, rerank, fallback)
    ↓
Monitoring & Logging
    ↓
Scheduler (retrain, update)
```

## Các Tầng Chính

### 1. Tầng Dữ Liệu (Data Layer)
- **Nguồn dữ liệu raw**: CSV files trong `model/data/published_data/`
- **Preprocessing pipeline**: Chuẩn hóa, deduplicate, filter
- **Storage**: Parquet files cho hiệu năng, JSON cho mappings
- **Versioning**: Hash và timestamp theo dõi phiên bản dữ liệu

### 2. Tầng Huấn Luyện (Training Layer)
- **Shared preprocessing**: Module `recsys/cf/data.py`
- **ALS pipeline**: Implicit matrix factorization
- **BPR pipeline**: Bayesian personalized ranking
- **Hyperparameter tuning**: Grid search với tracking
- **Artifacts management**: Lưu trữ có tổ chức theo model type

### 3. Tầng Đánh Giá (Evaluation Layer)
- **Metrics**: Recall@K, NDCG@K
- **Baseline comparison**: Popularity-based ranking
- **Validation strategy**: Leave-one-out temporal split
- **Reporting**: CSV summaries, visualization

### 4. Tầng Registry (Model Registry)
- **Best model tracking**: Theo NDCG@10
- **Version control**: Hash, timestamp, hyperparameters
- **Rollback support**: Giữ lịch sử các phiên bản
- **Metadata**: Performance metrics, data provenance

### 5. Tầng Dịch Vụ (Serving Layer)
- **Model loader**: Nạp model và mappings
- **Core recommender**: Top-K generation với filtering
- **Fallback logic**: Popularity cho cold-start users
- **Reranking**: Hybrid scoring với PhoBERT/attributes
- **API endpoint**: REST API cho integration

### 6. Tầng Giám Sát (Monitoring Layer)
- **Training logs**: Theo dõi quá trình huấn luyện
- **Service logs**: Request, latency, errors
- **Drift detection**: So sánh phân phối dữ liệu
- **Alert system**: Trigger retrain khi cần

### 7. Tầng Tự Động Hóa (Automation Layer)
- **Training scripts**: Chạy pipelines theo config
- **Registry updater**: Cập nhật best model
- **Scheduler**: Cron/Airflow jobs
- **CI/CD**: Testing và deployment

## Cấu Trúc Thư Mục

```
project/
├── model/
│   ├── data/
│   │   ├── published_data/          # Raw CSV files
│   │   ├── processed/               # Parquet, mappings
│   │   └── cache/                   # Temporary files
│   ├── tasks/                       # Task documentation (folder này)
│   └── phobert_recommendation.py    # Existing PhoBERT system
├── recsys/
│   └── cf/
│       ├── data.py                  # Shared preprocessing
│       ├── metrics.py               # Evaluation metrics
│       ├── als.py                   # ALS implementation
│       └── bpr.py                   # BPR implementation
├── artifacts/
│   └── cf/
│       ├── als/                     # ALS models, metrics
│       ├── bpr/                     # BPR models, metrics
│       └── registry.json            # Best model pointer
├── service/
│   ├── recommender/
│   │   ├── loader.py                # Model loading
│   │   ├── recommender.py           # Core recommendation logic
│   │   ├── rerank.py                # Hybrid reranking
│   │   └── api.py                   # REST API
│   └── config/
│       └── serving_config.yaml      # Serving configurations
├── scripts/
│   ├── train_cf.py                  # Training orchestration
│   ├── update_registry.py           # Registry management
│   └── evaluate_models.py           # Batch evaluation
├── logs/
│   ├── cf/                          # Training logs
│   └── service/                     # Service logs
└── reports/
    ├── cf_eval_summary.csv          # Performance summaries
    └── drift_analysis/              # Data drift reports
```

## Workflow Chính

### Workflow 1: Training & Evaluation
1. Load raw CSV → preprocessing → save parquet
2. Build CSR matrix, user-item mappings
3. Train ALS và BPR pipelines (parallel hoặc sequential)
4. Evaluate với metrics chuẩn
5. So sánh với baseline popularity
6. Update registry nếu performance tốt hơn
7. Save artifacts với versioning

### Workflow 2: Serving Recommendations
1. Load best model từ registry
2. Nhận request với `user_id`, `topk`, `exclude_seen`
3. Map user_id → u_idx
4. Generate scores = U[u] · V^T
5. Mask seen items (nếu exclude_seen=True)
6. Apply reranking nếu enabled
7. Return top-K product_ids
8. Log request và latency

### Workflow 3: Cold-Start Handling
1. Check user_id trong mappings
2. Nếu không tồn tại → fallback popularity
3. Popularity source: `num_sold_time` hoặc train frequency
4. Filter theo thuộc tính nếu có (brand, skin_type)
5. Return top-K popular items

### Workflow 4: Hybrid Reranking
1. Get CF scores từ ALS/BPR
2. Load PhoBERT embeddings cho top-K items
3. Compute content similarity với user profile
4. Load popularity signals (num_sold_time, avg_star)
5. Combine scores: α*CF + β*content + γ*popularity
6. Re-sort và return final top-K

### Workflow 5: Monitoring & Retraining
1. Scheduler chạy định kỳ (daily/weekly)
2. Load new interactions → preprocess
3. Detect data drift (distribution shift)
4. Trigger retrain nếu drift > threshold
5. Run evaluation pipeline
6. Update registry nếu model mới tốt hơn
7. Send alert/report

## Nguyên Tắc Thiết Kế

### Modularity
- Mỗi component độc lập, có interface rõ ràng
- Shared preprocessing giữa ALS và BPR
- Reusable metrics module

### Versioning
- Mọi artifact có version identifier
- Track data hash/timestamp
- Keep history cho rollback

### Monitoring
- Log mọi training run và serving request
- Track metrics theo thời gian
- Alert khi performance degradation

### Scalability
- Parquet cho large datasets
- CSR matrix cho sparse data
- Optional GPU support cho ALS

### Reproducibility
- Save random seeds
- Version control cho code và config
- Document data provenance

## Integration Points

### 1. PhoBERT System
- Reuse embeddings từ `content_based_embeddings/`
- Hybrid reranking kết hợp CF + semantic similarity
- Fallback cho cold-start items

### 2. Attribute Filtering
- Load từ `data_product_attribute.csv`
- Pre-filter theo brand, skin_type, ingredient
- Enhance diversity trong recommendations

### 3. Popularity Signals
- `num_sold_time` từ `data_product.csv`
- `avg_star` cho quality signal
- Boost popular items cho exploration

### 4. API Layer
- REST endpoints cho web/mobile integration
- Batch recommendation API
- Real-time single-user API

## Next Steps

Xem các file tasks chi tiết:
- `01_data_layer.md` - Tầng dữ liệu và preprocessing
- `02_training_pipelines.md` - ALS và BPR training
- `03_evaluation_metrics.md` - Đánh giá và benchmarking
- `04_model_registry.md` - Quản lý phiên bản model
- `05_serving_layer.md` - Dịch vụ gợi ý
- `06_monitoring_logging.md` - Giám sát và logging
- `07_automation_scheduling.md` - Tự động hóa và scheduling
- `08_hybrid_reranking.md` - Rerank và hybrid strategies
