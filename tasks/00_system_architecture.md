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

- **Nguồn dữ liệu raw**: CSV files trong `data/published_data/`
- **Preprocessing pipeline**: Chuẩn hóa, deduplicate, filter
- **Storage**: Parquet files cho hiệu năng, JSON cho mappings, PhoBERT embeddings (`data/processed/content_based_embeddings/product_embeddings.pt`)
- **Versioning**: Hash và timestamp theo dõi phiên bản dữ liệu và embedding metadata

### 2. Tầng Huấn Luyện (Training Layer)
- **Shared preprocessing**: Module `recsys/cf/data.py`
- **ALS pipeline**: Implicit matrix factorization (có hỗ trợ khởi tạo từ PhoBERT)
- **BPR pipeline**: Bayesian personalized ranking (tùy chọn fine-tune với content loss)
- **PhoBERT projection**: Giảm chiều 768→latent (TruncatedSVD) trong `recsys/bert`
- **Hyperparameter tuning**: Grid search với tracking
- **Artifacts management**: Lưu trữ có tổ chức theo model type và biến thể content-aware

### 3. Tầng Đánh Giá (Evaluation Layer)
- **Metrics**: Recall@K, NDCG@K, diversity (BERT cosine), semantic alignment, cold-start coverage
- **Baseline comparison**: Popularity-based ranking và CF thuần
- **Validation strategy**: Leave-one-out temporal split, cold-start holdout
- **Reporting**: CSV summaries, visualization, hybrid vs CF comparison dashboards

### 4. Tầng Registry (Model Registry)
- **Best model tracking**: Theo NDCG@10 và hybrid metrics
- **Version control**: Hash, timestamp, hyperparameters, embedding version
- **Rollback support**: Giữ lịch sử các phiên bản
- **Metadata**: Performance metrics, data provenance, compatibility checks CF ↔ PhoBERT

### 5. Tầng Dịch Vụ (Serving Layer)
- **Model loader**: Nạp model và mappings
- **Core recommender**: Top-K generation với filtering
- **PhoBERTEmbeddingLoader**: LRU cache, user profile aggregation
- **Fallback logic**: Popularity và content-based cho cold-start users
- **Reranking**: Hybrid scoring với PhoBERT/attributes
- **API endpoint**: REST API cho integration

### 6. Tầng Giám Sát (Monitoring Layer)
- **Training logs**: Theo dõi quá trình huấn luyện
- **Service logs**: Request, latency, errors
- **Drift detection**: So sánh phân phối dữ liệu và semantic drift embedding
- **Embedding freshness**: Kiểm tra tuổi đời PhoBERT embeddings, alert khi quá hạn
- **Alert system**: Trigger retrain/refetch khi cần

### 7. Tầng Tự Động Hóa (Automation Layer)
- **Training scripts**: Chạy pipelines theo config
- **Registry updater**: Cập nhật best model
- **BERT refresh**: `scripts/refresh_bert_embeddings.py` chạy hằng tuần (cron/Airflow)
- **Scheduler**: Cron/Airflow jobs (data refresh, drift detection, embeddings, training)
- **CI/CD**: Testing và deployment

## Cấu Trúc Thư Mục

```
project/
├── data/
│   ├── published_data/              # Raw CSV files
│   ├── processed/                   # Parquet, mappings, CF matrices
│   ├── processed_test/              # Evaluation subsets
│   └── content_based_embeddings/    # PhoBERT artifacts (product_embeddings.pt, metadata)
├── tasks/                           # Task documentation
├── model/
│   └── phobert_recommendation.py    # Legacy PhoBERT prototype
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
3. Generate PhoBERT embeddings & project vào latent space (nếu cần)
4. Train ALS và BPR pipelines (parallel hoặc sequential, optional BERT init)
5. Evaluate với metrics chuẩn + hybrid metrics
6. So sánh với baseline popularity và CF thuần
7. Update registry nếu performance tốt hơn (ghi nhận embedding version)
8. Save artifacts với versioning

### Workflow 2: Serving Recommendations
1. Load best model từ registry
2. Nhận request với `user_id`, `topk`, `exclude_seen`
3. Map user_id → u_idx
4. Load PhoBERT embeddings (theo version registry) → compute user profile
5. Generate scores = U[u] · V^T
6. Mask seen items (nếu exclude_seen=True)
7. Apply reranking (CF + PhoBERT + signals)
8. Return top-K product_ids
9. Log request, latency, embedding version

### Workflow 3: Cold-Start Handling
1. Check user_id trong mappings
2. Nếu không tồn tại → fallback popularity
3. Popularity source: `num_sold_time` hoặc train frequency
4. PhoBERT content-based recommendations dựa trên browsing/session
5. Filter theo thuộc tính nếu có (brand, skin_type)
6. Return top-K popular/content items

### Workflow 4: Hybrid Reranking
1. Get CF scores từ ALS/BPR
2. PhoBERTEmbeddingLoader tạo user profile (weighted_mean, tf_idf)
3. Compute content similarity với top-K items
4. Load popularity signals (num_sold_time, avg_star) và attribute match
5. Combine scores: α*CF + β*content + γ*popularity + δ*quality ± diversity penalty
6. Re-sort và return final top-K, log signals

### Workflow 5: Monitoring & Retraining
1. Scheduler chạy định kỳ (daily/weekly)
2. Load new interactions → preprocess
3. Detect data drift (distribution shift) và semantic drift (embeddings)
4. Check embedding freshness → trigger `refresh_bert_embeddings.py` nếu quá hạn
5. Trigger retrain nếu drift > threshold hoặc embeddings mới
6. Run evaluation pipeline (CF vs hybrid)
7. Update registry nếu model mới tốt hơn, ghi nhận embedding version
8. Send alert/report

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
