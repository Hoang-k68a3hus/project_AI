# Kiến Trúc Hệ Thống Gợi Ý (ALS + BPR + PhoBERT)

## Tổng Quan

Hệ thống gợi ý hybrid hiện đại hóa kết hợp **Collaborative Filtering (ALS, BPR)** với **Content-Based Filtering (PhoBERT)** và **Sentiment Analysis**. Kiến trúc được tối ưu cho dữ liệu thưa (sparsity ~99.9%) và rating skew (95% 5-star) thông qua chiến lược phân khúc người dùng (User Segmentation) và làm giàu tín hiệu (Signal Enrichment).

## Sơ Đồ Luồng Dữ Liệu

```
Raw Data (CSV)
    ↓
Data Processing Layer
    ├─ Sentiment Analysis (Rating + Comment Quality)
    ├─ User Segmentation (Trainable vs Cold-Start)
    └─ BERT Embedding Generation
    ↓
Training Pipelines
    ├─ ALS (Confidence-Weighted)
    └─ BPR (Hard Negative Mining)
    ↓
Model Registry (Versioning: Models + Embeddings)
    ↓
Serving Layer (FastAPI)
    ├─ Routing: Trainable → CF | Cold-Start → Content-Based
    └─ Reranking: Hybrid Scores
    ↓
Monitoring & Logging
```

## Các Tầng Chính

### 1. Tầng Dữ Liệu (Data Layer) - Task 01
- **Sentiment-Enhanced Confidence**: `confidence_score` = Rating (1-5) + Comment Quality (0-1). Giúp phân biệt "bare 5-star" và "genuine 5-star".
- **User Segmentation**:
  - **Trainable Users** (≥2 interactions): Dùng cho CF training (~8.6% users).
  - **Cold-Start Users** (<2 interactions): Dùng Content-Based (~91.4% users).
- **Hard Negative Mining**: Kết hợp Explicit (Rating ≤3) và Implicit (Popular items not bought).
- **Artifacts**: Parquet files, CSR matrices, User/Item Mappings, Global Stats (cho normalization).

### 2. Tầng Huấn Luyện (Training Layer) - Task 02
- **ALS Pipeline**: Matrix Factorization trên `confidence_score`.
  - **BERT Initialization**: Khởi tạo item factors từ PhoBERT embeddings để hỗ trợ sparse items.
  - **Regularization**: Tăng cường cho users ít tương tác.
- **BPR Pipeline**: Pairwise Ranking với chiến lược sampling 30% Hard Negatives (Explicit + Implicit).
- **Artifacts**: User/Item Factors (U, V), Metadata (Score Ranges cho normalization).

### 3. Tầng Đánh Giá (Evaluation Layer) - Task 03
- **Core Metrics**: Recall@K, NDCG@K (Test set chỉ chứa positive interactions rating ≥4).
- **Hybrid Metrics**:
  - **Diversity**: Dựa trên cosine similarity của PhoBERT embeddings.
  - **Semantic Alignment**: Độ tương đồng giữa CF recommendations và user profile.
  - **Cold-Start Coverage**: Khả năng recommend items ít phổ biến.
- **Validation**: Leave-one-out temporal split.

### 4. Tầng Registry (Model Registry) - Task 04
- **Dual Registry**: Quản lý version của cả **CF Models** và **BERT Embeddings**.
- **Metadata**: Tracking data hash, git commit, hyperparameters, và score distribution (min/max/p99) để phục vụ normalization tại serving.
- **Selection**: Tự động chọn best model dựa trên NDCG@10.

### 5. Tầng Dịch Vụ (Serving Layer) - Task 05
- **Smart Routing**:
  - **Trainable Users**: Gọi CF Model (ALS/BPR) → Rerank.
  - **Cold-Start Users**: Gọi Fallback (PhoBERT Item-Item Similarity + Popularity).
- **Hybrid Reranking**: Kết hợp điểm số:
  `Score = w1*CF + w2*Content + w3*Popularity + w4*Quality`
- **Optimization**: Caching user history, pre-computed similarity matrices.

### 6. Tầng Giám Sát (Monitoring Layer) - Task 06
- **Data Drift**: Theo dõi phân phối rating và sentiment.
- **Embedding Drift**: Kiểm tra semantic shift của sản phẩm mới.
- **Operational Metrics**: Latency, Fallback Rate, Error Rate.

### 7. Tầng Tự Động Hóa (Automation Layer) - Task 07
- **Scheduler**: Airflow/Cron cho định kỳ Retrain và Data Refresh.
- **CI/CD**: Automated testing cho data validation và model performance.

## Cấu Trúc Thư Mục

```
project/
├── data/
│   ├── published_data/              # Raw CSVs
│   ├── processed/                   # Parquet, CSR, Mappings
│   │   ├── interactions.parquet
│   │   ├── user_metadata.pkl        # Segmentation info
│   │   └── data_stats.json          # Global stats
│   └── content_based_embeddings/    # PhoBERT artifacts
├── tasks/                           # Documentation
├── recsys/
│   ├── cf/
│   │   ├── data.py                  # Data processing & Segmentation
│   │   ├── als.py                   # ALS with BERT Init
│   │   ├── bpr.py                   # BPR with Hard Negatives
│   │   └── metrics.py               # Hybrid Metrics
│   └── bert/                        # Embedding generation
├── artifacts/
│   └── cf/
│       ├── registry.json            # Model & Embedding Registry
│   │   └── ...
├── service/
│   ├── recommender/
│   │   ├── loader.py                # Model & Mapping Loader
│   │   ├── recommender.py           # Core Logic
│   │   ├── fallback.py              # Cold-Start Strategies
│   │   └── rerank.py                # Hybrid Scoring
│   └── api.py                       # FastAPI
└── scripts/                         # Automation Scripts
```

## Workflow Chính

### Workflow 1: Data Pipeline & Training
1. **Ingest**: Load raw CSV, validate timestamps & ratings.
2. **Enrich**: Tính `confidence_score` (Rating + Comment Quality).
3. **Segment**: Gán cờ `is_trainable` (≥2 interactions).
4. **Embed**: Generate/Update PhoBERT embeddings cho products.
5. **Train**:
   - Run ALS (Confidence-weighted, BERT-init).
   - Run BPR (Hard Negative Sampling).
6. **Evaluate**: So sánh NDCG@10 trên tập test (Trainable users only).
7. **Register**: Lưu model tốt nhất kèm metadata (Score ranges).

### Workflow 2: Serving Recommendations
1. **Request**: Nhận `user_id`.
2. **Check Segment**: User thuộc nhóm Trainable hay Cold-Start?
3. **Branch A (Trainable)**:
   - Retrieve CF Candidates (ALS/BPR).
   - Filter seen items.
   - **Rerank**: Combine CF score với Content Similarity & Popularity.
4. **Branch B (Cold-Start)**:
   - **Item-Item Similarity**: Tìm items giống lịch sử mua (dùng PhoBERT).
   - **Popularity Fallback**: Nếu lịch sử trống, dùng Top items (theo `num_sold_time`).
5. **Response**: Trả về Top-K items kèm giải thích (e.g., "Similar to X", "Popular").

### Workflow 3: Hybrid Reranking Detail
1. **Normalize**: Chuẩn hóa các điểm số (CF, Popularity, Quality) về [0,1] dùng global stats từ Registry.
2. **Compute Content Score**: Cosine similarity giữa User Profile (PhoBERT) và Candidate Item.
3. **Weighted Sum**: Áp dụng công thức tổng hợp.
4. **Diversity Penalty** (Optional): Trừ điểm nếu items quá giống nhau.

## Nguyên Tắc Thiết Kế

### Quality over Quantity
- Chỉ train CF trên users có đủ dữ liệu (≥2 interactions).
- Dùng Content-based để lấp đầy khoảng trống cho cold-start users.

### Semantic-Aware
- Tận dụng PhoBERT để hiểu ngữ nghĩa sản phẩm (Thành phần, Công dụng) thay vì chỉ dựa vào ID.
- Sentiment analysis giúp hiểu sâu hơn về rating.

### Reproducibility & Safety
- Versioning chặt chẽ từ Data → Embeddings → Model.
- Fallback mechanisms đảm bảo luôn có recommendations.

## Integration Points

### 1. PhoBERT System
- Cung cấp embeddings cho: BERT Initialization (Training), Content-Based Fallback (Serving), và Semantic Metrics (Evaluation).

### 2. Attribute System
- Dữ liệu thuộc tính (`data_product_attribute.csv`) dùng cho Hard Filtering và làm giàu text cho BERT.

### 3. Feedback Loop
- Logs từ Serving được đưa lại vào Data Pipeline để retrain và detect drift.

## Next Steps

Xem chi tiết implementation tại:
- `01_data_layer.md`: Xử lý dữ liệu, Sentiment, Segmentation.
- `02_training_pipelines.md`: ALS/BPR implementation details.
- `03_evaluation_metrics.md`: Hybrid metrics & Testing.
- `04_model_registry.md`: Quản lý version.
- `05_serving_layer.md`: Logic routing và fallback.
