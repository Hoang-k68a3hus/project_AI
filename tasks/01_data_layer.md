# Task 01: Tầng Dữ Liệu (Data Layer)

## Mục Tiêu

Xây dựng pipeline xử lý dữ liệu ổn định, có khả năng tái tạo (reproducible) và hiệu năng cao cho hệ thống CF. Pipeline này sẽ chuyển đổi raw CSV thành các định dạng tối ưu cho training và serving, tận dụng **Explicit Feedback (Rating)** và **Rich Metadata** để tạo đầu vào chất lượng cao cho ALS, BPR và PhoBERT.

## 🔄 Key Strategy Changes (Updated November 2025)

### Data Challenges Addressed:
1. **High Sparsity**: ~1.23 interactions/user (369K reviews / 300K users)
   - Most users are one-time buyers → CF has minimal overlap to learn from
   - **Trainable Users**: ~26,000 users with ≥2 interactions (~8.6% of total)
   - **Matrix Density**: ~0.11% for CF training (26k×2.2k with ~65k interactions)
2. **Rating Skew**: ~95% ratings are 5-star → Loss of discriminative power
   - Can't distinguish "truly loved" vs "just okay" products

### 1. Sentiment-Based Confidence Weighting
- **Problem**: 5-star ratings lack nuance when everyone gives 5 stars
- **Solution**: Enhance confidence scores using review comment quality
  - Base confidence = rating value (1-5)
  - **+0.2**: Comment length >10 words (thoughtful feedback)
  - **+0.3**: Contains positive keywords ("thấm nhanh", "hiệu quả", "thơm")
  - **+0.5**: Includes images (if data available)
  - **Result**: `confidence_score` = rating + quality_bonus (max ~6.0)
- **Usage**: ALS uses `confidence_score` instead of raw ratings

### 2. User Segmentation Strategy (UPDATED - Lowered to ≥2)
- **Trainable Users** (≥2 interactions):
  - Use for CF training (ALS/BPR) - minimum data for collaborative patterns
  - **~26,000 users (~8.6% of total)** - sufficient statistical base with BERT support
  - **Critical**: ALS must use BERT initialization + higher regularization (λ=0.1) to anchor sparse vectors
- **Cold-Start Users** (1 interaction or new users):
  - ~90% of users - skip CF training for these
  - Serve with content-based (PhoBERT item similarity) + popularity
- **Rationale**: Balance data hunger vs quality; BERT embeddings compensate for sparsity

### 3. Hard Negative Mining for BPR
- **Strategy 1**: Low ratings (≤3) as explicit hard negatives (if available)
- **Strategy 2**: Implicit hard negatives from popularity
  - Sample from Top-50 popular items user DIDN'T buy
  - Logic: "This product is hot, but you didn't buy it → you don't like it"
  - More informative than random negatives

### 4. Content-First Hybrid Approach
- **Shift**: Increase content-based weight (PhoBERT) relative to CF
  - For sparse data, semantic similarity more reliable than collaborative patterns
  - Recommended weights: `w_content=0.4`, `w_cf=0.3`, `w_popularity=0.2`, `w_quality=0.1`

### Legacy Strategy (Preserved for Reference):
- **ALS**: Sử dụng rating values (1-5) trực tiếp làm confidence scores thay vì binary matrix
- **BPR**: Hard Negative Mining - Tận dụng low ratings (≤3) làm negative examples thay vì chỉ random sampling
- **Test Set**: Chỉ chứa positive interactions (rating ≥4) để đo khả năng recommend items user sẽ thích



## Input Data Sources

### Raw CSV Files
Tất cả nằm trong `data/published_data/`:

1. **data_reviews_purchase.csv**
   - Columns: `user_id`, `product_id`, `rating`, `comment`, `cmt_date`
   - Rows: ~369K interactions
   - Issues: Trùng lặp user-item, missing timestamps, inconsistent types

2. **data_product.csv**
   - Columns: `product_id`, `product_name`, `brand`, `type`, `price`, `avg_star`, `num_sold_time`, `processed_description`
   - Rows: 2,244 products
   - Usage: Popularity baseline, metadata enrichment

3. **data_product_attribute.csv**
   - Columns: `product_id`, `ingredient`, `feature`, `skin_type`, `capacity`, `design`, `brand`, `expiry`, `origin`
   - Rows: 2,244 products
   - Usage: Attribute-based filtering, reranking signals

4. **data_shop.csv**
   - Columns: Shop metadata
   - Usage: Optional shop-level features (future)

## Preprocessing Steps

### Step 1: Data Validation & Cleaning

#### 1.1 Load và Audit
- **Encoding**: Đọc CSV với `encoding='utf-8'` (Vietnamese characters)
- **Type enforcement**:
  - `user_id`, `product_id`: int
  - `rating`: float (validate range 1.0-5.0)
  - `cmt_date`: parse thành datetime (format: DD/MM/YYYY hoặc auto-detect)
- **Missing values & "Time Travel" Fix**:
  - Drop rows với missing `user_id`, `product_id`, `rating`
  - **CRITICAL**: Drop rows với `cmt_date` = NaT/Null (KHÔNG điền placeholder)
    - Lý do: Tránh data leakage khi chia train/test - mô hình không được "nhìn thấy tương lai"
    - Log: Số lượng rows bị drop do missing timestamp
- **Rating validation**:
  - Chỉ giữ rows với `rating` trong khoảng [1.0, 5.0]
  - Drop hoàn toàn các giá trị ngoài range (không impute)
  - Log: Số lượng invalid ratings removed

#### 1.2 Deduplication
- **Rule**: Mỗi (user_id, product_id) chỉ giữ 1 interaction
- **Strategy**: Giữ interaction có `cmt_date` mới nhất
- **Fallback**: Nếu `cmt_date` trùng → giữ rating cao nhất
- **Log**: Số lượng duplicates removed

#### 1.3 Outlier Detection
- **User activity**: Identify users với >500 interactions (potential bots/scrapers)
- **Item popularity**: Flag items với <3 interactions (very cold items)
- **Rating distribution**: Check for rating bias (e.g., >90% ratings = 5)
- **Action**: Log outliers, quyết định filter sau

**✅ UPDATED**: `DataValidator` must enforce strict temporal validation and rating range checks

### Step 2: Explicit Feedback Feature Engineering

#### 2.0 Comment Quality Analysis (NEW - Addresses Rating Skew)
- **Problem**: 95% ratings are 5-star → need additional signal to distinguish quality
- **Solution**: Analyze `comment` column to compute quality bonus
  
```python
def compute_comment_quality_score(comment_text):
    """
    Compute quality bonus based on review comment.
    
    Returns: float [0.0, 1.0] quality score
    """
    if pd.isna(comment_text) or len(comment_text.strip()) == 0:
        return 0.0
    
    score = 0.0
    text_lower = comment_text.lower()
    
    # Length bonus (thoughtful reviews)
    word_count = len(comment_text.split())
    if word_count >= 10:
        score += 0.2
    elif word_count >= 5:
        score += 0.1
    
    # Positive keyword bonus
    positive_keywords = [
        'thấm nhanh', 'hiệu quả', 'thơm', 'mịn', 'sáng da',
        'trắng da', 'giảm mụn', 'không kích ứng', 'tốt',
        'rất thích', 'đáng mua', 'chất lượng', 'xuất sắc'
    ]
    keyword_matches = sum(1 for kw in positive_keywords if kw in text_lower)
    score += min(keyword_matches * 0.1, 0.3)  # Max 0.3 from keywords
    
    # Image bonus (if available in data)
    # if has_images:  # Implement if image flag exists
    #     score += 0.5
    
    return min(score, 1.0)  # Cap at 1.0

# Apply to all interactions
interactions_df['comment_quality'] = interactions_df['comment'].apply(compute_comment_quality_score)
interactions_df['confidence_score'] = interactions_df['rating'] + interactions_df['comment_quality']
# Result: confidence_score range [1.0, 6.0]
```

#### 2.1 ALS: Confidence-Weighted Matrix
- **Paradigm Shift**: Sử dụng Explicit Feedback với sentiment-based weighting
- **Matrix values**: `confidence_score` = rating + comment_quality (range 1.0-6.0)
  - Rating 5 + quality 0.0 → Confidence 5.0 (bare 5-star, suspicious)
  - Rating 5 + quality 1.0 → Confidence 6.0 (genuine 5-star with thoughtful review)
  - Rating 3 + quality 0.5 → Confidence 3.5 (mediocre but detailed feedback)
- **Rationale**: Distinguish "truly loved" products from "just okay" despite rating skew
- **Alternative**: Normalize to [0,1] → `normalized_conf = (confidence - 1) / 5`

#### 2.2 BPR: Positive Labels với Hard Negative Mining
- **Positive Signal Definition**: 
  - `rating >= 4` → Positive interaction (User thích sản phẩm)
  - Store in `is_positive` column (0/1)
  
- **Hard Negative Mining (UPDATED for Sparsity)**: 
  - **Strategy 1**: `rating <= 3` → Explicit hard negative (User đã mua nhưng thất vọng)
  - **Strategy 2**: Implicit hard negatives from popularity (NEW)
    - Identify Top-50 most popular items (by `num_sold_time`)
    - For each user, find popular items they DIDN'T interact with
    - Logic: "Hot product but you didn't buy → implicit negative preference"
  - Store both in `is_hard_negative` column with source flag
  
- **Sampling Strategy (for BPR training)**:
  - Positive samples: Items với `is_positive=1`
  - Negative samples: 30% hard negatives (explicit + implicit) + 70% random unseen
  - Rationale: Combat sparsity with popularity-informed negatives

#### 2.3 User Filtering (UPDATED - Lowered to ≥2)
- **Segment users by interaction count**:
  - **Trainable users**: ≥2 interactions **AND** ≥1 positive (rating ≥4)
    - **~26,000 users (~8.6% coverage)** with matrix density ~0.11%
    - These users provide minimal collaborative signal but BERT init compensates
    - Mark with `is_trainable_user = True`
  - **Cold-start users**: 1 interaction or no positives
    - Majority of users (~90%), but insufficient for CF
    - Mark with `is_trainable_user = False`
    - Will be served via content-based + popularity

- **Filtering for CF training**:
  - Train ALS/BPR only on `is_trainable_user = True`
  - **Special case**: User with exactly 2 interactions where both are negative (rating <4) → Force to cold-start
  - Log stats:
    - Trainable users: ~26,000 (~8.6% of total)
    - Cold-start users: ~274,000 (~91.4% of total)
    - Interactions from trainable users: ~65,000 (matrix density ~0.11%)

- **Iterative filtering**: Still apply min item interactions (≥5 positives) after user filtering

### Step 3: ID Mapping (Contiguous Indexing)

#### 3.1 User Mapping
- **Original**: `user_id` (sparse integers, gaps, range 1-304708)
- **Mapped**: `u_idx` (contiguous 0 to num_users-1)
- **Dict structure**: 
  ```
  {
    "user_to_idx": {original_id: idx, ...},
    "idx_to_user": {idx: original_id, ...}
  }
  ```

#### 3.2 Item Mapping
- **Original**: `product_id` (range 0-2243)
- **Mapped**: `i_idx` (contiguous 0 to num_items-1)
- **Dict structure**: Tương tự user mapping

#### 3.3 Apply Mapping
- Add columns `u_idx`, `i_idx` vào interactions DataFrame
- Validate: Không có missing mappings

#### 3.4 Save Mappings
- **Format**: JSON (human-readable, dễ debug)
- **Location**: `data/processed/user_item_mappings.json`
- **Include metadata**:
  - Timestamp tạo mappings
  - Số lượng users/items
  - Hash của raw data

### Step 4: Temporal Split (Leave-One-Out)

#### 4.1 Sort Per User
- Group interactions theo `u_idx`
- Sort mỗi group theo `cmt_date` (ascending)
- Handle ties: Nếu `cmt_date` trùng → sort theo `rating` desc

#### 4.2 Train/Test Split với Positive-Only Test
- **Train**: Tất cả interactions trừ latest positive
- **Test**: Latest **POSITIVE** interaction per user
  - **CRITICAL RULE**: Chỉ chọn tương tác cuối cùng làm test NẾU `rating >= 4`
  - Nếu latest interaction có rating < 4 → Lấy latest positive interaction (rating ≥4) trước đó
  - Rationale: Test set đo lường khả năng recommend items user sẽ **thích**, không phải items user sẽ ghét
- **Validation**: Optional - lấy 2nd latest positive làm val, remaining positives làm train

#### 4.3 Edge Cases (UPDATED for ≥2 Threshold)
- **Users with exactly 2 interactions**:
  - Both positive (≥4): Keep 1 for train, 1 for test → Valid trainable user
  - 1 positive, 1 negative: Keep positive for train, negative excluded → Train-only user (no test)
  - Both negative (<4): Force to cold-start (`is_trainable_user = False`)
- **Users with 1 positive**: No test data → Skip user in evaluation OR use as train-only
- **Users with 0 positives**: Already removed in Step 2.3
- **Users with latest interaction negative**: Take previous positive for test (if exists)

#### 4.4 Create Datasets
- `train_interactions`: DataFrame với columns [u_idx, i_idx, rating, is_positive, is_hard_negative, timestamp]
- `test_interactions`: Chỉ chứa positive interactions (rating ≥4)
- `val_interactions`: Optional, cũng chỉ chứa positives

### Step 5: Matrix Construction

#### 5.1 Dual CSR Matrices
- **X_train_confidence** (for ALS): 
  - Shape: (num_trainable_users, num_items)
  - Values: `confidence_score` (rating + comment_quality, range 1.0-6.0)
  - **Only includes trainable users** (≥3 interactions)
  - Library: `scipy.sparse.csr_matrix`
  - Usage: ALS training với sentiment-enhanced confidence weighting
  
- **X_train_binary** (for BPR - optional):
  - Shape: (num_trainable_users, num_items)
  - Values: Binary (1 for positive interactions only, 0 elsewhere)
  - Usage: BPR pairwise ranking

#### 5.2 User Positive Sets
- **Structure**: Dict `user_pos_train[u_idx] = set(i_idx, ...)`
- **Scope**: Only trainable users
- **Usage**: 
  - Negative sampling trong BPR (exclude positives)
  - Filtering seen items khi generate recommendations
  - Fast lookup O(1)

#### 5.3 Hard Negative Sets (UPDATED)
- **Structure**: Dict `user_hard_neg_train[u_idx] = {"explicit": set(...), "implicit": set(...)}`
- **Content**: 
  - `explicit`: Items với rating ≤3 (user đã mua nhưng thất vọng)
  - `implicit`: Popular items (Top-50) user DIDN'T buy
- **Usage**: 
  - BPR training: Sample 30% from combined hard negatives
  - Evaluation: Optional exclude from candidate pool
  - Analysis: Understand failure modes

#### 5.4 Item Popularity with Top-K Tracking
- **Count**: Số lần mỗi item xuất hiện trong train
- **Log-transform**: Apply `log(1 + count)`
- **Top-K popular items**: Store indices of Top-50 most popular items
  - Usage: Generate implicit hard negatives for cold-start users
  - Format: `top_k_popular_items = [i_idx1, i_idx2, ..., i_idx50]`

#### 5.5 User Segmentation Metadata
- **Structure**: Dict với user statistics
  ```python
  user_metadata = {
      "trainable_users": set(u_idx for users with ≥3 interactions),
      "cold_start_users": set(u_idx for users with 1-2 interactions),
      "user_interaction_counts": {u_idx: count, ...}
  }
  ```
- **Usage**: 
  - Serving layer decides CF vs content-based routing
  - Monitoring CF coverage

### Step 6: Save Processed Data

#### 6.1 Parquet Format
- **File**: `data/processed/interactions.parquet`
- **Content**: Full DataFrame với columns:
  - `user_id`, `product_id`, `u_idx`, `i_idx`
  - `rating`, `comment_quality`, `confidence_score`
  - `is_positive`, `is_hard_negative`, `timestamp`
  - `is_trainable_user` (NEW - flag for CF training eligibility)
  - `split` (train/val/test)
- **Advantages**: 
  - 10x faster read/write vs CSV
  - Type preservation
  - Compression (~50% size reduction)

#### 6.2 Mappings JSON
- **File**: `data/processed/user_item_mappings.json`
- **Structure**:
  ```json
  {
    "metadata": {
      "created_at": "2025-01-15T10:30:00",
      "num_users": 12000,
      "num_items": 2200,
      "data_hash": "abc123...",
      "positive_threshold": 4,
      "hard_negative_threshold": 3
    },
    "user_to_idx": {...},
    "idx_to_user": {...},
    "item_to_idx": {...},
    "idx_to_item": {...}
  }
  ```

#### 6.3 Matrix Files
- **X_train_confidence.npz**: CSR matrix với confidence scores (for ALS) - trainable users only
- **X_train_binary.npz**: CSR matrix với binary values (for BPR - optional) - trainable users only
- **user_pos_train.pkl**: Pickle dict với positive item sets (trainable users)
- **user_hard_neg_train.pkl**: Pickle dict với hard negative item sets (explicit + implicit)
- **item_popularity.npy**: NumPy array với log-transformed popularity scores
- **top_k_popular_items.json**: List of Top-50 popular item indices
- **user_metadata.pkl**: User segmentation data (trainable vs cold-start)

#### 6.4 Statistics Summary (UPDATED - Add Global Normalization Ranges)
- **File**: `data/processed/data_stats.json`
- **Content**:
  - Train/val/test sizes
  - Sparsity: nonzeros / (users * items)
  - Rating distribution (mean, std, quantiles per split)
  - Positive vs Hard negative counts
  - User/item interaction histograms (quantiles)
  - Filtered counts (users, items, interactions)
  - **NEW - Global Normalization Ranges** (Critical for Task 08):
    - `popularity`: {"min": X, "max": Y, "p01": A, "p99": B}
    - `quality`: {"min": 1.0, "max": 5.0, "mean": C, "std": D}
    - `confidence_score`: {"min": 1.0, "max": 6.0, "p01": E, "p99": F}
  - **Purpose**: Enable global normalization in hybrid reranking to prevent per-request bias

**Example structure**:
```json
{
  "train_size": 350000,
  "test_size": 15000,
  "sparsity": 0.0012,
  "trainable_users": {
    "count": 26000,
    "percentage": 8.6,
    "avg_interactions_per_user": 2.5,
    "matrix_density": 0.0011
  },
  "popularity": {
    "min": 0.0,
    "max": 9.21,
    "mean": 2.45,
    "std": 1.83,
    "p01": 0.0,
    "p50": 2.1,
    "p99": 7.8
  },
  "quality": {
    "min": 1.0,
    "max": 5.0,
    "mean": 4.67,
    "std": 0.52
  },
  "confidence_score": {
    "min": 1.0,
    "max": 6.0,
    "mean": 5.12,
    "std": 0.68,
    "p01": 3.2,
    "p99": 6.0
  }
}
```

### Step 7: Data Versioning

#### 7.1 Hash Calculation
- **Method**: MD5 hash của raw CSV files (sorted concatenation)
- **Purpose**: Track data changes, invalidate stale models
- **Storage**: In mappings JSON and model artifacts

#### 7.2 Timestamp Tracking
- **Creation time**: Khi nào data được processed
- **Usage**: Detect stale data, schedule retraining

#### 7.3 Version Registry
- **File**: `data/processed/versions.json`
- **Structure**:
  ```json
  {
    "v1": {
      "hash": "abc123",
      "timestamp": "2025-01-15T10:30:00",
      "filters": {"min_user_pos": 2, "min_item_pos": 5},
      "files": ["interactions.parquet", "mappings.json", ...]
    }
  }
  ```

## Output Artifacts

### Primary Files
1. `data/processed/interactions.parquet` - Full interaction data với `confidence_score`, `is_trainable_user`
2. `data/processed/user_item_mappings.json` - ID mappings với metadata (positive/hard_neg thresholds)
3. `data/processed/X_train_confidence.npz` - Sparse CSR matrix với confidence scores (for ALS, trainable users only)
4. `data/processed/X_train_binary.npz` - Sparse CSR matrix với binary values (for BPR - optional, trainable users)
5. `data/processed/user_pos_train.pkl` - User positive sets (trainable users)
6. `data/processed/user_hard_neg_train.pkl` - User hard negative sets (explicit + implicit)
7. `data/processed/item_popularity.npy` - Log-transformed popularity distribution
8. `data/processed/top_k_popular_items.json` - Top-50 popular items for implicit negatives
9. `data/processed/user_metadata.pkl` - User segmentation (trainable vs cold-start)

### Metadata Files
10. `data/processed/data_stats.json` - Statistics summary (includes trainable user %, sparsity, comment quality distribution)
11. `data/processed/versions.json` - Version tracking
12. `data/processed/preprocessing.log` - Processing logs

### Content Enrichment Files
13. `data/processed/product_attributes_enriched.parquet` - Standardized attributes + auxiliary signals
14. `data/processed/content_based_embeddings/product_embeddings.pt` - PhoBERT embeddings với rich text
15. `data/processed/content_based_embeddings/embedding_metadata.json` - Embedding version info

## Quality Checks

### Validation Tests
- [ ] No missing values trong key columns (u_idx, i_idx, rating)
- [ ] No NaT/Null timestamps (strict temporal validation)
- [ ] All ratings trong range [1.0, 5.0]
- [ ] u_idx range = [0, num_users-1], i_idx range = [0, num_items-1]
- [ ] CSR matrices shape matches (num_users, num_items)
- [ ] user_pos_train keys = all u_idx in train với positives
- [ ] user_hard_neg_train keys = subset of u_idx với hard negatives
- [ ] Test set: 1 positive interaction per user (hoặc 0 nếu user filtered)
- [ ] No data leakage: Test timestamps > Train timestamps per user
- [ ] Test set only contains positives (rating ≥4)
- [ ] Mappings reversible: user_to_idx → idx_to_user round-trip OK
- [ ] PhoBERT embeddings align với product_id trong mappings
- [ ] skin_type_standardized contains valid list values
- [ ] popularity_score và quality_score không có NaN

### Performance Benchmarks
- [ ] Parquet load time: <5 seconds cho 369K rows
- [ ] CSR matrix construction: <2 seconds
- [ ] Mapping lookup: O(1) constant time
- [ ] PhoBERT encoding: <10 minutes cho 2244 products

## Configuration Example

```yaml
# data_config.yaml
raw_data:
  base_path: "data/published_data"
  interactions: "data_reviews_purchase.csv"
  products: "data_product.csv"
  attributes: "attribute_based_embeddings/attribute_text_filtering.csv"

preprocessing:
  # Explicit feedback thresholds
  positive_threshold: 4  # rating >= 4 → positive
  hard_negative_threshold: 3  # rating <= 3 → hard negative
  
  # Filtering (UPDATED - Lowered to 2 for trainable users)
  min_user_interactions: 2  # Minimum total interactions for trainable user
  min_user_positives: 1  # Must have at least 1 positive (rating ≥4)
  min_item_positives: 5  # Items must have ≥5 positive interactions
  dedup_strategy: "keep_latest"  # or "keep_highest_rating"
  
  # Temporal validation
  drop_missing_timestamps: true  # CRITICAL: No placeholder dates
  validate_rating_range: true  # Enforce [1.0, 5.0]

temporal_split:
  method: "leave_one_out"  # or "leave_k_out", "timestamp"
  test_positive_only: true  # Only use positive interactions for test
  validation: false  # Enable val set?

matrix_construction:
  als_matrix: "ratings"  # Use rating values as confidence
  bpr_matrix: "binary"  # Binary matrix for BPR (optional)
  hard_negative_sampling_ratio: 0.3  # 30% hard neg, 70% random neg

content_enrichment:
  enable_bert: true
  bert_model: "vinai/phobert-base"
  bert_input_fields: ["product_name", "ingredient", "feature", "skin_type", "brand", "processed_description"]
  standardize_skin_type: true
  compute_auxiliary_signals: true
  log_transform_popularity: true

output:
  processed_path: "data/processed"
  format: "parquet"  # or "csv"
  save_ratings_matrix: true  # X_train_ratings.npz
  save_binary_matrix: false  # X_train_binary.npz (optional)
  save_hard_negatives: true  # user_hard_neg_train.pkl
  save_stats: true
  save_embeddings: true
  
versioning:
  enable: true
  hash_method: "md5"
```

## Module Interface

### Module: `recsys/cf/data.py`

#### Function 1: `load_raw_data(config)`
- **Input**: Config dict/object
- **Output**: Dict với keys: interactions_df, products_df, attributes_df
- **Purpose**: Load và validate raw CSVs với strict temporal validation

#### Function 2: `preprocess_interactions(df, config)`
- **Input**: Raw interactions DataFrame, config
- **Output**: Cleaned DataFrame (dedup, typed, filtered, no NaT timestamps)
- **Steps**: Validation, deduplication, filtering, rating range check
- **CRITICAL**: Drop rows với NaT timestamps, không impute

#### Function 3: `create_explicit_features(df, positive_threshold=4, hard_neg_threshold=3)`
- **Input**: Interactions DataFrame, thresholds
- **Output**: DataFrame với `is_positive`, `is_hard_negative` columns
- **Purpose**: Label explicit feedback cho ALS và BPR hard negative mining

#### Function 4: `create_mappings(df, user_col, item_col)`
- **Input**: DataFrame, column names
- **Output**: Dict với user/item mappings và metadata (thresholds, counts)
- **Purpose**: Contiguous ID mapping

#### Function 5: `temporal_split(df, method='leave_one_out', positive_only_test=True)`
- **Input**: Interactions DataFrame với timestamps, config
- **Output**: train_df, test_df, (val_df optional)
- **Purpose**: Split data theo temporal logic, test set chỉ chứa positives
- **CRITICAL**: Test = latest positive interaction (rating ≥4) per user

#### Function 6: `build_csr_matrix(df, num_users, num_items, value_col='rating')`
- **Input**: Interactions với u_idx/i_idx, dimensions, value column
- **Output**: scipy.sparse.csr_matrix
- **Purpose**: Efficient sparse matrix construction (ratings or binary)

#### Function 7: `build_user_pos_sets(df)`
- **Input**: Train interactions với `is_positive=1`
- **Output**: Dict[u_idx, Set[i_idx]]
- **Purpose**: Fast positive item lookup per user

#### Function 8: `build_user_hard_neg_sets(df)`
- **Input**: Train interactions với `is_hard_negative=1`
- **Output**: Dict[u_idx, Set[i_idx]]
- **Purpose**: Hard negative sets cho BPR training

#### Function 9: `save_processed_data(artifacts, output_path)`
- **Input**: Dict với all artifacts (df, matrices, mappings, stats, hard_negs)
- **Output**: None (save to disk)
- **Purpose**: Persist processed data với versioning

#### Function 10: `load_processed_data(version=None)`
- **Input**: Optional version identifier
- **Output**: Dict với loaded artifacts
- **Purpose**: Load processed data cho training/serving

### Module: `recsys/content/metadata.py`

#### Function 1: `standardize_skin_type(raw_text)`
- **Input**: Raw skin type text
- **Output**: List of standardized skin type tags
- **Purpose**: Chuẩn hóa skin_type cho hard filtering

#### Function 2: `compute_auxiliary_signals(attributes_df, reviews_df)`
- **Input**: Attributes và reviews DataFrames
- **Output**: Enriched attributes với popularity_score, quality_score
- **Purpose**: Prepare signals cho hybrid reranking

#### Function 3: `create_bert_input_text(products_df, attributes_df, fields)`
- **Input**: Product và attribute data, list of fields
- **Output**: DataFrame với `bert_input_text` column
- **Purpose**: Create rich text inputs cho PhoBERT encoding

## Component 8: BERT/PhoBERT Embeddings Pipeline

### Purpose
Tích hợp BERT embeddings vào data layer để hỗ trợ hybrid reranking và content-based fallback.

### Step 0: Metadata Standardization & Auxiliary Signals

#### 0.1 Standardize skin_type for Hard Filtering
```python
def standardize_skin_type(raw_text):
    """
    Chuẩn hóa skin_type từ text tự do sang danh sách chuẩn.
    
    Input: "Da mụn trứng cá, Da hỗn hợp..."
    Output: ['acne', 'combination']
    """
    skin_type_mapping = {
        'mụn': 'acne',
        'trứng cá': 'acne',
        'hỗn hợp': 'combination',
        'dầu': 'oily',
        'khô': 'dry',
        'nhạy cảm': 'sensitive',
        'thường': 'normal',
        'mọi loại': 'all'
    }
    
    if pd.isna(raw_text):
        return ['all']
    
    raw_lower = raw_text.lower()
    detected_types = []
    
    for keyword, standard_type in skin_type_mapping.items():
        if keyword in raw_lower:
            detected_types.append(standard_type)
    
    return detected_types if detected_types else ['all']

# Apply standardization
attributes_df['skin_type_standardized'] = attributes_df['skin_type'].apply(standardize_skin_type)
```

#### 0.2 Prepare Auxiliary Signals for Reranking
```python
# Popularity signal với log-transform
attributes_df['popularity_score'] = np.log1p(attributes_df['num_sold_time'].fillna(0))

# Quality signal
# Option 1: Từ attribute file
attributes_df['quality_score'] = attributes_df['is_5_star'].fillna(0)

# Option 2: Tính từ review data
product_quality = reviews_df.groupby('product_id')['rating'].agg([
    ('avg_rating', 'mean'),
    ('num_ratings', 'count'),
    ('pct_5star', lambda x: (x == 5).sum() / len(x))
]).reset_index()

# Merge back
attributes_df = attributes_df.merge(product_quality, on='product_id', how='left')

# Save enriched attributes
attributes_df.to_parquet('data/processed/product_attributes_enriched.parquet')
```

**Output Artifacts**:
- `data/processed/product_attributes_enriched.parquet` với columns:
  - `product_id`, `ingredient`, `feature`, `skin_type_standardized`
  - `popularity_score` (log-transformed)
  - `quality_score`, `avg_rating`, `pct_5star`
  - `price`, `brand`, `origin`, etc.

### Step 1: Extract Product Descriptions

#### Load Rich Product Text Data
```python
# Load product metadata
products_df = pd.read_csv('data/published_data/data_product.csv', encoding='utf-8')
attributes_df = pd.read_csv('data/published_data/attribute_based_embeddings/attribute_text_filtering.csv', encoding='utf-8')

# Merge để có full context
products_enriched = products_df.merge(attributes_df, on='product_id', how='left')

# Create "Super Text" for PhoBERT với Vietnamese context
products_enriched['bert_input_text'] = (
    'Tên: ' + products_enriched['product_name'] + ' [SEP] ' +
    'Thành phần: ' + products_enriched['ingredient'].fillna('Không rõ') + ' [SEP] ' +
    'Công dụng: ' + products_enriched['feature'].fillna('Không rõ') + ' [SEP] ' +
    'Loại da phù hợp: ' + products_enriched['skin_type'].fillna('Mọi loại da') + ' [SEP] ' +
    'Thương hiệu: ' + products_enriched['brand'].fillna('') + ' [SEP] ' +
    'Mô tả: ' + products_enriched['processed_description'].fillna('')
)
```

**Rationale**: 
- Nối nhiều trường metadata tạo ngữ cảnh phong phú
- PhoBERT sẽ học được semantic similarity sâu (ví dụ: sản phẩm khác brand nhưng cùng BHA + Da dầu → vector gần nhau)
- Token `[SEP]` giúp model phân biệt các trường thông tin

### Step 2: Generate BERT Embeddings

#### Module: `recsys/bert/embedding_generator.py`

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

class BERTEmbeddingGenerator:
    """
    Generate BERT/PhoBERT embeddings cho product descriptions.
    """
    
    def __init__(self, model_name='vinai/phobert-base', device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
    
    def encode_texts(self, texts, batch_size=32, max_length=256):
        """
        Encode texts thành embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size cho encoding
            max_length: Max token length
        
        Returns:
            np.array: (len(texts), hidden_dim) embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size)):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**encoded)
                
                # Mean pooling over sequence
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def save_embeddings(self, embeddings, product_ids, output_path):
        """
        Save embeddings với metadata.
        
        Args:
            embeddings: np.array (N, D)
            product_ids: List of product IDs
            output_path: Path to save .pt file
        """
        torch.save({
            'embeddings': torch.from_numpy(embeddings),
            'product_ids': product_ids,
            'model_name': self.tokenizer.name_or_path,
            'embedding_dim': embeddings.shape[1],
            'num_products': len(product_ids),
            'created_at': datetime.now().isoformat()
        }, output_path)
```

### Step 3: Embedding Generation Workflow

#### Script: `scripts/generate_bert_embeddings.py`
```python
"""
Generate BERT embeddings cho all products với rich metadata.

Usage:
    python scripts/generate_bert_embeddings.py \
        --model vinai/phobert-base \
        --output data/processed/content_based_embeddings/
"""

import argparse
import pandas as pd
from recsys.bert.embedding_generator import BERTEmbeddingGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vinai/phobert-base')
    parser.add_argument('--output', default='data/processed/content_based_embeddings/')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    # Load products with rich metadata
    products = pd.read_csv('data/published_data/data_product.csv', encoding='utf-8')
    attributes = pd.read_csv('data/published_data/attribute_based_embeddings/attribute_text_filtering.csv', encoding='utf-8')
    
    # Merge and create super text
    products_enriched = products.merge(attributes, on='product_id', how='left')
    products_enriched['bert_input_text'] = (
        'Tên: ' + products_enriched['product_name'] + ' [SEP] ' +
        'Thành phần: ' + products_enriched['ingredient'].fillna('Không rõ') + ' [SEP] ' +
        'Công dụng: ' + products_enriched['feature'].fillna('Không rõ') + ' [SEP] ' +
        'Loại da phù hợp: ' + products_enriched['skin_type'].fillna('Mọi loại da') + ' [SEP] ' +
        'Thương hiệu: ' + products_enriched['brand'].fillna('') + ' [SEP] ' +
        'Mô tả: ' + products_enriched['processed_description'].fillna('')
    )
    
    # Generate embeddings
    generator = BERTEmbeddingGenerator(model_name=args.model)
    embeddings = generator.encode_texts(
        products_enriched['bert_input_text'].tolist(),
        batch_size=args.batch_size
    )
    
    # Save
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, 'product_embeddings.pt')
    generator.save_embeddings(
        embeddings,
        products_enriched['product_id'].tolist(),
        output_file
    )
    
    print(f"Saved {len(embeddings)} embeddings to {output_file}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

if __name__ == '__main__':
    main()
```

### Step 4: User Profile Embeddings

#### Strategy 1: Interaction-Weighted Average
```python
def compute_user_profile_embedding(user_history_items, item_embeddings, item_to_idx):
    """
    Compute user profile bằng weighted average của item embeddings.
    
    Args:
        user_history_items: List[(product_id, weight)]
        item_embeddings: np.array (num_items, dim)
        item_to_idx: Dict mapping product_id -> idx
    
    Returns:
        np.array: (dim,) user profile embedding
    """
    history_embeddings = []
    weights = []
    
    for product_id, weight in user_history_items:
        if product_id in item_to_idx:
            idx = item_to_idx[product_id]
            history_embeddings.append(item_embeddings[idx])
            weights.append(weight)
    
    if not history_embeddings:
        # Fallback: zero embedding hoặc mean embedding
        return np.zeros(item_embeddings.shape[1])
    
    # Weighted average
    history_embeddings = np.array(history_embeddings)
    weights = np.array(weights).reshape(-1, 1)
    weights = weights / weights.sum()  # Normalize
    
    user_profile = (history_embeddings * weights).sum(axis=0)
    return user_profile
```

#### Strategy 2: TF-IDF Weighted
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_user_tfidf_profile(user_history_texts, item_embeddings, product_ids):
    """
    Compute user profile bằng TF-IDF weighted item embeddings.
    """
    # Compute TF-IDF weights
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(user_history_texts)
    
    # Weight embeddings by TF-IDF importance
    # ... (implementation)
```

### Step 5: Embedding Versioning

#### Metadata File: `data/processed/content_based_embeddings/embedding_metadata.json`
```json
{
  "version": "v1_20250115_103000",
  "model_name": "vinai/phobert-base",
  "embedding_dim": 768,
  "num_products": 2244,
  "created_at": "2025-01-15T10:30:00",
  "data_hash": "abc123...",
  "git_commit": "def456...",
  "files": {
    "product_embeddings": "product_embeddings.pt",
    "user_profiles": "user_profile_embeddings.pt"
  }
}
```

### Step 6: Sync with CF Data

#### Validation: Check Alignment
```python
def validate_embedding_alignment(mappings_path, embeddings_path):
    """
    Validate BERT embeddings align với CF item mappings.
    """
    # Load mappings
    with open(mappings_path) as f:
        mappings = json.load(f)
    
    # Load embeddings
    embeddings_data = torch.load(embeddings_path)
    
    cf_product_ids = set(mappings['item_to_idx'].keys())
    bert_product_ids = set(str(pid) for pid in embeddings_data['product_ids'])
    
    # Check coverage
    missing_in_bert = cf_product_ids - bert_product_ids
    extra_in_bert = bert_product_ids - cf_product_ids
    
    print(f"CF products: {len(cf_product_ids)}")
    print(f"BERT products: {len(bert_product_ids)}")
    print(f"Missing in BERT: {len(missing_in_bert)}")
    print(f"Extra in BERT: {len(extra_in_bert)}")
    
    if missing_in_bert:
        warnings.warn(f"Warning: {len(missing_in_bert)} products have CF embeddings but no BERT embeddings")
```

## Dependencies

```python
# requirements_data.txt
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
pyarrow>=10.0.0  # For parquet

# BERT dependencies
torch>=1.13.0
transformers>=4.25.0
sentencepiece>=0.1.96  # For PhoBERT tokenizer
```

## Error Handling

### Common Errors
1. **CSV encoding error** → Enforce UTF-8, log problematic rows
2. **Missing columns** → Raise clear error với expected schema
3. **Type conversion failure** → Log rows, fill with defaults hoặc drop
4. **Mapping collision** → Should not happen với unique IDs, validate
5. **Empty DataFrame after filtering** → Log warning, adjust thresholds

### Logging Strategy
- **Level INFO**: Summary stats (rows processed, filtered)
- **Level WARNING**: Outliers, missing data
- **Level ERROR**: Critical failures (corrupt file, schema mismatch)
- **Output**: `logs/data_processing.log` với rotation

## Monitoring Metrics

### Data Quality Metrics
- **Sparsity**: % of nonzero cells trong matrix
- **Coverage**: % users/items còn lại sau filtering
- **Rating distribution**: Mean, std, quantiles
- **Temporal spread**: Min/max timestamps, gaps

### Drift Detection (For Retraining)
- **Distribution shift**: KL divergence của rating distribution
- **Popularity shift**: Spearman correlation của item ranks
- **User growth**: % new users vs existing
- **Trigger**: Retrain nếu shift > threshold

## Timeline Estimate

- **Implementation**: 2-3 days
- **Testing & Validation**: 1 day
- **Documentation**: 0.5 day
- **Total**: ~4 days

## Success Criteria

- [ ] Pipeline chạy end-to-end không errors
- [ ] Output artifacts pass all quality checks
- [ ] Processing time <1 minute cho 369K interactions
- [ ] Reproducible: Same input → same output (với fixed random seed)
- [ ] Documented: Clear README và inline comments
- [ ] **NEW**: No NaT timestamps trong processed data
- [ ] **NEW**: Test set chỉ chứa positive interactions (rating ≥4)
- [ ] **NEW**: Rating matrix (X_train_ratings) có values trong [1.0, 5.0]
- [ ] **NEW**: Hard negative sets coverage ≥50% of users
- [ ] **NEW**: PhoBERT embeddings coverage 100% products
- [ ] **NEW**: skin_type_standardized chứa valid list values
