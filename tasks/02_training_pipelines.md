# Task 02: Training Pipelines (ALS & BPR)

## Mục Tiêu

Xây dựng hai pipelines huấn luyện Collaborative Filtering song song: ALS (Alternating Least Squares) và BPR (Bayesian Personalized Ranking). Mỗi pipeline sẽ train, evaluate, và persist artifacts độc lập, sau đó được so sánh để chọn best model.

## 🔄 Updated Training Strategy (November 2025)

### Data Context: High Sparsity + Rating Skew
- **Sparsity**: ~1.23 interactions/user → Most users are one-time buyers
- **Rating Skew**: ~95% are 5-star → Loss of discriminative power
- **Challenge**: Traditional CF struggles with minimal user overlap

### Key Changes from Task 01 Data Layer Updates:

1. **User Segmentation**
   - Train CF only on **trainable users** (≥3 interactions)
   - Skip cold-start users (1-2 interactions) - serve them with content-based
   - Result: Higher quality training data, better model convergence

2. **ALS: Sentiment-Enhanced Confidence**
   - Input: Confidence matrix (rating + comment_quality, range 1.0-6.0)
   - Distinguishes "genuine 5-star" from "bare 5-star"
   - Lower alpha scaling (5-10) due to higher confidence range

3. **BPR: Dual Hard Negative Mining**
   - Explicit: Low ratings (≤3) when available
   - Implicit: Top-50 popular items user didn't buy
   - Sampling: 30% hard negatives (mixed) + 70% random unseen
   - Improved ranking for sparse data

4. **Test Set: Trainable Users Only**
   - Evaluate CF models only on trainable users
   - Cold-start users evaluated separately on content-based metrics
   - Fair comparison of CF effectiveness



## Pipeline Overview

```
Preprocessed Data (từ Task 01)
    ↓
├─ ALS Pipeline ─────────────────┐
│  - Load CSR matrix              │
│  - Configure hyperparameters    │
│  - Train với implicit library   │
│  - Evaluate Recall@K, NDCG@K    │
│  - Save U, V, metrics           │
│                                 │
└─ BPR Pipeline ─────────────────┤
   - Load positive sets            │
   - Sample triplets (u, i+, j-)  │
   - Gradient descent updates     │
   - Evaluate metrics             │
   - Save U, V, metrics           │
                                  ↓
              Compare Metrics (NDCG@10)
                                  ↓
              Update Registry (best model)
```

## Module Architecture Overview

Code đã được tổ chức thành **class-based architecture** với các modules riêng biệt:

```
recsys/cf/model/
├── __init__.py
├── bert_enhanced_als.py      # BERT-enhanced ALS wrapper
├── als/
│   ├── __init__.py           # Package exports
│   ├── pre_data.py           # ALSMatrixPreparer - Step 1
│   ├── model_init.py         # ALSModelInitializer - Step 2
│   ├── trainer.py            # ALSTrainer - Step 3
│   ├── embeddings.py         # EmbeddingExtractor - Step 4
│   ├── recommender.py        # ALSRecommender - Step 5
│   ├── evaluation.py         # ALSEvaluator - Step 6
│   ├── artifact_saver.py     # Artifact saving - Step 7
│   ├── run_als_complete.py  # Complete pipeline script
│   └── train_als_complete.py
└── bpr/
    ├── __init__.py           # Package exports
    ├── pre_data.py           # BPRDataPreparer - Step 1
    ├── sampler.py            # TripletSampler, HardNegativeMixer - Step 2
    ├── model_init.py         # BPRModelInitializer - Step 3
    ├── trainer.py            # BPRTrainer - Step 4
    ├── artifact_saver.py     # Artifact saving - Step 7
    └── ...
```

## Shared Components

### 1. Data Loading (Updated: DataProcessor Orchestration)
- **Module**: `recsys/cf/data/data.py`
- **Class**: `DataProcessor`
- **Usage**:
  ```python
  from recsys.cf.data import DataProcessor
  
  processor = DataProcessor(base_path="data/published_data")
  
  # 1. Load & Validate
  df_clean, stats = processor.load_and_validate_interactions()
  
  # 2. Prepare ALS Matrix
  X_train_conf, als_stats = processor.prepare_als_matrix(
      df_clean, num_users, num_items
  )
  
  # 3. Prepare BPR Data
  df_bpr, hard_neg_sets = processor.prepare_bpr_labels(df_clean)
  user_pos_sets = processor.build_bpr_positive_sets(df_bpr)
  ```
- **Outputs**:
  - `X_train_confidence`: CSR matrix (ALS) - from `prepare_als_matrix`
  - `df_bpr`: DataFrame with `is_positive`, `is_hard_negative` columns (BPR)
  - `user_pos_sets`: Dict[u, Set[i]] (BPR sampling)
  - `hard_neg_sets`: Dict[u, Set[i]] (BPR sampling)
  - `mappings`: ID mappings from `processor.id_mapper`

### 1.1 ALS Data Preparation (Module: `recsys/cf/model/als/pre_data.py`)
- **Class**: `ALSMatrixPreparer`
- **Method**: `prepare_complete_als_data()`
- **Purpose**: Load processed data from Task 01 and prepare for ALS training
- **Outputs**: 
  - `X_train_implicit`: Transposed CSR matrix (items × users) for implicit library
  - `X_train_confidence`: Original confidence matrix (users × items)
  - `mappings`: ID mappings
  - `user_pos_train`: User positive sets
  - `metadata`: Data statistics

### 1.2 BPR Data Preparation (Module: `recsys/cf/model/bpr/pre_data.py`)
- **Class**: `BPRDataPreparer` / `BPRDataLoader`
- **Method**: `load_bpr_data()` or `prepare_bpr_data()`
- **Purpose**: Load positive sets, pairs, and hard negative sets for BPR training
- **Outputs**:
  - `positive_pairs`: Array of (u_idx, i_idx) pairs
  - `user_pos_sets`: Dict[u_idx, Set[i_idx]]
  - `hard_neg_sets`: Dict[u_idx, Set[i_idx]] (explicit + implicit)
  - `num_users`, `num_items`: Dimensions

### 2. Configuration Management
- **File**: `config/training_config.yaml`
- **Structure**:
  ```yaml
  als:
    factors: 64
    regularization: 0.01
    iterations: 15
    alpha: 10  # Confidence scaling (LOWER than binary: 10 vs 40)
    use_gpu: false
    random_seed: 42
    confidence_strategy: "direct"  # "direct" or "scaled"
  
  bpr:
    factors: 64
    learning_rate: 0.05
    regularization: 0.0001
    epochs: 50
    samples_per_epoch: 5  # Multiple của số positives
    negative_sampling: "hard_mixed"  # "hard_mixed", "uniform", or "popularity"
    hard_negative_ratio: 0.3  # 30% hard negatives, 70% random
    random_seed: 42
  
  evaluation:
    k_values: [10, 20]
    metrics: ["recall", "ndcg"]
  ```

### 3. Evaluation Module
- **Module**: `recsys/cf/metrics.py`
- **Functions**:
  - `recall_at_k(predictions, ground_truth, k)`
  - `ndcg_at_k(predictions, ground_truth, k)`
  - `evaluate_model(U, V, test_data, user_pos_train, k_values)`
- **Details**: Xem Task 03 - Evaluation Metrics

## ALS Pipeline

### Step 1: Matrix Preparation (UPDATED: Sentiment-Enhanced Confidence)

#### 1.1 Load Confidence Matrix
- **Module**: `recsys/cf/model/als/pre_data.py`
- **Class**: `ALSMatrixPreparer`
- **Method**: `prepare_complete_als_data()` or `quick_prepare_als_matrix()`
- **Input**: Processed data from Task 01 (Parquet files, NPZ matrices)
- **Logic**:
  - Loads `X_train_confidence.npz` (CSR matrix with confidence scores)
  - `confidence_score = rating + comment_quality` (Range: 1.0 - 6.0)
  - **Transpose**: Converts to item-user format for implicit library
- **Output**: 
  - `X_train_implicit`: Transposed CSR matrix (items × users) for `implicit` library
  - `X_train_confidence`: Original confidence matrix (users × items)
  - `mappings`: ID mappings
  - `user_pos_train`: User positive sets for filtering

**Usage**:
```python
from recsys.cf.model.als import ALSMatrixPreparer

preparer = ALSMatrixPreparer(base_path='data/processed')
data = preparer.prepare_complete_als_data()
X_train = data['X_train_implicit']  # Ready for implicit library
```

#### 1.2 Confidence Scaling Strategy
- **Implementation**: Handled in `ALSModelInitializer` with alpha recommendation
- **Range**: 
  - Raw: [1.0, 6.0] (Default) → alpha = 5-10 (lower due to higher range)
  - Normalized: [0.0, 1.0] → alpha = 20-40 (standard scaling)
- **Auto-recommendation**: `ALSModelInitializer.get_alpha_recommendation(confidence_range)`
- **Presets**: 
  - `'default'`: alpha=10 (for raw 1-6 range)
  - `'normalized'`: alpha=40 (for 0-1 range)
  - `'sparse_data'`: alpha=5, reg=0.1 (for very sparse data)

#### 1.3 Preference Matrix (Optional)
- **P**: Can derive binary preference from confidence
  - P[u,i] = 1 if confidence >= threshold (e.g., 4.5), else 0
  - Or continuous: P[u,i] = (confidence - 1) / 5 (normalize to [0,1])
- **Usage**: Target preferences trong ALS loss
- **Note**: With sentiment-enhanced confidence, ALS learns nuanced preferences beyond binary

### Step 2: Model Initialization

#### 2.1 Library Choice
- **Preferred**: `implicit` library (C++ backend, fast, GPU support)
  - Install: `pip install implicit`
  - Model: `implicit.als.AlternatingLeastSquares`
- **Module**: `recsys/cf/model/als/model_init.py`
- **Class**: `ALSModelInitializer`

#### 2.2 Hyperparameters & Presets
- **Module**: `recsys/cf/model/als/model_init.py`
- **Class**: `ALSModelInitializer` với configuration presets:
  - `'default'`: factors=64, reg=0.01, alpha=10 (for sentiment-enhanced confidence 1-6)
  - `'normalized'`: factors=64, reg=0.01, alpha=40 (for normalized 0-1)
  - `'high_quality'`: factors=128, reg=0.05, iterations=20
  - `'fast'`: factors=32, reg=0.01, iterations=10
  - `'sparse_data'`: factors=64, reg=0.1, alpha=5 (for ≥2 threshold, high sparsity)

**Hyperparameters**:
- **factors**: Embedding dimension (32/64/128) - Start với 64
- **regularization**: L2 penalty (0.01-0.1) - Higher for sparse data (0.1 for ≥2 threshold)
- **iterations**: ALS iterations (10-20) - Monitor convergence
- **alpha**: Confidence scaling (5-10 for raw 1-6, 20-40 for normalized 0-1)
- **random_seed**: For reproducibility (default: 42)
- **use_gpu**: GPU acceleration (requires cupy)

#### 2.3 Initialization Methods

**Method 1: Using Preset** (Recommended):
```python
from recsys.cf.model.als import ALSModelInitializer

initializer = ALSModelInitializer(preset='default')
model = initializer.initialize_model()
```

**Method 2: Custom Configuration**:
```python
from recsys.cf.model.als import ALSModelInitializer

config = {
    'factors': 64,
    'regularization': 0.01,
    'iterations': 15,
    'alpha': 10,  # Lower for sentiment-enhanced confidence (1-6 range)
    'random_state': 42,
    'use_gpu': False
}
initializer = ALSModelInitializer(config=config)
model = initializer.initialize_model()
```

**Method 3: Quick Initialization**:
```python
from recsys.cf.model.als import quick_initialize_als

model = quick_initialize_als(
    factors=64,
    regularization=0.01,
    iterations=15,
    alpha=10,
    use_gpu=False,
    random_state=42
)
```

**Method 4: Data-Driven Recommendations**:
```python
from recsys.cf.model.als import ALSModelInitializer

initializer = ALSModelInitializer()
recommended_config = initializer.recommend_config_for_data(
    num_users=26000,
    num_items=2231,
    nnz=65000,
    confidence_range=(1.0, 6.0)
)
# Returns: {'factors': 64, 'regularization': 0.1, 'alpha': 10, ...}
initializer.update_config(**recommended_config)
model = initializer.initialize_model()
```

### Step 3: Training

#### 3.1 Fit Model
- **Module**: `recsys/cf/model/als/trainer.py`
- **Class**: `ALSTrainer`
- **Input**: Transposed CSR matrix `X_train_implicit` (items × users)
  - `implicit` library expects item-user matrix format
- **Process**: 
  - Alternates giữa fixing U update V, và fixing V update U
  - Mỗi iteration solve least squares với regularization
- **Features**:
  - Progress tracking với iteration metrics
  - Optional checkpointing every N iterations
  - Memory usage monitoring
  - Validation-based early stopping (optional)

**Usage**:
```python
from recsys.cf.model.als import ALSTrainer

trainer = ALSTrainer(
    model=model,
    checkpoint_dir=Path('artifacts/cf/als/checkpoints'),
    checkpoint_interval=5,
    track_memory=True,
    enable_validation=False
)

# Fit model
results = trainer.fit(X_train_implicit)
# Returns: TrainingHistory with iteration metrics
```

#### 3.2 Progress Tracking
- **Logging**: 
  - Iteration number, wall-clock time
  - Memory usage (if `track_memory=True`)
  - Training loss (if available from implicit library)
- **Checkpointing**: 
  - Save intermediate U, V mỗi `checkpoint_interval` iterations
  - Location: `checkpoint_dir/checkpoint_iter_{N}.npz`
  - Cho early stopping nếu validation loss tăng
- **Validation Monitoring** (Optional):
  ```python
  def compute_val_metrics(model, iteration):
      # Custom validation logic
      return {'val_loss': 0.123, 'val_recall@10': 0.234}
  
  trainer.set_validation_data(X_val, compute_val_metrics)
  results = trainer.fit(X_train)
  ```

### Step 4: Extract Embeddings

#### 4.1 Extract Embeddings
- **Module**: `recsys/cf/model/als/embeddings.py`
- **Class**: `EmbeddingExtractor`
- **Methods**:
  - `get_embeddings()`: Extract U and V matrices
  - `normalize_embeddings()`: Optional L2 normalization
  - `compute_embedding_quality_score()`: Quality metrics

**Usage**:
```python
from recsys.cf.model.als import EmbeddingExtractor, extract_embeddings

# Method 1: Using class
extractor = EmbeddingExtractor(model, normalize=True)
U, V = extractor.get_embeddings()
print(f"U shape: {U.shape}, V shape: {V.shape}")
print(f"Normalized: {extractor.is_normalized}")

# Method 2: Quick function
U, V = extract_embeddings(model, normalize=True)
```

#### 4.2 User Embeddings
- **Matrix U**: Shape (num_users, factors)
- **Access**: `model.user_factors` (NumPy array) or `extractor.get_user_embeddings()`
- **Normalization**: Optional L2 normalize rows cho cosine similarity
- **Type**: `np.float32` for memory efficiency

#### 4.3 Item Embeddings
- **Matrix V**: Shape (num_items, factors)
- **Access**: `model.item_factors` or `extractor.get_item_embeddings()`
- **Normalization**: Match với U nếu normalized
- **Type**: `np.float32` for memory efficiency

### Step 5: Recommendation Generation (For Evaluation)

#### 5.1 Recommendation Module
- **Module**: `recsys/cf/model/als/recommender.py`
- **Class**: `ALSRecommender`
- **Features**:
  - Single user recommendations
  - Batch recommendations for multiple users
  - Automatic seen item filtering
  - ID mapping (u_idx ↔ user_id, i_idx ↔ product_id)

**Usage**:
```python
from recsys.cf.model.als import ALSRecommender, quick_recommend

# Method 1: Using class
recommender = ALSRecommender(
    user_factors=U,
    item_factors=V,
    user_to_idx=mappings['user_to_idx'],
    idx_to_user=mappings['idx_to_user'],
    item_to_idx=mappings['item_to_idx'],
    idx_to_item=mappings['idx_to_item'],
    user_pos_train=user_pos_train
)

# Single user
result = recommender.recommend(user_id='12345', k=10, filter_seen=True)
print(f"Top items: {result.item_ids[:5]}")
print(f"Scores: {result.scores[:5]}")

# Batch recommendations
results = recommender.recommend_batch(user_ids=test_users, k=10)

# Method 2: Quick function
quick_recs = quick_recommend(
    U, V, 
    user_ids=test_users, 
    k=10, 
    mappings=mappings,
    user_pos_train=user_pos_train
)
```

#### 5.2 Batch Scoring
- **Method**: `U @ V.T` → (num_users, num_items) score matrix
- **Optimization**: 
  - Compute chỉ cho test users (subset U)
  - Sử dụng sparse operations nếu possible
  - Batch processing for memory efficiency

#### 5.3 Filtering Seen Items
- **Logic**: Cho mỗi user u, mask scores tại indices trong `user_pos_train[u]`
- **Implementation**: Set scores = -inf hoặc remove từ candidates
- **Automatic**: Handled by `ALSRecommender` if `filter_seen=True`

#### 5.4 Top-K Selection
- **Method**: `np.argsort` hoặc `np.argpartition` (faster cho large K)
- **Output**: `RecommendationResult` object với:
  - `item_ids`: List of product IDs (mapped from i_idx)
  - `scores`: List of recommendation scores
  - `item_indices`: Original i_idx values
- **Map back**: Automatic via `idx_to_item` mapping

### Step 6: Evaluation

#### 6.1 Evaluation Module
- **Module**: `recsys/cf/model/als/evaluation.py`
- **Class**: `ALSEvaluator`
- **Features**:
  - Recall@K, NDCG@K computation
  - Popularity baseline comparison
  - Batch evaluation for multiple users
  - Comprehensive metrics summary

**Usage**:
```python
from recsys.cf.model.als import ALSEvaluator, quick_evaluate

# Method 1: Using class
evaluator = ALSEvaluator(
    user_factors=U,
    item_factors=V,
    user_to_idx=mappings['user_to_idx'],
    idx_to_user=mappings['idx_to_user'],
    item_to_idx=mappings['item_to_idx'],
    idx_to_item=mappings['idx_to_item'],
    user_pos_train=user_pos_train,
    user_pos_test=user_pos_test
)

results = evaluator.evaluate(k_values=[10, 20], compare_baseline=True)
results.print_summary()
print(f"Recall@10: {results.metrics['recall@10']:.3f}")
print(f"Improvement: {results.improvement['recall@10']}")

# Method 2: Quick function
metrics = quick_evaluate(
    U, V, 
    user_pos_test, 
    user_pos_train, 
    k_values=[10, 20]
)
```

#### 6.2 Metrics Computation
- **Recall@K**: % test items trong top-K recommendations
- **NDCG@K**: Discounted cumulative gain (xem Task 03)
- **K values**: [10, 20] (configurable)
- **Output**: `EvaluationResult` object với:
  - `metrics`: Dict with recall@k, ndcg@k for each k
  - `baseline_metrics`: Popularity baseline metrics
  - `improvement`: Percentage improvement over baseline

#### 6.3 Baseline Comparison
- **Popularity Baseline**: 
  - Rank items theo `item_popularity` từ train data
  - Hoặc `num_sold_time` từ `data_product.csv`
- **Class**: `PopularityBaseline` in `evaluation.py`
- **Purpose**: Verify CF beats naive popularity
- **Automatic**: Computed if `compare_baseline=True`

### Step 7: Save Artifacts

#### 7.1 Artifact Saving Module
- **Module**: `recsys/cf/model/als/artifact_saver.py`
- **Class**: `ALSArtifacts` (dataclass container)
- **Main Function**: `save_als_complete()` - One-line orchestrator
- **Features**:
  - Save all artifacts (embeddings, params, metrics, metadata)
  - **Score Range Computation** (Critical for Task 08)
  - Version tracking and metadata
  - Comprehensive error handling

**Usage**:
```python
from recsys.cf.model.als import save_als_complete, compute_score_range

# Complete save with score range
artifacts = save_als_complete(
    user_embeddings=U,
    item_embeddings=V,
    params={'factors': 64, 'regularization': 0.01, 'iterations': 15, 'alpha': 10},
    metrics={'recall@10': 0.234, 'ndcg@10': 0.189, ...},
    validation_user_indices=[10, 25, 42, ...],  # Critical for score range
    data_version_hash='abc123def456',
    output_dir='artifacts/cf/als'
)

print(artifacts.summary())
# Score range: [0.012, 1.123] for Task 08 normalization
```

#### 7.2 Model Files
- **Location**: `artifacts/cf/als/`
- **Files**:
  - `als_U.npy`: User embeddings (num_users, factors)
  - `als_V.npy`: Item embeddings (num_items, factors)
  - `als_model.pkl`: Serialized `implicit` model object (optional)

#### 7.3 Configuration
- **File**: `als_params.json`
- **Content**:
  ```json
  {
    "factors": 64,
    "regularization": 0.01,
    "iterations": 15,
    "alpha": 10,
    "random_seed": 42,
    "training_time_seconds": 45.2,
    "use_gpu": false,
    "dtype": "float32"
  }
  ```

#### 7.4 Metrics
- **File**: `als_metrics.json`
- **Content**:
  ```json
  {
    "recall@10": 0.234,
    "recall@20": 0.312,
    "ndcg@10": 0.189,
    "ndcg@20": 0.221,
    "baseline_recall@10": 0.145,
    "baseline_ndcg@10": 0.102,
    "improvement_recall@10": "61.4%"
  }
  ```

#### 7.5 Metadata (UPDATED - Add Score Range for Global Normalization)
- **File**: `als_metadata.json`
- **Content**:
  - Timestamp huấn luyện
  - Data version hash (từ Task 01)
  - Git commit hash (code version)
  - System info (CPU/GPU, memory, platform)
  - **NEW - CF Score Range** (Critical for Task 08 normalization):
    ```json
    {
      "score_range": {
        "method": "validation_set",
        "min": 0.0,
        "max": 1.48,
        "mean": 0.32,
        "std": 0.21,
        "p01": 0.01,
        "p99": 1.12,
        "num_samples": 50000
      }
    }
    ```
  - **Computation**: 
    - Function: `compute_score_range(U, V, validation_user_indices, user_pos_train)`
    - Runs `U @ V.T` on validation users, computes percentiles
    - Excludes seen items from score distribution
  - **Purpose**: Enable global normalization in hybrid reranking (Task 08)
  - **Usage**: Prevents score range mismatch between CF and content-based models

## BPR Pipeline

### Step 1: Data Preparation

#### 1.1 Load BPR Data
- **Module**: `recsys/cf/model/bpr/pre_data.py`
- **Class**: `BPRDataPreparer` / `BPRDataLoader`
- **Method**: `load_bpr_data()` or `prepare_bpr_data()`
- **Input**: Processed data from Task 01
- **Output**: 
  - `positive_pairs`: Array of (u_idx, i_idx) pairs
  - `user_pos_sets`: Dict[u_idx, Set[i_idx]]
  - `hard_neg_sets`: Dict[u_idx, Set[i_idx]] (explicit + implicit)
  - `num_users`, `num_items`: Dimensions

**Usage**:
```python
from recsys.cf.model.bpr import BPRDataLoader, load_bpr_data

# Method 1: Using class
loader = BPRDataLoader(base_path='data/processed')
data = loader.load_bpr_data()

# Method 2: Quick function
data = load_bpr_data(base_path='data/processed')
```

#### 1.2 Positive Pairs List
- **Format**: NumPy array of shape (N, 2) with columns [u_idx, i_idx]
- **Definition**: `(u, i)` where `rating >= 4.0` (positive threshold)
- **Source**: Derived from `user_pos_sets` or DataFrame
- **Usage**: Input for triplet sampling in training loop

### Step 2: Dual Hard Negative Mining Strategy (Implemented)

#### 2.1 Hard Negative Sampling Module
- **Module**: `recsys/cf/model/bpr/sampler.py`
- **Class**: `HardNegativeMixer`
- **Features**:
  - Dual-strategy: Explicit (rating ≤3) + Implicit (Top-K popular, not bought)
  - Mixed sampling: 30% hard negatives + 70% random
  - Efficient batch sampling
  - Statistics tracking

**Usage**:
```python
from recsys.cf.model.bpr import HardNegativeMixer, TripletSampler

# Initialize mixer
mixer = HardNegativeMixer(
    hard_neg_sets=hard_neg_sets,  # From Step 1
    hard_ratio=0.3,  # 30% hard, 70% random
    random_seed=42
)

# Sample negative for a user
neg_idx = mixer.sample_negative(
    user_idx=42,
    positive_set=user_pos_sets[42],
    num_items=2231
)

# Batch sampling
neg_indices = mixer.sample_negatives_batch(
    user_indices=np.array([10, 20, 30]),
    user_pos_sets=user_pos_sets,
    num_items=2231
)
```

#### 2.2 Triplet Sampling
- **Class**: `TripletSampler`
- **Method**: `sample_triplets()` or `sample_epoch()`
- **Logic**:
  - For each positive (u, i_pos), sample negative i_neg
  - 30% from `hard_neg_sets` (explicit + implicit)
  - 70% uniformly random from unseen items
- **Output**: Array of (u, i_pos, i_neg) triplets

**Usage**:
```python
from recsys.cf.model.bpr import TripletSampler

sampler = TripletSampler(
    positive_pairs=data['positive_pairs'],
    user_pos_sets=data['user_pos_sets'],
    hard_neg_mixer=mixer,
    num_items=data['num_items'],
    samples_per_epoch=5,  # multiplier of num_positives
    random_seed=42
)

# Sample triplets for one epoch
triplets = sampler.sample_epoch()
# Returns: (N, 3) array with columns [u_idx, i_pos_idx, i_neg_idx]
```

#### 2.3 Hard Negative Composition
- **Explicit**: Items with `rating <= 3.0` (User bought but disliked)
- **Implicit**: Top-50 popular items (by `num_sold_time`) user DID NOT interact with
- **Logic**: "Hot product but you didn't buy → implicit negative preference"
- **Source**: Prepared in Task 01 (`BPRDataPreparer.mine_hard_negatives`)

#### 2.4 Fallback Strategy
- **For users without hard negatives**: 
  - Fallback to 100% random sampling
  - `HardNegativeMixer` handles this automatically
  - Statistics tracked: `stats['fallback_to_random']`

#### 2.5 Samples Per Epoch
- **Rule**: `samples_per_epoch = num_positives * multiplier`
- **Default multiplier**: 5 (balance coverage vs speed)
- **Total samples**: ~1.8M triplets per epoch with multiplier=5
- **Tracking metrics**:
  - % triplets using explicit hard negatives
  - % triplets using implicit hard negatives
  - % triplets using random negatives
  - Available via `mixer.stats` or `sampler.get_sampling_stats()`

### Step 3: Model Initialization

#### 3.1 Model Initialization Module
- **Module**: `recsys/cf/model/bpr/model_init.py`
- **Class**: `BPRModelInitializer`
- **Features**:
  - Random initialization (Gaussian)
  - Optional BERT initialization for item embeddings
  - Preset configurations
  - Reproducible random seed

**Usage**:
```python
from recsys.cf.model.bpr import BPRModelInitializer, initialize_bpr_model

# Method 1: Using class
initializer = BPRModelInitializer(
    num_users=26000,
    num_items=2231,
    factors=64,
    random_seed=42
)
U, V = initializer.initialize_embeddings()

# Method 2: With BERT initialization
U, V = initializer.initialize_embeddings(
    bert_embeddings=bert_embeddings,  # (num_items, 768)
    item_mapping=item_to_idx
)

# Method 3: Quick function
U, V = initialize_bpr_model(
    num_users=26000,
    num_items=2231,
    factors=64,
    random_seed=42
)
```

#### 3.2 Embedding Matrices
- **U**: Shape (num_users, factors) - User embeddings
- **V**: Shape (num_items, factors) - Item embeddings
- **Initialization**: 
  - **Random**: Gaussian noise `np.random.randn * 0.01` (mean=0, std=0.01)
  - **BERT**: Optional initialization from PhoBERT embeddings (projected to factors dim)
  - Small values prevent exploding gradients
- **Random seed**: Fixed for reproducibility (default: 42)

#### 3.3 Hyperparameters
- **factors**: 64 (match ALS cho fair comparison)
- **learning_rate**: 0.05 (tune 0.01-0.1)
- **regularization**: 0.0001 (L2 penalty, tune 1e-5 to 1e-3)
- **epochs**: 50 (monitor validation early stopping)
- **lr_decay**: 0.9 (learning rate decay factor)
- **lr_decay_every**: 10 (decay every N epochs)

### Step 4: Training Loop

#### 4.1 Training Module
- **Module**: `recsys/cf/model/bpr/trainer.py`
- **Class**: `BPRTrainer`
- **Features**:
  - Mini-batch SGD with BPR loss
  - Hard negative mining (30% hard + 70% random)
  - Learning rate decay
  - Early stopping
  - Checkpoint saving
  - Training history tracking

**Usage**:
```python
from recsys.cf.model.bpr import BPRTrainer, train_bpr_model

# Method 1: Using class
trainer = BPRTrainer(
    U=U,
    V=V,
    learning_rate=0.05,
    regularization=0.0001,
    lr_decay=0.9,
    lr_decay_every=10,
    checkpoint_dir=Path('artifacts/cf/bpr/checkpoints'),
    checkpoint_interval=5,
    random_seed=42
)

results = trainer.fit(
    positive_pairs=data['positive_pairs'],
    user_pos_sets=data['user_pos_sets'],
    hard_neg_sets=data['hard_neg_sets'],
    num_items=data['num_items'],
    epochs=50,
    samples_per_epoch=5
)

# Access training history
history = trainer.history
print(f"Best epoch: {history.get_best_epoch('recall@10')}")

# Method 2: Quick function
U, V, history = train_bpr_model(
    U_init=U,
    V_init=V,
    positive_pairs=data['positive_pairs'],
    user_pos_sets=data['user_pos_sets'],
    hard_neg_sets=data['hard_neg_sets'],
    num_items=data['num_items'],
    epochs=50,
    learning_rate=0.05,
    regularization=0.0001
)
```

#### 4.2 BPR Loss & Updates
- **Loss**: `L = -log(sigmoid(x_uij)) + reg * (||U[u]||^2 + ||V[i]||^2 + ||V[j]||^2)`
- **Where**: `x_uij = U[u] @ V[i] - U[u] @ V[j]` (score difference)
- **Update Rules** (SGD):
  - `U[u] += lr * ((1 - sigmoid) * (V[i] - V[j]) - reg * U[u])`
  - `V[i] += lr * ((1 - sigmoid) * U[u] - reg * V[i])`
  - `V[j] += lr * (-(1 - sigmoid) * U[u] - reg * V[j])`

#### 4.3 Monitoring
- **Loss proxy**: Average `log(sigmoid(x_uij))` per epoch
- **Training History**: `TrainingHistory` dataclass tracks:
  - Epochs, losses, validation metrics, learning rates, durations
- **Validation metrics**: Recall@10 mỗi N epochs (configurable)
- **Early stopping**: Nếu validation Recall không cải thiện N epochs liên tiếp
- **Checkpointing**: Save U, V mỗi `checkpoint_interval` epochs

#### 4.4 Learning Rate Decay
- **Schedule**: `lr = lr_init * (lr_decay ** (epoch // lr_decay_every))`
- **Default**: `lr_decay=0.9`, `lr_decay_every=10`
- **Purpose**: Finer updates khi converge
- **Automatic**: Handled by `BPRTrainer` if configured

### Step 5: Recommendation Generation

#### 5.1 Recommendation Module
- **Note**: BPR recommendation generation similar to ALS
- **Formula**: `scores[u] = U[u] @ V.T` → (num_items,) per user
- **Filtering**: Mask seen items, argsort, top-K
- **Implementation**: Can reuse `ALSRecommender` class or implement similar logic

### Step 6: Evaluation

#### 6.1 Evaluation Module
- **Note**: BPR evaluation similar to ALS
- **Metrics**: Recall@K, NDCG@K
- **Baseline**: Compare với popularity baseline
- **Implementation**: Can reuse `ALSEvaluator` class or implement similar logic

#### 6.2 AUC (Optional)
- **Definition**: Area under ROC curve
- **Purpose**: Ranking quality across all pairs
- **Computation**: Expensive, optional

### Step 7: Save Artifacts

#### 7.1 Artifact Saving Module
- **Module**: `recsys/cf/model/bpr/artifact_saver.py`
- **Class**: `BPRArtifacts` (dataclass container)
- **Main Function**: `save_bpr_complete()` - One-line orchestrator
- **Features**:
  - Save all artifacts (embeddings, params, metrics, metadata)
  - **Score Range Computation** (Critical for Task 08)
  - Training history tracking
  - Version tracking and metadata

**Usage**:
```python
from recsys.cf.model.bpr import save_bpr_complete, compute_bpr_score_range

# Complete save with score range
artifacts = save_bpr_complete(
    user_embeddings=trainer.U,
    item_embeddings=trainer.V,
    params=trainer.get_params(),
    metrics={'recall@10': 0.234, 'ndcg@10': 0.189, ...},
    training_history=trainer.history,
    validation_user_indices=[10, 25, 42, ...],  # Critical for score range
    data_version_hash='abc123def456',
    output_dir='artifacts/cf/bpr'
)
```

#### 7.2 Files Structure (Mirror ALS)
- **Location**: `artifacts/cf/bpr/`
- **Files**:
  - `bpr_U.npy`, `bpr_V.npy`: User and item embeddings
  - `bpr_params.json`: Training hyperparameters
  - `bpr_metrics.json`: Evaluation metrics
  - `bpr_metadata.json`: Comprehensive metadata (UPDATED - Add Score Range)
  - `bpr_training_history.json`: Training curves (optional)

#### 7.3 Additional BPR-Specific Metadata
- **Sampling config**: hard_ratio, samples_per_epoch, sampling strategy
- **Convergence**: Epoch tốt nhất (early stopping)
- **Training curves**: Loss và metrics theo epoch (stored in `TrainingHistory`)
- **NEW - CF Score Range** (Critical for Task 08 normalization):
  ```json
  {
    "score_range": {
      "method": "validation_set",
      "min": -0.35,
      "max": 1.62,
      "mean": 0.48,
      "std": 0.28,
      "p01": -0.12,
      "p99": 1.38,
      "num_samples": 50000
    }
  }
  ```
- **Computation**: 
  - Function: `compute_bpr_score_range(U, V, validation_user_indices, user_pos_train)`
  - Runs `U @ V.T` on validation users, computes percentiles
  - Excludes seen items from score distribution
- **Purpose**: Enable global normalization in hybrid reranking (Task 08)

## Hyperparameter Tuning

### Grid Search Strategy

#### 1. ALS Grid (UPDATED for ≥2 Threshold)
```yaml
factors: [32, 64, 128]
regularization: [0.01, 0.05, 0.1]  # Higher for sparse data (≥2 threshold)
iterations: [10, 20]
alpha: [5, 10, 20]  # Lower alpha due to sentiment-enhanced confidence (1-6 range)
```
- **Total combinations**: 3×3×2×3 = 54 configs
- **Parallel**: Có thể train parallel nếu có resources
- **Rationale**: Higher regularization (λ=0.1) anchors sparse user vectors to BERT semantic space

#### 2. BPR Grid
```yaml
factors: [32, 64, 128]
learning_rate: [0.01, 0.05, 0.1]
regularization: [0.00001, 0.0001, 0.001]
epochs: [30, 50]  # With early stopping
```
- **Total**: 3×3×3×2 = 54 configs
- **Note**: Mỗi config chậm hơn ALS (~10-30 phút)

### Tuning Workflow

#### Stage 1: Coarse Search (UPDATED for ≥2 Threshold)
- **ALS**: 
  - Fix iterations=15, alpha=10 (lower due to confidence_score range 1-6)
  - Tune factors={32,64,128}, reg={0.05, 0.1, 0.15}
  - **Critical**: Higher reg anchors BERT initialization for sparse users (≥2 interactions)
- **BPR**:
  - Fix epochs=30
  - Tune factors={64}, lr={0.01,0.05,0.1}, reg={1e-4,1e-3}

#### Stage 2: Fine-Tune Best Config
- **Expand range**: Thêm intermediate values
- **ALS**: Tune alpha range [best±20]
- **BPR**: Tune learning rate decay, sampling strategy

#### Stage 3: Validation
- **Retrain** best config với train+val → evaluate trên test
- **Verify** không overfitting

### Tracking Experiments

#### Tool Option 1: CSV Log
- **File**: `experiments/cf_experiments.csv`
- **Columns**: model_type, factors, reg, lr, alpha, recall@10, ndcg@10, training_time

#### Tool Option 2: MLflow (Advanced)
- **Setup**: Local MLflow tracking server
- **Log**: Params, metrics, artifacts
- **UI**: Compare runs visually

## Pipeline Orchestration

### Complete Pipeline Scripts

#### ALS Complete Pipeline
- **Module**: `recsys/cf/model/als/run_als_complete.py`
- **Function**: `run_als_pipeline()`
- **Features**:
  - Single function call to run complete pipeline
  - Automatic data loading from Task 01 outputs
  - Model initialization, training, evaluation, artifact saving
  - Score range computation for Task 08
  - Comprehensive logging

**Usage**:
```python
from recsys.cf.model.als.run_als_complete import run_als_pipeline

# Run with default settings
artifacts = run_als_pipeline()

# Custom configuration
artifacts = run_als_pipeline(
    data_dir='data/processed',
    output_dir='artifacts/cf/als',
    factors=64,
    regularization=0.01,
    iterations=15,
    alpha=10,
    use_bert_init=True,
    bert_embeddings_path='data/processed/content_based_embeddings/product_embeddings.pt'
)
```

#### BPR Complete Pipeline
- **Module**: Similar structure to ALS
- **Function**: `run_bpr_pipeline()` (if implemented)
- **Features**: Similar to ALS pipeline

### CLI Scripts (Optional)

#### Script: `scripts/train_cf.py`

**CLI Arguments**:
```bash
python scripts/train_cf.py \
  --model als \                # or bpr
  --config config/als_config.yaml \
  --data-version latest \      # or specific hash
  --output artifacts/cf/als \
  --gpu                        # optional for ALS
  --bert-init                  # optional BERT initialization
```

**Workflow**:
1. Parse arguments
2. Load config
3. Load preprocessed data (Task 01)
4. Initialize model (with optional BERT init)
5. Train
6. Evaluate
7. Save artifacts (with score range)
8. Log results

### Parallel Training (Optional)
- **Script**: `scripts/train_both.sh`
- **Logic**: 
  ```bash
  python scripts/train_cf.py --model als &
  python scripts/train_cf.py --model bpr &
  wait
  ```
- **Resources**: Đảm bảo không vượt RAM/CPU limits

## Error Handling

### Common Issues

#### 1. Memory Errors
- **ALS**: CSR matrix too large
  - **Solution**: Subsample users hoặc use GPU
- **BPR**: Triplet sampling explodes
  - **Solution**: Reduce samples_per_epoch

#### 2. Convergence Issues
- **ALS**: Loss không giảm
  - **Check**: Alpha too high, regularization too low
- **BPR**: Loss oscillates
  - **Solution**: Lower learning rate, add LR decay

#### 3. Evaluation Crashes
- **Symptom**: OOM khi compute U @ V.T
- **Solution**: Batch users, compute scores incrementally

### Logging Strategy
- **Training logs**: `logs/cf/als.log`, `logs/cf/bpr.log`
- **Format**: Timestamp, level, message
- **Rotation**: Keep last 10 runs

## Performance Benchmarks

### Expected Training Times (CPU)
- **ALS**: 
  - 15 iterations, 12K users, 2.2K items: ~1-2 minutes
- **BPR**:
  - 50 epochs, 1.8M samples/epoch: ~20-30 minutes

### Expected Metrics (Baselines)
- **Popularity Baseline**: 
  - Recall@10: ~0.12-0.15
  - NDCG@10: ~0.08-0.10
- **ALS Target**:
  - Recall@10: >0.20 (+40% vs baseline)
  - NDCG@10: >0.15
- **BPR Target**:
  - Recall@10: >0.22 (slightly better than ALS)
  - NDCG@10: >0.16

## Pipeline Extension: BERT-Enhanced Training

### Strategy 1: BERT-Initialized Item Factors

#### Purpose
Khởi tạo ALS/BPR item embeddings từ BERT embeddings để transfer semantic knowledge.

**CRITICAL for ≥2 Threshold**: With ~26k trainable users (≥2 interactions) and matrix density ~0.11%, BERT initialization is essential. Higher regularization (λ=0.1) prevents user vectors from drifting too far from semantic space, especially for users with exactly 2 interactions.

#### Implementation: BERT-Enhanced ALS

- **Module**: `recsys/cf/model/bert_enhanced_als.py`
- **Class**: `BERTEnhancedALS`
- **Features**:
  - Loads PhoBERT embeddings from `.pt` file
  - Projects BERT embeddings (768-dim) to target dimension using SVD/PCA
  - Aligns embeddings with CSR matrix item ordering
  - Initializes ALS item factors from projected BERT embeddings
  - Validates for NaN/Inf values
  - Comprehensive metadata tracking

**Usage**:
```python
from recsys.cf.model.bert_enhanced_als import BERTEnhancedALS

# Initialize BERT-enhanced ALS
bert_als = BERTEnhancedALS(
    bert_embeddings_path='data/processed/content_based_embeddings/product_embeddings.pt',
    factors=64,
    projection_method='svd',  # or 'pca'
    regularization=0.1,  # Higher for sparse data
    iterations=15,
    alpha=10,
    random_state=42
)

# Fit model (automatically initializes item factors from BERT)
model = bert_als.fit(
    X_train=X_train_implicit,  # Transposed CSR matrix
    item_to_idx=mappings['item_to_idx']
)

# Access embeddings
U = model.user_factors
V = model.item_factors

# Get metadata
metadata = bert_als.get_training_metadata()
print(f"BERT init used: {metadata['bert_initialization']['enabled']}")
print(f"Explained variance: {metadata['bert_initialization']['explained_variance']:.3f}")
```

**Key Methods**:
- `project_bert_to_factors(target_dim)`: Projects BERT embeddings using SVD/PCA
- `align_embeddings_to_matrix(projected_embeddings, item_to_idx, num_items)`: Aligns with matrix ordering
- `fit(X_train, item_to_idx)`: Trains ALS with BERT initialization
- `get_training_metadata()`: Returns comprehensive metadata including BERT init info

#### Configuration Extension

**Python Configuration** (matches implementation):
```python
# ALS with BERT initialization
from recsys.cf.model.bert_enhanced_als import BERTEnhancedALS

bert_als = BERTEnhancedALS(
    bert_embeddings_path="data/processed/content_based_embeddings/product_embeddings.pt",
    factors=64,
    projection_method="svd",  # or "pca"
    regularization=0.1,  # Higher for sparse data
    iterations=15,
    alpha=10,
    random_state=42
)

# BPR with BERT initialization
from recsys.cf.model.bpr import BPRModelInitializer

initializer = BPRModelInitializer(
    num_users=26000,
    num_items=2231,
    factors=64
)

# Load BERT embeddings
import torch
bert_data = torch.load('data/processed/content_based_embeddings/product_embeddings.pt')
bert_embeddings = bert_data['embeddings'].numpy()

# Initialize with BERT
U, V = initializer.initialize_embeddings(
    bert_embeddings=bert_embeddings,
    item_mapping=item_to_idx
)
```

**YAML Configuration** (for external config files):
```yaml
als:
  factors: 64
  regularization: 0.1  # Higher for sparse data (≥2 threshold)
  iterations: 15
  alpha: 10  # Lower for sentiment-enhanced confidence (1-6 range)
  use_gpu: false
  random_seed: 42
  
  # BERT initialization
  bert_init:
    enabled: true
    embeddings_path: "data/processed/content_based_embeddings/product_embeddings.pt"
    projection_method: "svd"  # or "pca"
    freeze_first_iter: false  # Not implemented, always fine-tunes

bpr:
  factors: 64
  learning_rate: 0.05
  regularization: 0.0001
  epochs: 50
  samples_per_epoch: 5
  negative_sampling: "hard_mixed"  # 30% hard + 70% random
  hard_negative_ratio: 0.3
  random_seed: 42
  
  # BERT initialization
  bert_init:
    enabled: true
    embeddings_path: "data/processed/content_based_embeddings/product_embeddings.pt"
    projection_method: "svd"
```

### Strategy 2: Multi-Task Learning (Experimental)

#### Joint Optimization of CF + Content Loss

```python
import torch
import torch.nn as nn

class HybridCFBERTModel(nn.Module):
    """
    Multi-task model: CF loss + content similarity loss.
    """
    
    def __init__(self, num_users, num_items, factors=128, bert_dim=768):
        super().__init__()
        
        # CF embeddings
        self.user_embedding = nn.Embedding(num_users, factors)
        self.item_embedding = nn.Embedding(num_items, factors)
        
        # BERT projection
        self.bert_projection = nn.Linear(bert_dim, factors)
        
        # Initialize
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids, item_bert_embeddings):
        # CF score
        user_emb = self.user_embedding(user_ids)  # (B, factors)
        item_emb = self.item_embedding(item_ids)  # (B, factors)
        cf_score = (user_emb * item_emb).sum(dim=1)  # (B,)
        
        # Content score
        item_bert_proj = self.bert_projection(item_bert_embeddings)  # (B, factors)
        content_score = (user_emb * item_bert_proj).sum(dim=1)  # (B,)
        
        return cf_score, content_score
    
    def compute_loss(self, cf_score_pos, cf_score_neg, content_score_pos, content_score_neg, 
                     cf_weight=0.7, content_weight=0.3):
        """
        Multi-task BPR loss.
        """
        # CF BPR loss
        cf_loss = -torch.log(torch.sigmoid(cf_score_pos - cf_score_neg)).mean()
        
        # Content BPR loss
        content_loss = -torch.log(torch.sigmoid(content_score_pos - content_score_neg)).mean()
        
        # Combined
        total_loss = cf_weight * cf_loss + content_weight * content_loss
        
        return total_loss, cf_loss.item(), content_loss.item()
```

### Training Comparison: Cold-Start Performance

#### Evaluation: Test on Cold-Start Items

```python
def evaluate_cold_start_items(model, test_data, item_to_idx, cold_threshold=5):
    """
    Evaluate performance on cold-start items (few interactions).
    
    Args:
        cold_threshold: Items với <N interactions considered cold-start
    """
    # Identify cold-start items
    item_counts = test_data['product_id'].value_counts()
    cold_items = set(item_counts[item_counts < cold_threshold].index)
    
    # Filter test data
    cold_test = test_data[test_data['product_id'].isin(cold_items)]
    
    # Evaluate
    metrics = evaluate_model(model, cold_test, ...)
    
    return metrics

# Compare BERT-init vs Random-init
bert_als_cold_recall = evaluate_cold_start_items(bert_als_model, test_data, ...)
random_als_cold_recall = evaluate_cold_start_items(random_als_model, test_data, ...)

print(f"Cold-start Recall@10:")
print(f"  BERT-init ALS: {bert_als_cold_recall['recall@10']:.3f}")
print(f"  Random-init ALS: {random_als_cold_recall['recall@10']:.3f}")
print(f"  Improvement: {(bert_als_cold_recall['recall@10'] / random_als_cold_recall['recall@10'] - 1):.1%}")
```

### Training Artifacts: BERT Metadata

#### Extended Metadata File

```json
{
  "model_type": "als",
  "factors": 128,
  "regularization": 0.01,
  "iterations": 15,
  "alpha": 40,
  "random_seed": 42,
  "training_time_seconds": 102.8,
  
  "bert_initialization": {
    "enabled": true,
    "embeddings_path": "data/processed/content_based_embeddings/product_embeddings.pt",
    "embedding_version": "v1_20250115_103000",
    "projection_method": "svd",
    "original_dim": 768,
    "projected_dim": 128,
    "explained_variance": 0.873
  },
  
  "cold_start_performance": {
    "cold_threshold": 5,
    "recall@10": 0.189,
    "improvement_vs_random": 0.234
  }
}
```

## Dependencies

```python
# requirements_cf.txt
numpy>=1.23.0
scipy>=1.9.0
pandas>=1.5.0
implicit>=0.6.0  # For ALS
scikit-learn>=1.2.0  # For metrics
pyyaml>=6.0  # For config
tqdm>=4.64.0  # Progress bars

# BERT training
torch>=1.13.0
transformers>=4.25.0
```

## Timeline Estimate

- **Implementation ALS**: 1 day
- **Implementation BPR**: 2 days
- **Hyperparameter tuning**: 2-3 days (compute time)
- **Testing & debugging**: 1 day
- **Total**: ~6-7 days

## Success Criteria

- [ ] ALS pipeline trains và saves artifacts
- [ ] BPR pipeline trains và saves artifacts
- [ ] Both exceed popularity baseline by ≥20% Recall@10
- [ ] Artifacts có đầy đủ metadata (reproducible)
- [ ] Training scripts documented và configurable
- [ ] Error handling robust (memory, convergence)
