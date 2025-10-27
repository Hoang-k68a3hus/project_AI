# Task 02: Training Pipelines (ALS & BPR)

## Mục Tiêu

Xây dựng hai pipelines huấn luyện Collaborative Filtering song song: ALS (Alternating Least Squares) và BPR (Bayesian Personalized Ranking). Mỗi pipeline sẽ train, evaluate, và persist artifacts độc lập, sau đó được so sánh để chọn best model.

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

## Shared Components

### 1. Data Loading (Reuse từ Task 01)
- **Module**: `recsys/cf/data.py`
- **Functions**:
  - `load_processed_data(version)` → dict với matrices, mappings, splits
- **Outputs**:
  - `X_train`: CSR matrix (num_users, num_items)
  - `user_pos_train`: Dict[u_idx, Set[i_idx]]
  - `test_interactions`: DataFrame với held-out positives
  - `mappings`: User/item ID mappings
  - `item_popularity`: Array cho baseline

### 2. Configuration Management
- **File**: `config/training_config.yaml`
- **Structure**:
  ```yaml
  als:
    factors: 64
    regularization: 0.01
    iterations: 15
    alpha: 40  # Confidence scaling
    use_gpu: false
    random_seed: 42
  
  bpr:
    factors: 64
    learning_rate: 0.05
    regularization: 0.0001
    epochs: 50
    samples_per_epoch: 5  # Multiple của số positives
    negative_sampling: "uniform"  # or "popularity"
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

### Step 1: Matrix Preparation

#### 1.1 Load Preprocessed CSR Matrix
- **Input**: `X_train.npz` từ Task 01
- **Shape**: (num_users, num_items)
- **Values**: Binary (1 for positives) hoặc counts

#### 1.2 Confidence Matrix Construction
- **Formula**: `C = 1 + alpha * X_train`
- **Intuition**: 
  - Observed positives → high confidence (1 + alpha)
  - Unobserved (zeros) → low confidence (1)
- **Default alpha**: 40 (tune trong range 20-80)

#### 1.3 Preference Matrix
- **P**: Binary matrix (1 where X > 0, else 0)
- **Usage**: Ground truth preferences trong ALS loss

### Step 2: Model Initialization

#### 2.1 Library Choice
- **Preferred**: `implicit` library (C++ backend, fast, GPU support)
  - Install: `pip install implicit`
  - Model: `implicit.als.AlternatingLeastSquares`
- **Alternative**: Custom NumPy implementation (slower, educational)

#### 2.2 Hyperparameters
- **factors**: Embedding dimension (32/64/128)
  - Start với 64, tune sau
- **regularization**: L2 penalty (0.001 - 0.1)
  - Prevent overfitting, start 0.01
- **iterations**: ALS iterations (10-20)
  - Monitor loss convergence
- **alpha**: Confidence scaling (20-80)
  - Higher → trust positives more
- **random_seed**: For reproducibility

#### 2.3 Initialization
```python
from implicit.als import AlternatingLeastSquares

model = AlternatingLeastSquares(
    factors=64,
    regularization=0.01,
    iterations=15,
    alpha=40,
    random_state=42,
    use_gpu=False  # Set True nếu có CUDA
)
```

### Step 3: Training

#### 3.1 Fit Model
- **Input**: Transposed CSR matrix `X_train.T` (items × users)
  - `implicit` expects item-user matrix
- **Process**: 
  - Alternates giữa fixing U update V, và fixing V update U
  - Mỗi iteration solve least squares với regularization
- **Monitoring**: 
  - Log loss mỗi iteration (nếu available)
  - Estimate training time

#### 3.2 Progress Tracking
- **Logging**: 
  - Iteration number
  - Wall-clock time
  - Memory usage (optional)
- **Checkpointing**: 
  - Save intermediate U, V mỗi 5 iterations (optional)
  - Cho early stopping nếu validation loss tăng

### Step 4: Extract Embeddings

#### 4.1 User Embeddings
- **Matrix U**: Shape (num_users, factors)
- **Access**: `model.user_factors` (NumPy array)
- **Normalization**: Optional L2 normalize rows cho cosine similarity

#### 4.2 Item Embeddings
- **Matrix V**: Shape (num_items, factors)
- **Access**: `model.item_factors`
- **Normalization**: Match với U nếu normalized

### Step 5: Recommendation Generation (For Evaluation)

#### 5.1 Batch Scoring
- **Method**: `U @ V.T` → (num_users, num_items) score matrix
- **Optimization**: 
  - Compute chỉ cho test users (subset U)
  - Sử dụng sparse operations nếu possible

#### 5.2 Filtering Seen Items
- **Logic**: Cho mỗi user u, mask scores tại indices trong `user_pos_train[u]`
- **Implementation**: Set scores = -inf hoặc remove từ candidates

#### 5.3 Top-K Selection
- **Method**: `np.argsort` hoặc `np.argpartition` (faster cho large K)
- **Output**: Array (num_users, K) với item indices
- **Map back**: i_idx → product_id sử dụng `idx_to_item` mapping

### Step 6: Evaluation

#### 6.1 Metrics Computation
- **Recall@K**: % test items trong top-K recommendations
- **NDCG@K**: Discounted cumulative gain (xem Task 03)
- **K values**: [10, 20] (configurable)

#### 6.2 Baseline Comparison
- **Popularity Baseline**: 
  - Rank items theo `item_popularity` từ train data
  - Hoặc `num_sold_time` từ `data_product.csv`
- **Purpose**: Verify CF beats naive popularity

### Step 7: Save Artifacts

#### 7.1 Model Files
- **Location**: `artifacts/cf/als/`
- **Files**:
  - `als_U.npy`: User embeddings
  - `als_V.npy`: Item embeddings
  - `als_model.pkl`: Serialized `implicit` model object (optional)

#### 7.2 Configuration
- **File**: `als_params.json`
- **Content**:
  ```json
  {
    "factors": 64,
    "regularization": 0.01,
    "iterations": 15,
    "alpha": 40,
    "random_seed": 42,
    "training_time_seconds": 45.2
  }
  ```

#### 7.3 Metrics
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

#### 7.4 Metadata
- **File**: `als_metadata.json`
- **Content**:
  - Timestamp huấn luyện
  - Data version hash (từ Task 01)
  - Git commit hash (code version)
  - System info (CPU/GPU, memory)

## BPR Pipeline

### Step 1: Data Preparation

#### 1.1 Load Positive Sets
- **Input**: `user_pos_train.pkl` (Dict[u_idx, Set[i_idx]])
- **Purpose**: Fast lookup cho negative sampling

#### 1.2 Positive Pairs List
- **Format**: List of tuples `(u_idx, i_idx)` for all train positives
- **Size**: ~369K pairs (after filtering)

### Step 2: Negative Sampling Strategy

#### 2.1 Uniform Sampling
- **Method**: 
  - For each (u, i+), sample j- uniformly từ [0, num_items-1]
  - Reject nếu j- in `user_pos_train[u]` → resample
- **Pros**: Simple, unbiased
- **Cons**: Nhiều "easy" negatives (user không hứng thú)

#### 2.2 Popularity-Biased Sampling (Advanced)
- **Method**: 
  - Sample j- theo `item_popularity` distribution
  - Higher chance cho popular items
- **Pros**: Harder negatives, better ranking
- **Cons**: Slower, need tuned popularity exponent

#### 2.3 Samples Per Epoch
- **Rule**: `samples_per_epoch = num_positives * multiplier`
- **Default multiplier**: 5 (balance coverage vs speed)
- **Total samples**: ~1.8M triplets per epoch với multiplier=5

### Step 3: Model Initialization

#### 3.1 Embedding Matrices
- **U**: Shape (num_users, factors)
- **V**: Shape (num_items, factors)
- **Initialization**: 
  - Gaussian noise: `np.random.randn * 0.01`
  - Mean=0, std=0.01 (small values prevent exploding gradients)
- **Random seed**: Fix để reproducible

#### 3.2 Hyperparameters
- **factors**: 64 (match ALS cho fair comparison)
- **learning_rate**: 0.05 (tune 0.01-0.1)
- **regularization**: 0.0001 (L2 penalty, tune 1e-5 to 1e-3)
- **epochs**: 50 (monitor validation early stopping)

### Step 4: Training Loop

#### 4.1 Epoch Structure
```python
for epoch in range(epochs):
    # Shuffle triplets
    triplets = sample_triplets(user_pos_train, num_samples)
    
    for (u, i_pos, i_neg) in triplets:
        # Compute scores
        score_pos = U[u] @ V[i_pos]
        score_neg = U[u] @ V[i_neg]
        
        # BPR loss gradient
        x_uij = score_pos - score_neg
        sigmoid = 1 / (1 + np.exp(-x_uij))
        
        # Update U[u]
        grad_u = (1 - sigmoid) * (V[i_pos] - V[i_neg]) - reg * U[u]
        U[u] += lr * grad_u
        
        # Update V[i_pos]
        grad_v_pos = (1 - sigmoid) * U[u] - reg * V[i_pos]
        V[i_pos] += lr * grad_v_pos
        
        # Update V[i_neg]
        grad_v_neg = -(1 - sigmoid) * U[u] - reg * V[i_neg]
        V[i_neg] += lr * grad_v_neg
    
    # Evaluate on validation (optional)
    if epoch % 5 == 0:
        val_metrics = evaluate_model(U, V, val_data, k=10)
        log_metrics(epoch, val_metrics)
```

#### 4.2 Monitoring
- **Loss proxy**: Average `log(sigmoid(x_uij))` per epoch
- **Validation metrics**: Recall@10 mỗi 5 epochs
- **Early stopping**: Nếu validation Recall không cải thiện 3 epochs liên tiếp

#### 4.3 Learning Rate Decay (Optional)
- **Schedule**: `lr = lr_init * (0.9 ** (epoch // 10))`
- **Purpose**: Finer updates khi converge

### Step 5: Recommendation Generation

#### 5.1 Scoring
- **Formula**: `scores[u] = U[u] @ V.T`
- **Shape**: (num_items,) per user

#### 5.2 Filtering & Ranking
- Same as ALS: Mask seen items, argsort, top-K

### Step 6: Evaluation

#### 6.1 Compute Metrics
- Same metrics as ALS: Recall@K, NDCG@K
- Compare với popularity baseline

#### 6.2 AUC (Optional)
- **Definition**: Area under ROC curve
- **Purpose**: Ranking quality across all pairs
- **Computation**: Expensive, optional

### Step 7: Save Artifacts

#### 7.1 Files Structure (Mirror ALS)
- **Location**: `artifacts/cf/bpr/`
- **Files**:
  - `bpr_U.npy`, `bpr_V.npy`
  - `bpr_params.json`
  - `bpr_metrics.json`
  - `bpr_metadata.json`

#### 7.2 Additional BPR-Specific Metadata
- **Sampling config**: uniform/popularity, samples_per_epoch
- **Convergence**: Epoch tốt nhất (early stopping)
- **Training curves**: Loss và metrics theo epoch (CSV hoặc JSON)

## Hyperparameter Tuning

### Grid Search Strategy

#### 1. ALS Grid
```yaml
factors: [32, 64, 128]
regularization: [0.001, 0.01]
iterations: [10, 20]
alpha: [20, 40, 80]
```
- **Total combinations**: 3×2×2×3 = 36 configs
- **Parallel**: Có thể train parallel nếu có resources

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

#### Stage 1: Coarse Search
- **ALS**: 
  - Fix iterations=15, alpha=40
  - Tune factors={32,64,128}, reg={0.01, 0.1}
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

### Script: `scripts/train_cf.py`

#### CLI Arguments
```bash
python scripts/train_cf.py \
  --model als \                # or bpr
  --config config/als_config.yaml \
  --data-version latest \      # or specific hash
  --output artifacts/cf/als \
  --gpu                        # optional for ALS
```

#### Workflow
1. Parse arguments
2. Load config
3. Load preprocessed data (Task 01)
4. Initialize model
5. Train
6. Evaluate
7. Save artifacts
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
