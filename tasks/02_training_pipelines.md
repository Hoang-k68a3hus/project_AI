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

## Shared Components

### 1. Data Loading (Reuse từ Task 01)
- **Module**: `recsys/cf/data.py`
- **Functions**:
  - `load_processed_data(version)` → dict với matrices, mappings, splits
- **Outputs**:
  - `X_train_confidence`: CSR matrix với confidence scores (1-6) for ALS - **trainable users only**
  - `X_train_binary`: CSR matrix với binary values (optional) for BPR - **trainable users only**
  - `user_pos_train`: Dict[u_idx, Set[i_idx]] - positive interactions (trainable users)
  - `user_hard_neg_train`: Dict[u_idx, Dict["explicit"|"implicit", Set[i_idx]]] - dual hard negatives
  - `test_interactions`: DataFrame với held-out positives from **trainable users only**
  - `user_metadata`: Segmentation data (trainable vs cold-start users)
  - `mappings`: User/item ID mappings
  - `top_k_popular_items`: Top-50 popular item indices for implicit negatives
  - `item_popularity`: Log-transformed popularity array

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
- **Input**: `X_train_confidence.npz` từ Task 01
- **Shape**: (num_trainable_users, num_items)
- **Values**: Confidence scores (1.0-6.0) = rating + comment_quality
  - Range breakdown:
    - [1.0-2.0]: Low ratings with/without quality comments
    - [3.0-4.0]: Medium ratings with variable comment quality
    - [5.0-6.0]: High ratings, 6.0 indicates genuine enthusiasm (detailed review)
- **Rationale**: Combat 95% 5-star skew by incorporating review sentiment
- **Coverage**: Only trainable users (≥3 interactions) - better model quality

#### 1.2 Confidence Scaling Strategy
- **Option 1 (Direct)**: Use confidence scores directly
  - `C[u,i] = confidence_score` if observed
  - `C[u,i] = 1` for unobserved (low confidence baseline)
  
- **Option 2 (Scaled - Recommended)**: Apply moderate alpha scaling
  - `C = 1 + alpha * (confidence - 1) / 5` (normalize to [0,1] first)
  - Default alpha: **5-10** (much lower than binary case 40 due to richer signal)
  - Higher confidence values already encode quality

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

#### 7.4 Metadata (UPDATED - Add Score Range for Global Normalization)
- **File**: `als_metadata.json`
- **Content**:
  - Timestamp huấn luyện
  - Data version hash (từ Task 01)
  - Git commit hash (code version)
  - System info (CPU/GPU, memory)
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
        "p99": 1.12
      }
    }
    ```
  - **Computation**: Run U @ V.T on validation set, compute percentiles
  - **Purpose**: Enable global normalization in hybrid reranking (Task 08)

## BPR Pipeline

### Step 1: Data Preparation

#### 1.1 Load Positive Sets
- **Input**: `user_pos_train.pkl` (Dict[u_idx, Set[i_idx]])
- **Purpose**: Fast lookup cho negative sampling

#### 1.2 Positive Pairs List
- **Format**: List of tuples `(u_idx, i_idx)` for all train positives
- **Size**: ~369K pairs (after filtering)

### Step 2: Dual Hard Negative Mining Strategy (UPDATED)

#### 2.1 Hard Negative Sampling - Explicit + Implicit (PRIMARY)
- **Load hard negatives**: `user_hard_neg_train` with structure:
  ```python
  {
    u_idx: {
      "explicit": set([i_idx1, i_idx2]),  # Items with rating ≤3
      "implicit": set([i_idx3, i_idx4])   # Popular items user didn't buy
    }
  }
  ```

- **Sampling Strategy**:
  - For each (u, i+), sample j- from mix:
    - 15% from `explicit` hard negatives (if available)
    - 15% from `implicit` hard negatives (popular items not bought)
    - 70% uniform random from unseen items
  - Reject if j- in `user_pos_train[u]` → resample

- **Rationale**: 
  - **Explicit negatives**: "You bought this but hated it" - strong signal
  - **Implicit negatives**: "Everyone bought this but you didn't" - preference signal
  - **Random negatives**: Maintain diversity, avoid over-focusing on hard cases

#### 2.2 Implicit Negative Generation Details
- **Source**: `top_k_popular_items` (Top-50 from Task 01)
- **Logic**: For each trainable user, find popular items NOT in their history
  ```python
  implicit_negs = set(top_k_popular_items) - user_pos_train[u] - user_hard_neg_train[u]["explicit"]
  ```
- **Why effective for sparse data**: 
  - Popular items have high probability of being relevant
  - User NOT buying them signals negative preference
  - More informative than random unpopular items

#### 2.3 Fallback Strategy
- **For users without hard negatives**: 
  - Use 30% popularity-biased sampling + 70% uniform random
  - Popularity-biased: Sample from log-transformed `item_popularity` distribution

#### 2.4 Samples Per Epoch
- **Rule**: `samples_per_epoch = num_positives * multiplier`
- **Default multiplier**: 5 (balance coverage vs speed)
- **Total samples**: ~1.8M triplets per epoch with multiplier=5
- **Tracking metrics**:
  - % triplets using explicit hard negatives
  - % triplets using implicit hard negatives
  - % triplets using random negatives
  - Average hard negative quality score

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
  - `bpr_metadata.json` (UPDATED - Add Score Range)

#### 7.2 Additional BPR-Specific Metadata
- **Sampling config**: uniform/popularity, samples_per_epoch
- **Convergence**: Epoch tốt nhất (early stopping)
- **Training curves**: Loss và metrics theo epoch (CSV hoặc JSON)
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
      "p99": 1.38
    }
  }
  ```
- **Computation**: Run dot(U[user], V[items]) on validation set, compute percentiles
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

## Pipeline Extension: BERT-Enhanced Training

### Strategy 1: BERT-Initialized Item Factors

#### Purpose
Khởi tạo ALS/BPR item embeddings từ BERT embeddings để transfer semantic knowledge.

**CRITICAL for ≥2 Threshold**: With ~26k trainable users (≥2 interactions) and matrix density ~0.11%, BERT initialization is essential. Higher regularization (λ=0.1) prevents user vectors from drifting too far from semantic space, especially for users with exactly 2 interactions.

#### Implementation: ALS with BERT Init

```python
from sklearn.decomposition import TruncatedSVD

class BERTEnhancedALS:
    """
    ALS với item factors initialized từ BERT embeddings.
    """
    
    def __init__(self, bert_embeddings_path, factors=128, **als_params):
        self.factors = factors
        self.als_params = als_params
        
        # Load BERT embeddings
        bert_data = torch.load(bert_embeddings_path)
        self.bert_embeddings = bert_data['embeddings'].numpy()  # (num_items, 768)
        self.product_ids = bert_data['product_ids']
    
    def project_bert_to_factors(self, target_dim):
        """
        Project BERT embeddings (768-dim) xuống target_dim using SVD.
        """
        if self.bert_embeddings.shape[1] == target_dim:
            return self.bert_embeddings
        
        svd = TruncatedSVD(n_components=target_dim, random_state=42)
        projected = svd.fit_transform(self.bert_embeddings)
        
        print(f"BERT projection: {self.bert_embeddings.shape[1]} -> {target_dim}")
        print(f"Explained variance: {svd.explained_variance_ratio_.sum():.3f}")
        
        return projected
    
    def fit(self, X_train, item_to_idx):
        """
        Train ALS với BERT-initialized item factors.
        """
        from implicit.als import AlternatingLeastSquares
        
        # Initialize ALS model
        model = AlternatingLeastSquares(
            factors=self.factors,
            **self.als_params
        )
        
        # Project BERT embeddings
        projected_embeddings = self.project_bert_to_factors(self.factors)
        
        # Align với item ordering trong CSR matrix
        aligned_embeddings = np.zeros((X_train.shape[1], self.factors))
        for i, product_id in enumerate(self.product_ids):
            if str(product_id) in item_to_idx:
                idx = item_to_idx[str(product_id)]
                aligned_embeddings[idx] = projected_embeddings[i]
        
        # Initialize item_factors
        model.item_factors = aligned_embeddings.astype(np.float32)
        
        print("Item factors initialized from BERT embeddings")
        
        # Fit (ALS will fine-tune from BERT initialization)
        model.fit(X_train.T)  # Transpose for implicit library
        
        return model
```

#### Configuration Extension

```yaml
als:
  factors: 128
  regularization: 0.01
  iterations: 15
  alpha: 40
  use_gpu: false
  random_seed: 42
  
  # BERT initialization
  bert_init:
    enabled: true
    embeddings_path: "data/processed/content_based_embeddings/product_embeddings.pt"
    projection_method: "svd"  # or "pca", "random"
    freeze_first_iter: false  # Keep BERT init frozen for first iteration

bpr:
  factors: 128
  learning_rate: 0.05
  regularization: 0.0001
  epochs: 50
  samples_per_epoch: 5
  negative_sampling: "uniform"
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
