# Task 03: Evaluation & Metrics

## Mục Tiêu

Xây dựng module đánh giá toàn diện cho Collaborative Filtering, bao gồm metrics chuẩn RecSys, baseline comparisons, và statistical significance testing. Module này được chia sẻ giữa ALS và BPR pipelines.

## Module Structure

### Module: `recsys/cf/metrics.py`

#### Exported Functions
1. `recall_at_k(predictions, ground_truth, k)` → float
2. `ndcg_at_k(predictions, ground_truth, k)` → float
3. `precision_at_k(predictions, ground_truth, k)` → float
4. `mrr(predictions, ground_truth)` → float
5. `map_at_k(predictions, ground_truth, k)` → float
6. `coverage(predictions, num_total_items)` → float
7. `evaluate_model(U, V, test_data, user_pos_train, k_values)` → dict
8. `evaluate_baseline_popularity(test_data, item_popularity, k_values)` → dict
9. `compare_models(model_metrics, baseline_metrics)` → DataFrame

## Core Metrics

### 1. Recall@K

#### Definition
**Recall@K** đo lường tỷ lệ relevant items (test positives) xuất hiện trong top-K recommendations.

#### Formula
```
Recall@K = |Top-K ∩ Test_Items| / |Test_Items|
```

#### Implementation Details
- **Input**:
  - `predictions`: List of K item IDs (recommended)
  - `ground_truth`: Set of positive item IDs (test)
  - `k`: Integer cutoff
- **Output**: Float [0, 1]
- **Edge cases**:
  - Nếu `|Test_Items| = 0` → skip user (hoặc return NaN)
  - Nếu `K > num_predictions` → use len(predictions)

#### Interpretation
- **Recall@10 = 0.25**: 25% test items được tìm thấy trong top-10
- **Higher is better**: Tối đa = 1.0 (tất cả test items trong top-K)
- **Trade-off**: Recall tăng theo K, nhưng precision giảm

### 2. NDCG@K (Normalized Discounted Cumulative Gain)

#### Definition
**NDCG@K** đánh giá chất lượng ranking, với relevant items ở vị trí cao hơn được reward nhiều hơn.

#### Formula
```
DCG@K = Σ(i=1 to K) [rel_i / log2(i+1)]
IDCG@K = DCG@K của perfect ranking (all relevant items first)
NDCG@K = DCG@K / IDCG@K
```

#### Relevance Definition
- **Binary**: rel_i = 1 nếu item_i in test positives, else 0
- **Graded** (optional): rel_i = rating (1-5) nếu có explicit ratings

#### Implementation Details
- **Discounting**: `log2(position + 1)` → vị trí 1,2,3,... có weight 1.0, 0.63, 0.5, ...
- **Normalization**: Divide bằng IDCG (ideal DCG) để scale [0, 1]
- **Edge cases**:
  - Nếu không có relevant items trong top-K → DCG = 0, NDCG = 0
  - Nếu IDCG = 0 (no test positives) → skip user

#### Interpretation
- **NDCG@10 = 0.18**: Ranking quality = 18% của ideal ranking
- **Higher is better**: 1.0 = perfect ranking
- **Stricter than Recall**: Penalizes relevant items ở vị trí thấp

### 3. Precision@K

#### Definition
**Precision@K** đo tỷ lệ relevant items trong top-K.

#### Formula
```
Precision@K = |Top-K ∩ Test_Items| / K
```

#### Characteristics
- **Complement của Recall**: Recall = coverage, Precision = accuracy
- **Upper bound**: Limited bởi min(|Test_Items|, K)
- **Use case**: Quan trọng khi show ít items (e.g., K=5 trên mobile)

### 4. MRR (Mean Reciprocal Rank)

#### Definition
**MRR** đo vị trí trung bình của **first relevant item**.

#### Formula
```
RR_u = 1 / rank(first_relevant_item)
MRR = Average(RR_u) across all users
```

#### Implementation
- **Find rank**: Vị trí (1-indexed) của test item đầu tiên trong predictions
- **Reciprocal**: 1/rank → rank 1 = 1.0, rank 2 = 0.5, rank 10 = 0.1
- **Average**: Across users

#### Use Case
- **Search/QA systems**: User chỉ quan tâm item đầu tiên relevant
- **RecSys**: Ít dùng hơn Recall/NDCG (users scan nhiều items)

### 5. MAP@K (Mean Average Precision)

#### Definition
**MAP@K** là trung bình của Precision tại mỗi vị trí relevant item trong top-K.

#### Formula
```
AP@K = (1/|Rel_K|) * Σ(k=1 to K) [Precision@k * rel_k]
MAP@K = Average(AP@K) across users
```
Trong đó `Rel_K` = relevant items trong top-K

#### Interpretation
- **Combines**: Precision và ranking quality
- **Stricter than Recall**: Penalizes relevant items ở vị trí thấp
- **Range**: [0, 1], higher is better

### 6. Coverage

#### Definition
**Coverage** đo tỷ lệ unique items được recommend cho tất cả users.

#### Formula
```
Coverage = |Unique Items in All Recommendations| / |Total Items|
```

#### Purpose
- **Diversity metric**: High coverage → diverse recommendations
- **Business metric**: Expose more products → sales
- **Trade-off**: Accuracy vs diversity (popular items → low coverage)

#### Interpretation
- **Coverage = 0.3**: 30% sản phẩm được recommend ít nhất 1 lần
- **Baseline**: Popularity recommender có coverage rất thấp (<0.05)
- **Target**: CF thường có coverage 0.2-0.5

## Baseline Comparisons

### 1. Popularity Baseline

#### Method 1: Training Frequency
- **Source**: `item_popularity` array từ Task 01
- **Logic**: Rank items theo số lần xuất hiện trong train data
- **Pros**: Simple, data-driven
- **Cons**: Không personalized

#### Method 2: Product Metadata
- **Source**: `num_sold_time` từ `data_product.csv`
- **Logic**: Rank items theo số lượng đã bán (external signal)
- **Pros**: Reflects real-world popularity
- **Cons**: Có thể stale (data cũ)

#### Recommendation Logic
```python
def popularity_recommendations(test_users, item_popularity, k):
    # Rank items by popularity
    top_k_items = argsort(item_popularity)[::-1][:k]
    
    # Same recommendations for all users
    recommendations = {u: top_k_items for u in test_users}
    return recommendations
```

#### Expected Performance
- **Recall@10**: 0.12 - 0.15
- **NDCG@10**: 0.08 - 0.10
- **Coverage**: <0.05 (very low)

### 2. Random Baseline

#### Method
- **Logic**: Sample K items uniformly random (exclude seen)
- **Purpose**: Lower bound sanity check

#### Expected Performance
- **Recall@10**: ~0.01 (very low)
- **NDCG@10**: ~0.005

### 3. Comparison Metrics

#### Improvement Percentage
```
Improvement = (CF_Metric - Baseline_Metric) / Baseline_Metric * 100%
```

#### Statistical Significance
- **Test**: Paired t-test trên per-user metrics
- **Null hypothesis**: CF và Baseline có cùng mean metric
- **Threshold**: p-value < 0.05 → significant improvement

## Evaluation Workflow

### Function: `evaluate_model(U, V, test_data, user_pos_train, k_values)`

#### Step 1: Generate Recommendations
```python
# For all test users
test_users = test_data['u_idx'].unique()
all_recommendations = {}

for u in test_users:
    # Compute scores
    scores = U[u] @ V.T  # Shape: (num_items,)
    
    # Mask seen items
    seen_items = user_pos_train.get(u, set())
    scores[list(seen_items)] = -np.inf
    
    # Top-K
    top_k_indices = np.argsort(scores)[::-1][:max(k_values)]
    all_recommendations[u] = top_k_indices
```

#### Step 2: Prepare Ground Truth
```python
# Extract test positives per user
ground_truth = {}
for u in test_users:
    ground_truth[u] = set(test_data[test_data['u_idx'] == u]['i_idx'])
```

#### Step 3: Compute Metrics
```python
results = {}
for k in k_values:  # e.g., [10, 20]
    recalls = []
    ndcgs = []
    
    for u in test_users:
        pred_k = all_recommendations[u][:k]
        gt_u = ground_truth[u]
        
        recalls.append(recall_at_k(pred_k, gt_u, k))
        ndcgs.append(ndcg_at_k(pred_k, gt_u, k))
    
    # Average across users
    results[f'recall@{k}'] = np.mean(recalls)
    results[f'ndcg@{k}'] = np.mean(ndcgs)
```

#### Step 4: Compute Coverage
```python
all_recommended_items = set()
for recs in all_recommendations.values():
    all_recommended_items.update(recs)

results['coverage'] = len(all_recommended_items) / num_items
```

#### Step 5: Return Results
```python
return {
    'recall@10': 0.234,
    'recall@20': 0.312,
    'ndcg@10': 0.189,
    'ndcg@20': 0.221,
    'coverage': 0.287,
    'num_users_evaluated': len(test_users)
}
```

## Reporting & Visualization

### 1. Summary Table

#### CSV Format: `reports/cf_eval_summary.csv`
```csv
model,factors,reg,alpha,recall@10,recall@20,ndcg@10,ndcg@20,coverage,training_time
als,64,0.01,40,0.234,0.312,0.189,0.221,0.287,45.2
bpr,64,0.0001,NA,0.242,0.321,0.195,0.228,0.301,1824.5
popularity,NA,NA,NA,0.145,0.201,0.102,0.134,0.042,0.1
```

#### DataFrame Operations
```python
import pandas as pd

# Load all experiment results
df = pd.read_csv('reports/cf_eval_summary.csv')

# Sort by NDCG@10 (best model selection)
df_sorted = df.sort_values('ndcg@10', ascending=False)

# Filter models beating baseline
baseline_ndcg = df[df['model'] == 'popularity']['ndcg@10'].values[0]
df_better = df[df['ndcg@10'] > baseline_ndcg * 1.1]  # >10% improvement
```

### 2. Comparison Plots

#### Metric Bar Chart
- **X-axis**: Models (ALS, BPR, Popularity)
- **Y-axis**: Recall@10 / NDCG@10
- **Visualization**: Side-by-side bars
- **Highlight**: Best model

#### K-Value Sensitivity
- **X-axis**: K (5, 10, 20, 50)
- **Y-axis**: Recall@K
- **Lines**: ALS, BPR, Popularity
- **Purpose**: Show recall increases với K

#### Coverage vs Accuracy Trade-off
- **X-axis**: Coverage
- **Y-axis**: NDCG@10
- **Points**: Each experiment config
- **Insight**: High accuracy models có thể low coverage

### 3. Per-User Analysis (Advanced)

#### Distribution Plots
- **Metric**: Recall@10 per user (histogram)
- **Purpose**: Identify users với poor recommendations
- **Action**: Investigate cold-start, niche users

#### Stratification by Activity
- **Bins**: Users theo số interactions (low/medium/high)
- **Metrics**: Recall@10 per bin
- **Insight**: CF works better cho active users

## Statistical Testing

### Paired t-Test

#### Hypothesis
- **H0**: Mean difference giữa CF và Baseline = 0
- **H1**: CF có mean metric cao hơn Baseline

#### Implementation
```python
from scipy.stats import ttest_rel

# Per-user metrics
cf_recalls = [recall_at_k(cf_recs[u], gt[u], 10) for u in users]
baseline_recalls = [recall_at_k(pop_recs[u], gt[u], 10) for u in users]

# Paired t-test
t_stat, p_value = ttest_rel(cf_recalls, baseline_recalls)

if p_value < 0.05:
    print(f"CF significantly better than baseline (p={p_value:.4f})")
```

### Effect Size (Cohen's d)

#### Formula
```
d = (mean_CF - mean_Baseline) / pooled_std
```

#### Interpretation
- **d < 0.2**: Small effect
- **d = 0.5**: Medium effect
- **d > 0.8**: Large effect

## Error Analysis

### 1. Failure Cases

#### Cold-Start Users
- **Definition**: Users với <3 train interactions
- **Expected**: Low Recall (no signal)
- **Mitigation**: Fallback popularity, hybrid với content

#### Niche Items
- **Definition**: Items với <10 train interactions
- **Expected**: Never recommended (low embedding quality)
- **Mitigation**: Content-based boost, attribute filtering

### 2. Debugging Tools

#### Inspect Recommendations
```python
def inspect_user_recommendations(u_id, model, mappings, products_df, k=10):
    # Map user_id → u_idx
    u_idx = mappings['user_to_idx'][u_id]
    
    # Get recommendations
    scores = model.U[u_idx] @ model.V.T
    top_k = np.argsort(scores)[::-1][:k]
    
    # Map i_idx → product_id
    product_ids = [mappings['idx_to_item'][i] for i in top_k]
    
    # Enrich với metadata
    recs = products_df[products_df['product_id'].isin(product_ids)]
    print(recs[['product_id', 'product_name', 'brand', 'avg_star', 'num_sold_time']])
```

#### Embedding Visualization
- **Method**: t-SNE hoặc UMAP reduce item embeddings V → 2D
- **Plot**: Scatter với color by category/brand
- **Insight**: Kiểm tra embeddings có cluster theo attributes không

## Performance Optimization

### 1. Batch Evaluation

#### Problem
Computing `U @ V.T` cho all users → memory explosion

#### Solution
```python
# Evaluate in batches
batch_size = 1000
for i in range(0, num_users, batch_size):
    u_batch = range(i, min(i + batch_size, num_users))
    scores_batch = U[u_batch] @ V.T  # Shape: (batch_size, num_items)
    # Generate recs for batch
```

### 2. Top-K Optimization

#### Problem
`np.argsort` sorts all items (expensive cho large catalogs)

#### Solution
```python
# Use argpartition (O(n) vs O(n log n))
top_k_unsorted = np.argpartition(scores, -k)[-k:]
top_k_sorted = top_k_unsorted[np.argsort(scores[top_k_unsorted])][::-1]
```

### 3. Sparse Filtering

#### Problem
Masking seen items với `-inf` inefficient

#### Solution
```python
# Precompute candidate items per user
candidates = {u: set(range(num_items)) - user_pos_train[u] for u in users}

# Evaluate only candidates
for u in users:
    cand_indices = list(candidates[u])
    cand_scores = scores[cand_indices]
    top_k = cand_indices[np.argsort(cand_scores)[::-1][:k]]
```

## Dependencies

```python
# requirements_metrics.txt
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.9.0  # For statistical tests
scikit-learn>=1.2.0  # For metrics utilities
matplotlib>=3.6.0  # For plotting
seaborn>=0.12.0  # For nice plots
```

## Module Documentation

### Example Usage
```python
from recsys.cf.metrics import evaluate_model, evaluate_baseline_popularity

# Load model and data
U, V = load_model('artifacts/cf/als/')
test_data, user_pos_train, item_popularity = load_test_data()

# Evaluate CF model
cf_metrics = evaluate_model(
    U, V, test_data, user_pos_train, k_values=[10, 20]
)
print(f"ALS Recall@10: {cf_metrics['recall@10']:.3f}")

# Evaluate baseline
baseline_metrics = evaluate_baseline_popularity(
    test_data, item_popularity, k_values=[10, 20]
)
print(f"Baseline Recall@10: {baseline_metrics['recall@10']:.3f}")

# Compare
improvement = (cf_metrics['recall@10'] - baseline_metrics['recall@10']) / baseline_metrics['recall@10']
print(f"Improvement: {improvement:.1%}")
```

## Timeline Estimate

- **Implementation**: 1.5 days
- **Testing**: 0.5 day
- **Visualization**: 0.5 day
- **Documentation**: 0.5 day
- **Total**: ~3 days

## Success Criteria

- [ ] Module computes Recall@K, NDCG@K correctly (unit tests)
- [ ] Evaluation runs <30 seconds cho 12K users
- [ ] Baseline comparison shows CF beats popularity by ≥20%
- [ ] Statistical tests confirm significance (p < 0.05)
- [ ] Visualizations clear và informative
- [ ] Per-user analysis identifies failure cases
- [ ] Code documented với docstrings và examples
