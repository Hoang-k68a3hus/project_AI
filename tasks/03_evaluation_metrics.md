# Task 03: Evaluation & Metrics

## Mục Tiêu

Xây dựng module đánh giá toàn diện cho Collaborative Filtering, bao gồm metrics chuẩn RecSys, baseline comparisons, và statistical significance testing. Module này được chia sẻ giữa ALS và BPR pipelines.

## Module Architecture Overview

Code đã được tổ chức thành **class-based architecture** với các modules riêng biệt:

```
recsys/cf/evaluation/
├── __init__.py              # Package exports
├── metrics.py               # Core metrics (Recall, NDCG, Precision, MRR, MAP, Coverage)
├── model_evaluator.py       # ModelEvaluator, BatchModelEvaluator
├── baseline_evaluator.py    # PopularityBaseline, RandomBaseline, ItemSimilarityBaseline
├── hybrid_metrics.py        # Diversity, SemanticAlignment, ColdStartCoverage, Novelty
├── comparison.py            # ModelComparator, ReportGenerator, EvaluationVisualizer
└── statistical_tests.py    # StatisticalTester, BootstrapEstimator
```

## Module Structure

### Package: `recsys/cf/evaluation`

#### Core Exports

**Core Metrics** (`metrics.py`):
- `BaseMetric`: Abstract base class for all metrics
- `RecallAtK`, `PrecisionAtK`, `NDCGAtK`, `MRR`, `MAPAtK`, `HitRate`: Ranking metrics
- `Coverage`: Catalog coverage metric
- `MetricFactory`: Factory for creating standard metric sets
- Convenience functions: `recall_at_k()`, `ndcg_at_k()`, `precision_at_k()`, `mrr()`, `map_at_k()`, `coverage()`

**Model Evaluator** (`model_evaluator.py`):
- `ModelEvaluator`: Main class for evaluating CF models (ALS, BPR)
- `BatchModelEvaluator`: Evaluate and compare multiple models simultaneously
- Convenience functions: `evaluate_model()`, `load_and_evaluate()`

**Baseline Evaluators** (`baseline_evaluator.py`):
- `BaselineRecommender`: Abstract base class
- `PopularityBaseline`: Popularity-based recommendations
- `RandomBaseline`: Random recommendations
- `ItemSimilarityBaseline`: Content-based similarity baseline
- `BaselineComparator`: Compare baselines
- Convenience functions: `evaluate_baseline_popularity()`, `evaluate_baseline_random()`

**Hybrid Metrics** (`hybrid_metrics.py`):
- `HybridMetric`: Abstract base class
- `DiversityMetric`: Intra-list diversity using embeddings
- `SemanticAlignmentMetric`: Alignment with user content profile
- `ColdStartCoverageMetric`: Coverage of cold-start items
- `NoveltyMetric`: Recommendation novelty (long-tail items)
- `SerendipityMetric`: Serendipity score (surprising yet relevant)
- `HybridMetricCollection`: Collection of hybrid metrics for batch evaluation
- Convenience functions: `compute_diversity_bert()`, `compute_semantic_alignment()`, `compute_cold_start_coverage()`

**Comparison & Reporting** (`comparison.py`):
- `ModelComparator`: Statistical comparison of models
- `ReportGenerator`: Generate CSV/JSON/Markdown reports
- `EvaluationVisualizer`: Prepare data for visualizations
- Convenience functions: `compare_models()`, `generate_evaluation_report()`

**Statistical Testing** (`statistical_tests.py`):
- `StatisticalTester`: Paired t-tests, Wilcoxon, Cohen's d, multiple comparisons
- `BootstrapEstimator`: Bootstrap confidence intervals, permutation tests
- Convenience functions: `paired_t_test()`, `cohens_d()`

## Core Metrics

### 1. Recall@K

#### Definition
**Recall@K** đo lường tỷ lệ relevant items (test positives) xuất hiện trong top-K recommendations.

#### Formula
```
Recall@K = |Top-K ∩ Test_Items| / |Test_Items|
```

#### Implementation
- **Module**: `recsys/cf/evaluation/metrics.py`
- **Class**: `RecallAtK(BaseMetric)`
- **Method**: `compute(predictions, ground_truth, k=None)`

**Usage**:
```python
from recsys.cf.evaluation import RecallAtK, recall_at_k

# Method 1: Using class
metric = RecallAtK(k=10)
score = metric.compute(predictions=[1, 5, 3, 8, 2], ground_truth={3, 8, 10})

# Method 2: Convenience function
score = recall_at_k(predictions=[1, 5, 3, 8, 2], ground_truth={3, 8, 10}, k=10)
```

#### Implementation Details
- **Input**:
  - `predictions`: List/array of item indices (recommended)
  - `ground_truth`: Set of positive item indices (test)
  - `k`: Integer cutoff (optional, uses class default if not provided)
- **Output**: Float [0, 1]
- **Edge cases**:
  - Nếu `|Test_Items| = 0` → return 0.0 (no relevant items to find)
  - Nếu `K > num_predictions` → use len(predictions)
  - Input validation via `BaseMetric.validate_inputs()`

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

#### Implementation
- **Module**: `recsys/cf/evaluation/metrics.py`
- **Class**: `NDCGAtK(BaseMetric)`
- **Method**: `compute(predictions, ground_truth, k=None)`

**Usage**:
```python
from recsys.cf.evaluation import NDCGAtK, ndcg_at_k

# Method 1: Using class
metric = NDCGAtK(k=10)
score = metric.compute(predictions=[1, 5, 3, 8, 2], ground_truth={3, 8, 10})

# Method 2: Convenience function
score = ndcg_at_k(predictions=[1, 5, 3, 8, 2], ground_truth={3, 8, 10}, k=10)
```

#### Relevance Definition
- **Binary**: rel_i = 1 nếu item_i in test positives, else 0
- **Graded** (optional): rel_i = rating (1-5) nếu có explicit ratings (not implemented in current version)

#### Implementation Details
- **Discounting**: `log2(position + 1)` → vị trí 1,2,3,... có weight 1.0, 0.63, 0.5, ...
- **Normalization**: Divide bằng IDCG (ideal DCG) để scale [0, 1]
- **Edge cases**:
  - Nếu không có relevant items trong top-K → DCG = 0, NDCG = 0
  - Nếu IDCG = 0 (no test positives) → return 0.0
  - Input validation via `BaseMetric.validate_inputs()`

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

#### Implementation
- **Module**: `recsys/cf/evaluation/metrics.py`
- **Class**: `PrecisionAtK(BaseMetric)`
- **Method**: `compute(predictions, ground_truth, k=None)`

**Usage**:
```python
from recsys.cf/evaluation import PrecisionAtK, precision_at_k

# Method 1: Using class
metric = PrecisionAtK(k=10)
score = metric.compute(predictions=[1, 5, 3, 8, 2], ground_truth={3, 8, 10})

# Method 2: Convenience function
score = precision_at_k(predictions=[1, 5, 3, 8, 2], ground_truth={3, 8, 10}, k=10)
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
- **Module**: `recsys/cf/evaluation/metrics.py`
- **Class**: `MRR(BaseMetric)`
- **Method**: `compute(predictions, ground_truth)`

**Usage**:
```python
from recsys.cf.evaluation import MRR, mrr

# Method 1: Using class
metric = MRR()
score = metric.compute(predictions=[1, 5, 3, 8, 2], ground_truth={3, 8, 10})

# Method 2: Convenience function
score = mrr(predictions=[1, 5, 3, 8, 2], ground_truth={3, 8, 10})
```

#### Implementation Details
- **Find rank**: Vị trí (1-indexed) của test item đầu tiên trong predictions
- **Reciprocal**: 1/rank → rank 1 = 1.0, rank 2 = 0.5, rank 10 = 0.1
- **Average**: Across users (computed in `ModelEvaluator`)

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

#### Implementation
- **Module**: `recsys/cf/evaluation/metrics.py`
- **Class**: `MAPAtK(BaseMetric)`
- **Method**: `compute(predictions, ground_truth, k=None)`

**Usage**:
```python
from recsys.cf.evaluation import MAPAtK, map_at_k

# Method 1: Using class
metric = MAPAtK(k=10)
score = metric.compute(predictions=[1, 5, 3, 8, 2], ground_truth={3, 8, 10})

# Method 2: Convenience function
score = map_at_k(predictions=[1, 5, 3, 8, 2], ground_truth={3, 8, 10}, k=10)
```

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

#### Implementation
- **Module**: `recsys/cf/evaluation/metrics.py`
- **Class**: `Coverage(BaseMetric)`
- **Method**: `compute(recommendations, num_total_items)`

**Usage**:
```python
from recsys.cf.evaluation import Coverage, coverage

# Method 1: Using class
metric = Coverage()
score = metric.compute(
    recommendations={0: [1, 5, 3], 1: [2, 5, 8], 2: [3, 7, 9]},
    num_total_items=100
)

# Method 2: Convenience function
score = coverage(
    recommendations={0: [1, 5, 3], 1: [2, 5, 8], 2: [3, 7, 9]},
    num_total_items=100
)
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

#### Implementation
- **Module**: `recsys/cf/evaluation/baseline_evaluator.py`
- **Class**: `PopularityBaseline(BaselineRecommender)`
- **Methods**:
  - `recommend(user_idx, k, exclude_items)`: Generate recommendations for single user
  - `recommend_batch(user_indices, k, user_exclude_items)`: Batch recommendations
  - `evaluate(test_data, user_pos_train, k_values)`: Full evaluation

**Usage**:
```python
from recsys.cf.evaluation import PopularityBaseline, evaluate_baseline_popularity

# Method 1: Using class
baseline = PopularityBaseline(
    item_popularity=item_popularity,  # Array from Task 01
    num_items=2231
)
results = baseline.evaluate(
    test_data=test_data,
    user_pos_train=user_pos_train,
    k_values=[10, 20]
)

# Method 2: Convenience function
results = evaluate_baseline_popularity(
    test_data=test_data,
    item_popularity=item_popularity,
    user_pos_train=user_pos_train,
    k_values=[10, 20]
)
```

#### Method 1: Training Frequency
- **Source**: `item_popularity` array từ Task 01 (log-transformed interaction counts)
- **Logic**: Rank items theo số lần xuất hiện trong train data
- **Pros**: Simple, data-driven
- **Cons**: Không personalized

#### Method 2: Product Metadata
- **Source**: `num_sold_time` từ `data_product.csv`
- **Logic**: Rank items theo số lượng đã bán (external signal)
- **Pros**: Reflects real-world popularity
- **Cons**: Có thể stale (data cũ)
- **Note**: Can be implemented by passing `num_sold_time` as `item_popularity`

#### Recommendation Logic
- **Same recommendations for all users**: Top-K popular items (excluding seen items)
- **Efficient**: Pre-compute ranked list once, filter per user

#### Expected Performance
- **Recall@10**: 0.12 - 0.15
- **NDCG@10**: 0.08 - 0.10
- **Coverage**: <0.05 (very low)

### 2. Random Baseline

#### Implementation
- **Module**: `recsys/cf/evaluation/baseline_evaluator.py`
- **Class**: `RandomBaseline(BaselineRecommender)`
- **Method**: `recommend(user_idx, k, exclude_items)`: Random sampling

**Usage**:
```python
from recsys.cf.evaluation import RandomBaseline, evaluate_baseline_random

# Method 1: Using class
baseline = RandomBaseline(num_items=2231, random_seed=42)
results = baseline.evaluate(test_data, user_pos_train, k_values=[10, 20])

# Method 2: Convenience function
results = evaluate_baseline_random(
    test_data=test_data,
    num_items=2231,
    user_pos_train=user_pos_train,
    k_values=[10, 20],
    random_seed=42
)
```

#### Method
- **Logic**: Sample K items uniformly random (exclude seen)
- **Purpose**: Lower bound sanity check
- **Reproducible**: Uses random seed for consistency

#### Expected Performance
- **Recall@10**: ~0.01 (very low)
- **NDCG@10**: ~0.005

### 3. Item Similarity Baseline (Content-Based)

#### Implementation
- **Module**: `recsys/cf/evaluation/baseline_evaluator.py`
- **Class**: `ItemSimilarityBaseline(BaselineRecommender)`
- **Method**: Uses item-item similarity matrix (e.g., from BERT embeddings)

**Usage**:
```python
from recsys.cf.evaluation import ItemSimilarityBaseline

# Requires similarity matrix (e.g., from BERT embeddings)
similarity_matrix = compute_item_similarity(bert_embeddings)  # (num_items, num_items)

baseline = ItemSimilarityBaseline(
    similarity_matrix=similarity_matrix,
    num_items=2231
)
results = baseline.evaluate(test_data, user_pos_train, k_values=[10, 20])
```

### 4. Comparison Metrics

#### Improvement Percentage
```
Improvement = (CF_Metric - Baseline_Metric) / Baseline_Metric * 100%
```

#### Implementation
- **Module**: `recsys/cf/evaluation/comparison.py`
- **Class**: `ModelComparator`
- **Method**: `compute_improvement(model_name, baseline_name, metric)`

**Usage**:
```python
from recsys.cf.evaluation import ModelComparator

comparator = ModelComparator()
comparator.add_model_results('als', als_results)
comparator.add_baseline_results('popularity', pop_results)

improvement = comparator.compute_improvement('als', 'popularity', 'recall@10')
print(f"Improvement: {improvement['relative_percent']:.1f}%")
```

#### Statistical Significance
- **Module**: `recsys/cf/evaluation/statistical_tests.py`
- **Class**: `StatisticalTester`
- **Tests**: 
  - Paired t-test: `paired_t_test(sample1, sample2)`
  - Wilcoxon signed-rank test: `wilcoxon_test(sample1, sample2)` (non-parametric)
  - Effect size: `cohens_d(sample1, sample2)`
- **Null hypothesis**: CF và Baseline có cùng mean metric
- **Threshold**: p-value < 0.05 → significant improvement

**Usage**:
```python
from recsys.cf.evaluation import StatisticalTester

tester = StatisticalTester(significance_level=0.05)
result = tester.paired_t_test(
    model_scores=als_per_user_recall,
    baseline_scores=pop_per_user_recall,
    alternative='greater'  # Test if model > baseline
)

if result['significant']:
    print(f"Significant improvement (p={result['p_value']:.4f})")
```

## Evaluation Workflow

### Model Evaluator

#### Class: `ModelEvaluator`

- **Module**: `recsys/cf/evaluation/model_evaluator.py`
- **Features**:
  - Batch recommendation generation (memory efficient)
  - Multi-metric evaluation (Recall, NDCG, Precision, MRR, MAP)
  - Coverage analysis
  - Per-user metrics for analysis
  - Efficient seen-item filtering
  - Top-K optimization (argpartition for large K)

**Usage**:
```python
from recsys.cf.evaluation import ModelEvaluator, evaluate_model

# Method 1: Using class
evaluator = ModelEvaluator(
    U=U,  # User embeddings (num_users, factors)
    V=V,  # Item embeddings (num_items, factors)
    k_values=[10, 20],
    batch_size=1000,  # Memory optimization
    use_argpartition=True  # O(n) vs O(n log n) for top-K
)

results = evaluator.evaluate(
    test_data=test_data,  # DataFrame or Dict
    user_pos_train=user_pos_train,  # Dict[u_idx, Set[i_idx]]
    user_col='u_idx',
    item_col='i_idx'
)

# Access results
print(f"Recall@10: {results['recall@10']:.4f}")
print(f"NDCG@10: {results['ndcg@10']:.4f}")
print(f"Coverage: {results['coverage']:.4f}")

# Per-user metrics
per_user = evaluator.get_per_user_metrics()

# Method 2: Convenience function
results = evaluate_model(
    U=U,
    V=V,
    test_data=test_data,
    user_pos_train=user_pos_train,
    k_values=[10, 20]
)
```

#### Batch Model Evaluator

- **Class**: `BatchModelEvaluator`
- **Purpose**: Evaluate and compare multiple models simultaneously
- **Features**:
  - Parallel evaluation of multiple models
  - Consistent evaluation setup
  - Comparison table generation

**Usage**:
```python
from recsys.cf.evaluation import BatchModelEvaluator

batch_evaluator = BatchModelEvaluator(k_values=[10, 20])

# Add models
batch_evaluator.add_model('als', U_als, V_als)
batch_evaluator.add_model('bpr', U_bpr, V_bpr)

# Evaluate all models
results = batch_evaluator.evaluate_all(
    test_data=test_data,
    user_pos_train=user_pos_train
)

# Get comparison table
comparison = batch_evaluator.get_comparison_table()
print(comparison)
```

#### Step 1: Generate Recommendations (Batch Processing)
- **Method**: `generate_recommendations(test_users, user_pos_train, k)`
- **Optimization**: 
  - Batch processing: `U[batch] @ V.T` for memory efficiency
  - Top-K optimization: Uses `argpartition` (O(n)) instead of `argsort` (O(n log n)) when K < n/2
  - Efficient filtering: Masks seen items with `-np.inf`

#### Step 2: Prepare Ground Truth
- **Method**: `_prepare_ground_truth(test_data, user_col, item_col)`
- **Supports**: Both DataFrame and Dict formats
- **Output**: Dict[u_idx, Set[i_idx]]

#### Step 3: Compute Metrics
- **Method**: `_compute_metrics(recommendations, ground_truth, k_values)`
- **Metrics**: Uses `MetricFactory.create_standard_metrics(k_values)` to create metric instances
- **Per-user**: Stores per-user metrics for statistical testing

#### Step 4: Compute Coverage
- **Method**: Uses `Coverage` metric class
- **Input**: All recommendations dict
- **Output**: Coverage score [0, 1]

#### Step 5: Return Results
```python
{
    'recall@10': 0.234,
    'recall@20': 0.312,
    'ndcg@10': 0.189,
    'ndcg@20': 0.221,
    'precision@10': 0.156,
    'mrr': 0.342,
    'map@10': 0.178,
    'coverage': 0.287,
    'num_users_evaluated': len(test_users),
    'evaluation_time_seconds': 12.5
}
```

## Reporting & Visualization

### Report Generator

#### Implementation
- **Module**: `recsys/cf/evaluation/comparison.py`
- **Class**: `ReportGenerator`
- **Features**:
  - Generate CSV/JSON/Markdown reports
  - Summary tables
  - Comparison tables with improvements
  - Statistical significance annotations

**Usage**:
```python
from recsys.cf.evaluation import ReportGenerator, generate_evaluation_report

# Method 1: Using class
generator = ReportGenerator(output_dir='reports/')
generator.add_results('als', als_results, metadata={'factors': 64, 'reg': 0.01})
generator.add_results('bpr', bpr_results, metadata={'factors': 64, 'lr': 0.05})
generator.add_results('popularity', pop_results, metadata={})

# Generate reports
generator.generate_csv_report('cf_eval_summary.csv')
generator.generate_json_report('cf_eval_summary.json')
generator.generate_markdown_report('cf_eval_summary.md')

# Method 2: Convenience function
generate_evaluation_report(
    results={'als': als_results, 'bpr': bpr_results, 'popularity': pop_results},
    output_path='reports/cf_eval_summary.csv',
    format='csv'
)
```

### 1. Summary Table

#### CSV Format: `reports/cf_eval_summary.csv`
```csv
model,factors,reg,alpha,recall@10,recall@20,ndcg@10,ndcg@20,coverage,training_time
als,64,0.01,10,0.234,0.312,0.189,0.221,0.287,45.2
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

#### Evaluation Visualizer

- **Module**: `recsys/cf/evaluation/comparison.py`
- **Class**: `EvaluationVisualizer`
- **Features**: Prepare data for visualization (does not create plots, but prepares data)

**Usage**:
```python
from recsys.cf.evaluation import EvaluationVisualizer

visualizer = EvaluationVisualizer()

# Prepare data for bar chart
bar_data = visualizer.prepare_bar_chart_data(
    results={'als': als_results, 'bpr': bpr_results, 'popularity': pop_results},
    metrics=['recall@10', 'ndcg@10']
)

# Prepare data for K-value sensitivity
k_sensitivity_data = visualizer.prepare_k_sensitivity_data(
    results={'als': als_results, 'bpr': bpr_results},
    metric='recall'
)

# Prepare data for coverage vs accuracy
tradeoff_data = visualizer.prepare_tradeoff_data(
    results={'als': als_results, 'bpr': bpr_results},
    accuracy_metric='ndcg@10'
)
```

#### Metric Bar Chart
- **X-axis**: Models (ALS, BPR, Popularity)
- **Y-axis**: Recall@10 / NDCG@10
- **Visualization**: Side-by-side bars
- **Highlight**: Best model
- **Note**: Use `EvaluationVisualizer` to prepare data, then plot with matplotlib/seaborn

#### K-Value Sensitivity
- **X-axis**: K (5, 10, 20, 50)
- **Y-axis**: Recall@K
- **Lines**: ALS, BPR, Popularity
- **Purpose**: Show recall increases với K
- **Data preparation**: Use `prepare_k_sensitivity_data()`

#### Coverage vs Accuracy Trade-off
- **X-axis**: Coverage
- **Y-axis**: NDCG@10
- **Points**: Each experiment config
- **Insight**: High accuracy models có thể low coverage
- **Data preparation**: Use `prepare_tradeoff_data()`

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

### Statistical Tester

#### Implementation
- **Module**: `recsys/cf/evaluation/statistical_tests.py`
- **Class**: `StatisticalTester`
- **Features**:
  - Paired t-tests
  - Wilcoxon signed-rank tests (non-parametric)
  - Effect size (Cohen's d, Glass's delta)
  - Confidence intervals
  - Multiple comparison corrections (Bonferroni, Holm-Bonferroni)

**Usage**:
```python
from recsys.cf.evaluation import StatisticalTester, paired_t_test, cohens_d

# Method 1: Using class
tester = StatisticalTester(significance_level=0.05)

# Paired t-test
result = tester.paired_t_test(
    sample1=cf_per_user_recall,  # Model per-user metrics
    sample2=baseline_per_user_recall,  # Baseline per-user metrics
    alternative='greater'  # Test if model > baseline
)

if result['significant']:
    print(f"Significant improvement (p={result['p_value']:.4f})")
    print(f"Mean difference: {result['mean_diff']:.4f}")

# Wilcoxon test (non-parametric, more robust)
wilcoxon_result = tester.wilcoxon_test(
    sample1=cf_per_user_recall,
    sample2=baseline_per_user_recall,
    alternative='greater'
)

# Effect size
effect_size = tester.cohens_d(cf_per_user_recall, baseline_per_user_recall)
print(f"Effect size (Cohen's d): {effect_size['cohens_d']:.3f}")

# Method 2: Convenience functions
result = paired_t_test(cf_per_user_recall, baseline_per_user_recall)
d = cohens_d(cf_per_user_recall, baseline_per_user_recall)
```

### Paired t-Test

#### Hypothesis
- **H0**: Mean difference giữa CF và Baseline = 0
- **H1**: CF có mean metric cao hơn Baseline (one-sided) hoặc different (two-sided)

#### Implementation Details
- **Test**: `scipy.stats.ttest_rel` (paired t-test)
- **Input**: Per-user metrics from both model and baseline
- **Output**: t-statistic, p-value, significance flag, mean difference, std difference
- **Edge cases**: Handles small sample sizes (<3) with warnings

### Effect Size (Cohen's d)

#### Formula
```
d = (mean_CF - mean_Baseline) / pooled_std
```

#### Implementation
- **Method**: `cohens_d(sample1, sample2)`
- **Output**: Cohen's d value and interpretation (small/medium/large)

#### Interpretation
- **d < 0.2**: Small effect
- **d = 0.5**: Medium effect
- **d > 0.8**: Large effect

### Bootstrap Estimator

#### Implementation
- **Class**: `BootstrapEstimator`
- **Features**:
  - Bootstrap confidence intervals
  - Permutation tests
  - Non-parametric significance testing

**Usage**:
```python
from recsys.cf.evaluation import BootstrapEstimator

estimator = BootstrapEstimator(n_bootstrap=1000, random_seed=42)

# Bootstrap confidence interval
ci = estimator.bootstrap_ci(
    sample=cf_per_user_recall,
    confidence=0.95
)
print(f"95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")

# Permutation test
perm_result = estimator.permutation_test(
    sample1=cf_per_user_recall,
    sample2=baseline_per_user_recall,
    n_permutations=1000
)
```

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

#### Implementation
- **Class**: `ModelEvaluator` với `batch_size` parameter
- **Default**: `batch_size=1000` (configurable)
- **Method**: `generate_recommendations()` processes users in batches

**Usage**:
```python
evaluator = ModelEvaluator(
    U=U,
    V=V,
    k_values=[10, 20],
    batch_size=1000  # Process 1000 users at a time
)
```

#### Problem
Computing `U @ V.T` cho all users → memory explosion

#### Solution
- **Automatic**: `ModelEvaluator` handles batching internally
- **Batch processing**: `U[batch] @ V.T` for memory efficiency
- **Configurable**: Adjust `batch_size` based on available memory

### 2. Top-K Optimization

#### Implementation
- **Class**: `ModelEvaluator` với `use_argpartition` parameter
- **Default**: `use_argpartition=True`
- **Method**: `_get_top_k()` uses argpartition when K < n/2

**Usage**:
```python
evaluator = ModelEvaluator(
    U=U,
    V=V,
    k_values=[10, 20],
    use_argpartition=True  # Use O(n) argpartition instead of O(n log n) argsort
)
```

#### Problem
`np.argsort` sorts all items (expensive cho large catalogs)

#### Solution
- **Automatic**: `ModelEvaluator` uses `argpartition` when K < n/2
- **Algorithm**: O(n) vs O(n log n) for large catalogs
- **Fallback**: Uses full `argsort` when K is large

### 3. Sparse Filtering

#### Implementation
- **Method**: `_get_top_k()` with `exclude_indices` parameter
- **Efficiency**: Masks excluded items with `-np.inf` before argpartition/argsort
- **Optimization**: Only processes valid indices

#### Problem
Masking seen items với `-inf` inefficient for very sparse data

#### Solution
- **Current**: Efficient masking in `ModelEvaluator._get_top_k()`
- **Alternative**: Could precompute candidate sets (not currently implemented)
- **Note**: Current implementation is efficient for typical sparsity levels

## Hybrid Metrics: CF + BERT Evaluation

### 7. Diversity (Intra-List Diversity)

#### Definition
**Diversity** đo lường mức độ khác biệt giữa các items trong recommendation list.

#### Formula (Content-Based)
```
Diversity = 1 - (1/K(K-1)) * ΣΣ similarity(i, j)  for i ≠ j
```

#### Implementation
- **Module**: `recsys/cf/evaluation/hybrid_metrics.py`
- **Class**: `DiversityMetric(HybridMetric)`
- **Method**: `compute(recommendations, embeddings, item_to_idx)`
- **Similarity**: Cosine similarity between BERT embeddings

**Usage**:
```python
from recsys.cf.evaluation import DiversityMetric, compute_diversity_bert

# Method 1: Using class
diversity_metric = DiversityMetric(similarity_type='cosine')
score = diversity_metric.compute(
    recommendations=[1, 5, 3, 8, 2],
    embeddings=bert_embeddings,  # (num_items, 768)
    item_to_idx=item_to_idx  # Optional mapping
)

# Batch computation
batch_results = diversity_metric.compute_batch(
    all_recommendations={0: [1, 5, 3], 1: [2, 5, 8]},
    embeddings=bert_embeddings,
    item_to_idx=item_to_idx
)
# Returns: {'mean': 0.45, 'std': 0.12, 'min': 0.32, 'max': 0.58}

# Method 2: Convenience function
score = compute_diversity_bert(
    recommendations=[1, 5, 3, 8, 2],
    bert_embeddings=bert_embeddings,
    item_to_idx=item_to_idx
)
```

#### Implementation Details
- **Embeddings**: Uses BERT/PhoBERT embeddings for semantic similarity
- **Normalization**: L2 normalization for cosine similarity
- **Pairwise computation**: Computes all pairwise similarities
- **Edge cases**: Returns 0.0 if <2 items, handles missing embeddings gracefully

#### Interpretation
- **Diversity = 0.3**: Items trong list tương đối similar (avg similarity = 0.7)
- **Diversity = 0.6**: Items khá diverse (avg similarity = 0.4)
- **Trade-off**: High diversity có thể giảm accuracy (recommend less similar items)

### 8. Semantic Alignment Score

#### Definition
Đo lường mức độ CF recommendations align với user content preferences.

#### Formula
```
Alignment = (1/K) * Σ cosine_similarity(user_profile_bert, item_bert_i)
```

#### Implementation
- **Module**: `recsys/cf/evaluation/hybrid_metrics.py`
- **Class**: `SemanticAlignmentMetric(HybridMetric)`
- **Method**: `compute(user_profile_emb, recommendations, embeddings, item_to_idx)`

**Usage**:
```python
from recsys.cf.evaluation import SemanticAlignmentMetric, compute_semantic_alignment

# Method 1: Using class
alignment_metric = SemanticAlignmentMetric()
score = alignment_metric.compute(
    user_profile_emb=user_profile_bert,  # (768,) user profile embedding
    recommendations=[1, 5, 3, 8, 2],
    embeddings=bert_embeddings,
    item_to_idx=item_to_idx
)

# Batch computation
batch_results = alignment_metric.compute_batch(
    user_profiles={0: profile_0, 1: profile_1},
    all_recommendations={0: [1, 5, 3], 1: [2, 5, 8]},
    embeddings=bert_embeddings,
    item_to_idx=item_to_idx
)

# Method 2: Convenience function
score = compute_semantic_alignment(
    user_profile_emb=user_profile_bert,
    recommendations=[1, 5, 3, 8, 2],
    item_embeddings=bert_embeddings,
    item_to_idx=item_to_idx
)
```

#### Implementation Details
- **User Profile**: Weighted average of user's historical item embeddings
- **Similarity**: Cosine similarity between user profile and each recommended item
- **Normalization**: L2 normalization for both user profile and item embeddings
- **Output**: Mean similarity across all recommendations

#### Use Case
- **Evaluate CF quality**: CF recommendations có semantically relevant không?
- **Compare models**: BERT-init ALS có higher alignment với user preferences không?

### 9. Cold-Start Coverage

#### Definition
% cold-start items được recommend ít nhất 1 lần.

#### Formula
```
ColdStartCoverage = |Unique Cold Items in All Recs| / |Total Cold Items|
```

#### Implementation
- **Module**: `recsys/cf/evaluation/hybrid_metrics.py`
- **Class**: `ColdStartCoverageMetric(HybridMetric)`
- **Method**: `compute(all_recommendations, item_counts, cold_threshold)`

**Usage**:
```python
from recsys.cf.evaluation import ColdStartCoverageMetric, compute_cold_start_coverage

# Method 1: Using class
cold_metric = ColdStartCoverageMetric(cold_threshold=5)
score = cold_metric.compute(
    all_recommendations={0: [1, 5, 3], 1: [2, 5, 8]},
    item_counts=item_interaction_counts  # Series or Dict
)

# Method 2: Convenience function
score = compute_cold_start_coverage(
    all_recommendations={0: [1, 5, 3], 1: [2, 5, 8]},
    item_counts=item_interaction_counts,
    cold_threshold=5
)
```

#### Implementation Details
- **Cold Threshold**: Items với <N interactions considered cold-start (default: 5)
- **Identification**: Filters items by interaction count
- **Coverage**: Percentage of cold items that appear in at least one recommendation list
- **Edge cases**: Returns 0.0 if no cold items exist

### 10. Novelty

#### Definition
**Novelty** đo lường mức độ unpopular/surprising của recommendations (long-tail items).

#### Formula
```
Novelty@K = (1/K) * Σ log2(num_users / item_popularity_i)
```

#### Implementation
- **Module**: `recsys/cf/evaluation/hybrid_metrics.py`
- **Class**: `NoveltyMetric(HybridMetric)`
- **Method**: `compute(recommendations, item_popularity, num_users, k=None)`

**Usage**:
```python
from recsys.cf.evaluation import NoveltyMetric

novelty_metric = NoveltyMetric(k=10)
score = novelty_metric.compute(
    recommendations=[1, 5, 3, 8, 2],
    item_popularity=item_popularity,  # Array of popularity scores
    num_users=26000
)

# Batch computation
batch_results = novelty_metric.compute_batch(
    all_recommendations={0: [1, 5, 3], 1: [2, 5, 8]},
    item_popularity=item_popularity,
    num_users=26000
)
```

#### Interpretation
- **High novelty**: Recommending unpopular (long-tail) items
- **Low novelty**: Recommending popular items
- **Trade-off**: High novelty có thể giảm accuracy (less popular items may be less relevant)

### 11. Serendipity

#### Definition
**Serendipity** đo lường mức độ surprising yet relevant của recommendations.

#### Formula
```
Serendipity = (1/K) * Σ [relevant_i * (1 - expected_i)]
```
Trong đó:
- `relevant_i`: 1 nếu item_i in ground truth, else 0
- `expected_i`: Probability item_i appears in baseline recommendations

#### Implementation
- **Module**: `recsys/cf/evaluation/hybrid_metrics.py`
- **Class**: `SerendipityMetric(HybridMetric)`
- **Method**: `compute(recommendations, ground_truth, baseline_recommendations, k=None)`

**Usage**:
```python
from recsys.cf.evaluation import SerendipityMetric

serendipity_metric = SerendipityMetric(k=10)
score = serendipity_metric.compute(
    recommendations=[1, 5, 3, 8, 2],
    ground_truth={3, 8, 10},
    baseline_recommendations=[100, 101, 102, 103, 104]  # Popular items
)
```

#### Interpretation
- **High serendipity**: Novel and relevant recommendations ("pleasant surprises")
- **Measures**: Unexpectedness combined with relevance
- **Use case**: Evaluate if model recommends items user wouldn't find themselves but would like

### Hybrid Evaluation Workflow

#### Hybrid Metric Collection

- **Module**: `recsys/cf/evaluation/hybrid_metrics.py`
- **Class**: `HybridMetricCollection`
- **Purpose**: Evaluate multiple hybrid metrics together

**Usage**:
```python
from recsys.cf.evaluation import (
    HybridMetricCollection,
    DiversityMetric,
    SemanticAlignmentMetric,
    ColdStartCoverageMetric
)

# Create collection
collection = HybridMetricCollection(
    metrics=[
        DiversityMetric(),
        SemanticAlignmentMetric(),
        ColdStartCoverageMetric(cold_threshold=5)
    ]
)

# Evaluate
results = collection.evaluate_all(
    all_recommendations=recommendations,
    embeddings=bert_embeddings,
    user_profiles=user_profiles,
    item_counts=item_counts,
    item_to_idx=item_to_idx
)
```

#### Complete Hybrid Evaluation

**Workflow**:
1. **Standard CF Evaluation**: Use `ModelEvaluator` for Recall, NDCG, etc.
2. **Generate Recommendations**: Get top-K for all test users
3. **Compute Hybrid Metrics**: 
   - Diversity per user → aggregate
   - Semantic alignment per user → aggregate
   - Cold-start coverage (global)
4. **Combine Results**: Merge CF metrics + hybrid metrics

**Example**:
```python
from recsys.cf.evaluation import (
    ModelEvaluator,
    DiversityMetric,
    SemanticAlignmentMetric,
    ColdStartCoverageMetric
)

# Step 1: Standard CF evaluation
cf_evaluator = ModelEvaluator(U, V, k_values=[10, 20])
cf_results = cf_evaluator.evaluate(test_data, user_pos_train)

# Step 2: Get recommendations
recommendations = cf_evaluator.generate_recommendations(
    test_users=test_users,
    user_pos_train=user_pos_train,
    k=10
)

# Step 3: Hybrid metrics
diversity_metric = DiversityMetric()
diversity_results = diversity_metric.compute_batch(
    all_recommendations=recommendations,
    embeddings=bert_embeddings,
    item_to_idx=item_to_idx
)

alignment_metric = SemanticAlignmentMetric()
alignment_results = alignment_metric.compute_batch(
    user_profiles=user_profiles,
    all_recommendations=recommendations,
    embeddings=bert_embeddings,
    item_to_idx=item_to_idx
)

cold_metric = ColdStartCoverageMetric(cold_threshold=5)
cold_coverage = cold_metric.compute(
    all_recommendations=recommendations,
    item_counts=item_counts
)

# Step 4: Combine
hybrid_results = {
    **cf_results,
    'diversity@10': diversity_results['mean'],
    'diversity_std': diversity_results['std'],
    'semantic_alignment@10': alignment_results['mean'],
    'alignment_std': alignment_results['std'],
    'cold_start_coverage': cold_coverage
}
```

### Comparison: CF vs CF+BERT Reranking

#### Evaluation Script: `scripts/compare_cf_hybrid.py`

```python
"""
Compare pure CF vs CF+BERT hybrid reranking.

Usage:
    python scripts/compare_cf_hybrid.py \
        --cf-model artifacts/cf/als/v2_20250116_141500 \
        --bert-embeddings data/processed/content_based_embeddings/product_embeddings.pt
"""

import argparse
import pandas as pd
from recsys.cf.metrics import evaluate_hybrid_model

def main():
    # Load CF model
    U, V = load_cf_model(args.cf_model)
    
    # Load BERT embeddings
    bert_data = torch.load(args.bert_embeddings)
    bert_embeddings = bert_data['embeddings'].numpy()
    
    # Load test data
    test_data = load_test_data()
    
    # Evaluate pure CF
    print("Evaluating Pure CF...")
    cf_results = evaluate_hybrid_model(U, V, bert_embeddings, test_data, user_profiles=None)
    
    # Evaluate CF + BERT reranking
    print("Evaluating CF + BERT Reranking...")
    # Generate user profiles
    user_profiles = compute_all_user_profiles(test_data, bert_embeddings)
    
    # Rerank CF recommendations
    hybrid_results = evaluate_with_reranking(U, V, bert_embeddings, test_data, user_profiles)
    
    # Compare
    comparison = pd.DataFrame({
        'Pure CF': cf_results,
        'CF + BERT': hybrid_results,
        'Improvement': {k: (hybrid_results[k] - cf_results[k]) / cf_results[k] * 100 
                        for k in cf_results.keys()}
    }).T
    
    print("\nComparison:")
    print(comparison)
    
    # Save report
    comparison.to_csv('reports/cf_vs_hybrid_comparison.csv')

if __name__ == '__main__':
    main()
```

#### Expected Results

```
Metric                  | Pure CF | CF + BERT | Improvement
------------------------|---------|-----------|------------
recall@10               | 0.245   | 0.252     | +2.9%
ndcg@10                 | 0.195   | 0.208     | +6.7%
coverage                | 0.310   | 0.298     | -3.9%
diversity@10            | 0.352   | 0.418     | +18.8%
semantic_alignment@10   | 0.412   | 0.531     | +28.9%
cold_start_coverage     | 0.087   | 0.142     | +63.2%
```

**Insights**:
- Hybrid reranking improves **diversity** (+18.8%) và **semantic alignment** (+28.9%)
- **Cold-start coverage** tăng đáng kể (+63.2%) do BERT embeddings cho cold items
- Trade-off: Coverage giảm nhẹ (-3.9%) vì reranking prioritize quality over popularity

## Dependencies

```python
# requirements_metrics.txt
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.9.0  # For statistical tests
scikit-learn>=1.2.0  # For metrics utilities
matplotlib>=3.6.0  # For plotting
seaborn>=0.12.0  # For nice plots

# BERT evaluation
torch>=1.13.0
```

## Module Documentation

### Example Usage

**Complete Evaluation Workflow**:
```python
from recsys.cf.evaluation import (
    ModelEvaluator,
    PopularityBaseline,
    ModelComparator,
    StatisticalTester,
    ReportGenerator
)

# Load model and data
U, V = load_model('artifacts/cf/als/')
test_data, user_pos_train, item_popularity = load_test_data()

# Step 1: Evaluate CF model
cf_evaluator = ModelEvaluator(U, V, k_values=[10, 20])
cf_results = cf_evaluator.evaluate(test_data, user_pos_train)
cf_per_user = cf_evaluator.get_per_user_metrics()

print(f"ALS Recall@10: {cf_results['recall@10']:.3f}")
print(f"ALS NDCG@10: {cf_results['ndcg@10']:.3f}")

# Step 2: Evaluate baseline
baseline = PopularityBaseline(item_popularity, num_items=2231)
baseline_results = baseline.evaluate(test_data, user_pos_train, k_values=[10, 20])
baseline_per_user = baseline.get_per_user_metrics()

print(f"Baseline Recall@10: {baseline_results['recall@10']:.3f}")

# Step 3: Compare with statistical testing
comparator = ModelComparator()
comparator.add_model_results('als', cf_results, cf_per_user)
comparator.add_baseline_results('popularity', baseline_results, baseline_per_user)

comparison_table = comparator.get_comparison_table()
print(comparison_table)

# Statistical significance
tester = StatisticalTester()
stat_result = tester.paired_t_test(
    cf_per_user['recall@10'],
    baseline_per_user['recall@10'],
    alternative='greater'
)

if stat_result['significant']:
    print(f"Significant improvement (p={stat_result['p_value']:.4f})")

# Step 4: Generate report
generator = ReportGenerator(output_dir='reports/')
generator.add_results('als', cf_results)
generator.add_results('popularity', baseline_results)
generator.generate_csv_report('cf_eval_summary.csv')
```

**Quick Evaluation** (Convenience Functions):
```python
from recsys.cf.evaluation import (
    evaluate_model,
    evaluate_baseline_popularity,
    compare_models
)

# Quick evaluation
cf_metrics = evaluate_model(
    U, V, test_data, user_pos_train, k_values=[10, 20]
)

baseline_metrics = evaluate_baseline_popularity(
    test_data, item_popularity, user_pos_train, k_values=[10, 20]
)

# Quick comparison
comparison = compare_models(
    {'als': cf_metrics},
    {'popularity': baseline_metrics}
)
print(comparison)
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
