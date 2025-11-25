"""
Evaluation Module for Collaborative Filtering.

This package provides comprehensive evaluation capabilities for CF models:

Core Metrics:
- recall_at_k: Recall at K
- ndcg_at_k: Normalized Discounted Cumulative Gain at K
- precision_at_k: Precision at K
- mrr: Mean Reciprocal Rank
- map_at_k: Mean Average Precision at K
- coverage: Catalog coverage

Hybrid Metrics (CF + BERT):
- DiversityMetric: Intra-list diversity using embeddings
- SemanticAlignmentMetric: Alignment with user content profile
- ColdStartCoverageMetric: Coverage of cold-start items
- NoveltyMetric: Recommendation novelty

Evaluators:
- ModelEvaluator: Evaluate CF models (ALS, BPR)
- BatchModelEvaluator: Compare multiple models
- PopularityBaseline: Popularity-based baseline
- RandomBaseline: Random baseline
- ItemSimilarityBaseline: Content-based baseline
- BaselineComparator: Compare baselines

Comparison & Reporting:
- ModelComparator: Statistical comparison of models
- ReportGenerator: Generate CSV/JSON/Markdown reports
- EvaluationVisualizer: Prepare data for visualizations

Statistical Testing:
- StatisticalTester: Paired t-tests, Wilcoxon, Cohen's d, multiple comparisons
- BootstrapEstimator: Bootstrap confidence intervals, permutation tests

Example:
    >>> from recsys.cf.evaluation import (
    ...     ModelEvaluator, 
    ...     PopularityBaseline,
    ...     evaluate_model,
    ...     evaluate_baseline_popularity,
    ...     compare_models,
    ...     StatisticalTester
    ... )
    >>> 
    >>> # Evaluate model
    >>> evaluator = ModelEvaluator(U, V, k_values=[10, 20])
    >>> results = evaluator.evaluate(test_data, user_pos_train)
    >>> 
    >>> # Evaluate baseline
    >>> baseline = PopularityBaseline(item_popularity)
    >>> baseline_results = baseline.evaluate(test_data, user_pos_train, k_values=[10, 20])
    >>> 
    >>> # Compare
    >>> comparison = compare_models(
    ...     {'als': results},
    ...     {'popularity': baseline_results}
    ... )
    >>> 
    >>> # Statistical significance testing
    >>> tester = StatisticalTester()
    >>> result = tester.paired_t_test(model_scores, baseline_scores)
"""

# Core Metrics
from .metrics import (
    # Base classes
    BaseMetric,
    MetricFactory,
    
    # Ranking Metrics
    RecallAtK,
    PrecisionAtK,
    NDCGAtK,
    MRR,
    MAPAtK,
    HitRate,
    
    # Coverage
    Coverage,
    
    # Convenience functions
    recall_at_k,
    ndcg_at_k,
    precision_at_k,
    mrr,
    map_at_k,
    coverage,
)

# Hybrid Metrics (CF + BERT)
from .hybrid_metrics import (
    # Base class
    HybridMetric,
    
    # Metrics
    DiversityMetric,
    NoveltyMetric,
    SemanticAlignmentMetric,
    ColdStartCoverageMetric,
    SerendipityMetric,
    
    # Collection
    HybridMetricCollection,
    
    # Convenience functions
    compute_diversity_bert,
    compute_semantic_alignment,
    compute_cold_start_coverage,
)

# Model Evaluator
from .model_evaluator import (
    ModelEvaluator,
    BatchModelEvaluator,
    
    # Convenience functions
    evaluate_model,
    load_and_evaluate,
)

# Baseline Evaluators
from .baseline_evaluator import (
    # Base class
    BaselineRecommender,
    
    # Baselines
    PopularityBaseline,
    RandomBaseline,
    ItemSimilarityBaseline,
    
    # Comparator
    BaselineComparator,
    
    # Convenience functions
    evaluate_baseline_popularity,
    evaluate_baseline_random,
)

# Comparison & Reporting
from .comparison import (
    ModelComparator,
    ReportGenerator,
    EvaluationVisualizer,
    
    # Convenience functions
    compare_models,
    generate_evaluation_report,
)

# Statistical Testing
from .statistical_tests import (
    StatisticalTester,
    BootstrapEstimator,
    paired_t_test,
    cohens_d,
)


__all__ = [
    # Core Metrics
    'BaseMetric',
    'MetricFactory',
    'RecallAtK',
    'PrecisionAtK',
    'NDCGAtK',
    'MRR',
    'MAPAtK',
    'HitRate',
    'Coverage',
    'recall_at_k',
    'ndcg_at_k',
    'precision_at_k',
    'mrr',
    'map_at_k',
    'coverage',
    
    # Hybrid Metrics
    'HybridMetric',
    'DiversityMetric',
    'NoveltyMetric',
    'SemanticAlignmentMetric',
    'ColdStartCoverageMetric',
    'SerendipityMetric',
    'HybridMetricCollection',
    'compute_diversity_bert',
    'compute_semantic_alignment',
    'compute_cold_start_coverage',
    
    # Model Evaluator
    'ModelEvaluator',
    'BatchModelEvaluator',
    'evaluate_model',
    'load_and_evaluate',
    
    # Baselines
    'BaselineRecommender',
    'PopularityBaseline',
    'RandomBaseline',
    'ItemSimilarityBaseline',
    'BaselineComparator',
    'evaluate_baseline_popularity',
    'evaluate_baseline_random',
    
    # Comparison & Reporting
    'ModelComparator',
    'ReportGenerator',
    'EvaluationVisualizer',
    'compare_models',
    'generate_evaluation_report',
    
    # Statistical Testing
    'StatisticalTester',
    'BootstrapEstimator',
    'paired_t_test',
    'cohens_d',
]
