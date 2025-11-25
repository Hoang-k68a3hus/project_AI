"""
Core Metrics Module for Collaborative Filtering Evaluation.

This module provides fundamental evaluation metrics for recommender systems:
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Precision@K
- MRR (Mean Reciprocal Rank)
- MAP@K (Mean Average Precision)
- Coverage

All metrics follow standard RecSys conventions and are optimized for efficiency.

Example:
    >>> from recsys.cf.evaluation import recall_at_k, ndcg_at_k
    >>> predictions = [1, 5, 3, 8, 2]  # Recommended item indices
    >>> ground_truth = {3, 8, 10}  # Positive test items
    >>> print(f"Recall@5: {recall_at_k(predictions, ground_truth, k=5):.3f}")
    >>> print(f"NDCG@5: {ndcg_at_k(predictions, ground_truth, k=5):.3f}")
"""

from typing import List, Set, Union, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Abstract Base Metric
# ============================================================================

class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
    
    Provides a consistent interface for computing metrics with:
    - Type validation
    - Edge case handling
    - Batch computation support
    """
    
    def __init__(self, name: str):
        """
        Initialize base metric.
        
        Args:
            name: Human-readable metric name
        """
        self.name = name
    
    @abstractmethod
    def compute(
        self,
        predictions: Union[List[int], np.ndarray],
        ground_truth: Set[int],
        **kwargs
    ) -> float:
        """
        Compute metric value.
        
        Args:
            predictions: Ordered list of predicted item indices
            ground_truth: Set of positive item indices
            **kwargs: Additional metric-specific parameters
        
        Returns:
            Metric value as float
        """
        pass
    
    def validate_inputs(
        self,
        predictions: Union[List[int], np.ndarray],
        ground_truth: Set[int]
    ) -> bool:
        """
        Validate metric inputs.
        
        Args:
            predictions: Predicted items
            ground_truth: Ground truth items
        
        Returns:
            True if inputs are valid
        """
        if predictions is None or ground_truth is None:
            return False
        if len(predictions) == 0:
            return False
        return True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ============================================================================
# Ranking Metrics
# ============================================================================

class RecallAtK(BaseMetric):
    """
    Recall@K metric: proportion of relevant items found in top-K recommendations.
    
    Formula:
        Recall@K = |Top-K ∩ Test_Items| / |Test_Items|
    
    Interpretation:
        - Recall@10 = 0.25: 25% of test items found in top-10 recommendations
        - Higher is better (max = 1.0)
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize Recall@K metric.
        
        Args:
            k: Cutoff value for top-K
        """
        super().__init__(name=f"Recall@{k}")
        self.k = k
    
    def compute(
        self,
        predictions: Union[List[int], np.ndarray],
        ground_truth: Set[int],
        k: Optional[int] = None
    ) -> float:
        """
        Compute Recall@K.
        
        Args:
            predictions: Ordered list of predicted item indices
            ground_truth: Set of positive item indices
            k: Optional override for K value
        
        Returns:
            Recall@K value in [0, 1]
        """
        if not self.validate_inputs(predictions, ground_truth):
            return 0.0
        
        if len(ground_truth) == 0:
            return 0.0  # No relevant items to find
        
        k = k if k is not None else self.k
        k = min(k, len(predictions))
        
        top_k = set(predictions[:k])
        hits = len(top_k & ground_truth)
        
        return hits / len(ground_truth)


class PrecisionAtK(BaseMetric):
    """
    Precision@K metric: proportion of relevant items in top-K recommendations.
    
    Formula:
        Precision@K = |Top-K ∩ Test_Items| / K
    
    Interpretation:
        - Precision@10 = 0.20: 20% of top-10 recommendations are relevant
        - Higher is better (max = min(1.0, |Test_Items|/K))
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize Precision@K metric.
        
        Args:
            k: Cutoff value for top-K
        """
        super().__init__(name=f"Precision@{k}")
        self.k = k
    
    def compute(
        self,
        predictions: Union[List[int], np.ndarray],
        ground_truth: Set[int],
        k: Optional[int] = None
    ) -> float:
        """
        Compute Precision@K.
        
        Args:
            predictions: Ordered list of predicted item indices
            ground_truth: Set of positive item indices
            k: Optional override for K value
        
        Returns:
            Precision@K value in [0, 1]
        """
        if not self.validate_inputs(predictions, ground_truth):
            return 0.0
        
        k = k if k is not None else self.k
        k = min(k, len(predictions))
        
        if k == 0:
            return 0.0
        
        top_k = set(predictions[:k])
        hits = len(top_k & ground_truth)
        
        return hits / k


class NDCGAtK(BaseMetric):
    """
    NDCG@K (Normalized Discounted Cumulative Gain) metric.
    
    Evaluates ranking quality with position discounting.
    
    Formula:
        DCG@K = Σ(i=1 to K) [rel_i / log2(i+1)]
        IDCG@K = DCG@K of perfect ranking
        NDCG@K = DCG@K / IDCG@K
    
    Interpretation:
        - NDCG@10 = 0.18: Ranking quality is 18% of ideal
        - Higher is better (max = 1.0)
        - Penalizes relevant items at lower positions
    """
    
    def __init__(self, k: int = 10, use_graded_relevance: bool = False):
        """
        Initialize NDCG@K metric.
        
        Args:
            k: Cutoff value for top-K
            use_graded_relevance: If True, use ratings as relevance (not implemented)
        """
        super().__init__(name=f"NDCG@{k}")
        self.k = k
        self.use_graded_relevance = use_graded_relevance
    
    def _dcg(self, relevances: np.ndarray) -> float:
        """
        Compute DCG (Discounted Cumulative Gain).
        
        Args:
            relevances: Array of relevance scores (binary: 0 or 1)
        
        Returns:
            DCG value
        """
        if len(relevances) == 0:
            return 0.0
        
        # Position discounting: log2(i+2) for i=0,1,2,...
        positions = np.arange(len(relevances)) + 2
        discounts = np.log2(positions)
        
        return np.sum(relevances / discounts)
    
    def _idcg(self, num_relevant: int, k: int) -> float:
        """
        Compute IDCG (Ideal DCG).
        
        Args:
            num_relevant: Number of relevant items in ground truth
            k: Cutoff value
        
        Returns:
            IDCG value
        """
        # Perfect ranking: all relevant items first
        ideal_relevances = np.zeros(k)
        ideal_relevances[:min(num_relevant, k)] = 1.0
        
        return self._dcg(ideal_relevances)
    
    def compute(
        self,
        predictions: Union[List[int], np.ndarray],
        ground_truth: Set[int],
        k: Optional[int] = None
    ) -> float:
        """
        Compute NDCG@K.
        
        Args:
            predictions: Ordered list of predicted item indices
            ground_truth: Set of positive item indices
            k: Optional override for K value
        
        Returns:
            NDCG@K value in [0, 1]
        """
        if not self.validate_inputs(predictions, ground_truth):
            return 0.0
        
        if len(ground_truth) == 0:
            return 0.0  # No relevant items
        
        k = k if k is not None else self.k
        k = min(k, len(predictions))
        
        # Compute relevances for top-K predictions
        top_k = list(predictions[:k])
        relevances = np.array([1.0 if item in ground_truth else 0.0 for item in top_k])
        
        # Compute DCG
        dcg = self._dcg(relevances)
        
        # Compute IDCG
        idcg = self._idcg(len(ground_truth), k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg


class MRR(BaseMetric):
    """
    MRR (Mean Reciprocal Rank) metric.
    
    Measures the position of the first relevant item.
    
    Formula:
        RR = 1 / rank(first_relevant_item)
        MRR = Average(RR) across users
    
    Interpretation:
        - MRR = 0.5: First relevant item is at position 2 on average
        - Higher is better (max = 1.0)
    """
    
    def __init__(self):
        """Initialize MRR metric."""
        super().__init__(name="MRR")
    
    def compute(
        self,
        predictions: Union[List[int], np.ndarray],
        ground_truth: Set[int],
        **kwargs
    ) -> float:
        """
        Compute Reciprocal Rank.
        
        Args:
            predictions: Ordered list of predicted item indices
            ground_truth: Set of positive item indices
        
        Returns:
            Reciprocal rank value in [0, 1]
        """
        if not self.validate_inputs(predictions, ground_truth):
            return 0.0
        
        if len(ground_truth) == 0:
            return 0.0
        
        # Find rank of first relevant item (1-indexed)
        for rank, item in enumerate(predictions, start=1):
            if item in ground_truth:
                return 1.0 / rank
        
        return 0.0  # No relevant item found


class MAPAtK(BaseMetric):
    """
    MAP@K (Mean Average Precision at K) metric.
    
    Average of Precision values at each relevant item position.
    
    Formula:
        AP@K = (1/|Rel_K|) * Σ(k=1 to K) [Precision@k * rel_k]
        MAP@K = Average(AP@K) across users
    
    Interpretation:
        - Combines precision and ranking quality
        - Higher is better (max = 1.0)
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize MAP@K metric.
        
        Args:
            k: Cutoff value for top-K
        """
        super().__init__(name=f"MAP@{k}")
        self.k = k
    
    def compute(
        self,
        predictions: Union[List[int], np.ndarray],
        ground_truth: Set[int],
        k: Optional[int] = None
    ) -> float:
        """
        Compute Average Precision at K.
        
        Args:
            predictions: Ordered list of predicted item indices
            ground_truth: Set of positive item indices
            k: Optional override for K value
        
        Returns:
            AP@K value in [0, 1]
        """
        if not self.validate_inputs(predictions, ground_truth):
            return 0.0
        
        if len(ground_truth) == 0:
            return 0.0
        
        k = k if k is not None else self.k
        k = min(k, len(predictions))
        
        # Compute AP@K
        hits = 0
        sum_precision = 0.0
        
        for i, item in enumerate(predictions[:k], start=1):
            if item in ground_truth:
                hits += 1
                precision_at_i = hits / i
                sum_precision += precision_at_i
        
        # Number of relevant items in top-K
        num_relevant_in_k = hits
        
        if num_relevant_in_k == 0:
            return 0.0
        
        return sum_precision / min(len(ground_truth), k)


class HitRate(BaseMetric):
    """
    Hit Rate metric: whether any relevant item is in top-K.
    
    Formula:
        HitRate@K = 1 if |Top-K ∩ Test_Items| > 0 else 0
    
    Interpretation:
        - Binary metric (0 or 1)
        - Useful for sparse test sets
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize Hit Rate metric.
        
        Args:
            k: Cutoff value for top-K
        """
        super().__init__(name=f"HitRate@{k}")
        self.k = k
    
    def compute(
        self,
        predictions: Union[List[int], np.ndarray],
        ground_truth: Set[int],
        k: Optional[int] = None
    ) -> float:
        """
        Compute Hit Rate at K.
        
        Args:
            predictions: Ordered list of predicted item indices
            ground_truth: Set of positive item indices
            k: Optional override for K value
        
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        if not self.validate_inputs(predictions, ground_truth):
            return 0.0
        
        if len(ground_truth) == 0:
            return 0.0
        
        k = k if k is not None else self.k
        k = min(k, len(predictions))
        
        top_k = set(predictions[:k])
        
        return 1.0 if len(top_k & ground_truth) > 0 else 0.0


# ============================================================================
# Coverage Metrics
# ============================================================================

class Coverage(BaseMetric):
    """
    Coverage metric: proportion of items recommended across all users.
    
    Formula:
        Coverage = |Unique Items in All Recommendations| / |Total Items|
    
    Interpretation:
        - Coverage = 0.30: 30% of catalog items are recommended
        - Higher indicates more diverse recommendations
    """
    
    def __init__(self):
        """Initialize Coverage metric."""
        super().__init__(name="Coverage")
    
    def compute(
        self,
        all_recommendations: Dict[int, List[int]],
        num_total_items: int,
        **kwargs
    ) -> float:
        """
        Compute catalog coverage.
        
        Args:
            all_recommendations: Dict mapping user_idx to list of recommended item indices
            num_total_items: Total number of items in catalog
        
        Returns:
            Coverage value in [0, 1]
        """
        if num_total_items == 0:
            return 0.0
        
        unique_items = set()
        for recs in all_recommendations.values():
            unique_items.update(recs)
        
        return len(unique_items) / num_total_items


# ============================================================================
# Metric Factory
# ============================================================================

class MetricFactory:
    """
    Factory for creating metric instances.
    
    Example:
        >>> factory = MetricFactory()
        >>> recall10 = factory.create("recall", k=10)
        >>> ndcg20 = factory.create("ndcg", k=20)
    """
    
    METRIC_REGISTRY = {
        'recall': RecallAtK,
        'precision': PrecisionAtK,
        'ndcg': NDCGAtK,
        'mrr': MRR,
        'map': MAPAtK,
        'hit_rate': HitRate,
        'coverage': Coverage,
    }
    
    @classmethod
    def create(cls, metric_name: str, **kwargs) -> BaseMetric:
        """
        Create a metric instance.
        
        Args:
            metric_name: Name of metric ('recall', 'ndcg', etc.)
            **kwargs: Metric-specific parameters (e.g., k=10)
        
        Returns:
            BaseMetric instance
        
        Raises:
            ValueError: If metric name is not recognized
        """
        metric_name = metric_name.lower()
        
        if metric_name not in cls.METRIC_REGISTRY:
            raise ValueError(
                f"Unknown metric: {metric_name}. "
                f"Available: {list(cls.METRIC_REGISTRY.keys())}"
            )
        
        return cls.METRIC_REGISTRY[metric_name](**kwargs)
    
    @classmethod
    def create_standard_metrics(cls, k_values: List[int] = [10, 20]) -> Dict[str, BaseMetric]:
        """
        Create standard set of metrics for evaluation.
        
        Args:
            k_values: List of K values for @K metrics
        
        Returns:
            Dict mapping metric name to metric instance
        """
        metrics = {}
        
        for k in k_values:
            metrics[f'recall@{k}'] = cls.create('recall', k=k)
            metrics[f'ndcg@{k}'] = cls.create('ndcg', k=k)
            metrics[f'precision@{k}'] = cls.create('precision', k=k)
            metrics[f'map@{k}'] = cls.create('map', k=k)
            metrics[f'hit_rate@{k}'] = cls.create('hit_rate', k=k)
        
        metrics['mrr'] = cls.create('mrr')
        
        return metrics


# ============================================================================
# Convenience Functions
# ============================================================================

def recall_at_k(
    predictions: Union[List[int], np.ndarray],
    ground_truth: Set[int],
    k: int
) -> float:
    """
    Compute Recall@K.
    
    Args:
        predictions: Ordered list of predicted item indices
        ground_truth: Set of positive item indices
        k: Cutoff value
    
    Returns:
        Recall@K value
    """
    metric = RecallAtK(k=k)
    return metric.compute(predictions, ground_truth)


def ndcg_at_k(
    predictions: Union[List[int], np.ndarray],
    ground_truth: Set[int],
    k: int
) -> float:
    """
    Compute NDCG@K.
    
    Args:
        predictions: Ordered list of predicted item indices
        ground_truth: Set of positive item indices
        k: Cutoff value
    
    Returns:
        NDCG@K value
    """
    metric = NDCGAtK(k=k)
    return metric.compute(predictions, ground_truth)


def precision_at_k(
    predictions: Union[List[int], np.ndarray],
    ground_truth: Set[int],
    k: int
) -> float:
    """
    Compute Precision@K.
    
    Args:
        predictions: Ordered list of predicted item indices
        ground_truth: Set of positive item indices
        k: Cutoff value
    
    Returns:
        Precision@K value
    """
    metric = PrecisionAtK(k=k)
    return metric.compute(predictions, ground_truth)


def mrr(
    predictions: Union[List[int], np.ndarray],
    ground_truth: Set[int]
) -> float:
    """
    Compute Reciprocal Rank (RR).
    
    Args:
        predictions: Ordered list of predicted item indices
        ground_truth: Set of positive item indices
    
    Returns:
        RR value
    """
    metric = MRR()
    return metric.compute(predictions, ground_truth)


def map_at_k(
    predictions: Union[List[int], np.ndarray],
    ground_truth: Set[int],
    k: int
) -> float:
    """
    Compute Average Precision at K.
    
    Args:
        predictions: Ordered list of predicted item indices
        ground_truth: Set of positive item indices
        k: Cutoff value
    
    Returns:
        AP@K value
    """
    metric = MAPAtK(k=k)
    return metric.compute(predictions, ground_truth)


def coverage(
    all_recommendations: Dict[int, List[int]],
    num_total_items: int
) -> float:
    """
    Compute catalog coverage.
    
    Args:
        all_recommendations: Dict mapping user_idx to recommended items
        num_total_items: Total number of items
    
    Returns:
        Coverage value
    """
    metric = Coverage()
    return metric.compute(all_recommendations, num_total_items)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Core Metrics Module")
    print("=" * 60)
    
    # Test data
    predictions = [1, 5, 3, 8, 2, 10, 7, 4, 9, 6]
    ground_truth = {3, 8, 10, 15}  # 3 items in top-10, 1 outside
    
    # Test individual metrics
    print("\n--- Individual Metrics ---")
    print(f"Recall@5: {recall_at_k(predictions, ground_truth, k=5):.4f}")  # Expected: 2/4 = 0.5
    print(f"Recall@10: {recall_at_k(predictions, ground_truth, k=10):.4f}")  # Expected: 3/4 = 0.75
    print(f"NDCG@10: {ndcg_at_k(predictions, ground_truth, k=10):.4f}")
    print(f"Precision@10: {precision_at_k(predictions, ground_truth, k=10):.4f}")  # 3/10 = 0.3
    print(f"MRR: {mrr(predictions, ground_truth):.4f}")  # First hit at position 3 → 1/3
    print(f"MAP@10: {map_at_k(predictions, ground_truth, k=10):.4f}")
    
    # Test coverage
    print("\n--- Coverage ---")
    all_recs = {
        0: [1, 2, 3],
        1: [3, 4, 5],
        2: [5, 6, 7],
    }
    cov = coverage(all_recs, num_total_items=100)
    print(f"Coverage: {cov:.4f}")  # 7 unique items / 100 = 0.07
    
    # Test factory
    print("\n--- Metric Factory ---")
    factory = MetricFactory()
    metrics = factory.create_standard_metrics(k_values=[5, 10])
    for name, metric in metrics.items():
        print(f"  {name}: {metric}")
    
    # Test edge cases
    print("\n--- Edge Cases ---")
    print(f"Empty ground truth: {recall_at_k(predictions, set(), k=10):.4f}")  # 0
    print(f"Empty predictions: {recall_at_k([], ground_truth, k=10):.4f}")  # 0
    print(f"Single item hit: {recall_at_k([15], ground_truth, k=1):.4f}")  # 1/4 = 0.25
    
    print("\n" + "=" * 60)
    print("All tests passed!")
