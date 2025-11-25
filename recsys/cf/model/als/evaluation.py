"""
Step 6: Evaluation for ALS Model

This module implements evaluation metrics and baseline comparison for collaborative
filtering models, specifically designed for the ALS (Alternating Least Squares) pipeline.

Key Components:
    1. Recall@K: Percentage of test items found in top-K recommendations
    2. NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
    3. PopularityBaseline: Naive recommender for comparison
    4. ALSEvaluator: Batch evaluation orchestration

Usage:
    >>> from recsys.cf.model.als.evaluation import ALSEvaluator
    >>> 
    >>> # Initialize evaluator
    >>> evaluator = ALSEvaluator(
    ...     user_factors=U,
    ...     item_factors=V,
    ...     user_to_idx=mappings['user_to_idx'],
    ...     idx_to_user=mappings['idx_to_user'],
    ...     item_to_idx=mappings['item_to_idx'],
    ...     idx_to_item=mappings['idx_to_item'],
    ...     user_pos_train=user_pos_train,
    ...     user_pos_test=user_pos_test
    ... )
    >>> 
    >>> # Evaluate
    >>> results = evaluator.evaluate(k_values=[10, 20])
    >>> print(f"Recall@10: {results['recall@10']:.3f}")
    >>> print(f"NDCG@10: {results['ndcg@10']:.3f}")

Author: Copilot
Date: 2025-01-15
"""

import numpy as np
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import time
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Metric Functions
# ============================================================================

def recall_at_k(predictions: np.ndarray, ground_truth: Set[int], k: int) -> float:
    """
    Compute Recall@K: Percentage of test items found in top-K recommendations.
    
    Formula:
        Recall@K = |predicted ∩ ground_truth| / |ground_truth|
    
    Args:
        predictions: Array of predicted item indices, ranked by score (descending)
        ground_truth: Set of ground truth item indices
        k: Number of top predictions to consider
    
    Returns:
        Recall@K score in range [0, 1]
    
    Example:
        >>> predictions = np.array([10, 3, 7, 15, 2])  # Top-5 items
        >>> ground_truth = {3, 7, 9}  # User interacted with 3 items in test
        >>> recall = recall_at_k(predictions, ground_truth, k=5)
        >>> print(f"Recall@5: {recall:.3f}")  # 2/3 = 0.667
        Recall@5: 0.667
    """
    if len(ground_truth) == 0:
        logger.warning("Empty ground truth - returning recall=0")
        return 0.0
    
    # Get top-K predictions
    top_k = predictions[:k]
    
    # Count hits
    hits = len(set(top_k) & ground_truth)
    
    # Recall = hits / total relevant
    recall = hits / len(ground_truth)
    
    return recall


def ndcg_at_k(predictions: np.ndarray, ground_truth: Set[int], k: int) -> float:
    """
    Compute NDCG@K: Normalized Discounted Cumulative Gain.
    
    NDCG measures ranking quality by giving higher weight to relevant items
    at top positions.
    
    Formula:
        DCG@K = Σ(rel_i / log2(i+1)) for i=1 to K
        IDCG@K = DCG@K for perfect ranking
        NDCG@K = DCG@K / IDCG@K
    
    Where:
        rel_i = 1 if prediction[i] in ground_truth, else 0
    
    Args:
        predictions: Array of predicted item indices, ranked by score (descending)
        ground_truth: Set of ground truth item indices
        k: Number of top predictions to consider
    
    Returns:
        NDCG@K score in range [0, 1]
    
    Example:
        >>> predictions = np.array([10, 3, 7, 15, 2])  # Top-5
        >>> ground_truth = {3, 7, 9}
        >>> ndcg = ndcg_at_k(predictions, ground_truth, k=5)
        >>> print(f"NDCG@5: {ndcg:.3f}")
        NDCG@5: 0.819
    
    Notes:
        - Binary relevance: item is either relevant (1) or not (0)
        - Discount factor: log2(position + 1), positions start at 1
        - Perfect NDCG = 1.0 when all relevant items are at top
    """
    if len(ground_truth) == 0:
        logger.warning("Empty ground truth - returning ndcg=0")
        return 0.0
    
    # Get top-K predictions
    top_k = predictions[:k]
    
    # Compute DCG@K
    dcg = 0.0
    for i, item_idx in enumerate(top_k):
        if item_idx in ground_truth:
            # Position starts at 1, not 0
            position = i + 1
            # Discount factor: log2(position + 1)
            dcg += 1.0 / np.log2(position + 1)
    
    # Compute Ideal DCG@K (perfect ranking)
    num_relevant = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
    
    # Normalize
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    
    return ndcg


def compute_metrics_single_user(
    predictions: np.ndarray,
    ground_truth: Set[int],
    k_values: List[int]
) -> Dict[str, float]:
    """
    Compute all metrics for a single user across multiple K values.
    
    Args:
        predictions: Ranked item indices for the user
        ground_truth: Ground truth item indices for the user
        k_values: List of K values to evaluate (e.g., [10, 20])
    
    Returns:
        Dictionary with keys like 'recall@10', 'ndcg@10', etc.
    
    Example:
        >>> predictions = np.array([10, 3, 7, 15, 2, 8, 1, 12, 5, 20])
        >>> ground_truth = {3, 7, 9}
        >>> metrics = compute_metrics_single_user(predictions, ground_truth, k_values=[5, 10])
        >>> print(metrics)
        {'recall@5': 0.667, 'ndcg@5': 0.819, 'recall@10': 0.667, 'ndcg@10': 0.630}
    """
    metrics = {}
    
    for k in k_values:
        metrics[f'recall@{k}'] = recall_at_k(predictions, ground_truth, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(predictions, ground_truth, k)
    
    return metrics


# ============================================================================
# Baseline: Popularity Recommender
# ============================================================================

class PopularityBaseline:
    """
    Naive popularity-based recommender for baseline comparison.
    
    Recommends the most popular items (highest interaction count or num_sold_time)
    to all users, ignoring personalization.
    
    Attributes:
        item_popularity: Dict mapping item_idx to popularity score
        popular_items_ranked: Array of item indices sorted by popularity (descending)
    
    Example:
        >>> # Option 1: Popularity from training data
        >>> baseline = PopularityBaseline()
        >>> baseline.fit_from_train(user_pos_train, num_items=2200)
        >>> 
        >>> # Option 2: Popularity from product metadata
        >>> baseline = PopularityBaseline()
        >>> baseline.fit_from_metadata(product_df, item_to_idx, popularity_col='num_sold_time')
        >>> 
        >>> # Recommend
        >>> recs = baseline.recommend(user_id='12345', k=10, filter_seen=True, seen_items={5, 10})
        >>> print(recs)  # Top-10 popular items excluding seen
    """
    
    def __init__(self):
        self.item_popularity: Dict[int, float] = {}
        self.popular_items_ranked: Optional[np.ndarray] = None
        self.fitted = False
    
    def fit_from_train(self, user_pos_train: Dict[int, Set[int]], num_items: int):
        """
        Fit popularity baseline from training interactions.
        
        Popularity = number of users who interacted with each item.
        
        Args:
            user_pos_train: Dict mapping user_idx to set of positive item indices
            num_items: Total number of items
        
        Example:
            >>> user_pos_train = {0: {5, 10, 15}, 1: {5, 20}, 2: {10}}
            >>> baseline.fit_from_train(user_pos_train, num_items=100)
            >>> # Item 5: 2 users, Item 10: 2 users, Item 15: 1 user, Item 20: 1 user
        """
        logger.info(f"Fitting PopularityBaseline from {len(user_pos_train)} users")
        
        # Count item occurrences
        item_counts = defaultdict(int)
        for items in user_pos_train.values():
            for item_idx in items:
                item_counts[item_idx] += 1
        
        # Store as dict
        self.item_popularity = dict(item_counts)
        
        # Rank items by popularity
        self.popular_items_ranked = np.array(
            sorted(self.item_popularity.keys(), 
                   key=lambda x: self.item_popularity[x], 
                   reverse=True)
        )
        
        self.fitted = True
        
        logger.info(f"Fitted PopularityBaseline: {len(self.item_popularity)} items with interactions")
        logger.info(f"Top-5 popular items: {self.popular_items_ranked[:5].tolist()}")
        logger.info(f"Top-5 popularities: {[self.item_popularity[i] for i in self.popular_items_ranked[:5]]}")
    
    def fit_from_metadata(
        self,
        product_df,
        item_to_idx: Dict[str, int],
        popularity_col: str = 'num_sold_time'
    ):
        """
        Fit popularity baseline from product metadata (e.g., num_sold_time).
        
        Args:
            product_df: DataFrame with columns ['product_id', popularity_col]
            item_to_idx: Mapping from product_id to item_idx
            popularity_col: Column name for popularity score
        
        Example:
            >>> import pandas as pd
            >>> product_df = pd.DataFrame({
            ...     'product_id': ['A', 'B', 'C'],
            ...     'num_sold_time': [1000, 500, 1500]
            ... })
            >>> item_to_idx = {'A': 0, 'B': 1, 'C': 2}
            >>> baseline.fit_from_metadata(product_df, item_to_idx, 'num_sold_time')
        """
        logger.info(f"Fitting PopularityBaseline from product metadata: {popularity_col}")
        
        self.item_popularity = {}
        
        for _, row in product_df.iterrows():
            product_id = str(row['product_id'])
            if product_id in item_to_idx:
                item_idx = item_to_idx[product_id]
                popularity = row[popularity_col]
                self.item_popularity[item_idx] = popularity
        
        # Rank items
        self.popular_items_ranked = np.array(
            sorted(self.item_popularity.keys(), 
                   key=lambda x: self.item_popularity[x], 
                   reverse=True)
        )
        
        self.fitted = True
        
        logger.info(f"Fitted PopularityBaseline: {len(self.item_popularity)} items")
        logger.info(f"Top-5 popular items: {self.popular_items_ranked[:5].tolist()}")
    
    def recommend(
        self,
        user_id: Optional[str] = None,
        k: int = 10,
        filter_seen: bool = True,
        seen_items: Optional[Set[int]] = None
    ) -> np.ndarray:
        """
        Generate top-K popular items (ignoring user preferences).
        
        Args:
            user_id: User ID (ignored, for API compatibility)
            k: Number of recommendations
            filter_seen: Whether to filter seen items
            seen_items: Set of item indices to exclude
        
        Returns:
            Array of top-K item indices
        
        Example:
            >>> recs = baseline.recommend(k=10, filter_seen=True, seen_items={0, 1, 2})
            >>> print(recs.shape)  # (10,)
        """
        if not self.fitted:
            raise ValueError("PopularityBaseline not fitted. Call fit_from_train() or fit_from_metadata() first.")
        
        # Start with all popular items
        candidates = self.popular_items_ranked.copy()
        
        # Filter seen items if needed
        if filter_seen and seen_items is not None:
            mask = np.array([item not in seen_items for item in candidates])
            candidates = candidates[mask]
        
        # Return top-K
        return candidates[:k]
    
    def evaluate(
        self,
        user_pos_test: Dict[int, Set[int]],
        user_pos_train: Optional[Dict[int, Set[int]]] = None,
        k_values: List[int] = [10, 20],
        filter_seen: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate popularity baseline on test set.
        
        Args:
            user_pos_test: Dict mapping user_idx to test item indices
            user_pos_train: Dict mapping user_idx to train item indices (for filtering)
            k_values: List of K values to evaluate
            filter_seen: Whether to filter training items
        
        Returns:
            Dictionary with average metrics across all test users
        
        Example:
            >>> metrics = baseline.evaluate(user_pos_test, user_pos_train, k_values=[10, 20])
            >>> print(f"Baseline Recall@10: {metrics['recall@10']:.3f}")
        """
        if not self.fitted:
            raise ValueError("PopularityBaseline not fitted")
        
        logger.info(f"Evaluating PopularityBaseline on {len(user_pos_test)} test users")
        
        all_metrics = defaultdict(list)
        
        for user_idx, ground_truth in user_pos_test.items():
            # Get seen items
            seen_items = user_pos_train.get(user_idx, set()) if filter_seen and user_pos_train else None
            
            # Get predictions
            predictions = self.recommend(k=max(k_values), filter_seen=filter_seen, seen_items=seen_items)
            
            # Compute metrics
            user_metrics = compute_metrics_single_user(predictions, ground_truth, k_values)
            
            for metric_name, value in user_metrics.items():
                all_metrics[metric_name].append(value)
        
        # Average across users
        avg_metrics = {
            metric_name: np.mean(values)
            for metric_name, values in all_metrics.items()
        }
        
        logger.info("PopularityBaseline metrics:")
        for metric_name, value in sorted(avg_metrics.items()):
            logger.info(f"  {metric_name}: {value:.4f}")
        
        return avg_metrics


# ============================================================================
# Evaluator: Batch Evaluation for ALS
# ============================================================================

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_type: str
    metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    improvement: Dict[str, float]
    num_test_users: int
    k_values: List[int]
    evaluation_time: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def save(self, output_path: Path):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")
    
    def print_summary(self):
        """Print formatted summary of evaluation results."""
        print("\n" + "="*70)
        print(f"EVALUATION RESULTS: {self.model_type.upper()}")
        print("="*70)
        
        print(f"\nTest Users: {self.num_test_users}")
        print(f"K Values: {self.k_values}")
        print(f"Evaluation Time: {self.evaluation_time:.2f}s")
        
        print("\n" + "-"*70)
        print(f"{'Metric':<20} {'Model':<12} {'Baseline':<12} {'Improvement':<15}")
        print("-"*70)
        
        for k in self.k_values:
            recall_key = f'recall@{k}'
            ndcg_key = f'ndcg@{k}'
            
            print(f"{recall_key:<20} {self.metrics[recall_key]:<12.4f} "
                  f"{self.baseline_metrics[recall_key]:<12.4f} "
                  f"{self.improvement[recall_key]:<15}")
            
            print(f"{ndcg_key:<20} {self.metrics[ndcg_key]:<12.4f} "
                  f"{self.baseline_metrics[ndcg_key]:<12.4f} "
                  f"{self.improvement[ndcg_key]:<15}")
        
        print("="*70 + "\n")


class ALSEvaluator:
    """
    Batch evaluator for ALS (or BPR) collaborative filtering models.
    
    Computes Recall@K and NDCG@K metrics, compares with popularity baseline,
    and generates comprehensive evaluation reports.
    
    Attributes:
        user_factors: User embedding matrix U (num_users, factors)
        item_factors: Item embedding matrix V (num_items, factors)
        mappings: ID mappings (user_to_idx, idx_to_user, item_to_idx, idx_to_item)
        user_pos_train: Training positive sets (for filtering)
        user_pos_test: Test positive sets (ground truth)
    
    Example:
        >>> evaluator = ALSEvaluator(
        ...     user_factors=U,
        ...     item_factors=V,
        ...     user_to_idx=mappings['user_to_idx'],
        ...     idx_to_user=mappings['idx_to_user'],
        ...     item_to_idx=mappings['item_to_idx'],
        ...     idx_to_item=mappings['idx_to_item'],
        ...     user_pos_train=user_pos_train,
        ...     user_pos_test=user_pos_test
        ... )
        >>> 
        >>> results = evaluator.evaluate(k_values=[10, 20])
        >>> results.print_summary()
    """
    
    def __init__(
        self,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        user_to_idx: Dict[str, int],
        idx_to_user: Dict[int, str],
        item_to_idx: Dict[str, int],
        idx_to_item: Dict[int, str],
        user_pos_train: Dict[int, Set[int]],
        user_pos_test: Dict[int, Set[int]]
    ):
        """
        Initialize evaluator.
        
        Args:
            user_factors: User embeddings (num_users, factors)
            item_factors: Item embeddings (num_items, factors)
            user_to_idx: Mapping from user_id to user_idx
            idx_to_user: Mapping from user_idx to user_id
            item_to_idx: Mapping from product_id to item_idx
            idx_to_item: Mapping from item_idx to product_id
            user_pos_train: Training positive sets
            user_pos_test: Test positive sets (ground truth)
        """
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.user_to_idx = user_to_idx
        self.idx_to_user = idx_to_user
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        self.user_pos_train = user_pos_train
        self.user_pos_test = user_pos_test
        
        logger.info(f"Initialized ALSEvaluator:")
        logger.info(f"  Users: {user_factors.shape[0]}")
        logger.info(f"  Items: {item_factors.shape[0]}")
        logger.info(f"  Factors: {user_factors.shape[1]}")
        logger.info(f"  Test users: {len(user_pos_test)}")
    
    def _compute_scores(self, user_idx: int) -> np.ndarray:
        """Compute scores for all items for a single user."""
        return self.user_factors[user_idx] @ self.item_factors.T
    
    def _filter_seen_items(self, scores: np.ndarray, user_idx: int) -> np.ndarray:
        """Filter seen items by setting scores to -inf."""
        filtered_scores = scores.copy()
        seen_items = self.user_pos_train.get(user_idx, set())
        if seen_items:
            filtered_scores[list(seen_items)] = -np.inf
        return filtered_scores
    
    def _get_top_k_predictions(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Get top-K item indices from scores."""
        # Use argpartition for efficiency
        if k >= len(scores):
            k = len(scores)
        
        top_k_unsorted = np.argpartition(scores, -k)[-k:]
        top_k_sorted = top_k_unsorted[np.argsort(scores[top_k_unsorted])[::-1]]
        
        return top_k_sorted
    
    def evaluate(
        self,
        k_values: List[int] = [10, 20],
        filter_seen: bool = True,
        compare_baseline: bool = True,
        baseline_source: str = 'train',
        product_df = None,
        model_type: str = 'als'
    ) -> EvaluationResult:
        """
        Evaluate CF model on test set and compare with popularity baseline.
        
        Args:
            k_values: List of K values to evaluate
            filter_seen: Whether to filter training items
            compare_baseline: Whether to compute popularity baseline
            baseline_source: 'train' or 'metadata' for baseline popularity
            product_df: Product metadata (needed if baseline_source='metadata')
            model_type: Model name for reporting ('als' or 'bpr')
        
        Returns:
            EvaluationResult with metrics, baseline, and improvement
        
        Example:
            >>> results = evaluator.evaluate(k_values=[10, 20], compare_baseline=True)
            >>> print(f"Recall@10: {results.metrics['recall@10']:.3f}")
            >>> print(f"Improvement: {results.improvement['recall@10']}")
        """
        logger.info(f"Starting evaluation for {model_type.upper()} model")
        logger.info(f"K values: {k_values}")
        logger.info(f"Filter seen: {filter_seen}")
        
        start_time = time.time()
        
        # Evaluate CF model
        all_metrics = defaultdict(list)
        
        for user_idx, ground_truth in self.user_pos_test.items():
            # Compute scores
            scores = self._compute_scores(user_idx)
            
            # Filter seen items
            if filter_seen:
                scores = self._filter_seen_items(scores, user_idx)
            
            # Get top-K predictions
            max_k = max(k_values)
            predictions = self._get_top_k_predictions(scores, max_k)
            
            # Compute metrics
            user_metrics = compute_metrics_single_user(predictions, ground_truth, k_values)
            
            for metric_name, value in user_metrics.items():
                all_metrics[metric_name].append(value)
        
        # Average across users
        cf_metrics = {
            metric_name: np.mean(values)
            for metric_name, values in all_metrics.items()
        }
        
        logger.info(f"{model_type.upper()} metrics:")
        for metric_name, value in sorted(cf_metrics.items()):
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Evaluate baseline
        baseline_metrics = {}
        improvement = {}
        
        if compare_baseline:
            logger.info("Computing popularity baseline...")
            
            baseline = PopularityBaseline()
            
            if baseline_source == 'train':
                baseline.fit_from_train(self.user_pos_train, num_items=self.item_factors.shape[0])
            elif baseline_source == 'metadata':
                if product_df is None:
                    raise ValueError("product_df required for baseline_source='metadata'")
                baseline.fit_from_metadata(product_df, self.item_to_idx, popularity_col='num_sold_time')
            else:
                raise ValueError(f"Unknown baseline_source: {baseline_source}")
            
            baseline_metrics = baseline.evaluate(
                self.user_pos_test,
                self.user_pos_train if filter_seen else None,
                k_values=k_values,
                filter_seen=filter_seen
            )
            
            # Compute improvement
            for metric_name in cf_metrics.keys():
                cf_value = cf_metrics[metric_name]
                baseline_value = baseline_metrics[metric_name]
                
                if baseline_value > 0:
                    improvement_pct = ((cf_value / baseline_value) - 1) * 100
                    improvement[metric_name] = f"+{improvement_pct:.1f}%"
                else:
                    improvement[metric_name] = "N/A"
        
        evaluation_time = time.time() - start_time
        
        # Create result
        result = EvaluationResult(
            model_type=model_type,
            metrics=cf_metrics,
            baseline_metrics=baseline_metrics,
            improvement=improvement,
            num_test_users=len(self.user_pos_test),
            k_values=k_values,
            evaluation_time=evaluation_time,
            metadata={
                'filter_seen': filter_seen,
                'baseline_source': baseline_source if compare_baseline else None,
                'num_users': self.user_factors.shape[0],
                'num_items': self.item_factors.shape[0],
                'factors': self.user_factors.shape[1]
            }
        )
        
        logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
        
        return result


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_evaluate(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_pos_test: Dict[int, Set[int]],
    user_pos_train: Optional[Dict[int, Set[int]]] = None,
    k_values: List[int] = [10, 20],
    model_type: str = 'als'
) -> Dict[str, float]:
    """
    Quick evaluation without full ID mappings (test users only).
    
    Args:
        user_factors: User embeddings
        item_factors: Item embeddings
        user_pos_test: Test positive sets
        user_pos_train: Train positive sets (for filtering)
        k_values: K values to evaluate
        model_type: Model name
    
    Returns:
        Dictionary with average metrics
    
    Example:
        >>> metrics = quick_evaluate(U, V, user_pos_test, user_pos_train, k_values=[10, 20])
        >>> print(f"Recall@10: {metrics['recall@10']:.3f}")
    """
    logger.info(f"Quick evaluation for {model_type.upper()}")
    
    all_metrics = defaultdict(list)
    
    for user_idx, ground_truth in user_pos_test.items():
        # Compute scores
        scores = user_factors[user_idx] @ item_factors.T
        
        # Filter seen items
        if user_pos_train and user_idx in user_pos_train:
            seen_items = list(user_pos_train[user_idx])
            scores[seen_items] = -np.inf
        
        # Get top-K
        max_k = max(k_values)
        top_k_unsorted = np.argpartition(scores, -max_k)[-max_k:]
        predictions = top_k_unsorted[np.argsort(scores[top_k_unsorted])[::-1]]
        
        # Compute metrics
        user_metrics = compute_metrics_single_user(predictions, ground_truth, k_values)
        
        for metric_name, value in user_metrics.items():
            all_metrics[metric_name].append(value)
    
    # Average
    avg_metrics = {
        metric_name: np.mean(values)
        for metric_name, values in all_metrics.items()
    }
    
    return avg_metrics


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ALS Evaluation Module - Demo")
    print("="*70)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Synthetic data
    np.random.seed(42)
    
    num_users = 100
    num_items = 50
    factors = 16
    num_test_users = 20
    
    # Generate embeddings
    U = np.random.randn(num_users, factors) * 0.1
    V = np.random.randn(num_items, factors) * 0.1
    
    # Generate train/test sets
    user_pos_train = {}
    user_pos_test = {}
    
    for u in range(num_users):
        # Random positive items
        num_train = np.random.randint(3, 10)
        num_test = np.random.randint(1, 5)
        
        all_items = set(range(num_items))
        train_items = set(np.random.choice(list(all_items), num_train, replace=False))
        remaining = all_items - train_items
        test_items = set(np.random.choice(list(remaining), min(num_test, len(remaining)), replace=False))
        
        user_pos_train[u] = train_items
        user_pos_test[u] = test_items
    
    # ID mappings
    user_to_idx = {f'user_{i}': i for i in range(num_users)}
    idx_to_user = {i: f'user_{i}' for i in range(num_users)}
    item_to_idx = {f'item_{i}': i for i in range(num_items)}
    idx_to_item = {i: f'item_{i}' for i in range(num_items)}
    
    print("\n" + "-"*70)
    print("Example 1: Metric Functions")
    print("-"*70)
    
    predictions = np.array([10, 3, 7, 15, 2, 8, 1, 12, 5, 20])
    ground_truth = {3, 7, 9}
    
    recall_5 = recall_at_k(predictions, ground_truth, k=5)
    ndcg_5 = ndcg_at_k(predictions, ground_truth, k=5)
    recall_10 = recall_at_k(predictions, ground_truth, k=10)
    ndcg_10 = ndcg_at_k(predictions, ground_truth, k=10)
    
    print(f"Predictions: {predictions[:10]}")
    print(f"Ground truth: {ground_truth}")
    print(f"Recall@5: {recall_5:.3f}")
    print(f"NDCG@5: {ndcg_5:.3f}")
    print(f"Recall@10: {recall_10:.3f}")
    print(f"NDCG@10: {ndcg_10:.3f}")
    
    print("\n" + "-"*70)
    print("Example 2: Popularity Baseline")
    print("-"*70)
    
    baseline = PopularityBaseline()
    baseline.fit_from_train(user_pos_train, num_items=num_items)
    
    baseline_metrics = baseline.evaluate(
        user_pos_test,
        user_pos_train,
        k_values=[10, 20],
        filter_seen=True
    )
    
    print("\nBaseline metrics:")
    for k, v in sorted(baseline_metrics.items()):
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "-"*70)
    print("Example 3: ALSEvaluator - Full Evaluation")
    print("-"*70)
    
    evaluator = ALSEvaluator(
        user_factors=U,
        item_factors=V,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
        user_pos_train=user_pos_train,
        user_pos_test=user_pos_test
    )
    
    results = evaluator.evaluate(
        k_values=[10, 20],
        filter_seen=True,
        compare_baseline=True,
        baseline_source='train',
        model_type='als'
    )
    
    results.print_summary()
    
    print("\n" + "-"*70)
    print("Example 4: Quick Evaluate")
    print("-"*70)
    
    quick_metrics = quick_evaluate(
        U, V,
        user_pos_test,
        user_pos_train,
        k_values=[10, 20],
        model_type='als'
    )
    
    print("\nQuick metrics:")
    for k, v in sorted(quick_metrics.items()):
        print(f"  {k}: {v:.4f}")
    
    print("\n" + "-"*70)
    print("Example 5: Save Results")
    print("-"*70)
    
    output_path = Path("als_evaluation_results.json")
    results.save(output_path)
    print(f"Results saved to {output_path}")
    
    # Load back
    with open(output_path, 'r') as f:
        loaded = json.load(f)
    print(f"Loaded metrics: {loaded['metrics']}")
    
    # Cleanup
    output_path.unlink()
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)
