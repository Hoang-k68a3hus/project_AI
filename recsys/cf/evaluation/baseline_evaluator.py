"""
Baseline Evaluator Module for Collaborative Filtering.

This module provides baseline recommenders for comparison:
- PopularityBaseline: Recommend most popular items
- RandomBaseline: Random item recommendations
- ItemSimilarityBaseline: Item-item similarity using content

These baselines provide lower bounds for CF model evaluation.

Example:
    >>> from recsys.cf.evaluation import PopularityBaseline, evaluate_baseline_popularity
    >>> baseline = PopularityBaseline(item_popularity)
    >>> results = baseline.evaluate(test_data)
"""

from typing import Dict, List, Set, Tuple, Optional, Union, Any
import numpy as np
from scipy import sparse
import logging
from datetime import datetime
from abc import ABC, abstractmethod

from .metrics import recall_at_k, ndcg_at_k, precision_at_k, mrr, map_at_k, coverage

logger = logging.getLogger(__name__)


# ============================================================================
# Abstract Baseline
# ============================================================================

class BaselineRecommender(ABC):
    """
    Abstract base class for baseline recommenders.
    """
    
    def __init__(self, name: str, num_items: int):
        """
        Initialize baseline recommender.
        
        Args:
            name: Baseline name
            num_items: Total number of items
        """
        self.name = name
        self.num_items = num_items
    
    @abstractmethod
    def recommend(
        self,
        user_idx: int,
        k: int,
        exclude_items: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Generate recommendations for a user.
        
        Args:
            user_idx: User index
            k: Number of recommendations
            exclude_items: Items to exclude (seen items)
        
        Returns:
            List of recommended item indices
        """
        pass
    
    def recommend_batch(
        self,
        user_indices: List[int],
        k: int,
        user_exclude_items: Optional[Dict[int, Set[int]]] = None
    ) -> Dict[int, List[int]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_indices: List of user indices
            k: Number of recommendations
            user_exclude_items: Dict mapping user_idx to excluded items
        
        Returns:
            Dict mapping user_idx to recommendations
        """
        recommendations = {}
        exclude_items = user_exclude_items or {}
        
        for u_idx in user_indices:
            excluded = exclude_items.get(u_idx, set())
            recommendations[u_idx] = self.recommend(u_idx, k, excluded)
        
        return recommendations
    
    def evaluate(
        self,
        test_data: Union[Dict[int, Set[int]], 'pd.DataFrame'],
        user_pos_train: Optional[Dict[int, Set[int]]] = None,
        k_values: List[int] = [10, 20],
        user_col: str = 'u_idx',
        item_col: str = 'i_idx'
    ) -> Dict[str, Any]:
        """
        Evaluate baseline on test data.
        
        Args:
            test_data: Test data (Dict or DataFrame)
            user_pos_train: Training positive items per user (for filtering)
            k_values: K values for evaluation
            user_col: User column name
            item_col: Item column name
        
        Returns:
            Dict with evaluation metrics
        """
        start_time = datetime.now()
        
        # Prepare ground truth
        ground_truth = self._prepare_ground_truth(test_data, user_col, item_col)
        test_users = list(ground_truth.keys())
        max_k = max(k_values)
        
        logger.info(f"Evaluating {self.name} on {len(test_users)} test users...")
        
        # Generate recommendations
        user_pos_train = user_pos_train or {}
        recommendations = self.recommend_batch(test_users, max_k, user_pos_train)
        
        # Compute metrics
        results = self._compute_metrics(recommendations, ground_truth, k_values)
        
        # Coverage
        results['coverage'] = coverage(recommendations, self.num_items)
        
        # Metadata
        results['baseline_name'] = self.name
        results['num_users_evaluated'] = len(test_users)
        results['evaluation_time_seconds'] = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            f"{self.name} evaluation: Recall@{k_values[0]}={results[f'recall@{k_values[0]}']:.4f}, "
            f"Coverage={results['coverage']:.4f}"
        )
        
        return results
    
    def _prepare_ground_truth(
        self,
        test_data: Union[Dict[int, Set[int]], 'pd.DataFrame'],
        user_col: str,
        item_col: str
    ) -> Dict[int, Set[int]]:
        """Prepare ground truth from test data."""
        if isinstance(test_data, dict):
            return test_data
        
        import pandas as pd
        if isinstance(test_data, pd.DataFrame):
            ground_truth = {}
            for u_idx in test_data[user_col].unique():
                mask = test_data[user_col] == u_idx
                items = set(test_data.loc[mask, item_col])
                ground_truth[u_idx] = items
            return ground_truth
        
        raise ValueError(f"Unsupported test_data type: {type(test_data)}")
    
    def _compute_metrics(
        self,
        recommendations: Dict[int, List[int]],
        ground_truth: Dict[int, Set[int]],
        k_values: List[int]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        results = {}
        
        for k in k_values:
            recalls = []
            ndcgs = []
            precisions = []
            maps = []
            
            for u_idx, gt in ground_truth.items():
                if len(gt) == 0:
                    continue
                
                recs = recommendations.get(u_idx, [])
                recalls.append(recall_at_k(recs, gt, k))
                ndcgs.append(ndcg_at_k(recs, gt, k))
                precisions.append(precision_at_k(recs, gt, k))
                maps.append(map_at_k(recs, gt, k))
            
            if len(recalls) > 0:
                results[f'recall@{k}'] = np.mean(recalls)
                results[f'ndcg@{k}'] = np.mean(ndcgs)
                results[f'precision@{k}'] = np.mean(precisions)
                results[f'map@{k}'] = np.mean(maps)
            else:
                results[f'recall@{k}'] = 0.0
                results[f'ndcg@{k}'] = 0.0
                results[f'precision@{k}'] = 0.0
                results[f'map@{k}'] = 0.0
        
        # MRR
        mrr_values = []
        for u_idx, gt in ground_truth.items():
            if len(gt) == 0:
                continue
            recs = recommendations.get(u_idx, [])
            mrr_values.append(mrr(recs, gt))
        
        results['mrr'] = np.mean(mrr_values) if mrr_values else 0.0
        
        return results
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', num_items={self.num_items})"


# ============================================================================
# Popularity Baseline
# ============================================================================

class PopularityBaseline(BaselineRecommender):
    """
    Popularity-based baseline recommender.
    
    Recommends the most popular items (by interaction count) to all users.
    
    Expected Performance:
        - Recall@10: 0.12 - 0.15
        - NDCG@10: 0.08 - 0.10
        - Coverage: <0.05 (very low)
    
    Example:
        >>> item_popularity = np.array([100, 50, 200, ...])  # Interaction counts
        >>> baseline = PopularityBaseline(item_popularity)
        >>> recs = baseline.recommend(user_idx=0, k=10)
    """
    
    def __init__(
        self,
        item_popularity: np.ndarray,
        name: str = "Popularity"
    ):
        """
        Initialize popularity baseline.
        
        Args:
            item_popularity: Array of item popularity scores (e.g., interaction counts)
            name: Baseline name
        """
        super().__init__(name=name, num_items=len(item_popularity))
        
        self.item_popularity = item_popularity
        
        # Precompute global ranking
        self._global_ranking = np.argsort(item_popularity)[::-1]
        
        logger.info(
            f"PopularityBaseline initialized: {self.num_items} items, "
            f"top item popularity={item_popularity[self._global_ranking[0]]:.1f}"
        )
    
    def recommend(
        self,
        user_idx: int,
        k: int,
        exclude_items: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Recommend top-K most popular items.
        
        Args:
            user_idx: User index (not used - same recs for all users)
            k: Number of recommendations
            exclude_items: Items to exclude
        
        Returns:
            List of top-K popular item indices
        """
        if exclude_items is None or len(exclude_items) == 0:
            return self._global_ranking[:k].tolist()
        
        # Filter excluded items
        recommendations = []
        for item_idx in self._global_ranking:
            if item_idx not in exclude_items:
                recommendations.append(item_idx)
            if len(recommendations) >= k:
                break
        
        return recommendations
    
    def get_popularity_stats(self) -> Dict[str, Any]:
        """
        Get statistics about item popularity.
        
        Returns:
            Dict with popularity statistics
        """
        return {
            'min': float(np.min(self.item_popularity)),
            'max': float(np.max(self.item_popularity)),
            'mean': float(np.mean(self.item_popularity)),
            'median': float(np.median(self.item_popularity)),
            'std': float(np.std(self.item_popularity)),
            'top_10_items': self._global_ranking[:10].tolist(),
            'top_10_popularity': self.item_popularity[self._global_ranking[:10]].tolist()
        }


# ============================================================================
# Random Baseline
# ============================================================================

class RandomBaseline(BaselineRecommender):
    """
    Random baseline recommender.
    
    Recommends random items to users. Serves as a lower bound.
    
    Expected Performance:
        - Recall@10: ~0.01 (very low)
        - NDCG@10: ~0.005
        - Coverage: High (random sampling covers catalog)
    
    Example:
        >>> baseline = RandomBaseline(num_items=1000, seed=42)
        >>> recs = baseline.recommend(user_idx=0, k=10)
    """
    
    def __init__(
        self,
        num_items: int,
        seed: Optional[int] = None,
        name: str = "Random"
    ):
        """
        Initialize random baseline.
        
        Args:
            num_items: Total number of items
            seed: Random seed for reproducibility
            name: Baseline name
        """
        super().__init__(name=name, num_items=num_items)
        
        self.rng = np.random.RandomState(seed)
        
        logger.info(f"RandomBaseline initialized: {num_items} items, seed={seed}")
    
    def recommend(
        self,
        user_idx: int,
        k: int,
        exclude_items: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Recommend random items.
        
        Args:
            user_idx: User index
            k: Number of recommendations
            exclude_items: Items to exclude
        
        Returns:
            List of random item indices
        """
        if exclude_items is None:
            exclude_items = set()
        
        # Available items
        available = list(set(range(self.num_items)) - exclude_items)
        
        if len(available) <= k:
            return available
        
        # Random sample
        return self.rng.choice(available, size=k, replace=False).tolist()


# ============================================================================
# Item Similarity Baseline (Content-based)
# ============================================================================

class ItemSimilarityBaseline(BaselineRecommender):
    """
    Item-item similarity baseline using content embeddings.
    
    Recommends items similar to user's history.
    
    Example:
        >>> bert_embeddings = np.load('product_embeddings.npy')
        >>> baseline = ItemSimilarityBaseline(bert_embeddings, user_histories)
        >>> recs = baseline.recommend(user_idx=0, k=10)
    """
    
    def __init__(
        self,
        item_embeddings: np.ndarray,
        user_histories: Dict[int, Set[int]],
        aggregation: str = 'mean',
        name: str = "ItemSimilarity"
    ):
        """
        Initialize item similarity baseline.
        
        Args:
            item_embeddings: Item content embeddings (num_items, embedding_dim)
            user_histories: Dict mapping user_idx to set of interacted item indices
            aggregation: How to aggregate user profile ('mean', 'max')
            name: Baseline name
        """
        super().__init__(name=name, num_items=len(item_embeddings))
        
        self.item_embeddings = item_embeddings
        self.user_histories = user_histories
        self.aggregation = aggregation
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        self._normalized_embeddings = item_embeddings / norms
        
        logger.info(
            f"ItemSimilarityBaseline initialized: {self.num_items} items, "
            f"embedding_dim={item_embeddings.shape[1]}, aggregation={aggregation}"
        )
    
    def _compute_user_profile(self, user_idx: int) -> Optional[np.ndarray]:
        """
        Compute user content profile from history.
        
        Args:
            user_idx: User index
        
        Returns:
            User profile embedding or None
        """
        history = self.user_histories.get(user_idx, set())
        
        if len(history) == 0:
            return None
        
        # Get embeddings for history items
        history_indices = list(history)
        history_embeddings = self._normalized_embeddings[history_indices]
        
        # Aggregate
        if self.aggregation == 'mean':
            profile = np.mean(history_embeddings, axis=0)
        elif self.aggregation == 'max':
            profile = np.max(history_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        # Normalize profile
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        
        return profile
    
    def recommend(
        self,
        user_idx: int,
        k: int,
        exclude_items: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Recommend items similar to user's history.
        
        Args:
            user_idx: User index
            k: Number of recommendations
            exclude_items: Items to exclude
        
        Returns:
            List of recommended item indices
        """
        profile = self._compute_user_profile(user_idx)
        
        if profile is None:
            # No history - return empty
            return []
        
        # Compute similarities
        similarities = self._normalized_embeddings @ profile
        
        # Mask excluded items
        if exclude_items:
            similarities[list(exclude_items)] = -np.inf
        
        # Get top-K
        top_k = np.argsort(similarities)[::-1][:k]
        
        return top_k.tolist()


# ============================================================================
# Baseline Comparison
# ============================================================================

class BaselineComparator:
    """
    Compare multiple baselines.
    
    Example:
        >>> comparator = BaselineComparator()
        >>> comparator.add_baseline('popularity', PopularityBaseline(item_pop))
        >>> comparator.add_baseline('random', RandomBaseline(num_items))
        >>> results = comparator.compare_all(test_data, user_pos_train)
    """
    
    def __init__(self, k_values: List[int] = [10, 20]):
        """
        Initialize baseline comparator.
        
        Args:
            k_values: K values for evaluation
        """
        self.k_values = k_values
        self.baselines: Dict[str, BaselineRecommender] = {}
        self.results: Dict[str, Dict] = {}
    
    def add_baseline(self, name: str, baseline: BaselineRecommender) -> None:
        """
        Add a baseline to compare.
        
        Args:
            name: Baseline identifier
            baseline: BaselineRecommender instance
        """
        self.baselines[name] = baseline
        logger.info(f"Added baseline '{name}'")
    
    def compare_all(
        self,
        test_data: Union[Dict[int, Set[int]], 'pd.DataFrame'],
        user_pos_train: Optional[Dict[int, Set[int]]] = None
    ) -> Dict[str, Dict]:
        """
        Evaluate and compare all baselines.
        
        Args:
            test_data: Test data
            user_pos_train: Training positive items per user
        
        Returns:
            Dict mapping baseline name to results
        """
        for name, baseline in self.baselines.items():
            logger.info(f"Evaluating baseline '{name}'...")
            results = baseline.evaluate(test_data, user_pos_train, self.k_values)
            self.results[name] = results
        
        return self.results
    
    def get_comparison_table(self) -> 'pd.DataFrame':
        """
        Get comparison table of all baselines.
        
        Returns:
            DataFrame with baselines as rows and metrics as columns
        """
        import pandas as pd
        
        rows = []
        for name, metrics in self.results.items():
            row = {'baseline': name}
            
            for k in self.k_values:
                row[f'recall@{k}'] = metrics.get(f'recall@{k}', 0)
                row[f'ndcg@{k}'] = metrics.get(f'ndcg@{k}', 0)
            row['mrr'] = metrics.get('mrr', 0)
            row['coverage'] = metrics.get('coverage', 0)
            
            rows.append(row)
        
        return pd.DataFrame(rows)


# ============================================================================
# Convenience Functions
# ============================================================================

def evaluate_baseline_popularity(
    test_data: Union[Dict[int, Set[int]], 'pd.DataFrame'],
    item_popularity: np.ndarray,
    k_values: List[int] = [10, 20],
    user_pos_train: Optional[Dict[int, Set[int]]] = None
) -> Dict[str, Any]:
    """
    Evaluate popularity baseline.
    
    Args:
        test_data: Test data
        item_popularity: Item popularity scores
        k_values: K values for evaluation
        user_pos_train: Training positive items per user
    
    Returns:
        Dict with evaluation metrics
    """
    baseline = PopularityBaseline(item_popularity)
    return baseline.evaluate(test_data, user_pos_train, k_values)


def evaluate_baseline_random(
    test_data: Union[Dict[int, Set[int]], 'pd.DataFrame'],
    num_items: int,
    k_values: List[int] = [10, 20],
    user_pos_train: Optional[Dict[int, Set[int]]] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluate random baseline.
    
    Args:
        test_data: Test data
        num_items: Total number of items
        k_values: K values for evaluation
        user_pos_train: Training positive items per user
        seed: Random seed
    
    Returns:
        Dict with evaluation metrics
    """
    baseline = RandomBaseline(num_items, seed=seed)
    return baseline.evaluate(test_data, user_pos_train, k_values)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Baseline Evaluator Module")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create mock data
    num_users = 100
    num_items = 50
    
    # Create mock popularity
    item_popularity = np.random.randint(1, 100, size=num_items)
    
    # Create mock test data
    test_data = {}
    user_pos_train = {}
    
    for u in range(num_users):
        # Random train positives
        train_items = set(np.random.choice(num_items, size=np.random.randint(5, 15), replace=False))
        user_pos_train[u] = train_items
        
        # Random test positives
        remaining = list(set(range(num_items)) - train_items)
        if len(remaining) > 0:
            num_test = min(np.random.randint(1, 4), len(remaining))
            test_items = set(np.random.choice(remaining, size=num_test, replace=False))
            test_data[u] = test_items
    
    print(f"\nMock data: {num_users} users, {num_items} items")
    
    # Test PopularityBaseline
    print("\n--- PopularityBaseline ---")
    pop_baseline = PopularityBaseline(item_popularity)
    pop_results = pop_baseline.evaluate(test_data, user_pos_train, k_values=[5, 10])
    
    print("Results:")
    for key, value in pop_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif key != 'baseline_name':
            print(f"  {key}: {value}")
    
    # Test RandomBaseline
    print("\n--- RandomBaseline ---")
    random_baseline = RandomBaseline(num_items, seed=42)
    random_results = random_baseline.evaluate(test_data, user_pos_train, k_values=[5, 10])
    
    print("Results:")
    for key, value in random_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # Test ItemSimilarityBaseline
    print("\n--- ItemSimilarityBaseline ---")
    item_embeddings = np.random.randn(num_items, 64)
    sim_baseline = ItemSimilarityBaseline(item_embeddings, user_pos_train)
    sim_results = sim_baseline.evaluate(test_data, user_pos_train, k_values=[5, 10])
    
    print("Results:")
    for key, value in sim_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # Test BaselineComparator
    print("\n--- BaselineComparator ---")
    comparator = BaselineComparator(k_values=[5, 10])
    comparator.add_baseline('popularity', pop_baseline)
    comparator.add_baseline('random', random_baseline)
    comparator.add_baseline('item_sim', sim_baseline)
    
    all_results = comparator.compare_all(test_data, user_pos_train)
    
    print("\nComparison:")
    for name, metrics in all_results.items():
        print(f"  {name}: Recall@10={metrics['recall@10']:.4f}, Coverage={metrics['coverage']:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
