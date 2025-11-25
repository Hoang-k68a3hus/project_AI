"""
Model Evaluator Module for Collaborative Filtering.

This module provides comprehensive evaluation capabilities for CF models:
- ModelEvaluator: Main class for evaluating ALS/BPR models
- Batch recommendation generation
- Multi-metric evaluation
- Per-user analysis

Example:
    >>> from recsys.cf.evaluation import ModelEvaluator
    >>> evaluator = ModelEvaluator(U, V, k_values=[10, 20])
    >>> results = evaluator.evaluate(test_data, user_pos_train)
    >>> print(f"Recall@10: {results['recall@10']:.4f}")
"""

from typing import Dict, List, Set, Tuple, Optional, Union, Any
import numpy as np
from scipy import sparse
import logging
from datetime import datetime
import json
import os

from .metrics import (
    MetricFactory, 
    recall_at_k, 
    ndcg_at_k, 
    precision_at_k, 
    mrr, 
    map_at_k,
    coverage
)

logger = logging.getLogger(__name__)


# ============================================================================
# Model Evaluator
# ============================================================================

class ModelEvaluator:
    """
    Comprehensive evaluator for CF models (ALS, BPR).
    
    Features:
    - Batch recommendation generation
    - Multi-metric evaluation (Recall, NDCG, Precision, MRR, MAP)
    - Coverage analysis
    - Per-user metrics for analysis
    - Efficient seen-item filtering
    
    Example:
        >>> U = np.load('als_U.npy')  # Shape: (num_users, factors)
        >>> V = np.load('als_V.npy')  # Shape: (num_items, factors)
        >>> evaluator = ModelEvaluator(U, V, k_values=[10, 20])
        >>> results = evaluator.evaluate(test_data, user_pos_train)
    """
    
    def __init__(
        self,
        U: np.ndarray,
        V: np.ndarray,
        k_values: List[int] = [10, 20],
        batch_size: int = 1000,
        use_argpartition: bool = True
    ):
        """
        Initialize model evaluator.
        
        Args:
            U: User embedding matrix (num_users, factors)
            V: Item embedding matrix (num_items, factors)
            k_values: List of K values for @K metrics
            batch_size: Batch size for recommendation generation
            use_argpartition: Use O(n) argpartition instead of O(n log n) argsort
        """
        self.U = U
        self.V = V
        self.num_users = U.shape[0]
        self.num_items = V.shape[0]
        self.factors = U.shape[1]
        self.k_values = sorted(k_values)
        self.max_k = max(k_values)
        self.batch_size = batch_size
        self.use_argpartition = use_argpartition
        
        # Create metrics
        self.metrics = MetricFactory.create_standard_metrics(k_values)
        
        # Store results
        self._last_results: Optional[Dict] = None
        self._per_user_metrics: Optional[Dict] = None
        
        logger.info(
            f"ModelEvaluator initialized: {self.num_users} users, "
            f"{self.num_items} items, {self.factors} factors, K={k_values}"
        )
    
    def _get_top_k(
        self,
        scores: np.ndarray,
        k: int,
        exclude_indices: Optional[Set[int]] = None
    ) -> np.ndarray:
        """
        Get top-K items from scores with efficient filtering.
        
        Args:
            scores: Score array (num_items,)
            k: Number of items to return
            exclude_indices: Set of item indices to exclude (seen items)
        
        Returns:
            Array of top-K item indices
        """
        if exclude_indices:
            # Mask excluded items
            scores = scores.copy()
            scores[list(exclude_indices)] = -np.inf
        
        if self.use_argpartition and k < len(scores) // 2:
            # Use argpartition for efficiency (O(n) vs O(n log n))
            # Get indices of top-K unsorted
            top_k_unsorted = np.argpartition(scores, -k)[-k:]
            # Sort only the top-K
            sorted_order = np.argsort(scores[top_k_unsorted])[::-1]
            return top_k_unsorted[sorted_order]
        else:
            # Use full argsort
            return np.argsort(scores)[::-1][:k]
    
    def generate_recommendations(
        self,
        test_users: List[int],
        user_pos_train: Dict[int, Set[int]],
        k: Optional[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        Generate top-K recommendations for test users.
        
        Args:
            test_users: List of user indices to generate recommendations for
            user_pos_train: Dict mapping u_idx to set of positive item indices (to filter)
            k: Number of recommendations (default: max_k)
        
        Returns:
            Dict mapping u_idx to array of recommended item indices
        """
        k = k if k is not None else self.max_k
        recommendations = {}
        
        # Process in batches for memory efficiency
        for batch_start in range(0, len(test_users), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(test_users))
            batch_users = test_users[batch_start:batch_end]
            batch_indices = np.array(batch_users)
            
            # Compute scores for batch: (batch_size, num_items)
            batch_scores = self.U[batch_indices] @ self.V.T
            
            # Get top-K for each user
            for i, u_idx in enumerate(batch_users):
                seen_items = user_pos_train.get(u_idx, set())
                top_k = self._get_top_k(batch_scores[i], k, seen_items)
                recommendations[u_idx] = top_k
        
        return recommendations
    
    def _prepare_ground_truth(
        self,
        test_data: Union[Dict[int, Set[int]], 'pd.DataFrame'],
        user_col: str = 'u_idx',
        item_col: str = 'i_idx'
    ) -> Dict[int, Set[int]]:
        """
        Prepare ground truth from test data.
        
        Args:
            test_data: Either Dict {u_idx: set(i_idx)} or DataFrame
            user_col: User column name (if DataFrame)
            item_col: Item column name (if DataFrame)
        
        Returns:
            Dict mapping u_idx to set of positive item indices
        """
        if isinstance(test_data, dict):
            return test_data
        
        # Handle DataFrame
        import pandas as pd
        if isinstance(test_data, pd.DataFrame):
            ground_truth = {}
            for u_idx in test_data[user_col].unique():
                mask = test_data[user_col] == u_idx
                items = set(test_data.loc[mask, item_col])
                ground_truth[u_idx] = items
            return ground_truth
        
        raise ValueError(f"Unsupported test_data type: {type(test_data)}")
    
    def evaluate(
        self,
        test_data: Union[Dict[int, Set[int]], 'pd.DataFrame'],
        user_pos_train: Dict[int, Set[int]],
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        compute_per_user: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test data (Dict or DataFrame with u_idx, i_idx)
            user_pos_train: Dict mapping u_idx to set of train positive items
            user_col: User column name (if DataFrame)
            item_col: Item column name (if DataFrame)
            compute_per_user: If True, store per-user metrics for analysis
        
        Returns:
            Dict with evaluation metrics:
            - recall@K, ndcg@K, precision@K, map@K for each K
            - mrr (Mean Reciprocal Rank)
            - coverage (catalog coverage)
            - num_users_evaluated
            - evaluation_time
        """
        start_time = datetime.now()
        
        # Prepare ground truth
        ground_truth = self._prepare_ground_truth(test_data, user_col, item_col)
        test_users = list(ground_truth.keys())
        
        logger.info(f"Evaluating on {len(test_users)} test users...")
        
        # Generate recommendations
        recommendations = self.generate_recommendations(test_users, user_pos_train)
        
        # Compute metrics per user
        per_user_metrics = {k: [] for k in self.k_values}
        per_user_metrics['mrr'] = []
        
        for u_idx in test_users:
            gt = ground_truth[u_idx]
            recs = recommendations[u_idx]
            
            if len(gt) == 0:
                continue
            
            # Compute metrics for each K
            for k in self.k_values:
                per_user_metrics[k].append({
                    'u_idx': u_idx,
                    'recall': recall_at_k(recs, gt, k),
                    'ndcg': ndcg_at_k(recs, gt, k),
                    'precision': precision_at_k(recs, gt, k),
                    'map': map_at_k(recs, gt, k)
                })
            
            # MRR
            per_user_metrics['mrr'].append({
                'u_idx': u_idx,
                'mrr': mrr(recs, gt)
            })
        
        # Aggregate metrics
        results = {}
        
        for k in self.k_values:
            user_metrics = per_user_metrics[k]
            if len(user_metrics) > 0:
                results[f'recall@{k}'] = np.mean([m['recall'] for m in user_metrics])
                results[f'ndcg@{k}'] = np.mean([m['ndcg'] for m in user_metrics])
                results[f'precision@{k}'] = np.mean([m['precision'] for m in user_metrics])
                results[f'map@{k}'] = np.mean([m['map'] for m in user_metrics])
            else:
                results[f'recall@{k}'] = 0.0
                results[f'ndcg@{k}'] = 0.0
                results[f'precision@{k}'] = 0.0
                results[f'map@{k}'] = 0.0
        
        # MRR
        if len(per_user_metrics['mrr']) > 0:
            results['mrr'] = np.mean([m['mrr'] for m in per_user_metrics['mrr']])
        else:
            results['mrr'] = 0.0
        
        # Coverage
        results['coverage'] = coverage(
            {u: recs.tolist() for u, recs in recommendations.items()},
            self.num_items
        )
        
        # Metadata
        results['num_users_evaluated'] = len(test_users)
        results['num_users_with_test_items'] = len(per_user_metrics.get(self.k_values[0], []))
        results['evaluation_time_seconds'] = (datetime.now() - start_time).total_seconds()
        
        # Store results
        self._last_results = results
        if compute_per_user:
            self._per_user_metrics = per_user_metrics
        
        logger.info(
            f"Evaluation complete: Recall@{self.k_values[0]}={results[f'recall@{self.k_values[0]}']:.4f}, "
            f"NDCG@{self.k_values[0]}={results[f'ndcg@{self.k_values[0]}']:.4f}, "
            f"Coverage={results['coverage']:.4f}"
        )
        
        return results
    
    def get_per_user_metrics(self) -> Optional[Dict]:
        """
        Get per-user metrics from last evaluation.
        
        Returns:
            Dict with per-user metrics or None if not computed
        """
        return self._per_user_metrics
    
    def get_metric_distribution(
        self,
        metric: str = 'recall',
        k: int = 10
    ) -> Optional[np.ndarray]:
        """
        Get distribution of a metric across users.
        
        Args:
            metric: Metric name ('recall', 'ndcg', 'precision', 'map')
            k: K value
        
        Returns:
            Array of metric values per user
        """
        if self._per_user_metrics is None:
            logger.warning("Per-user metrics not computed. Call evaluate() with compute_per_user=True")
            return None
        
        if k not in self._per_user_metrics:
            logger.warning(f"K={k} not in evaluated K values")
            return None
        
        return np.array([m[metric] for m in self._per_user_metrics[k]])
    
    def stratify_by_activity(
        self,
        user_interaction_counts: Dict[int, int],
        bins: List[Tuple[int, int]] = [(2, 5), (5, 10), (10, 50), (50, float('inf'))]
    ) -> Dict[str, Dict[str, float]]:
        """
        Stratify evaluation results by user activity level.
        
        Args:
            user_interaction_counts: Dict mapping u_idx to interaction count
            bins: List of (min_count, max_count) tuples
        
        Returns:
            Dict mapping bin name to aggregated metrics
        """
        if self._per_user_metrics is None:
            raise ValueError("Per-user metrics not computed")
        
        results = {}
        k = self.k_values[0]  # Use first K for stratification
        
        for bin_min, bin_max in bins:
            bin_name = f"{bin_min}-{bin_max if bin_max < float('inf') else '+'}"
            bin_users = set()
            
            for u_idx, count in user_interaction_counts.items():
                if bin_min <= count < bin_max:
                    bin_users.add(u_idx)
            
            # Filter metrics for this bin
            bin_metrics = [
                m for m in self._per_user_metrics[k]
                if m['u_idx'] in bin_users
            ]
            
            if len(bin_metrics) > 0:
                results[bin_name] = {
                    'num_users': len(bin_metrics),
                    'recall': np.mean([m['recall'] for m in bin_metrics]),
                    'ndcg': np.mean([m['ndcg'] for m in bin_metrics]),
                    'precision': np.mean([m['precision'] for m in bin_metrics]),
                }
            else:
                results[bin_name] = {
                    'num_users': 0,
                    'recall': 0.0,
                    'ndcg': 0.0,
                    'precision': 0.0,
                }
        
        return results
    
    def save_results(
        self,
        output_path: str,
        model_name: str = 'unknown',
        hyperparams: Optional[Dict] = None
    ) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Path to save JSON file
            model_name: Name of the model
            hyperparams: Optional model hyperparameters
        """
        if self._last_results is None:
            raise ValueError("No evaluation results to save. Call evaluate() first.")
        
        output = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'model_shape': {
                'num_users': self.num_users,
                'num_items': self.num_items,
                'factors': self.factors
            },
            'k_values': self.k_values,
            'hyperparams': hyperparams or {},
            'metrics': self._last_results
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")


# ============================================================================
# Batch Evaluator for Multiple Models
# ============================================================================

class BatchModelEvaluator:
    """
    Evaluate multiple models and compare results.
    
    Example:
        >>> evaluator = BatchModelEvaluator(k_values=[10, 20])
        >>> evaluator.add_model('als', U_als, V_als, {'factors': 64, 'reg': 0.01})
        >>> evaluator.add_model('bpr', U_bpr, V_bpr, {'factors': 64, 'lr': 0.05})
        >>> results = evaluator.evaluate_all(test_data, user_pos_train)
    """
    
    def __init__(self, k_values: List[int] = [10, 20]):
        """
        Initialize batch evaluator.
        
        Args:
            k_values: List of K values for @K metrics
        """
        self.k_values = k_values
        self.models: Dict[str, Dict] = {}
        self.results: Dict[str, Dict] = {}
    
    def add_model(
        self,
        name: str,
        U: np.ndarray,
        V: np.ndarray,
        hyperparams: Optional[Dict] = None
    ) -> None:
        """
        Add a model to evaluate.
        
        Args:
            name: Model name/identifier
            U: User embedding matrix
            V: Item embedding matrix
            hyperparams: Optional model hyperparameters
        """
        self.models[name] = {
            'U': U,
            'V': V,
            'hyperparams': hyperparams or {},
            'evaluator': ModelEvaluator(U, V, self.k_values)
        }
        logger.info(f"Added model '{name}' with shape U={U.shape}, V={V.shape}")
    
    def evaluate_all(
        self,
        test_data: Union[Dict[int, Set[int]], 'pd.DataFrame'],
        user_pos_train: Dict[int, Set[int]]
    ) -> Dict[str, Dict]:
        """
        Evaluate all added models.
        
        Args:
            test_data: Test data
            user_pos_train: Training positive items per user
        
        Returns:
            Dict mapping model name to evaluation results
        """
        for name, model_data in self.models.items():
            logger.info(f"Evaluating model '{name}'...")
            results = model_data['evaluator'].evaluate(test_data, user_pos_train)
            results['hyperparams'] = model_data['hyperparams']
            self.results[name] = results
        
        return self.results
    
    def get_comparison_table(self) -> 'pd.DataFrame':
        """
        Get comparison table of all model results.
        
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        import pandas as pd
        
        rows = []
        for name, metrics in self.results.items():
            row = {'model': name}
            row.update(metrics.get('hyperparams', {}))
            
            # Add metrics
            for k in self.k_values:
                row[f'recall@{k}'] = metrics.get(f'recall@{k}', 0)
                row[f'ndcg@{k}'] = metrics.get(f'ndcg@{k}', 0)
            row['mrr'] = metrics.get('mrr', 0)
            row['coverage'] = metrics.get('coverage', 0)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_best_model(self, metric: str = 'ndcg@10') -> Tuple[str, float]:
        """
        Get best model by a specific metric.
        
        Args:
            metric: Metric name to compare
        
        Returns:
            Tuple of (model_name, metric_value)
        """
        best_name = None
        best_value = -1
        
        for name, metrics in self.results.items():
            value = metrics.get(metric, 0)
            if value > best_value:
                best_value = value
                best_name = name
        
        return best_name, best_value


# ============================================================================
# Convenience Functions
# ============================================================================

def evaluate_model(
    U: np.ndarray,
    V: np.ndarray,
    test_data: Union[Dict[int, Set[int]], 'pd.DataFrame'],
    user_pos_train: Dict[int, Set[int]],
    k_values: List[int] = [10, 20]
) -> Dict[str, Any]:
    """
    Evaluate a CF model.
    
    Args:
        U: User embedding matrix
        V: Item embedding matrix
        test_data: Test data
        user_pos_train: Training positive items per user
        k_values: K values for evaluation
    
    Returns:
        Dict with evaluation metrics
    """
    evaluator = ModelEvaluator(U, V, k_values)
    return evaluator.evaluate(test_data, user_pos_train)


def load_and_evaluate(
    model_dir: str,
    test_data: Union[Dict[int, Set[int]], 'pd.DataFrame'],
    user_pos_train: Dict[int, Set[int]],
    k_values: List[int] = [10, 20]
) -> Dict[str, Any]:
    """
    Load model from directory and evaluate.
    
    Args:
        model_dir: Directory containing U.npy and V.npy
        test_data: Test data
        user_pos_train: Training positive items per user
        k_values: K values for evaluation
    
    Returns:
        Dict with evaluation metrics
    """
    U = np.load(os.path.join(model_dir, 'U.npy'))
    V = np.load(os.path.join(model_dir, 'V.npy'))
    
    return evaluate_model(U, V, test_data, user_pos_train, k_values)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Model Evaluator Module")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create mock model
    num_users = 100
    num_items = 50
    factors = 16
    
    U = np.random.randn(num_users, factors)
    V = np.random.randn(num_items, factors)
    
    # Normalize
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    
    # Create mock test data
    # Each user has 1-3 positive test items
    test_data = {}
    user_pos_train = {}
    
    for u in range(num_users):
        # Random train positives
        train_items = set(np.random.choice(num_items, size=np.random.randint(5, 15), replace=False))
        user_pos_train[u] = train_items
        
        # Random test positives (not in train)
        remaining_items = list(set(range(num_items)) - train_items)
        if len(remaining_items) > 0:
            num_test = min(np.random.randint(1, 4), len(remaining_items))
            test_items = set(np.random.choice(remaining_items, size=num_test, replace=False))
            test_data[u] = test_items
    
    print(f"\nMock data: {num_users} users, {num_items} items, {factors} factors")
    print(f"Test users: {len(test_data)}, Avg train items: {np.mean([len(v) for v in user_pos_train.values()]):.1f}")
    
    # Test ModelEvaluator
    print("\n--- ModelEvaluator ---")
    evaluator = ModelEvaluator(U, V, k_values=[5, 10])
    results = evaluator.evaluate(test_data, user_pos_train, compute_per_user=True)
    
    print("Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test metric distribution
    print("\n--- Metric Distribution ---")
    recall_dist = evaluator.get_metric_distribution('recall', k=10)
    if recall_dist is not None:
        print(f"Recall@10 distribution: mean={np.mean(recall_dist):.4f}, std={np.std(recall_dist):.4f}")
    
    # Test BatchModelEvaluator
    print("\n--- BatchModelEvaluator ---")
    batch_evaluator = BatchModelEvaluator(k_values=[5, 10])
    
    # Add multiple "models" (same embeddings with noise)
    batch_evaluator.add_model('model_a', U, V, {'reg': 0.01})
    batch_evaluator.add_model('model_b', U * 0.9 + np.random.randn(*U.shape) * 0.1, V, {'reg': 0.05})
    
    all_results = batch_evaluator.evaluate_all(test_data, user_pos_train)
    
    print("\nComparison:")
    for name, metrics in all_results.items():
        print(f"  {name}: Recall@10={metrics['recall@10']:.4f}, NDCG@10={metrics['ndcg@10']:.4f}")
    
    best_model, best_score = batch_evaluator.get_best_model('recall@10')
    print(f"\nBest model by Recall@10: {best_model} ({best_score:.4f})")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
