"""
ALS Recommendation Generation Module (Task 02 - Step 5)

This module handles recommendation generation from trained ALS embeddings:
- Batch scoring for multiple users efficiently
- Seen-item filtering to avoid recommending purchased items
- Top-K selection using optimized algorithms
- ID mapping from internal indices to product IDs
- Support for filtering by item attributes

Key Features:
- Efficient batch computation: U @ V.T for all users at once
- Memory-optimized scoring for large user bases (chunked processing)
- Fast top-K selection using np.argpartition
- Flexible seen-item filtering strategies
- Integration with ID mappings from Task 01

Author: Copilot AI Assistant
Date: November 23, 2025
"""

import logging
from typing import Dict, Any, Optional, List, Set, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.sparse import csr_matrix

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """
    Container for recommendation results.
    
    Attributes:
        user_id: Original user ID (string)
        user_idx: Internal user index (integer)
        item_indices: Recommended item indices (internal)
        item_ids: Recommended product IDs (original)
        scores: Predicted scores for recommended items
        rank: Rank of each recommendation (1-indexed)
    """
    user_id: str
    user_idx: int
    item_indices: np.ndarray
    item_ids: List[str]
    scores: np.ndarray
    rank: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'user_id': self.user_id,
            'user_idx': self.user_idx,
            'item_indices': self.item_indices.tolist(),
            'item_ids': self.item_ids,
            'scores': self.scores.tolist(),
            'rank': self.rank.tolist()
        }
    
    def __repr__(self) -> str:
        return (
            f"RecommendationResult(user_id={self.user_id}, "
            f"num_items={len(self.item_ids)}, "
            f"top_score={self.scores[0]:.4f})"
        )


class ALSRecommender:
    """
    Generate recommendations from trained ALS embeddings.
    
    This class provides:
    1. Batch scoring for efficient recommendation generation
    2. Seen-item filtering to avoid redundant recommendations
    3. Fast top-K selection algorithms
    4. ID mapping between internal indices and product IDs
    5. Support for user/item attribute filtering
    
    Attributes:
        user_factors: User embeddings (U matrix)
        item_factors: Item embeddings (V matrix)
        user_to_idx: Mapping from user_id to internal index
        idx_to_user: Mapping from internal index to user_id
        item_to_idx: Mapping from product_id to internal index
        idx_to_item: Mapping from internal index to product_id
        user_pos_train: Dict of seen items per user (for filtering)
    """
    
    def __init__(self, 
                 user_factors: np.ndarray,
                 item_factors: np.ndarray,
                 user_to_idx: Dict[str, int],
                 idx_to_user: Dict[int, str],
                 item_to_idx: Dict[str, int],
                 idx_to_item: Dict[int, str],
                 user_pos_train: Optional[Dict[int, Set[int]]] = None):
        """
        Initialize ALS recommender.
        
        Args:
            user_factors: User embeddings (num_users, factors)
            item_factors: Item embeddings (num_items, factors)
            user_to_idx: Mapping from user_id (str) to index (int)
            idx_to_user: Mapping from index (int) to user_id (str)
            item_to_idx: Mapping from product_id (str) to index (int)
            idx_to_item: Mapping from index (int) to product_id (str)
            user_pos_train: Dict of seen item indices per user index
        
        Example:
            >>> from recsys.cf.model.als import EmbeddingExtractor, ALSRecommender
            >>> extractor = EmbeddingExtractor(model, normalize=True)
            >>> U, V = extractor.get_embeddings()
            >>> recommender = ALSRecommender(
            ...     user_factors=U,
            ...     item_factors=V,
            ...     user_to_idx=mappings['user_to_idx'],
            ...     idx_to_user=mappings['idx_to_user'],
            ...     item_to_idx=mappings['item_to_idx'],
            ...     idx_to_item=mappings['idx_to_item'],
            ...     user_pos_train=user_pos_train
            ... )
        """
        self.user_factors = user_factors
        self.item_factors = item_factors
        self.user_to_idx = user_to_idx
        self.idx_to_user = idx_to_user
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        self.user_pos_train = user_pos_train or {}
        
        self.num_users = user_factors.shape[0]
        self.num_items = item_factors.shape[0]
        self.factors = user_factors.shape[1]
        
        logger.info(
            f"ALSRecommender initialized: "
            f"users={self.num_users}, items={self.num_items}, "
            f"factors={self.factors}, "
            f"seen_items_loaded={len(self.user_pos_train)}"
        )
    
    def _get_user_idx(self, user_id: Union[str, int]) -> int:
        """
        Get internal user index from user ID.
        
        Args:
            user_id: User ID (string) or index (int)
        
        Returns:
            Internal user index
        
        Raises:
            ValueError: If user_id not found
        """
        if isinstance(user_id, int):
            if user_id < 0 or user_id >= self.num_users:
                raise ValueError(f"User index {user_id} out of range [0, {self.num_users})")
            return user_id
        
        if user_id not in self.user_to_idx:
            raise ValueError(f"User ID '{user_id}' not found in mappings")
        
        return self.user_to_idx[user_id]
    
    def _get_item_idx(self, item_id: Union[str, int]) -> int:
        """
        Get internal item index from product ID.
        
        Args:
            item_id: Product ID (string) or index (int)
        
        Returns:
            Internal item index
        
        Raises:
            ValueError: If item_id not found
        """
        if isinstance(item_id, int):
            if item_id < 0 or item_id >= self.num_items:
                raise ValueError(f"Item index {item_id} out of range [0, {self.num_items})")
            return item_id
        
        if item_id not in self.item_to_idx:
            raise ValueError(f"Item ID '{item_id}' not found in mappings")
        
        return self.item_to_idx[item_id]
    
    def compute_scores(self, user_idx: Union[int, np.ndarray]) -> np.ndarray:
        """
        Compute scores for all items for given user(s).
        
        Args:
            user_idx: Single user index or array of indices
        
        Returns:
            Score matrix:
                - If user_idx is int: (num_items,)
                - If user_idx is array: (len(user_idx), num_items)
        
        Example:
            >>> scores = recommender.compute_scores(user_idx=42)
            >>> print(f"Max score: {scores.max():.4f}")
        """
        if isinstance(user_idx, int):
            # Single user
            return self.user_factors[user_idx] @ self.item_factors.T
        else:
            # Multiple users (batch)
            return self.user_factors[user_idx] @ self.item_factors.T
    
    def filter_seen_items(self, 
                         scores: np.ndarray, 
                         user_idx: int,
                         strategy: str = 'mask') -> np.ndarray:
        """
        Filter seen items from recommendation candidates.
        
        Args:
            scores: Score array (num_items,)
            user_idx: User index
            strategy: Filtering strategy
                - 'mask': Set seen item scores to -inf
                - 'remove': Remove seen items (not implemented, use mask)
        
        Returns:
            Filtered scores (same shape as input)
        
        Example:
            >>> scores = recommender.compute_scores(42)
            >>> filtered = recommender.filter_seen_items(scores, user_idx=42)
        """
        if user_idx not in self.user_pos_train:
            # No seen items to filter
            return scores
        
        seen_items = self.user_pos_train[user_idx]
        
        if strategy == 'mask':
            # Create copy to avoid modifying original
            filtered_scores = scores.copy()
            # Set seen item scores to -inf
            filtered_scores[list(seen_items)] = -np.inf
            return filtered_scores
        else:
            raise ValueError(f"Unknown filtering strategy: {strategy}")
    
    def select_top_k(self, 
                    scores: np.ndarray, 
                    k: int,
                    method: str = 'partition') -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top-K items by score.
        
        Args:
            scores: Score array (num_items,)
            k: Number of items to select
            method: Selection method
                - 'partition': Use np.argpartition (faster for large K)
                - 'sort': Use np.argsort (simpler, slower)
        
        Returns:
            Tuple of (top_k_indices, top_k_scores) sorted by score descending
        
        Example:
            >>> scores = recommender.compute_scores(42)
            >>> indices, scores_k = recommender.select_top_k(scores, k=10)
        """
        if k > len(scores):
            k = len(scores)
            logger.warning(f"k={k} exceeds num_items, using k={len(scores)}")
        
        if method == 'partition':
            # Use argpartition for efficiency (O(n) instead of O(n log n))
            # Get indices of k largest values (unsorted)
            top_k_unsorted = np.argpartition(scores, -k)[-k:]
            
            # Sort the top-k by score descending
            top_k_sorted_indices = top_k_unsorted[np.argsort(scores[top_k_unsorted])[::-1]]
            top_k_scores = scores[top_k_sorted_indices]
            
        elif method == 'sort':
            # Use argsort (simpler but slower)
            top_k_sorted_indices = np.argsort(scores)[::-1][:k]
            top_k_scores = scores[top_k_sorted_indices]
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        return top_k_sorted_indices, top_k_scores
    
    def recommend(self, 
                 user_id: Union[str, int],
                 k: int = 10,
                 filter_seen: bool = True,
                 return_scores: bool = True) -> RecommendationResult:
        """
        Generate top-K recommendations for a single user.
        
        Args:
            user_id: User ID (string) or index (int)
            k: Number of recommendations
            filter_seen: Whether to filter out seen items
            return_scores: Whether to include scores in result
        
        Returns:
            RecommendationResult with top-K items
        
        Example:
            >>> result = recommender.recommend(user_id='12345', k=10)
            >>> print(f"Top item: {result.item_ids[0]} (score: {result.scores[0]:.4f})")
        """
        # Get user index
        user_idx = self._get_user_idx(user_id)
        user_id_str = user_id if isinstance(user_id, str) else self.idx_to_user[user_idx]
        
        # Compute scores
        scores = self.compute_scores(user_idx)
        
        # Filter seen items
        if filter_seen:
            scores = self.filter_seen_items(scores, user_idx)
        
        # Select top-K
        top_k_indices, top_k_scores = self.select_top_k(scores, k)
        
        # Map indices to product IDs
        top_k_item_ids = [self.idx_to_item[idx] for idx in top_k_indices]
        
        # Create result
        result = RecommendationResult(
            user_id=user_id_str,
            user_idx=user_idx,
            item_indices=top_k_indices,
            item_ids=top_k_item_ids,
            scores=top_k_scores if return_scores else np.array([]),
            rank=np.arange(1, len(top_k_indices) + 1)
        )
        
        return result
    
    def recommend_batch(self,
                       user_ids: List[Union[str, int]],
                       k: int = 10,
                       filter_seen: bool = True,
                       batch_size: Optional[int] = None) -> List[RecommendationResult]:
        """
        Generate recommendations for multiple users efficiently.
        
        Args:
            user_ids: List of user IDs or indices
            k: Number of recommendations per user
            filter_seen: Whether to filter out seen items
            batch_size: Process users in batches (None = all at once)
        
        Returns:
            List of RecommendationResult objects
        
        Example:
            >>> results = recommender.recommend_batch(
            ...     user_ids=['12345', '67890'],
            ...     k=10
            ... )
            >>> for result in results:
            ...     print(f"User {result.user_id}: {len(result.item_ids)} items")
        """
        logger.info(f"Generating batch recommendations for {len(user_ids)} users...")
        
        results = []
        
        if batch_size is None:
            batch_size = len(user_ids)
        
        # Process in batches
        for i in range(0, len(user_ids), batch_size):
            batch_user_ids = user_ids[i:i+batch_size]
            
            # Get user indices
            batch_user_indices = [self._get_user_idx(uid) for uid in batch_user_ids]
            
            # Compute scores for batch (users × items)
            batch_scores = self.compute_scores(np.array(batch_user_indices))
            
            # Process each user in batch
            for j, (user_id, user_idx) in enumerate(zip(batch_user_ids, batch_user_indices)):
                scores = batch_scores[j] if len(batch_user_ids) > 1 else batch_scores
                
                # Filter seen items
                if filter_seen:
                    scores = self.filter_seen_items(scores, user_idx)
                
                # Select top-K
                top_k_indices, top_k_scores = self.select_top_k(scores, k)
                
                # Map to product IDs
                user_id_str = user_id if isinstance(user_id, str) else self.idx_to_user[user_idx]
                top_k_item_ids = [self.idx_to_item[idx] for idx in top_k_indices]
                
                # Create result
                result = RecommendationResult(
                    user_id=user_id_str,
                    user_idx=user_idx,
                    item_indices=top_k_indices,
                    item_ids=top_k_item_ids,
                    scores=top_k_scores,
                    rank=np.arange(1, len(top_k_indices) + 1)
                )
                results.append(result)
        
        logger.info(f"Batch recommendations complete: {len(results)} users processed")
        
        return results
    
    def get_similar_items(self,
                         item_id: Union[str, int],
                         k: int = 10,
                         exclude_self: bool = True) -> RecommendationResult:
        """
        Find similar items using item embeddings.
        
        Args:
            item_id: Item ID (string) or index (int)
            k: Number of similar items to return
            exclude_self: Whether to exclude the query item itself
        
        Returns:
            RecommendationResult with similar items (user_id/user_idx will be None)
        
        Example:
            >>> similar = recommender.get_similar_items(item_id='100', k=5)
            >>> print(f"Items similar to {item_id}: {similar.item_ids}")
        """
        # Get item index
        item_idx = self._get_item_idx(item_id)
        item_id_str = item_id if isinstance(item_id, str) else self.idx_to_item[item_idx]
        
        # Compute similarity with all items
        similarities = self.item_factors[item_idx] @ self.item_factors.T
        
        # Exclude self if requested
        if exclude_self:
            similarities[item_idx] = -np.inf
        
        # Select top-K
        k_adjusted = k + 1 if not exclude_self else k
        top_k_indices, top_k_scores = self.select_top_k(similarities, k_adjusted)
        
        # Remove self if it's in results and we wanted to exclude it
        if not exclude_self and item_idx in top_k_indices:
            mask = top_k_indices != item_idx
            top_k_indices = top_k_indices[mask]
            top_k_scores = top_k_scores[mask]
        
        # Map to product IDs
        top_k_item_ids = [self.idx_to_item[idx] for idx in top_k_indices]
        
        # Create result (user fields are None for item-item similarity)
        result = RecommendationResult(
            user_id=None,
            user_idx=None,
            item_indices=top_k_indices,
            item_ids=top_k_item_ids,
            scores=top_k_scores,
            rank=np.arange(1, len(top_k_indices) + 1)
        )
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get recommender statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'factors': self.factors,
            'users_with_seen_items': len(self.user_pos_train),
            'avg_seen_items_per_user': np.mean([len(items) for items in self.user_pos_train.values()]) if self.user_pos_train else 0
        }
        
        return stats
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ALSRecommender(users={self.num_users}, "
            f"items={self.num_items}, "
            f"factors={self.factors})"
        )


def quick_recommend(user_factors: np.ndarray,
                   item_factors: np.ndarray,
                   user_ids: List[Union[str, int]],
                   k: int = 10,
                   mappings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Quick convenience function for generating recommendations.
    
    Args:
        user_factors: User embeddings (U)
        item_factors: Item embeddings (V)
        user_ids: List of user IDs or indices
        k: Number of recommendations
        mappings: Optional ID mappings dict (if None, uses integer indices)
    
    Returns:
        List of recommendation dictionaries
    
    Example:
        >>> from recsys.cf.model.als import extract_embeddings, quick_recommend
        >>> U, V = extract_embeddings(model, normalize=True)
        >>> recs = quick_recommend(U, V, user_ids=[0, 1, 2], k=10)
    """
    # Create dummy mappings if not provided
    if mappings is None:
        num_users, num_items = user_factors.shape[0], item_factors.shape[0]
        mappings = {
            'user_to_idx': {str(i): i for i in range(num_users)},
            'idx_to_user': {i: str(i) for i in range(num_users)},
            'item_to_idx': {str(i): i for i in range(num_items)},
            'idx_to_item': {i: str(i) for i in range(num_items)}
        }
    
    # Create recommender
    recommender = ALSRecommender(
        user_factors=user_factors,
        item_factors=item_factors,
        user_to_idx=mappings['user_to_idx'],
        idx_to_user=mappings['idx_to_user'],
        item_to_idx=mappings['item_to_idx'],
        idx_to_item=mappings['idx_to_item']
    )
    
    # Generate recommendations
    results = recommender.recommend_batch(user_ids, k=k)
    
    return [result.to_dict() for result in results]


# Main execution example
if __name__ == "__main__":
    import traceback
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("ALS Recommender Demo")
    print("=" * 60)
    
    # Create synthetic data
    print("\nCreating synthetic embeddings and mappings...")
    np.random.seed(42)
    
    num_users, num_items, factors = 1000, 500, 64
    
    # Random embeddings (normalized)
    U = np.random.randn(num_users, factors).astype(np.float32)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    
    V = np.random.randn(num_items, factors).astype(np.float32)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    
    # Create ID mappings
    user_to_idx = {f"U{i:04d}": i for i in range(num_users)}
    idx_to_user = {i: f"U{i:04d}" for i in range(num_users)}
    item_to_idx = {f"I{i:03d}": i for i in range(num_items)}
    idx_to_item = {i: f"I{i:03d}" for i in range(num_items)}
    
    # Create seen items for filtering
    user_pos_train = {}
    for u in range(num_users):
        num_seen = np.random.randint(5, 20)
        seen_items = np.random.choice(num_items, num_seen, replace=False)
        user_pos_train[u] = set(seen_items)
    
    print(f"Data created: U={U.shape}, V={V.shape}")
    
    # Example 1: Single user recommendation
    print("\n" + "=" * 60)
    print("Example 1: Single User Recommendation")
    print("=" * 60)
    
    try:
        recommender = ALSRecommender(
            user_factors=U,
            item_factors=V,
            user_to_idx=user_to_idx,
            idx_to_user=idx_to_user,
            item_to_idx=item_to_idx,
            idx_to_item=idx_to_item,
            user_pos_train=user_pos_train
        )
        
        user_id = "U0042"
        result = recommender.recommend(user_id, k=10)
        
        print(f"\nRecommendations for {user_id}:")
        for rank, (item_id, score) in enumerate(zip(result.item_ids, result.scores), 1):
            print(f"  {rank}. {item_id}: {score:.4f}")
        
    except Exception as e:
        print(f"Example 1 failed: {e}")
        traceback.print_exc()
    
    # Example 2: Batch recommendations
    print("\n" + "=" * 60)
    print("Example 2: Batch Recommendations")
    print("=" * 60)
    
    try:
        test_users = [f"U{i:04d}" for i in [10, 20, 30]]
        results = recommender.recommend_batch(test_users, k=5)
        
        print(f"\nBatch recommendations for {len(results)} users:")
        for result in results:
            print(f"\n{result.user_id}:")
            for item_id, score in zip(result.item_ids[:3], result.scores[:3]):
                print(f"  - {item_id}: {score:.4f}")
        
    except Exception as e:
        print(f"Example 2 failed: {e}")
        traceback.print_exc()
    
    # Example 3: Similar items
    print("\n" + "=" * 60)
    print("Example 3: Similar Items")
    print("=" * 60)
    
    try:
        item_id = "I100"
        similar = recommender.get_similar_items(item_id, k=5)
        
        print(f"\nItems similar to {item_id}:")
        for rank, (sim_item, score) in enumerate(zip(similar.item_ids, similar.scores), 1):
            print(f"  {rank}. {sim_item}: similarity={score:.4f}")
        
    except Exception as e:
        print(f"Example 3 failed: {e}")
        traceback.print_exc()
    
    # Example 4: Statistics
    print("\n" + "=" * 60)
    print("Example 4: Recommender Statistics")
    print("=" * 60)
    
    try:
        stats = recommender.get_statistics()
        print("\nRecommender statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Example 4 failed: {e}")
        traceback.print_exc()
    
    # Example 5: Quick recommend convenience function
    print("\n" + "=" * 60)
    print("Example 5: Quick Recommend Function")
    print("=" * 60)
    
    try:
        mappings = {
            'user_to_idx': user_to_idx,
            'idx_to_user': idx_to_user,
            'item_to_idx': item_to_idx,
            'idx_to_item': idx_to_item
        }
        
        quick_results = quick_recommend(
            U, V, 
            user_ids=["U0010", "U0020"],
            k=5,
            mappings=mappings
        )
        
        print(f"\nQuick recommendations:")
        for rec in quick_results:
            print(f"  {rec['user_id']}: {rec['item_ids'][:3]}")
        
    except Exception as e:
        print(f"Example 5 failed: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
