"""
Hybrid Metrics Module for CF + BERT Evaluation.

This module provides specialized metrics for evaluating hybrid recommender systems:
- Diversity (Intra-List Diversity)
- Semantic Alignment Score
- Cold-Start Coverage
- Novelty

These metrics complement standard ranking metrics by measuring content quality
and diversity of recommendations.

Example:
    >>> from recsys.cf.evaluation import DiversityMetric, SemanticAlignmentMetric
    >>> diversity = DiversityMetric()
    >>> score = diversity.compute(recommendations, bert_embeddings, item_to_idx)
"""

from typing import List, Set, Dict, Optional, Union, Any
from abc import ABC, abstractmethod
import numpy as np
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Abstract Base for Hybrid Metrics
# ============================================================================

class HybridMetric(ABC):
    """
    Abstract base class for hybrid (CF + Content) metrics.
    """
    
    def __init__(self, name: str):
        """
        Initialize hybrid metric.
        
        Args:
            name: Human-readable metric name
        """
        self.name = name
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        """Compute metric value."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ============================================================================
# Diversity Metrics
# ============================================================================

class DiversityMetric(HybridMetric):
    """
    Intra-List Diversity metric using content embeddings.
    
    Measures how different the items in a recommendation list are from each other.
    
    Formula:
        Diversity = 1 - (1/K(K-1)) * ΣΣ similarity(i, j) for i ≠ j
    
    Interpretation:
        - Diversity = 0.3: Items are relatively similar (avg similarity = 0.7)
        - Diversity = 0.6: Items are quite diverse (avg similarity = 0.4)
        - Higher is better for diverse recommendations
    """
    
    def __init__(self, similarity_type: str = 'cosine'):
        """
        Initialize Diversity metric.
        
        Args:
            similarity_type: Type of similarity ('cosine', 'euclidean')
        """
        super().__init__(name="Diversity")
        self.similarity_type = similarity_type
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
        
        Returns:
            Cosine similarity in [-1, 1]
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def compute(
        self,
        recommendations: List[int],
        embeddings: np.ndarray,
        item_to_idx: Optional[Dict[int, int]] = None,
        **kwargs
    ) -> float:
        """
        Compute diversity using BERT/content embeddings.
        
        Args:
            recommendations: List of recommended item indices or IDs
            embeddings: np.array of shape (num_items, embedding_dim)
            item_to_idx: Optional mapping from item ID to embedding index
        
        Returns:
            Diversity score in [0, 1], higher = more diverse
        """
        if len(recommendations) < 2:
            return 0.0
        
        # Get embeddings for recommended items
        embs = []
        for item in recommendations:
            if item_to_idx is not None:
                if item in item_to_idx:
                    idx = item_to_idx[item]
                    if idx < len(embeddings):
                        embs.append(embeddings[idx])
            else:
                # Assume item is already an index
                if item < len(embeddings):
                    embs.append(embeddings[item])
        
        if len(embs) < 2:
            return 0.0
        
        embs = np.array(embs)
        
        # Normalize embeddings
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # Avoid division by zero
        embs_norm = embs / norms
        
        # Compute pairwise cosine similarities
        similarities = []
        n = len(embs_norm)
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embs_norm[i], embs_norm[j])
                similarities.append(sim)
        
        if len(similarities) == 0:
            return 0.0
        
        # Diversity = 1 - average similarity
        avg_similarity = np.mean(similarities)
        diversity = 1 - avg_similarity
        
        # Clamp to [0, 1] (similarities can be negative for cosine)
        diversity = max(0.0, min(1.0, diversity))
        
        return diversity
    
    def compute_batch(
        self,
        all_recommendations: Dict[int, List[int]],
        embeddings: np.ndarray,
        item_to_idx: Optional[Dict[int, int]] = None
    ) -> Dict[str, float]:
        """
        Compute diversity for all users.
        
        Args:
            all_recommendations: Dict mapping user_idx to recommended items
            embeddings: Item embeddings
            item_to_idx: Optional mapping from item ID to embedding index
        
        Returns:
            Dict with 'mean', 'std', 'min', 'max' diversity values
        """
        diversities = []
        
        for recs in all_recommendations.values():
            div = self.compute(recs, embeddings, item_to_idx)
            diversities.append(div)
        
        if len(diversities) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': float(np.mean(diversities)),
            'std': float(np.std(diversities)),
            'min': float(np.min(diversities)),
            'max': float(np.max(diversities))
        }


class NoveltyMetric(HybridMetric):
    """
    Novelty metric: how unpopular/surprising the recommendations are.
    
    Formula:
        Novelty@K = (1/K) * Σ log2(num_users / item_popularity_i)
    
    Interpretation:
        - High novelty: Recommending unpopular (long-tail) items
        - Low novelty: Recommending popular items
        - Trade-off with accuracy
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize Novelty metric.
        
        Args:
            k: Number of recommendations to consider
        """
        super().__init__(name=f"Novelty@{k}")
        self.k = k
    
    def compute(
        self,
        recommendations: List[int],
        item_popularity: np.ndarray,
        num_users: Optional[int] = None,
        k: Optional[int] = None
    ) -> float:
        """
        Compute novelty score.
        
        Args:
            recommendations: List of recommended item indices
            item_popularity: Array of item interaction counts
            num_users: Total number of users (for normalization)
            k: Optional override for K value
        
        Returns:
            Novelty score (higher = more novel)
        """
        k = k if k is not None else self.k
        k = min(k, len(recommendations))
        
        if k == 0:
            return 0.0
        
        if num_users is None:
            num_users = max(item_popularity) + 1  # Estimate
        
        novelty_sum = 0.0
        valid_items = 0
        
        for item in recommendations[:k]:
            if item < len(item_popularity):
                pop = item_popularity[item]
                if pop > 0:
                    # Self-information: log2(N / popularity)
                    novelty_sum += np.log2(num_users / pop)
                    valid_items += 1
        
        if valid_items == 0:
            return 0.0
        
        return novelty_sum / valid_items


# ============================================================================
# Semantic Alignment Metrics
# ============================================================================

class SemanticAlignmentMetric(HybridMetric):
    """
    Semantic Alignment Score: how well CF recommendations match user content preferences.
    
    Formula:
        Alignment = (1/K) * Σ cosine_similarity(user_profile_emb, item_emb_i)
    
    Interpretation:
        - High alignment: CF recommendations match user's content preferences
        - Useful for validating BERT-initialized embeddings
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize Semantic Alignment metric.
        
        Args:
            k: Number of recommendations to consider
        """
        super().__init__(name=f"SemanticAlignment@{k}")
        self.k = k
    
    def compute(
        self,
        user_profile_emb: np.ndarray,
        recommendations: List[int],
        item_embeddings: np.ndarray,
        item_to_idx: Optional[Dict[int, int]] = None,
        k: Optional[int] = None
    ) -> float:
        """
        Compute semantic alignment.
        
        Args:
            user_profile_emb: User's content profile embedding
            recommendations: List of recommended item indices or IDs
            item_embeddings: Item content embeddings
            item_to_idx: Optional mapping from item ID to embedding index
            k: Optional override for K value
        
        Returns:
            Alignment score in [0, 1]
        """
        k = k if k is not None else self.k
        k = min(k, len(recommendations))
        
        if k == 0 or len(user_profile_emb) == 0:
            return 0.0
        
        # Normalize user profile
        user_norm = np.linalg.norm(user_profile_emb)
        if user_norm == 0:
            return 0.0
        user_profile_normalized = user_profile_emb / user_norm
        
        # Compute similarities
        similarities = []
        for item in recommendations[:k]:
            if item_to_idx is not None:
                if item not in item_to_idx:
                    continue
                idx = item_to_idx[item]
            else:
                idx = item
            
            if idx >= len(item_embeddings):
                continue
            
            item_emb = item_embeddings[idx]
            item_norm = np.linalg.norm(item_emb)
            
            if item_norm > 0:
                item_normalized = item_emb / item_norm
                sim = np.dot(user_profile_normalized, item_normalized)
                similarities.append(sim)
        
        if len(similarities) == 0:
            return 0.0
        
        # Average similarity (clamp to [0, 1])
        avg_sim = np.mean(similarities)
        return max(0.0, min(1.0, avg_sim))
    
    @staticmethod
    def compute_user_profile(
        user_history_items: List[int],
        item_embeddings: np.ndarray,
        item_to_idx: Optional[Dict[int, int]] = None,
        aggregation: str = 'mean'
    ) -> np.ndarray:
        """
        Compute user content profile from item history.
        
        Args:
            user_history_items: List of item indices/IDs in user's history
            item_embeddings: Item content embeddings
            item_to_idx: Optional mapping from item ID to embedding index
            aggregation: How to aggregate ('mean', 'max')
        
        Returns:
            User profile embedding
        """
        embs = []
        
        for item in user_history_items:
            if item_to_idx is not None:
                if item not in item_to_idx:
                    continue
                idx = item_to_idx[item]
            else:
                idx = item
            
            if idx < len(item_embeddings):
                embs.append(item_embeddings[idx])
        
        if len(embs) == 0:
            return np.zeros(item_embeddings.shape[1])
        
        embs = np.array(embs)
        
        if aggregation == 'mean':
            return np.mean(embs, axis=0)
        elif aggregation == 'max':
            return np.max(embs, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


# ============================================================================
# Cold-Start Coverage Metrics
# ============================================================================

class ColdStartCoverageMetric(HybridMetric):
    """
    Cold-Start Coverage: percentage of cold-start items recommended.
    
    Formula:
        ColdStartCoverage = |Unique Cold Items in All Recs| / |Total Cold Items|
    
    Interpretation:
        - High coverage: System can expose cold items
        - Important for catalog freshness
    """
    
    def __init__(self, cold_threshold: int = 5):
        """
        Initialize Cold-Start Coverage metric.
        
        Args:
            cold_threshold: Items with <N interactions are considered cold
        """
        super().__init__(name="ColdStartCoverage")
        self.cold_threshold = cold_threshold
    
    def compute(
        self,
        all_recommendations: Dict[int, List[int]],
        item_counts: np.ndarray,
        cold_threshold: Optional[int] = None
    ) -> float:
        """
        Compute cold-start item coverage.
        
        Args:
            all_recommendations: Dict mapping user_idx to recommended item indices
            item_counts: Array of interaction counts per item
            cold_threshold: Optional override for threshold
        
        Returns:
            Coverage of cold items in [0, 1]
        """
        threshold = cold_threshold if cold_threshold is not None else self.cold_threshold
        
        # Identify cold items
        cold_items = set(np.where(item_counts < threshold)[0])
        
        if len(cold_items) == 0:
            return 0.0  # No cold items to cover
        
        # Collect recommended cold items
        recommended_cold = set()
        for recs in all_recommendations.values():
            for item in recs:
                if item in cold_items:
                    recommended_cold.add(item)
        
        coverage = len(recommended_cold) / len(cold_items)
        return coverage
    
    def get_cold_item_stats(
        self,
        item_counts: np.ndarray,
        cold_threshold: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about cold items.
        
        Args:
            item_counts: Array of interaction counts per item
            cold_threshold: Optional override for threshold
        
        Returns:
            Dict with cold item statistics
        """
        threshold = cold_threshold if cold_threshold is not None else self.cold_threshold
        
        cold_items = np.where(item_counts < threshold)[0]
        warm_items = np.where(item_counts >= threshold)[0]
        
        return {
            'cold_threshold': threshold,
            'num_cold_items': len(cold_items),
            'num_warm_items': len(warm_items),
            'cold_percentage': len(cold_items) / len(item_counts) * 100 if len(item_counts) > 0 else 0,
            'cold_items': cold_items.tolist()
        }


# ============================================================================
# Serendipity Metrics
# ============================================================================

class SerendipityMetric(HybridMetric):
    """
    Serendipity: how surprising yet relevant the recommendations are.
    
    Combines unexpectedness (different from popularity baseline) with relevance.
    
    Formula:
        Serendipity = (1/K) * Σ [relevant_i * (1 - expected_i)]
    
    Interpretation:
        - High serendipity: Novel and relevant recommendations
        - Measures "pleasant surprises"
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize Serendipity metric.
        
        Args:
            k: Number of recommendations to consider
        """
        super().__init__(name=f"Serendipity@{k}")
        self.k = k
    
    def compute(
        self,
        recommendations: List[int],
        ground_truth: Set[int],
        baseline_recommendations: List[int],
        k: Optional[int] = None
    ) -> float:
        """
        Compute serendipity score.
        
        Args:
            recommendations: Model's recommended items
            ground_truth: Set of relevant items
            baseline_recommendations: Baseline (e.g., popularity) recommendations
            k: Optional override for K value
        
        Returns:
            Serendipity score in [0, 1]
        """
        k = k if k is not None else self.k
        k = min(k, len(recommendations))
        
        if k == 0:
            return 0.0
        
        baseline_set = set(baseline_recommendations[:k])
        
        serendipity_sum = 0.0
        for item in recommendations[:k]:
            is_relevant = 1.0 if item in ground_truth else 0.0
            is_unexpected = 1.0 if item not in baseline_set else 0.0
            serendipity_sum += is_relevant * is_unexpected
        
        return serendipity_sum / k


# ============================================================================
# Hybrid Metric Collection
# ============================================================================

class HybridMetricCollection:
    """
    Collection of hybrid metrics for comprehensive evaluation.
    
    Example:
        >>> collection = HybridMetricCollection(k_values=[10, 20])
        >>> results = collection.evaluate_all(
        ...     all_recommendations=recs,
        ...     item_embeddings=embeddings,
        ...     ...
        ... )
    """
    
    def __init__(
        self,
        k_values: List[int] = [10, 20],
        cold_threshold: int = 5,
        include_serendipity: bool = False
    ):
        """
        Initialize metric collection.
        
        Args:
            k_values: List of K values for @K metrics
            cold_threshold: Threshold for cold-start items
            include_serendipity: Whether to include serendipity (requires baseline)
        """
        self.k_values = k_values
        self.cold_threshold = cold_threshold
        
        # Initialize metrics
        self.diversity = DiversityMetric()
        self.novelty_metrics = {k: NoveltyMetric(k=k) for k in k_values}
        self.alignment_metrics = {k: SemanticAlignmentMetric(k=k) for k in k_values}
        self.cold_start_coverage = ColdStartCoverageMetric(cold_threshold=cold_threshold)
        
        if include_serendipity:
            self.serendipity_metrics = {k: SerendipityMetric(k=k) for k in k_values}
        else:
            self.serendipity_metrics = {}
    
    def evaluate_diversity(
        self,
        all_recommendations: Dict[int, List[int]],
        item_embeddings: np.ndarray,
        item_to_idx: Optional[Dict[int, int]] = None
    ) -> Dict[str, float]:
        """
        Evaluate diversity for all users.
        
        Returns:
            Dict with diversity statistics
        """
        return self.diversity.compute_batch(
            all_recommendations, item_embeddings, item_to_idx
        )
    
    def evaluate_novelty(
        self,
        all_recommendations: Dict[int, List[int]],
        item_popularity: np.ndarray,
        num_users: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate novelty for all users.
        
        Returns:
            Dict with novelty@K for each K
        """
        results = {}
        
        for k, metric in self.novelty_metrics.items():
            novelties = []
            for recs in all_recommendations.values():
                nov = metric.compute(recs, item_popularity, num_users, k)
                novelties.append(nov)
            
            if novelties:
                results[f'novelty@{k}'] = float(np.mean(novelties))
                results[f'novelty@{k}_std'] = float(np.std(novelties))
        
        return results
    
    def evaluate_alignment(
        self,
        all_recommendations: Dict[int, List[int]],
        user_profiles: Dict[int, np.ndarray],
        item_embeddings: np.ndarray,
        item_to_idx: Optional[Dict[int, int]] = None
    ) -> Dict[str, float]:
        """
        Evaluate semantic alignment for all users.
        
        Args:
            all_recommendations: User recommendations
            user_profiles: Dict mapping user_idx to profile embedding
            item_embeddings: Item content embeddings
            item_to_idx: Optional mapping
        
        Returns:
            Dict with alignment@K for each K
        """
        results = {}
        
        for k, metric in self.alignment_metrics.items():
            alignments = []
            for u_idx, recs in all_recommendations.items():
                if u_idx in user_profiles:
                    align = metric.compute(
                        user_profiles[u_idx], recs, item_embeddings, item_to_idx, k
                    )
                    alignments.append(align)
            
            if alignments:
                results[f'alignment@{k}'] = float(np.mean(alignments))
                results[f'alignment@{k}_std'] = float(np.std(alignments))
        
        return results
    
    def evaluate_all(
        self,
        all_recommendations: Dict[int, List[int]],
        item_embeddings: np.ndarray,
        item_popularity: np.ndarray,
        item_counts: np.ndarray,
        user_profiles: Optional[Dict[int, np.ndarray]] = None,
        item_to_idx: Optional[Dict[int, int]] = None,
        num_users: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate all hybrid metrics.
        
        Args:
            all_recommendations: Dict mapping user_idx to recommended items
            item_embeddings: Item content embeddings
            item_popularity: Item popularity scores
            item_counts: Item interaction counts
            user_profiles: Optional user profile embeddings
            item_to_idx: Optional item ID to index mapping
            num_users: Total number of users
        
        Returns:
            Dict with all metric results
        """
        results = {}
        
        # Diversity
        diversity_results = self.evaluate_diversity(
            all_recommendations, item_embeddings, item_to_idx
        )
        results['diversity'] = diversity_results['mean']
        results['diversity_std'] = diversity_results['std']
        
        # Novelty
        novelty_results = self.evaluate_novelty(
            all_recommendations, item_popularity, num_users
        )
        results.update(novelty_results)
        
        # Alignment (if user profiles provided)
        if user_profiles is not None:
            alignment_results = self.evaluate_alignment(
                all_recommendations, user_profiles, item_embeddings, item_to_idx
            )
            results.update(alignment_results)
        
        # Cold-start coverage
        results['cold_start_coverage'] = self.cold_start_coverage.compute(
            all_recommendations, item_counts
        )
        
        return results


# ============================================================================
# Convenience Functions
# ============================================================================

def compute_diversity_bert(
    recommendations: List[int],
    bert_embeddings: np.ndarray,
    item_to_idx: Optional[Dict[int, int]] = None
) -> float:
    """
    Compute diversity using BERT embeddings.
    
    Args:
        recommendations: List of recommended item IDs
        bert_embeddings: np.array (num_items, 768)
        item_to_idx: Dict mapping product_id -> embedding idx
    
    Returns:
        Diversity score [0, 1], higher = more diverse
    """
    metric = DiversityMetric()
    return metric.compute(recommendations, bert_embeddings, item_to_idx)


def compute_semantic_alignment(
    user_profile_emb: np.ndarray,
    recommendations: List[int],
    item_embeddings: np.ndarray,
    item_to_idx: Optional[Dict[int, int]] = None
) -> float:
    """
    Compute semantic alignment of recommendations with user profile.
    
    Args:
        user_profile_emb: User's content profile embedding
        recommendations: List of recommended items
        item_embeddings: Item content embeddings
        item_to_idx: Optional mapping from item ID to embedding index
    
    Returns:
        Alignment score [0, 1]
    """
    metric = SemanticAlignmentMetric()
    return metric.compute(user_profile_emb, recommendations, item_embeddings, item_to_idx)


def compute_cold_start_coverage(
    all_recommendations: Dict[int, List[int]],
    item_counts: np.ndarray,
    cold_threshold: int = 5
) -> float:
    """
    Compute coverage of cold-start items.
    
    Args:
        all_recommendations: Dict {user_id: [recommended_items]}
        item_counts: Array with item interaction counts
        cold_threshold: Items with <N interactions = cold
    
    Returns:
        Coverage [0, 1]
    """
    metric = ColdStartCoverageMetric(cold_threshold=cold_threshold)
    return metric.compute(all_recommendations, item_counts)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Hybrid Metrics Module")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create mock embeddings
    num_items = 100
    embedding_dim = 64
    item_embeddings = np.random.randn(num_items, embedding_dim)
    
    # Normalize
    norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    item_embeddings = item_embeddings / norms
    
    # Create mock recommendations
    all_recommendations = {
        0: [1, 5, 10, 15, 20, 25, 30, 35, 40, 45],
        1: [2, 6, 11, 16, 21, 26, 31, 36, 41, 46],
        2: [3, 7, 12, 17, 22, 27, 32, 37, 42, 47],
    }
    
    # Create mock popularity
    item_popularity = np.random.randint(1, 100, size=num_items)
    item_counts = item_popularity.copy()
    
    # Test Diversity
    print("\n--- Diversity ---")
    diversity = DiversityMetric()
    div_score = diversity.compute([1, 5, 10, 15, 20], item_embeddings)
    print(f"Diversity (5 items): {div_score:.4f}")
    
    div_batch = diversity.compute_batch(all_recommendations, item_embeddings)
    print(f"Batch diversity: mean={div_batch['mean']:.4f}, std={div_batch['std']:.4f}")
    
    # Test Novelty
    print("\n--- Novelty ---")
    novelty = NoveltyMetric(k=10)
    nov_score = novelty.compute([1, 5, 10, 15, 20], item_popularity, num_users=1000)
    print(f"Novelty@10: {nov_score:.4f}")
    
    # Test Semantic Alignment
    print("\n--- Semantic Alignment ---")
    user_profile = np.random.randn(embedding_dim)
    user_profile = user_profile / np.linalg.norm(user_profile)
    
    alignment = SemanticAlignmentMetric(k=10)
    align_score = alignment.compute(user_profile, [1, 5, 10, 15, 20], item_embeddings)
    print(f"Alignment@10: {align_score:.4f}")
    
    # Test Cold-Start Coverage
    print("\n--- Cold-Start Coverage ---")
    cold_coverage = ColdStartCoverageMetric(cold_threshold=10)
    cold_score = cold_coverage.compute(all_recommendations, item_counts)
    print(f"Cold-start coverage: {cold_score:.4f}")
    
    cold_stats = cold_coverage.get_cold_item_stats(item_counts)
    print(f"Cold items: {cold_stats['num_cold_items']} ({cold_stats['cold_percentage']:.1f}%)")
    
    # Test Full Collection
    print("\n--- Full Metric Collection ---")
    collection = HybridMetricCollection(k_values=[5, 10], cold_threshold=10)
    
    user_profiles = {
        0: np.random.randn(embedding_dim),
        1: np.random.randn(embedding_dim),
        2: np.random.randn(embedding_dim),
    }
    # Normalize profiles
    for k in user_profiles:
        user_profiles[k] = user_profiles[k] / np.linalg.norm(user_profiles[k])
    
    all_results = collection.evaluate_all(
        all_recommendations=all_recommendations,
        item_embeddings=item_embeddings,
        item_popularity=item_popularity,
        item_counts=item_counts,
        user_profiles=user_profiles,
        num_users=1000
    )
    
    print("Full evaluation results:")
    for key, value in all_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
