"""
BPR + Content Ensemble Model

This module combines BPR collaborative filtering with content-based features
for improved recommendations, especially for cold-start scenarios.

Architecture:
1. BPR Component: User-item embeddings learned from interactions
2. Content Component: PhoBERT text embeddings + product attributes
3. Fusion Layer: Learned weighting of both components

Key Features:
- Late fusion: Combine BPR scores with content similarity
- Attention fusion: Learn context-dependent weights
- MLP fusion: Neural network to combine features
- Automatic cold-start handling

Author: VieComRec Team
Date: 2025-11-26
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Union

import numpy as np

logger = logging.getLogger(__name__)


class FusionType(Enum):
    """Fusion strategies for combining BPR and content scores."""
    WEIGHTED = "weighted"       # Simple weighted average
    ATTENTION = "attention"     # Attention-based weighting
    MLP = "mlp"                 # MLP-based fusion
    ADAPTIVE = "adaptive"       # User-adaptive weighting


@dataclass
class EnsembleConfig:
    """
    Configuration for BPR + Content Ensemble.
    
    Attributes:
        fusion_type: How to combine BPR and content scores
        bpr_weight: Base weight for BPR scores (for weighted fusion)
        content_weight: Base weight for content scores
        popularity_weight: Weight for popularity component
        cold_start_threshold: Interactions threshold for cold-start
        mlp_hidden_dims: Hidden dimensions for MLP fusion
        attention_dim: Dimension for attention fusion
        temperature: Temperature for softmax in attention
        normalize_scores: Whether to normalize scores before fusion
    """
    fusion_type: FusionType = FusionType.ADAPTIVE
    bpr_weight: float = 0.4
    content_weight: float = 0.4
    popularity_weight: float = 0.2
    cold_start_threshold: int = 2
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    attention_dim: int = 32
    temperature: float = 1.0
    normalize_scores: bool = True


class ContentScorer:
    """
    Content-based scoring using BERT embeddings and product attributes.
    
    Computes content similarity between user profile (aggregated from history)
    and candidate items using:
    1. Text similarity: Cosine similarity of BERT embeddings
    2. Attribute matching: Overlap in skin_type, category, etc.
    3. Brand affinity: Preference for brands user has bought before
    """
    
    def __init__(
        self,
        item_embeddings: np.ndarray,
        item_attributes: Optional[Dict[int, Dict[str, Any]]] = None,
        item_brands: Optional[Dict[int, str]] = None,
        text_weight: float = 0.7,
        attribute_weight: float = 0.2,
        brand_weight: float = 0.1
    ):
        """
        Initialize content scorer.
        
        Args:
            item_embeddings: BERT embeddings (num_items, embed_dim)
            item_attributes: Dict of item_idx -> {attribute: value}
            item_brands: Dict of item_idx -> brand name
            text_weight: Weight for text similarity
            attribute_weight: Weight for attribute matching
            brand_weight: Weight for brand affinity
        """
        self.item_embeddings = item_embeddings
        self.item_attributes = item_attributes or {}
        self.item_brands = item_brands or {}
        
        self.text_weight = text_weight
        self.attribute_weight = attribute_weight
        self.brand_weight = brand_weight
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.normalized_embeddings = item_embeddings / norms
        
        self.num_items = len(item_embeddings)
        
        logger.info(f"ContentScorer initialized:")
        logger.info(f"  Items: {self.num_items:,}")
        logger.info(f"  Embedding dim: {item_embeddings.shape[1]}")
        logger.info(f"  Weights: text={text_weight}, attr={attribute_weight}, brand={brand_weight}")
    
    def compute_user_profile(
        self,
        user_items: Set[int],
        item_weights: Optional[Dict[int, float]] = None
    ) -> np.ndarray:
        """
        Compute user content profile by aggregating item embeddings.
        
        Args:
            user_items: Set of item indices user has interacted with
            item_weights: Optional weights per item (e.g., from rating/confidence)
        
        Returns:
            User profile embedding (embed_dim,)
        """
        if not user_items:
            # Return mean embedding for cold-start
            return self.normalized_embeddings.mean(axis=0)
        
        # Aggregate item embeddings (weighted if available)
        item_list = list(user_items)
        embeddings = self.normalized_embeddings[item_list]
        
        if item_weights:
            weights = np.array([item_weights.get(i, 1.0) for i in item_list])
            weights = weights / weights.sum()
            profile = np.average(embeddings, axis=0, weights=weights)
        else:
            profile = embeddings.mean(axis=0)
        
        # Normalize
        norm = np.linalg.norm(profile)
        if norm > 1e-8:
            profile = profile / norm
        
        return profile
    
    def compute_text_similarity(
        self,
        user_profile: np.ndarray,
        candidate_items: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute text similarity between user profile and items.
        
        Args:
            user_profile: User content profile (embed_dim,)
            candidate_items: Optional subset of items to score
        
        Returns:
            Similarity scores (num_candidates,) or (num_items,)
        """
        if candidate_items is not None:
            embeddings = self.normalized_embeddings[candidate_items]
        else:
            embeddings = self.normalized_embeddings
        
        # Cosine similarity
        similarities = embeddings @ user_profile
        
        return similarities
    
    def compute_attribute_score(
        self,
        user_items: Set[int],
        item_idx: int
    ) -> float:
        """
        Compute attribute matching score.
        
        Measures overlap between candidate item's attributes and
        attributes of items user has interacted with.
        
        Args:
            user_items: User's item history
            item_idx: Candidate item
        
        Returns:
            Attribute match score [0, 1]
        """
        if not self.item_attributes or item_idx not in self.item_attributes:
            return 0.5  # Neutral score
        
        item_attrs = self.item_attributes[item_idx]
        
        # Collect user's preferred attributes
        user_attrs: Dict[str, Set] = {}
        for u_item in user_items:
            if u_item in self.item_attributes:
                for key, value in self.item_attributes[u_item].items():
                    if key not in user_attrs:
                        user_attrs[key] = set()
                    if isinstance(value, (list, set)):
                        user_attrs[key].update(value)
                    else:
                        user_attrs[key].add(value)
        
        if not user_attrs:
            return 0.5
        
        # Compute overlap
        matches = 0
        total = 0
        
        for key, user_values in user_attrs.items():
            if key in item_attrs:
                item_value = item_attrs[key]
                if isinstance(item_value, (list, set)):
                    if any(v in user_values for v in item_value):
                        matches += 1
                else:
                    if item_value in user_values:
                        matches += 1
                total += 1
        
        return matches / total if total > 0 else 0.5
    
    def compute_brand_score(
        self,
        user_items: Set[int],
        item_idx: int
    ) -> float:
        """
        Compute brand affinity score.
        
        Args:
            user_items: User's item history
            item_idx: Candidate item
        
        Returns:
            Brand score [0, 1]
        """
        if not self.item_brands or item_idx not in self.item_brands:
            return 0.5
        
        item_brand = self.item_brands[item_idx]
        
        # Get user's preferred brands
        user_brands = set()
        for u_item in user_items:
            if u_item in self.item_brands:
                user_brands.add(self.item_brands[u_item])
        
        if not user_brands:
            return 0.5
        
        return 1.0 if item_brand in user_brands else 0.3
    
    def score_items(
        self,
        user_items: Set[int],
        item_weights: Optional[Dict[int, float]] = None,
        candidate_items: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Score items for a user using content features.
        
        Args:
            user_items: User's item history
            item_weights: Optional weights per historical item
            candidate_items: Optional subset to score
        
        Returns:
            Content scores (num_items,) or (num_candidates,)
        """
        # Compute user profile
        user_profile = self.compute_user_profile(user_items, item_weights)
        
        # Text similarity (main component)
        text_scores = self.compute_text_similarity(user_profile, candidate_items)
        
        # Initialize combined scores
        scores = self.text_weight * text_scores
        
        # Add attribute and brand scores if available
        if self.item_attributes or self.item_brands:
            items_to_score = candidate_items if candidate_items is not None else np.arange(self.num_items)
            
            for i, item_idx in enumerate(items_to_score):
                attr_score = self.compute_attribute_score(user_items, item_idx)
                brand_score = self.compute_brand_score(user_items, item_idx)
                
                scores[i] += self.attribute_weight * attr_score
                scores[i] += self.brand_weight * brand_score
        
        return scores


class PopularityScorer:
    """
    Popularity-based scoring component.
    
    Uses item popularity (num_sold, avg_rating) as a baseline/fallback.
    """
    
    def __init__(
        self,
        item_popularity: Dict[int, float],
        item_ratings: Optional[Dict[int, float]] = None,
        popularity_weight: float = 0.7,
        rating_weight: float = 0.3
    ):
        """
        Initialize popularity scorer.
        
        Args:
            item_popularity: Dict of item_idx -> popularity score (e.g., num_sold)
            item_ratings: Dict of item_idx -> average rating
            popularity_weight: Weight for popularity
            rating_weight: Weight for rating
        """
        self.item_popularity = item_popularity
        self.item_ratings = item_ratings or {}
        self.popularity_weight = popularity_weight
        self.rating_weight = rating_weight
        
        # Normalize popularity scores
        if item_popularity:
            pop_values = np.array(list(item_popularity.values()))
            self.pop_min = pop_values.min()
            self.pop_max = pop_values.max()
            self.pop_range = self.pop_max - self.pop_min
            if self.pop_range < 1e-8:
                self.pop_range = 1.0
        else:
            self.pop_min, self.pop_max, self.pop_range = 0, 1, 1
        
        logger.info(f"PopularityScorer initialized with {len(item_popularity)} items")
    
    def score_item(self, item_idx: int) -> float:
        """Score a single item by popularity."""
        pop = self.item_popularity.get(item_idx, self.pop_min)
        pop_normalized = (pop - self.pop_min) / self.pop_range
        
        rating = self.item_ratings.get(item_idx, 3.0) / 5.0  # Normalize to [0, 1]
        
        return self.popularity_weight * pop_normalized + self.rating_weight * rating
    
    def score_items(self, item_indices: np.ndarray) -> np.ndarray:
        """Score multiple items by popularity."""
        return np.array([self.score_item(idx) for idx in item_indices])
    
    def get_top_popular(self, k: int, exclude_items: Optional[Set[int]] = None) -> List[int]:
        """Get top-k most popular items."""
        sorted_items = sorted(
            self.item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        result = []
        for item_idx, _ in sorted_items:
            if exclude_items and item_idx in exclude_items:
                continue
            result.append(item_idx)
            if len(result) >= k:
                break
        
        return result


class BPRContentEnsemble:
    """
    Ensemble model combining BPR and content-based recommendations.
    
    Supports multiple fusion strategies:
    1. Weighted: Fixed weight combination
    2. Attention: Learn attention weights based on user features
    3. MLP: Neural network fusion
    4. Adaptive: User-dependent weighting based on interaction count
    
    Usage:
        >>> ensemble = BPRContentEnsemble(
        ...     bpr_U=user_embeddings,
        ...     bpr_V=item_embeddings,
        ...     content_scorer=content_scorer,
        ...     popularity_scorer=popularity_scorer,
        ...     config=EnsembleConfig(fusion_type=FusionType.ADAPTIVE)
        ... )
        >>> recommendations = ensemble.recommend(user_idx=123, k=10)
    """
    
    def __init__(
        self,
        bpr_U: np.ndarray,
        bpr_V: np.ndarray,
        content_scorer: ContentScorer,
        popularity_scorer: Optional[PopularityScorer] = None,
        user_interaction_counts: Optional[Dict[int, int]] = None,
        user_pos_sets: Optional[Dict[int, Set[int]]] = None,
        config: Optional[EnsembleConfig] = None
    ):
        """
        Initialize BPR + Content Ensemble.
        
        Args:
            bpr_U: BPR user embeddings (num_users, factors)
            bpr_V: BPR item embeddings (num_items, factors)
            content_scorer: Content-based scorer
            popularity_scorer: Popularity scorer (optional)
            user_interaction_counts: Dict of user_idx -> interaction count
            user_pos_sets: Dict of user_idx -> set of positive items
            config: Ensemble configuration
        """
        self.bpr_U = bpr_U
        self.bpr_V = bpr_V
        self.content_scorer = content_scorer
        self.popularity_scorer = popularity_scorer
        self.user_interaction_counts = user_interaction_counts or {}
        self.user_pos_sets = user_pos_sets or {}
        self.config = config or EnsembleConfig()
        
        self.num_users, self.factors = bpr_U.shape
        self.num_items = len(bpr_V)
        
        # Score normalization stats
        self._compute_score_stats()
        
        # Initialize fusion components
        if self.config.fusion_type == FusionType.MLP:
            self._init_mlp_fusion()
        elif self.config.fusion_type == FusionType.ATTENTION:
            self._init_attention_fusion()
        
        logger.info(f"BPRContentEnsemble initialized:")
        logger.info(f"  Users: {self.num_users:,}, Items: {self.num_items:,}")
        logger.info(f"  Fusion type: {self.config.fusion_type.value}")
        logger.info(f"  Cold-start threshold: {self.config.cold_start_threshold}")
    
    def _compute_score_stats(self):
        """Compute statistics for score normalization."""
        # Sample BPR scores for normalization
        sample_users = np.random.choice(self.num_users, min(1000, self.num_users), replace=False)
        sample_scores = []
        
        for u in sample_users:
            scores = self.bpr_U[u] @ self.bpr_V.T
            sample_scores.extend(scores[:100])
        
        sample_scores = np.array(sample_scores)
        self.bpr_mean = sample_scores.mean()
        self.bpr_std = sample_scores.std()
        if self.bpr_std < 1e-8:
            self.bpr_std = 1.0
    
    def _init_mlp_fusion(self):
        """Initialize MLP fusion network weights."""
        # Input: [bpr_score, content_score, popularity_score, user_features]
        input_dim = 3 + self.factors  # scores + user embedding
        
        self.mlp_weights = []
        prev_dim = input_dim
        
        for hidden_dim in self.config.mlp_hidden_dims:
            W = np.random.randn(prev_dim, hidden_dim) * 0.01
            b = np.zeros(hidden_dim)
            self.mlp_weights.append((W, b))
            prev_dim = hidden_dim
        
        # Output layer
        W_out = np.random.randn(prev_dim, 1) * 0.01
        b_out = np.zeros(1)
        self.mlp_weights.append((W_out, b_out))
    
    def _init_attention_fusion(self):
        """Initialize attention fusion weights."""
        # Learn attention over [bpr, content, popularity]
        self.attention_query = np.random.randn(self.factors, self.config.attention_dim) * 0.01
        self.attention_keys = np.random.randn(3, self.config.attention_dim) * 0.01
    
    def _normalize_scores(self, scores: np.ndarray, source: str = 'bpr') -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if not self.config.normalize_scores:
            return scores
        
        if source == 'bpr':
            normalized = (scores - self.bpr_mean) / self.bpr_std
            # Sigmoid to [0, 1]
            return 1.0 / (1.0 + np.exp(-normalized))
        else:
            # Min-max normalization
            min_s, max_s = scores.min(), scores.max()
            if max_s - min_s < 1e-8:
                return np.ones_like(scores) * 0.5
            return (scores - min_s) / (max_s - min_s)
    
    def _get_adaptive_weights(
        self,
        user_idx: int
    ) -> Tuple[float, float, float]:
        """
        Get adaptive fusion weights based on user interaction count.
        
        More interactions → higher BPR weight
        Fewer interactions → higher content/popularity weight
        
        Args:
            user_idx: User index
        
        Returns:
            Tuple of (bpr_weight, content_weight, popularity_weight)
        """
        interaction_count = self.user_interaction_counts.get(user_idx, 0)
        
        if interaction_count < self.config.cold_start_threshold:
            # Cold-start: mostly content + popularity
            return 0.1, 0.6, 0.3
        elif interaction_count < 5:
            # Low activity: balanced
            return 0.3, 0.5, 0.2
        elif interaction_count < 10:
            # Medium activity: favor BPR
            return 0.5, 0.35, 0.15
        else:
            # High activity: mostly BPR
            return 0.6, 0.3, 0.1
    
    def _mlp_forward(
        self,
        bpr_scores: np.ndarray,
        content_scores: np.ndarray,
        popularity_scores: np.ndarray,
        user_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass through MLP fusion.
        
        Args:
            bpr_scores: BPR scores (num_items,)
            content_scores: Content scores (num_items,)
            popularity_scores: Popularity scores (num_items,)
            user_embedding: User BPR embedding (factors,)
        
        Returns:
            Fused scores (num_items,)
        """
        num_items = len(bpr_scores)
        
        # Construct input: [bpr, content, pop, user_emb] for each item
        # Shape: (num_items, 3 + factors)
        inputs = np.column_stack([
            bpr_scores.reshape(-1, 1),
            content_scores.reshape(-1, 1),
            popularity_scores.reshape(-1, 1),
            np.tile(user_embedding, (num_items, 1))
        ])
        
        # Forward through MLP
        x = inputs
        for i, (W, b) in enumerate(self.mlp_weights[:-1]):
            x = x @ W + b
            x = np.maximum(0, x)  # ReLU
        
        # Output layer
        W_out, b_out = self.mlp_weights[-1]
        output = (x @ W_out + b_out).flatten()
        
        return output
    
    def _attention_forward(
        self,
        bpr_scores: np.ndarray,
        content_scores: np.ndarray,
        popularity_scores: np.ndarray,
        user_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass through attention fusion.
        
        Learns to weight [bpr, content, popularity] based on user embedding.
        
        Args:
            bpr_scores: BPR scores (num_items,)
            content_scores: Content scores (num_items,)
            popularity_scores: Popularity scores (num_items,)
            user_embedding: User BPR embedding (factors,)
        
        Returns:
            Fused scores (num_items,)
        """
        # Query from user embedding
        query = user_embedding @ self.attention_query  # (attention_dim,)
        
        # Compute attention scores
        attention_logits = self.attention_keys @ query  # (3,)
        attention_weights = np.exp(attention_logits / self.config.temperature)
        attention_weights = attention_weights / attention_weights.sum()
        
        # Weighted combination
        fused = (
            attention_weights[0] * bpr_scores +
            attention_weights[1] * content_scores +
            attention_weights[2] * popularity_scores
        )
        
        return fused
    
    def score_items(
        self,
        user_idx: int,
        candidate_items: Optional[np.ndarray] = None,
        return_components: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Score items for a user using ensemble.
        
        Args:
            user_idx: User index
            candidate_items: Optional subset of items to score
            return_components: If True, return individual score components
        
        Returns:
            Fused scores, or dict of component scores if return_components=True
        """
        # Get user history
        user_items = self.user_pos_sets.get(user_idx, set())
        
        # Items to score
        if candidate_items is not None:
            items = candidate_items
        else:
            items = np.arange(self.num_items)
        
        # 1. BPR scores
        bpr_scores = self.bpr_U[user_idx] @ self.bpr_V[items].T
        bpr_scores = self._normalize_scores(bpr_scores, 'bpr')
        
        # 2. Content scores
        content_scores = self.content_scorer.score_items(
            user_items,
            candidate_items=items
        )
        content_scores = self._normalize_scores(content_scores, 'content')
        
        # 3. Popularity scores
        if self.popularity_scorer:
            popularity_scores = self.popularity_scorer.score_items(items)
        else:
            popularity_scores = np.ones(len(items)) * 0.5
        
        # 4. Fusion
        if self.config.fusion_type == FusionType.WEIGHTED:
            fused = (
                self.config.bpr_weight * bpr_scores +
                self.config.content_weight * content_scores +
                self.config.popularity_weight * popularity_scores
            )
        
        elif self.config.fusion_type == FusionType.ADAPTIVE:
            w_bpr, w_content, w_pop = self._get_adaptive_weights(user_idx)
            fused = w_bpr * bpr_scores + w_content * content_scores + w_pop * popularity_scores
        
        elif self.config.fusion_type == FusionType.MLP:
            user_emb = self.bpr_U[user_idx]
            fused = self._mlp_forward(bpr_scores, content_scores, popularity_scores, user_emb)
        
        elif self.config.fusion_type == FusionType.ATTENTION:
            user_emb = self.bpr_U[user_idx]
            fused = self._attention_forward(bpr_scores, content_scores, popularity_scores, user_emb)
        
        else:
            fused = bpr_scores  # Fallback
        
        if return_components:
            return {
                'fused': fused,
                'bpr': bpr_scores,
                'content': content_scores,
                'popularity': popularity_scores
            }
        
        return fused
    
    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_seen: bool = True,
        exclude_items: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate top-k recommendations for a user.
        
        Args:
            user_idx: User index
            k: Number of recommendations
            exclude_seen: Whether to exclude items user has interacted with
            exclude_items: Additional items to exclude
        
        Returns:
            List of (item_idx, score) tuples
        """
        # Get scores
        scores = self.score_items(user_idx)
        
        # Build exclusion set
        excluded = set()
        if exclude_seen:
            excluded.update(self.user_pos_sets.get(user_idx, set()))
        if exclude_items:
            excluded.update(exclude_items)
        
        # Mask excluded items
        for item in excluded:
            if item < len(scores):
                scores[item] = -np.inf
        
        # Get top-k
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        return [(int(idx), float(scores[idx])) for idx in top_k_indices if scores[idx] > -np.inf]
    
    def batch_recommend(
        self,
        user_indices: List[int],
        k: int = 10,
        exclude_seen: bool = True
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_indices: List of user indices
            k: Number of recommendations per user
            exclude_seen: Whether to exclude seen items
        
        Returns:
            Dict of user_idx -> list of (item_idx, score)
        """
        results = {}
        for user_idx in user_indices:
            results[user_idx] = self.recommend(user_idx, k, exclude_seen)
        return results
    
    def evaluate(
        self,
        user_pos_test: Dict[int, Set[int]],
        k_values: List[int] = [10, 20],
        num_eval_users: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate ensemble model.
        
        Args:
            user_pos_test: Dict of user -> set of test items
            k_values: Values of K for metrics
            num_eval_users: Maximum users to evaluate
        
        Returns:
            Dict of metric -> value
        """
        rng = np.random.default_rng(42)
        eval_users = list(user_pos_test.keys())
        
        if len(eval_users) > num_eval_users:
            eval_users = rng.choice(eval_users, num_eval_users, replace=False)
        
        metrics = {}
        
        for k in k_values:
            recalls = []
            ndcgs = []
            
            for user_idx in eval_users:
                test_items = user_pos_test.get(user_idx, set())
                if not test_items:
                    continue
                
                # Get recommendations
                recs = self.recommend(user_idx, k, exclude_seen=True)
                rec_items = set([r[0] for r in recs])
                
                # Recall@K
                hits = len(rec_items & test_items)
                recall = hits / min(k, len(test_items))
                recalls.append(recall)
                
                # NDCG@K
                dcg = 0.0
                for rank, (item, _) in enumerate(recs):
                    if item in test_items:
                        dcg += 1.0 / np.log2(rank + 2)
                
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(test_items))))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcgs.append(ndcg)
            
            metrics[f'Recall@{k}'] = np.mean(recalls) if recalls else 0.0
            metrics[f'NDCG@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
        
        return metrics
    
    def get_user_fusion_weights(self, user_idx: int) -> Dict[str, float]:
        """Get fusion weights for a specific user (for adaptive fusion)."""
        if self.config.fusion_type == FusionType.ADAPTIVE:
            w_bpr, w_content, w_pop = self._get_adaptive_weights(user_idx)
            return {
                'bpr_weight': w_bpr,
                'content_weight': w_content,
                'popularity_weight': w_pop,
                'interaction_count': self.user_interaction_counts.get(user_idx, 0)
            }
        else:
            return {
                'bpr_weight': self.config.bpr_weight,
                'content_weight': self.config.content_weight,
                'popularity_weight': self.config.popularity_weight
            }


def create_ensemble_from_artifacts(
    bpr_artifact_dir: Path,
    bert_embeddings_path: Path,
    item_popularity: Dict[int, float],
    user_pos_sets: Dict[int, Set[int]],
    user_interaction_counts: Dict[int, int],
    item_attributes: Optional[Dict[int, Dict]] = None,
    item_brands: Optional[Dict[int, str]] = None,
    config: Optional[EnsembleConfig] = None
) -> BPRContentEnsemble:
    """
    Create ensemble model from saved artifacts.
    
    Args:
        bpr_artifact_dir: Directory containing BPR_U.npy, BPR_V.npy
        bert_embeddings_path: Path to BERT embeddings (.npy or .pt)
        item_popularity: Dict of item popularity scores
        user_pos_sets: User positive item sets
        user_interaction_counts: User interaction counts
        item_attributes: Optional item attributes
        item_brands: Optional item brands
        config: Ensemble configuration
    
    Returns:
        Configured BPRContentEnsemble
    """
    # Load BPR embeddings
    bpr_U = np.load(bpr_artifact_dir / 'BPR_U.npy')
    bpr_V = np.load(bpr_artifact_dir / 'BPR_V.npy')
    
    # Load BERT embeddings
    if bert_embeddings_path.suffix == '.pt':
        import torch
        bert_emb = torch.load(bert_embeddings_path)
        if isinstance(bert_emb, torch.Tensor):
            bert_emb = bert_emb.numpy()
    else:
        bert_emb = np.load(bert_embeddings_path)
    
    # Create scorers
    content_scorer = ContentScorer(
        item_embeddings=bert_emb,
        item_attributes=item_attributes,
        item_brands=item_brands
    )
    
    popularity_scorer = PopularityScorer(item_popularity)
    
    # Create ensemble
    ensemble = BPRContentEnsemble(
        bpr_U=bpr_U,
        bpr_V=bpr_V,
        content_scorer=content_scorer,
        popularity_scorer=popularity_scorer,
        user_interaction_counts=user_interaction_counts,
        user_pos_sets=user_pos_sets,
        config=config
    )
    
    return ensemble
