"""
Advanced Negative Sampling Module for BPR

This module implements intelligent negative sampling strategies:
1. Contextual Negatives: Similar items user hasn't bought (via BERT similarity)
2. Sentiment-Contrasted Negatives: Items with opposite sentiment to user preference
3. Cold-Start Popular Negatives: Popular items for cold-start users
4. Dynamic Sampling: Adaptive weights based on model predictions (curriculum learning)

Key Features:
- Multi-strategy sampling with configurable ratios
- Dynamic difficulty adjustment (harder negatives as training progresses)
- Text/BERT similarity-based contextual negatives
- Sentiment-aware negative selection

Author: VieComRec Team
Date: 2025-11-26
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Optional, Any, Union
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SamplingStrategy:
    """
    Configuration for negative sampling strategy.
    
    Attributes:
        hard_ratio: Fraction from explicit hard negatives (rating ≤ 3)
        contextual_ratio: Fraction from contextual negatives (BERT similar but not bought)
        sentiment_contrast_ratio: Fraction from sentiment-contrasted items
        popular_ratio: Fraction from cold-start popular items
        random_ratio: Fraction from random sampling (computed as remainder)
        
    Note: Ratios should sum to 1.0 (random_ratio is auto-computed)
    """
    hard_ratio: float = 0.25
    contextual_ratio: float = 0.20
    sentiment_contrast_ratio: float = 0.15
    popular_ratio: float = 0.10
    # random_ratio is computed as: 1.0 - sum(other ratios)
    
    def __post_init__(self):
        total = self.hard_ratio + self.contextual_ratio + self.sentiment_contrast_ratio + self.popular_ratio
        if total > 1.0:
            raise ValueError(f"Sum of explicit ratios ({total:.2f}) exceeds 1.0")
        self.random_ratio = 1.0 - total
        
    def to_dict(self) -> Dict[str, float]:
        return {
            'hard_ratio': self.hard_ratio,
            'contextual_ratio': self.contextual_ratio,
            'sentiment_contrast_ratio': self.sentiment_contrast_ratio,
            'popular_ratio': self.popular_ratio,
            'random_ratio': self.random_ratio
        }


@dataclass
class DynamicSamplingConfig:
    """
    Configuration for dynamic/adaptive sampling.
    
    As training progresses, we want to:
    1. Increase difficulty (sample harder negatives)
    2. Focus on items the model still ranks incorrectly
    
    Attributes:
        enable_dynamic: Whether to enable dynamic sampling
        warmup_epochs: Number of epochs before enabling dynamic sampling
        difficulty_schedule: How difficulty increases ('linear', 'exponential', 'cosine')
        initial_difficulty: Starting difficulty multiplier (0.0 = easy, 1.0 = hard)
        final_difficulty: Ending difficulty multiplier
        resample_hard_negatives: Re-sample negatives that model ranks highly
        resample_threshold: Score threshold for resampling (items scored > threshold)
    """
    enable_dynamic: bool = True
    warmup_epochs: int = 5
    difficulty_schedule: str = 'linear'  # 'linear', 'exponential', 'cosine'
    initial_difficulty: float = 0.3
    final_difficulty: float = 0.8
    resample_hard_negatives: bool = True
    resample_threshold: float = 0.5  # Re-sample if neg score > pos_score * threshold


class ContextualNegativeSampler:
    """
    Sample contextual negatives based on item similarity.
    
    Contextual negatives are items that are semantically similar to what
    the user likes, but the user hasn't interacted with. These are more
    informative than random negatives because they force the model to
    make fine-grained distinctions.
    
    Sources:
    1. BERT/PhoBERT similarity: Items with high text similarity to user history
    2. Category similarity: Items in same category as user preferences
    3. Attribute similarity: Items with similar attributes (skin_type, etc.)
    """
    
    def __init__(
        self,
        item_embeddings: Optional[np.ndarray] = None,
        item_categories: Optional[Dict[int, str]] = None,
        item_attributes: Optional[Dict[int, Dict[str, Any]]] = None,
        top_k_similar: int = 50,
        random_seed: int = 42
    ):
        """
        Initialize contextual negative sampler.
        
        Args:
            item_embeddings: BERT embeddings (num_items, embed_dim) for text similarity
            item_categories: Dict mapping item_idx -> category string
            item_attributes: Dict mapping item_idx -> attributes dict
            top_k_similar: Number of similar items to consider per item
            random_seed: Random seed
        """
        self.item_embeddings = item_embeddings
        self.item_categories = item_categories or {}
        self.item_attributes = item_attributes or {}
        self.top_k_similar = top_k_similar
        self.rng = np.random.default_rng(random_seed)
        
        # Pre-compute similarity indices if embeddings provided
        self.similar_items_cache: Dict[int, np.ndarray] = {}
        
        if item_embeddings is not None:
            logger.info(f"ContextualNegativeSampler: Computing item similarities...")
            self._precompute_similarities()
            logger.info(f"  Cached similar items for {len(self.similar_items_cache)} items")
        
        # Category-based similar items
        self.category_items: Dict[str, Set[int]] = {}
        if item_categories:
            for item_idx, category in item_categories.items():
                if category not in self.category_items:
                    self.category_items[category] = set()
                self.category_items[category].add(item_idx)
            logger.info(f"  Categories indexed: {len(self.category_items)}")
        
        # Stats
        self.stats = {
            'bert_similar_samples': 0,
            'category_similar_samples': 0,
            'fallback_samples': 0
        }
    
    def _precompute_similarities(self, batch_size: int = 1000):
        """
        Pre-compute top-K similar items for each item using BERT embeddings.
        
        Uses batched cosine similarity for memory efficiency.
        """
        if self.item_embeddings is None:
            return
        
        num_items = len(self.item_embeddings)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        normalized = self.item_embeddings / norms
        
        # Process in batches to avoid memory issues
        for start_idx in range(0, num_items, batch_size):
            end_idx = min(start_idx + batch_size, num_items)
            batch = normalized[start_idx:end_idx]
            
            # Compute similarities: (batch_size, num_items)
            similarities = batch @ normalized.T
            
            # For each item in batch, get top-K similar (excluding self)
            for i, global_idx in enumerate(range(start_idx, end_idx)):
                sims = similarities[i]
                sims[global_idx] = -np.inf  # Exclude self
                
                # Get top-K indices
                top_k_indices = np.argpartition(sims, -self.top_k_similar)[-self.top_k_similar:]
                top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])[::-1]]
                
                self.similar_items_cache[global_idx] = top_k_indices
    
    def sample_contextual_negative(
        self,
        user_idx: int,
        positive_item: int,
        user_pos_set: Set[int],
        num_items: int,
        use_bert: bool = True
    ) -> int:
        """
        Sample a contextual negative for a user-item pair.
        
        Strategy:
        1. Try BERT-similar items to positive_item that user hasn't bought
        2. Fallback to category-similar items
        3. Fallback to random
        
        Args:
            user_idx: User index
            positive_item: Positive item index (to find similar items)
            user_pos_set: Set of items user has interacted with
            num_items: Total number of items
            use_bert: Whether to use BERT similarity (vs category only)
        
        Returns:
            Sampled contextual negative item index
        """
        # Strategy 1: BERT-similar items
        if use_bert and positive_item in self.similar_items_cache:
            similar_items = self.similar_items_cache[positive_item]
            
            # Filter out items user already has
            valid_similar = [i for i in similar_items if i not in user_pos_set]
            
            if valid_similar:
                self.stats['bert_similar_samples'] += 1
                return int(self.rng.choice(valid_similar[:min(10, len(valid_similar))]))
        
        # Strategy 2: Category-similar items
        if positive_item in self.item_categories:
            category = self.item_categories[positive_item]
            if category in self.category_items:
                category_set = self.category_items[category]
                valid_category = category_set - user_pos_set - {positive_item}
                
                if valid_category:
                    self.stats['category_similar_samples'] += 1
                    return int(self.rng.choice(list(valid_category)))
        
        # Strategy 3: Random fallback
        self.stats['fallback_samples'] += 1
        for _ in range(100):
            neg = self.rng.integers(0, num_items)
            if neg not in user_pos_set:
                return int(neg)
        
        # Final fallback
        all_items = set(range(num_items))
        valid = list(all_items - user_pos_set)
        return int(self.rng.choice(valid)) if valid else 0
    
    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()
    
    def reset_stats(self):
        self.stats = {
            'bert_similar_samples': 0,
            'category_similar_samples': 0,
            'fallback_samples': 0
        }


class SentimentContrastSampler:
    """
    Sample negatives with contrasting sentiment to user preferences.
    
    Logic: If a user tends to like products with positive sentiment (good reviews),
    then items with negative sentiment (bad reviews) are good hard negatives.
    
    This helps the model learn that sentiment matters for recommendations.
    """
    
    def __init__(
        self,
        item_sentiment_scores: Optional[Dict[int, float]] = None,
        sentiment_threshold_positive: float = 0.6,
        sentiment_threshold_negative: float = 0.4,
        random_seed: int = 42
    ):
        """
        Initialize sentiment contrast sampler.
        
        Args:
            item_sentiment_scores: Dict mapping item_idx -> avg sentiment score [0, 1]
            sentiment_threshold_positive: Score above this is "positive sentiment"
            sentiment_threshold_negative: Score below this is "negative sentiment"
            random_seed: Random seed
        """
        self.item_sentiment = item_sentiment_scores or {}
        self.threshold_positive = sentiment_threshold_positive
        self.threshold_negative = sentiment_threshold_negative
        self.rng = np.random.default_rng(random_seed)
        
        # Pre-compute positive and negative sentiment item sets
        self.positive_sentiment_items: Set[int] = set()
        self.negative_sentiment_items: Set[int] = set()
        
        for item_idx, score in self.item_sentiment.items():
            if score >= self.threshold_positive:
                self.positive_sentiment_items.add(item_idx)
            elif score <= self.threshold_negative:
                self.negative_sentiment_items.add(item_idx)
        
        logger.info(f"SentimentContrastSampler:")
        logger.info(f"  Positive sentiment items: {len(self.positive_sentiment_items)}")
        logger.info(f"  Negative sentiment items: {len(self.negative_sentiment_items)}")
        
        self.stats = {'contrast_samples': 0, 'fallback_samples': 0}
    
    def sample_contrast_negative(
        self,
        user_idx: int,
        positive_item: int,
        user_pos_set: Set[int],
        num_items: int
    ) -> int:
        """
        Sample a negative with contrasting sentiment.
        
        If positive_item has positive sentiment → sample from negative sentiment items
        If positive_item has negative sentiment → sample from positive sentiment items
        
        Args:
            user_idx: User index
            positive_item: Positive item index
            user_pos_set: Items user has interacted with
            num_items: Total items
        
        Returns:
            Sampled negative item index
        """
        pos_sentiment = self.item_sentiment.get(positive_item, 0.5)
        
        # Determine target sentiment pool
        if pos_sentiment >= self.threshold_positive:
            # User likes positive → sample from negative sentiment
            target_pool = self.negative_sentiment_items - user_pos_set
        elif pos_sentiment <= self.threshold_negative:
            # User interacted with negative → sample from positive sentiment
            target_pool = self.positive_sentiment_items - user_pos_set
        else:
            # Neutral → random from either extreme
            if self.rng.random() < 0.5:
                target_pool = self.negative_sentiment_items - user_pos_set
            else:
                target_pool = self.positive_sentiment_items - user_pos_set
        
        if target_pool:
            self.stats['contrast_samples'] += 1
            return int(self.rng.choice(list(target_pool)))
        
        # Fallback to random
        self.stats['fallback_samples'] += 1
        for _ in range(100):
            neg = self.rng.integers(0, num_items)
            if neg not in user_pos_set:
                return int(neg)
        
        all_items = set(range(num_items))
        valid = list(all_items - user_pos_set)
        return int(self.rng.choice(valid)) if valid else 0
    
    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()
    
    def reset_stats(self):
        self.stats = {'contrast_samples': 0, 'fallback_samples': 0}


class PopularNegativeSampler:
    """
    Sample from popular items for cold-start scenarios.
    
    Popular items that a user hasn't interacted with are informative negatives
    because they test whether the model can distinguish user-specific preferences
    from global popularity.
    """
    
    def __init__(
        self,
        item_popularity: Optional[Dict[int, float]] = None,
        top_k_popular: int = 100,
        random_seed: int = 42
    ):
        """
        Initialize popular negative sampler.
        
        Args:
            item_popularity: Dict mapping item_idx -> popularity score (e.g., num_sold)
            top_k_popular: Number of top popular items to consider
            random_seed: Random seed
        """
        self.rng = np.random.default_rng(random_seed)
        
        # Sort items by popularity
        if item_popularity:
            sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
            self.popular_items = np.array([item for item, _ in sorted_items[:top_k_popular]])
            self.popularity_weights = np.array([pop for _, pop in sorted_items[:top_k_popular]])
            # Normalize weights
            self.popularity_weights = self.popularity_weights / self.popularity_weights.sum()
        else:
            self.popular_items = np.array([])
            self.popularity_weights = np.array([])
        
        logger.info(f"PopularNegativeSampler: {len(self.popular_items)} popular items indexed")
        
        self.stats = {'popular_samples': 0, 'fallback_samples': 0}
    
    def sample_popular_negative(
        self,
        user_idx: int,
        user_pos_set: Set[int],
        num_items: int,
        weighted: bool = True
    ) -> int:
        """
        Sample a popular item not in user's history.
        
        Args:
            user_idx: User index
            user_pos_set: Items user has interacted with
            num_items: Total items
            weighted: If True, sample proportional to popularity
        
        Returns:
            Sampled negative item index
        """
        if len(self.popular_items) > 0:
            # Filter out items user already has
            valid_mask = ~np.isin(self.popular_items, list(user_pos_set))
            valid_popular = self.popular_items[valid_mask]
            
            if len(valid_popular) > 0:
                if weighted and len(self.popularity_weights) > 0:
                    valid_weights = self.popularity_weights[valid_mask]
                    valid_weights = valid_weights / valid_weights.sum()
                    self.stats['popular_samples'] += 1
                    return int(self.rng.choice(valid_popular, p=valid_weights))
                else:
                    self.stats['popular_samples'] += 1
                    return int(self.rng.choice(valid_popular))
        
        # Fallback to random
        self.stats['fallback_samples'] += 1
        for _ in range(100):
            neg = self.rng.integers(0, num_items)
            if neg not in user_pos_set:
                return int(neg)
        
        all_items = set(range(num_items))
        valid = list(all_items - user_pos_set)
        return int(self.rng.choice(valid)) if valid else 0
    
    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()
    
    def reset_stats(self):
        self.stats = {'popular_samples': 0, 'fallback_samples': 0}


class AdvancedTripletSampler:
    """
    Advanced triplet sampler with multiple negative sampling strategies.
    
    Combines:
    1. Hard negatives (explicit rating ≤ 3)
    2. Contextual negatives (BERT-similar items not bought)
    3. Sentiment-contrast negatives (opposite sentiment)
    4. Popular negatives (cold-start items)
    5. Random negatives (uniform sampling)
    6. Dynamic sampling (adaptive difficulty)
    
    Usage:
        >>> sampler = AdvancedTripletSampler(
        ...     positive_pairs=pairs,
        ...     user_pos_sets=pos_sets,
        ...     num_items=2200,
        ...     hard_neg_sets=hard_negs,
        ...     item_embeddings=bert_embeddings,  # For contextual
        ...     item_sentiment_scores=sentiment,   # For sentiment contrast
        ...     item_popularity=popularity,        # For popular negatives
        ...     strategy=SamplingStrategy(hard_ratio=0.25, contextual_ratio=0.20)
        ... )
        >>> triplets = sampler.sample_epoch()
    """
    
    def __init__(
        self,
        positive_pairs: np.ndarray,
        user_pos_sets: Dict[int, Set[int]],
        num_items: int,
        hard_neg_sets: Optional[Dict[int, Set[int]]] = None,
        item_embeddings: Optional[np.ndarray] = None,
        item_categories: Optional[Dict[int, str]] = None,
        item_sentiment_scores: Optional[Dict[int, float]] = None,
        item_popularity: Optional[Dict[int, float]] = None,
        strategy: Optional[SamplingStrategy] = None,
        dynamic_config: Optional[DynamicSamplingConfig] = None,
        samples_per_positive: int = 5,
        random_seed: int = 42
    ):
        """
        Initialize advanced triplet sampler.
        
        Args:
            positive_pairs: Array of (u_idx, i_idx) positive pairs
            user_pos_sets: Dict mapping u_idx -> Set of positive items
            num_items: Total number of items
            hard_neg_sets: Dict mapping u_idx -> Set of hard negative items
            item_embeddings: BERT embeddings for contextual sampling
            item_categories: Item categories for contextual sampling
            item_sentiment_scores: Item sentiment scores for contrast sampling
            item_popularity: Item popularity for popular sampling
            strategy: Sampling strategy configuration
            dynamic_config: Dynamic sampling configuration
            samples_per_positive: Samples per positive pair per epoch
            random_seed: Random seed
        """
        self.positive_pairs = positive_pairs
        self.user_pos_sets = user_pos_sets
        self.num_items = num_items
        self.samples_per_positive = samples_per_positive
        self.rng = np.random.default_rng(random_seed)
        
        self.num_positives = len(positive_pairs)
        self.samples_per_epoch = self.num_positives * samples_per_positive
        
        # Strategy configuration
        self.strategy = strategy or SamplingStrategy()
        self.dynamic_config = dynamic_config or DynamicSamplingConfig()
        
        # Initialize sub-samplers
        # 1. Hard negative sampler
        self.hard_neg_sets = hard_neg_sets or {}
        self.hard_neg_arrays = {
            u: np.array(list(items)) for u, items in self.hard_neg_sets.items()
            if len(items) > 0
        }
        
        # 2. Contextual negative sampler
        self.contextual_sampler = ContextualNegativeSampler(
            item_embeddings=item_embeddings,
            item_categories=item_categories,
            random_seed=random_seed
        )
        
        # 3. Sentiment contrast sampler
        self.sentiment_sampler = SentimentContrastSampler(
            item_sentiment_scores=item_sentiment_scores,
            random_seed=random_seed
        )
        
        # 4. Popular negative sampler
        self.popular_sampler = PopularNegativeSampler(
            item_popularity=item_popularity,
            random_seed=random_seed
        )
        
        # Training state
        self.current_epoch = 0
        self.current_difficulty = self.dynamic_config.initial_difficulty
        
        # Statistics
        self.stats = {
            'hard_samples': 0,
            'contextual_samples': 0,
            'sentiment_samples': 0,
            'popular_samples': 0,
            'random_samples': 0,
            'resampled_hard': 0
        }
        
        logger.info(f"AdvancedTripletSampler initialized:")
        logger.info(f"  Positive pairs: {self.num_positives:,}")
        logger.info(f"  Samples per epoch: {self.samples_per_epoch:,}")
        logger.info(f"  Strategy: {self.strategy.to_dict()}")
        logger.info(f"  Dynamic sampling: {self.dynamic_config.enable_dynamic}")
    
    def _get_current_difficulty(self) -> float:
        """
        Compute current difficulty based on epoch and schedule.
        
        Returns difficulty multiplier [initial_difficulty, final_difficulty]
        """
        if not self.dynamic_config.enable_dynamic:
            return self.strategy.hard_ratio
        
        if self.current_epoch < self.dynamic_config.warmup_epochs:
            return self.dynamic_config.initial_difficulty
        
        # Progress after warmup (0 to 1)
        epochs_after_warmup = self.current_epoch - self.dynamic_config.warmup_epochs
        max_epochs = 50  # Assume ~50 epochs total
        progress = min(epochs_after_warmup / max_epochs, 1.0)
        
        initial = self.dynamic_config.initial_difficulty
        final = self.dynamic_config.final_difficulty
        
        if self.dynamic_config.difficulty_schedule == 'linear':
            return initial + (final - initial) * progress
        elif self.dynamic_config.difficulty_schedule == 'exponential':
            return initial * (final / initial) ** progress
        elif self.dynamic_config.difficulty_schedule == 'cosine':
            return initial + (final - initial) * (1 - np.cos(progress * np.pi)) / 2
        else:
            return initial
    
    def _select_strategy(self) -> str:
        """
        Select sampling strategy based on configured ratios.
        
        Returns one of: 'hard', 'contextual', 'sentiment', 'popular', 'random'
        """
        difficulty = self._get_current_difficulty()
        
        # Adjust ratios based on current difficulty
        # Higher difficulty → more hard/contextual, less random
        hard_ratio = self.strategy.hard_ratio * (1 + difficulty - 0.5)
        contextual_ratio = self.strategy.contextual_ratio * (1 + difficulty - 0.5)
        sentiment_ratio = self.strategy.sentiment_contrast_ratio
        popular_ratio = self.strategy.popular_ratio
        
        # Normalize
        total = hard_ratio + contextual_ratio + sentiment_ratio + popular_ratio
        random_ratio = max(0, 1.0 - min(total, 0.9))  # Keep at least 10% random
        
        # Sample strategy
        r = self.rng.random()
        cumsum = 0
        
        cumsum += hard_ratio / (total + random_ratio)
        if r < cumsum:
            return 'hard'
        
        cumsum += contextual_ratio / (total + random_ratio)
        if r < cumsum:
            return 'contextual'
        
        cumsum += sentiment_ratio / (total + random_ratio)
        if r < cumsum:
            return 'sentiment'
        
        cumsum += popular_ratio / (total + random_ratio)
        if r < cumsum:
            return 'popular'
        
        return 'random'
    
    def sample_negative(
        self,
        user_idx: int,
        positive_item: int,
        user_pos_set: Set[int]
    ) -> int:
        """
        Sample a single negative item using multi-strategy approach.
        
        Args:
            user_idx: User index
            positive_item: The positive item (for contextual sampling)
            user_pos_set: Items user has interacted with
        
        Returns:
            Sampled negative item index
        """
        strategy = self._select_strategy()
        
        if strategy == 'hard':
            neg = self._sample_hard_negative(user_idx, user_pos_set)
            if neg is not None:
                self.stats['hard_samples'] += 1
                return neg
            # Fallback to random
            strategy = 'random'
        
        if strategy == 'contextual':
            neg = self.contextual_sampler.sample_contextual_negative(
                user_idx, positive_item, user_pos_set, self.num_items
            )
            self.stats['contextual_samples'] += 1
            return neg
        
        if strategy == 'sentiment':
            neg = self.sentiment_sampler.sample_contrast_negative(
                user_idx, positive_item, user_pos_set, self.num_items
            )
            self.stats['sentiment_samples'] += 1
            return neg
        
        if strategy == 'popular':
            neg = self.popular_sampler.sample_popular_negative(
                user_idx, user_pos_set, self.num_items
            )
            self.stats['popular_samples'] += 1
            return neg
        
        # Random sampling
        self.stats['random_samples'] += 1
        return self._sample_random_negative(user_pos_set)
    
    def _sample_hard_negative(
        self,
        user_idx: int,
        user_pos_set: Set[int]
    ) -> Optional[int]:
        """Sample from explicit hard negatives (rating ≤ 3)."""
        if user_idx not in self.hard_neg_arrays:
            return None
        
        hard_arr = self.hard_neg_arrays[user_idx]
        valid_hard = hard_arr[~np.isin(hard_arr, list(user_pos_set))]
        
        if len(valid_hard) > 0:
            return int(self.rng.choice(valid_hard))
        return None
    
    def _sample_random_negative(self, user_pos_set: Set[int]) -> int:
        """Sample uniformly random negative."""
        for _ in range(100):
            neg = self.rng.integers(0, self.num_items)
            if neg not in user_pos_set:
                return int(neg)
        
        all_items = set(range(self.num_items))
        valid = list(all_items - user_pos_set)
        return int(self.rng.choice(valid)) if valid else 0
    
    def sample_epoch(
        self,
        shuffle: bool = True,
        model_scores: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Sample triplets for one epoch.
        
        Args:
            shuffle: Whether to shuffle triplets
            model_scores: Optional (U, V) embeddings for dynamic resampling
        
        Returns:
            Array of shape (samples_per_epoch, 3) with [u, i_pos, i_neg]
        """
        # Sample positive pairs with replacement
        pair_indices = self.rng.choice(
            self.num_positives,
            size=self.samples_per_epoch,
            replace=True
        )
        
        sampled_pairs = self.positive_pairs[pair_indices]
        users = sampled_pairs[:, 0]
        positives = sampled_pairs[:, 1]
        
        # Sample negatives
        negatives = np.zeros(self.samples_per_epoch, dtype=np.int64)
        
        for i in range(self.samples_per_epoch):
            user_idx = int(users[i])
            pos_item = int(positives[i])
            pos_set = self.user_pos_sets.get(user_idx, set())
            
            negatives[i] = self.sample_negative(user_idx, pos_item, pos_set)
        
        # Dynamic resampling: Re-sample negatives that model scores too high
        if (self.dynamic_config.enable_dynamic and 
            self.dynamic_config.resample_hard_negatives and
            model_scores is not None):
            
            U, V = model_scores
            negatives = self._resample_easy_negatives(
                users, positives, negatives, U, V
            )
        
        triplets = np.column_stack([users, positives, negatives])
        
        if shuffle:
            self.rng.shuffle(triplets)
        
        return triplets
    
    def _resample_easy_negatives(
        self,
        users: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        max_resample_ratio: float = 0.3
    ) -> np.ndarray:
        """
        Re-sample negatives that the model scores too highly.
        
        This implements curriculum learning: as training progresses,
        we focus on harder examples (negatives the model still gets wrong).
        
        Args:
            users: User indices
            positives: Positive item indices
            negatives: Current negative item indices
            U: User embeddings (num_users, factors)
            V: Item embeddings (num_items, factors)
            max_resample_ratio: Maximum fraction to resample
        
        Returns:
            Updated negative indices
        """
        # Compute scores
        pos_scores = np.sum(U[users] * V[positives], axis=1)
        neg_scores = np.sum(U[users] * V[negatives], axis=1)
        
        # Identify "easy" negatives (neg score too close to or above pos score)
        threshold = self.dynamic_config.resample_threshold
        easy_mask = neg_scores > pos_scores * threshold
        
        # Limit resampling
        num_easy = easy_mask.sum()
        max_resample = int(len(negatives) * max_resample_ratio)
        
        if num_easy > max_resample:
            # Randomly select subset to resample
            easy_indices = np.where(easy_mask)[0]
            resample_indices = self.rng.choice(easy_indices, max_resample, replace=False)
            easy_mask = np.zeros(len(negatives), dtype=bool)
            easy_mask[resample_indices] = True
        
        # Re-sample easy negatives with higher difficulty
        for idx in np.where(easy_mask)[0]:
            user_idx = int(users[idx])
            pos_item = int(positives[idx])
            pos_set = self.user_pos_sets.get(user_idx, set())
            
            # Try harder sampling strategies
            for _ in range(3):  # Multiple attempts
                # Force hard or contextual strategy
                if self.rng.random() < 0.5:
                    neg = self._sample_hard_negative(user_idx, pos_set)
                else:
                    neg = self.contextual_sampler.sample_contextual_negative(
                        user_idx, pos_item, pos_set, self.num_items
                    )
                
                if neg is not None:
                    # Check if new negative is harder
                    new_score = np.dot(U[user_idx], V[neg])
                    if new_score < neg_scores[idx]:
                        negatives[idx] = neg
                        self.stats['resampled_hard'] += 1
                        break
        
        return negatives
    
    def set_epoch(self, epoch: int):
        """Update current epoch for dynamic difficulty adjustment."""
        self.current_epoch = epoch
        self.current_difficulty = self._get_current_difficulty()
    
    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get comprehensive sampling statistics."""
        total = sum([
            self.stats['hard_samples'],
            self.stats['contextual_samples'],
            self.stats['sentiment_samples'],
            self.stats['popular_samples'],
            self.stats['random_samples']
        ])
        
        return {
            **self.stats,
            'total_samples': total,
            'hard_ratio': self.stats['hard_samples'] / total if total > 0 else 0,
            'contextual_ratio': self.stats['contextual_samples'] / total if total > 0 else 0,
            'sentiment_ratio': self.stats['sentiment_samples'] / total if total > 0 else 0,
            'popular_ratio': self.stats['popular_samples'] / total if total > 0 else 0,
            'random_ratio': self.stats['random_samples'] / total if total > 0 else 0,
            'current_epoch': self.current_epoch,
            'current_difficulty': self.current_difficulty,
            'contextual_stats': self.contextual_sampler.get_stats(),
            'sentiment_stats': self.sentiment_sampler.get_stats(),
            'popular_stats': self.popular_sampler.get_stats()
        }
    
    def reset_stats(self):
        """Reset all statistics."""
        self.stats = {
            'hard_samples': 0,
            'contextual_samples': 0,
            'sentiment_samples': 0,
            'popular_samples': 0,
            'random_samples': 0,
            'resampled_hard': 0
        }
        self.contextual_sampler.reset_stats()
        self.sentiment_sampler.reset_stats()
        self.popular_sampler.reset_stats()


# Convenience function
def create_advanced_sampler(
    positive_pairs: np.ndarray,
    user_pos_sets: Dict[int, Set[int]],
    num_items: int,
    hard_neg_sets: Optional[Dict[int, Set[int]]] = None,
    item_embeddings: Optional[np.ndarray] = None,
    item_sentiment_scores: Optional[Dict[int, float]] = None,
    item_popularity: Optional[Dict[int, float]] = None,
    hard_ratio: float = 0.25,
    contextual_ratio: float = 0.20,
    sentiment_ratio: float = 0.15,
    popular_ratio: float = 0.10,
    enable_dynamic: bool = True,
    random_seed: int = 42
) -> AdvancedTripletSampler:
    """
    Create an advanced triplet sampler with configurable strategies.
    
    Args:
        positive_pairs: (N, 2) array of [u_idx, i_idx]
        user_pos_sets: Dict of positive sets per user
        num_items: Total items
        hard_neg_sets: Hard negative sets (rating ≤ 3)
        item_embeddings: BERT embeddings for contextual sampling
        item_sentiment_scores: Sentiment scores for contrast sampling
        item_popularity: Popularity scores for popular sampling
        hard_ratio: Fraction from hard negatives
        contextual_ratio: Fraction from contextual negatives
        sentiment_ratio: Fraction from sentiment contrast
        popular_ratio: Fraction from popular items
        enable_dynamic: Enable dynamic difficulty adjustment
        random_seed: Random seed
    
    Returns:
        Configured AdvancedTripletSampler
    """
    strategy = SamplingStrategy(
        hard_ratio=hard_ratio,
        contextual_ratio=contextual_ratio,
        sentiment_contrast_ratio=sentiment_ratio,
        popular_ratio=popular_ratio
    )
    
    dynamic_config = DynamicSamplingConfig(
        enable_dynamic=enable_dynamic,
        warmup_epochs=5,
        difficulty_schedule='cosine',
        initial_difficulty=0.3,
        final_difficulty=0.8
    )
    
    return AdvancedTripletSampler(
        positive_pairs=positive_pairs,
        user_pos_sets=user_pos_sets,
        num_items=num_items,
        hard_neg_sets=hard_neg_sets,
        item_embeddings=item_embeddings,
        item_sentiment_scores=item_sentiment_scores,
        item_popularity=item_popularity,
        strategy=strategy,
        dynamic_config=dynamic_config,
        random_seed=random_seed
    )
