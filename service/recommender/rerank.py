"""
Hybrid Reranking Module.

This module provides reranking utilities for combining CF scores
with content-based signals, popularity, and quality metrics.

Includes:
- HybridReranker: Main class for hybrid reranking
- Normalization utilities (min-max, robust, global)
- Diversity penalty using BERT similarity
- Signal computation (CF, content, popularity, quality)

Example:
    >>> from service.recommender.rerank import HybridReranker, get_reranker
    >>> reranker = get_reranker()  # Singleton
    >>> reranked = reranker.rerank(cf_recs, user_id, user_history)
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging
import time

if TYPE_CHECKING:
    from .phobert_loader import PhoBERTEmbeddingLoader

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RerankerConfig:
    """Configuration for HybridReranker."""
    
    # Weights for trainable users (≥2 interactions)
    weights_trainable: Dict[str, float] = field(default_factory=lambda: {
        'cf': 0.30,         # SECONDARY - Collaborative signal
        'content': 0.40,    # PRIMARY - PhoBERT semantic similarity  
        'popularity': 0.20, # TERTIARY - Trending items
        'quality': 0.10     # BONUS - High-rated products
    })
    
    # Weights for cold-start users (<2 interactions)
    weights_cold_start: Dict[str, float] = field(default_factory=lambda: {
        'content': 0.60,    # DOMINANT - Only reliable signal
        'popularity': 0.30, # Social proof
        'quality': 0.10     # Bonus
    })
    
    # Diversity settings
    diversity_enabled: bool = True
    diversity_penalty: float = 0.1      # Penalty for similar items
    diversity_threshold: float = 0.85   # BERT similarity threshold
    
    # User profile strategy (from PhoBERTEmbeddingLoader)
    user_profile_strategy: str = 'weighted_mean'  # mean, weighted_mean, recency
    
    # Candidate multiplier (generate more for reranking)
    candidate_multiplier: int = 5
    
    # Normalization ranges (global, not local per-request)
    cf_score_min: float = 0.0
    cf_score_max: float = 1.5      # ALS/BPR scores typically in [0, 1.5]
    content_score_min: float = -1.0 # Cosine similarity range
    content_score_max: float = 1.0
    quality_min: float = 1.0        # Rating range
    quality_max: float = 5.0
    popularity_p01: float = 0.0     # From data_stats.json
    popularity_p99: float = 6.0     # From data_stats.json


@dataclass
class RerankedResult:
    """Result of a reranking operation."""
    recommendations: List[Dict[str, Any]]
    latency_ms: float
    diversity_score: float
    weights_used: Dict[str, float]
    num_candidates: int
    num_output: int


# ============================================================================
# HybridReranker Class
# ============================================================================

class HybridReranker:
    """
    Hybrid reranker combining CF, content, popularity, quality signals.
    
    Uses PhoBERTEmbeddingLoader for content similarity and applies
    global normalization for consistent scoring across requests.
    
    Features:
    - Weighted combination of multiple signals
    - Global normalization (not local per-request)
    - Diversity penalty using BERT similarity
    - Adaptive weights based on user interaction count
    - Cold-start user handling
    
    Example:
        >>> reranker = HybridReranker(phobert_loader, item_metadata)
        >>> result = reranker.rerank(cf_recs, user_id, user_history)
    """
    
    _instance: Optional['HybridReranker'] = None
    
    def __init__(
        self,
        phobert_loader: Optional['PhoBERTEmbeddingLoader'] = None,
        item_metadata: Optional[pd.DataFrame] = None,
        config: Optional[RerankerConfig] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize HybridReranker.
        
        Args:
            phobert_loader: PhoBERTEmbeddingLoader instance
            item_metadata: Product metadata DataFrame
            config: RerankerConfig instance
            config_path: Path to config YAML file
        """
        self.phobert_loader = phobert_loader
        self.metadata = item_metadata
        
        # Load or create config
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            self.config = RerankerConfig()
        
        # Load global stats for normalization
        self._load_global_stats()
        
        # Cache for item popularity scores
        self._popularity_cache: Optional[Dict[int, float]] = None
        self._quality_cache: Optional[Dict[int, float]] = None
        
        logger.info(
            f"HybridReranker initialized: "
            f"diversity={self.config.diversity_enabled}, "
            f"strategy={self.config.user_profile_strategy}"
        )
    
    def _load_config(self, config_path: str) -> RerankerConfig:
        """Load config from YAML file."""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            rerank_cfg = data.get('reranking', {})
            
            return RerankerConfig(
                weights_trainable=rerank_cfg.get('weights_trainable', 
                    RerankerConfig().weights_trainable),
                weights_cold_start=rerank_cfg.get('weights_cold_start',
                    RerankerConfig().weights_cold_start),
                diversity_enabled=rerank_cfg.get('diversity', {}).get('enabled', True),
                diversity_penalty=rerank_cfg.get('diversity', {}).get('penalty', 0.1),
                diversity_threshold=rerank_cfg.get('diversity', {}).get('threshold', 0.85),
                user_profile_strategy=rerank_cfg.get('user_profile_strategy', 'weighted_mean'),
                candidate_multiplier=rerank_cfg.get('candidate_multiplier', 5),
            )
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return RerankerConfig()
    
    def _load_global_stats(self) -> None:
        """Load global statistics for normalization from data_stats.json."""
        try:
            stats_path = Path('data/processed/data_stats.json')
            if stats_path.exists():
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                # Extract popularity range
                if 'popularity_range' in stats:
                    pop_range = stats['popularity_range']
                    self.config.popularity_p01 = pop_range.get('min', 0.0)
                    self.config.popularity_p99 = pop_range.get('max', 6.0)
                
                # Extract CF score range if available (from model metadata)
                if 'cf_score_range' in stats:
                    cf_range = stats['cf_score_range']
                    self.config.cf_score_min = cf_range.get('min', 0.0)
                    self.config.cf_score_max = cf_range.get('max', 1.5)
                
                logger.info(
                    f"Loaded global stats: popularity=[{self.config.popularity_p01:.2f}, "
                    f"{self.config.popularity_p99:.2f}]"
                )
            else:
                logger.warning("data_stats.json not found, using default normalization ranges")
        except Exception as e:
            logger.warning(f"Failed to load global stats: {e}")
    
    def _ensure_phobert(self) -> bool:
        """Ensure PhoBERT loader is available."""
        if self.phobert_loader is not None:
            return self.phobert_loader.is_loaded()
        
        try:
            from .phobert_loader import get_phobert_loader
            self.phobert_loader = get_phobert_loader()
            return self.phobert_loader.is_loaded()
        except Exception as e:
            logger.warning(f"Could not load PhoBERT: {e}")
            return False
    
    def _ensure_metadata(self) -> bool:
        """Ensure item metadata is loaded."""
        if self.metadata is not None and not self.metadata.empty:
            return True
        
        try:
            from .loader import get_loader
            loader = get_loader()
            if loader.item_metadata is None:
                loader.load_item_metadata()
            self.metadata = loader.item_metadata
            return self.metadata is not None
        except Exception as e:
            logger.warning(f"Could not load item metadata: {e}")
            return False
    
    def _get_popularity_scores(self) -> Dict[int, float]:
        """Get cached popularity scores."""
        if self._popularity_cache is not None:
            return self._popularity_cache
        
        if not self._ensure_metadata():
            return {}
        
        self._popularity_cache = {}
        
        # Look for popularity column
        pop_col = None
        for col in ['popularity_score', 'num_sold_time', 'total_sold']:
            if col in self.metadata.columns:
                pop_col = col
                break
        
        if pop_col is None:
            return {}
        
        for _, row in self.metadata.iterrows():
            pid = row['product_id']
            pop = row[pop_col]
            if pd.notna(pop):
                self._popularity_cache[int(pid)] = float(pop)
        
        return self._popularity_cache
    
    def _get_quality_scores(self) -> Dict[int, float]:
        """Get cached quality scores."""
        if self._quality_cache is not None:
            return self._quality_cache
        
        if not self._ensure_metadata():
            return {}
        
        self._quality_cache = {}
        
        # Look for quality column
        qual_col = None
        for col in ['quality_score', 'avg_star', 'avg_rating']:
            if col in self.metadata.columns:
                qual_col = col
                break
        
        if qual_col is None:
            return {}
        
        for _, row in self.metadata.iterrows():
            pid = row['product_id']
            qual = row[qual_col]
            if pd.notna(qual):
                self._quality_cache[int(pid)] = float(qual)
        
        return self._quality_cache
    
    def _normalize_global(
        self,
        values: Dict[int, float],
        signal_type: str
    ) -> Dict[int, float]:
        """
        Normalize values using global ranges (not local per-request).
        
        This ensures consistent normalization across different requests.
        
        Args:
            values: Dict of product_id -> raw value
            signal_type: 'cf', 'content', 'popularity', 'quality'
        
        Returns:
            Dict of product_id -> normalized value in [0, 1]
        """
        if not values:
            return {}
        
        # Get ranges based on signal type
        if signal_type == 'cf':
            min_val = self.config.cf_score_min
            max_val = self.config.cf_score_max
        elif signal_type == 'content':
            min_val = self.config.content_score_min
            max_val = self.config.content_score_max
        elif signal_type == 'popularity':
            min_val = self.config.popularity_p01
            max_val = self.config.popularity_p99
        elif signal_type == 'quality':
            min_val = self.config.quality_min
            max_val = self.config.quality_max
        else:
            # Unknown signal, use local min-max
            vals = list(values.values())
            min_val, max_val = min(vals), max(vals)
        
        normalized = {}
        range_val = max_val - min_val
        
        for pid, val in values.items():
            if range_val > 0:
                # Clip to range and normalize
                clipped = np.clip(val, min_val, max_val)
                normalized[pid] = (clipped - min_val) / range_val
            else:
                normalized[pid] = 0.5
        
        return normalized
    
    def _compute_signals(
        self,
        candidate_ids: List[int],
        cf_scores: Dict[int, float],
        user_history: Optional[List[int]] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute all signals for candidates.
        
        Args:
            candidate_ids: List of candidate product IDs
            cf_scores: Dict of product_id -> CF score
            user_history: User's interaction history
        
        Returns:
            Dict of signal_name -> {product_id: score}
        """
        signals = {}
        
        # CF scores (already provided)
        signals['cf'] = {pid: cf_scores.get(pid, 0.0) for pid in candidate_ids}
        
        # Content similarity (PhoBERT)
        signals['content'] = {}
        if user_history and self._ensure_phobert():
            user_profile = self.phobert_loader.compute_user_profile(
                user_history,
                strategy=self.config.user_profile_strategy
            )
            
            if user_profile is not None and len(user_profile) > 0:
                for pid in candidate_ids:
                    emb = self.phobert_loader.get_embedding_normalized(pid)
                    if emb is not None and len(emb) > 0:
                        # Cosine similarity (both vectors should be normalized)
                        sim = float(np.dot(user_profile, emb))
                        signals['content'][pid] = sim
                    else:
                        signals['content'][pid] = 0.0
            else:
                signals['content'] = {pid: 0.0 for pid in candidate_ids}
        else:
            signals['content'] = {pid: 0.0 for pid in candidate_ids}
        
        # Popularity
        pop_scores = self._get_popularity_scores()
        signals['popularity'] = {
            pid: pop_scores.get(pid, 0.0) for pid in candidate_ids
        }
        
        # Quality
        qual_scores = self._get_quality_scores()
        signals['quality'] = {
            pid: qual_scores.get(pid, 3.0) for pid in candidate_ids
        }
        
        return signals
    
    def _normalize_signals(
        self,
        signals: Dict[str, Dict[int, float]]
    ) -> Dict[str, Dict[int, float]]:
        """
        Normalize all signals to [0, 1] using global ranges.
        
        Args:
            signals: Dict of signal_name -> {product_id: raw_score}
        
        Returns:
            Dict of signal_name -> {product_id: normalized_score}
        """
        normalized = {}
        
        for signal_name, scores in signals.items():
            normalized[signal_name] = self._normalize_global(scores, signal_name)
        
        return normalized
    
    def _combine_scores(
        self,
        normalized_signals: Dict[str, Dict[int, float]],
        weights: Dict[str, float]
    ) -> Dict[int, float]:
        """
        Combine normalized signals with weighted sum.
        
        Args:
            normalized_signals: Dict of signal_name -> {product_id: normalized_score}
            weights: Dict of signal_name -> weight
        
        Returns:
            Dict of product_id -> final_score
        """
        # Get all product IDs
        product_ids = set()
        for scores in normalized_signals.values():
            product_ids.update(scores.keys())
        
        final_scores = {}
        for pid in product_ids:
            score = 0.0
            for signal_name, weight in weights.items():
                if signal_name in normalized_signals:
                    score += weight * normalized_signals[signal_name].get(pid, 0.0)
            final_scores[pid] = score
        
        return final_scores
    
    def _apply_diversity_penalty(
        self,
        scores: Dict[int, float],
        candidate_ids: List[int]
    ) -> Tuple[Dict[int, float], float]:
        """
        Apply diversity penalty to reduce similar items in ranking.
        
        Uses MMR-style penalty based on BERT similarity.
        
        Args:
            scores: Dict of product_id -> score
            candidate_ids: Ordered list of candidate IDs
        
        Returns:
            Tuple of (penalized_scores, diversity_score)
        """
        if not self.config.diversity_enabled or not self._ensure_phobert():
            return scores, 0.0
        
        penalty = self.config.diversity_penalty
        threshold = self.config.diversity_threshold
        
        penalized = scores.copy()
        
        # Sort by score
        sorted_ids = sorted(candidate_ids, key=lambda x: scores.get(x, 0), reverse=True)
        
        selected = []
        pairwise_sims = []
        
        for pid in sorted_ids:
            # Get embedding for current product
            emb_pid = self.phobert_loader.get_embedding_normalized(pid)
            
            if emb_pid is None or len(emb_pid) == 0:
                selected.append(pid)
                continue
            
            if selected:
                # Compute max similarity to already selected items
                max_sim = 0.0
                for sel_pid in selected:
                    emb_sel = self.phobert_loader.get_embedding_normalized(sel_pid)
                    if emb_sel is not None and len(emb_sel) > 0:
                        sim = float(np.dot(emb_pid, emb_sel))
                        pairwise_sims.append(sim)
                        max_sim = max(max_sim, sim)
                
                # Apply penalty if too similar
                if max_sim > threshold:
                    penalized[pid] *= (1 - penalty * (max_sim - threshold) / (1 - threshold))
            
            selected.append(pid)
        
        # Compute diversity score (1 - avg similarity)
        diversity_score = 1.0 - np.mean(pairwise_sims) if pairwise_sims else 1.0
        
        return penalized, diversity_score
    
    def rerank(
        self,
        cf_recommendations: List[Dict[str, Any]],
        user_id: Optional[int] = None,
        user_history: Optional[List[int]] = None,
        topk: Optional[int] = None,
        is_cold_start: bool = False
    ) -> RerankedResult:
        """
        Rerank CF recommendations with hybrid signals.
        
        Args:
            cf_recommendations: List of recommendation dicts from CFRecommender
            user_id: User ID for logging
            user_history: User's interaction history for content similarity
            topk: Number of items to return (None = all)
            is_cold_start: Whether user is cold-start (uses different weights)
        
        Returns:
            RerankedResult with reranked recommendations
        """
        start_time = time.perf_counter()
        
        if not cf_recommendations:
            return RerankedResult(
                recommendations=[],
                latency_ms=0,
                diversity_score=0,
                weights_used={},
                num_candidates=0,
                num_output=0
            )
        
        num_candidates = len(cf_recommendations)
        
        # Extract candidate IDs and CF scores
        candidate_ids = [rec['product_id'] for rec in cf_recommendations]
        cf_scores = {rec['product_id']: rec.get('score', 0.0) for rec in cf_recommendations}
        
        # Select weights based on user type
        if is_cold_start:
            weights = self.config.weights_cold_start
        else:
            weights = self.config.weights_trainable
        
        # Compute signals
        signals = self._compute_signals(candidate_ids, cf_scores, user_history)
        
        # Normalize signals using global ranges
        normalized = self._normalize_signals(signals)
        
        # Combine scores
        final_scores = self._combine_scores(normalized, weights)
        
        # Apply diversity penalty
        final_scores, diversity_score = self._apply_diversity_penalty(
            final_scores, candidate_ids
        )
        
        # Update recommendations with signal info
        for rec in cf_recommendations:
            pid = rec['product_id']
            rec['cf_score'] = rec.get('score', 0.0)
            rec['final_score'] = final_scores.get(pid, 0.0)
            rec['signals'] = {
                name: signals[name].get(pid, 0.0)
                for name in signals.keys()
            }
            rec['signals_normalized'] = {
                name: normalized[name].get(pid, 0.0)
                for name in normalized.keys()
            }
        
        # Sort by final score
        cf_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Update ranks
        for i, rec in enumerate(cf_recommendations):
            rec['rank'] = i + 1
        
        # Truncate to topk if specified
        if topk is not None:
            cf_recommendations = cf_recommendations[:topk]
        
        latency = (time.perf_counter() - start_time) * 1000
        
        logger.debug(
            f"Reranked {num_candidates} candidates for user {user_id}: "
            f"output={len(cf_recommendations)}, diversity={diversity_score:.3f}, "
            f"latency={latency:.1f}ms"
        )
        
        return RerankedResult(
            recommendations=cf_recommendations,
            latency_ms=latency,
            diversity_score=diversity_score,
            weights_used=weights,
            num_candidates=num_candidates,
            num_output=len(cf_recommendations)
        )
    
    def rerank_cold_start(
        self,
        recommendations: List[Dict[str, Any]],
        user_history: Optional[List[int]] = None,
        topk: Optional[int] = None
    ) -> RerankedResult:
        """
        Rerank cold-start recommendations (content + popularity focus).
        
        Args:
            recommendations: Fallback recommendations
            user_history: User's sparse history (if any)
            topk: Number of items to return
        
        Returns:
            RerankedResult with reranked recommendations
        """
        return self.rerank(
            cf_recommendations=recommendations,
            user_history=user_history,
            topk=topk,
            is_cold_start=True
        )
    
    def update_config(
        self,
        weights_trainable: Optional[Dict[str, float]] = None,
        weights_cold_start: Optional[Dict[str, float]] = None,
        diversity_enabled: Optional[bool] = None,
        diversity_penalty: Optional[float] = None,
        diversity_threshold: Optional[float] = None
    ) -> None:
        """
        Update reranker configuration dynamically.
        
        Args:
            weights_trainable: New weights for trainable users
            weights_cold_start: New weights for cold-start users
            diversity_enabled: Enable/disable diversity
            diversity_penalty: Diversity penalty factor
            diversity_threshold: Similarity threshold for penalty
        """
        if weights_trainable is not None:
            self.config.weights_trainable = weights_trainable
        if weights_cold_start is not None:
            self.config.weights_cold_start = weights_cold_start
        if diversity_enabled is not None:
            self.config.diversity_enabled = diversity_enabled
        if diversity_penalty is not None:
            self.config.diversity_penalty = diversity_penalty
        if diversity_threshold is not None:
            self.config.diversity_threshold = diversity_threshold
        
        logger.info(f"Reranker config updated: {self.config}")
    
    def clear_cache(self) -> None:
        """Clear cached popularity and quality scores."""
        self._popularity_cache = None
        self._quality_cache = None
        logger.info("Reranker cache cleared")


# Singleton accessor
_reranker_instance: Optional[HybridReranker] = None


def get_reranker(
    phobert_loader: Optional['PhoBERTEmbeddingLoader'] = None,
    item_metadata: Optional[pd.DataFrame] = None,
    config_path: Optional[str] = None
) -> HybridReranker:
    """
    Get singleton HybridReranker instance.
    
    Args:
        phobert_loader: Optional PhoBERTEmbeddingLoader
        item_metadata: Optional item metadata DataFrame
        config_path: Optional path to config YAML
    
    Returns:
        HybridReranker singleton
    """
    global _reranker_instance
    
    if _reranker_instance is None:
        _reranker_instance = HybridReranker(
            phobert_loader=phobert_loader,
            item_metadata=item_metadata,
            config_path=config_path
        )
    
    return _reranker_instance


# ============================================================================
# Normalization Helpers
# ============================================================================

def min_max_normalize(values: List[float]) -> List[float]:
    """
    Normalize values to [0, 1] using min-max scaling.
    
    Args:
        values: List of numeric values
    
    Returns:
        Normalized values
    """
    if not values:
        return []
    
    arr = np.array(values)
    v_min, v_max = arr.min(), arr.max()
    
    if v_max > v_min:
        return ((arr - v_min) / (v_max - v_min)).tolist()
    
    return [0.5] * len(values)


def robust_normalize(
    values: List[float],
    p01: Optional[float] = None,
    p99: Optional[float] = None
) -> List[float]:
    """
    Normalize values using robust percentile scaling.
    
    Args:
        values: List of numeric values
        p01: 1st percentile (computed if None)
        p99: 99th percentile (computed if None)
    
    Returns:
        Normalized values clipped to [0, 1]
    """
    if not values:
        return []
    
    arr = np.array(values)
    
    if p01 is None:
        p01 = np.percentile(arr, 1)
    if p99 is None:
        p99 = np.percentile(arr, 99)
    
    if p99 > p01:
        normalized = (arr - p01) / (p99 - p01)
        return np.clip(normalized, 0, 1).tolist()
    
    return [0.5] * len(values)


# ============================================================================
# Reranking Functions
# ============================================================================

def rerank_with_signals(
    recommendations: List[Dict[str, Any]],
    user_id: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    score_range: Optional[Dict[str, float]] = None,
    phobert_loader: Optional['PhoBERTEmbeddingLoader'] = None
) -> List[Dict[str, Any]]:
    """
    Rerank recommendations using weighted combination of signals.
    
    Formula:
        final_score = α * CF_score + β * popularity + γ * quality + δ * content
    
    Args:
        recommendations: List of recommendation dicts with 'score', 'product_id'
        user_id: Optional user ID for personalized content similarity
        weights: Dict with signal weights, e.g.:
            {'cf': 0.5, 'popularity': 0.2, 'quality': 0.2, 'content': 0.1}
        score_range: Optional CF score normalization range
        phobert_loader: Optional PhoBERT loader for content similarity
    
    Returns:
        Reranked recommendations with 'final_score' and updated 'rank'
    """
    if not recommendations:
        return []
    
    # Default weights
    if weights is None:
        weights = {
            'cf': 0.6,
            'popularity': 0.2,
            'quality': 0.2,
            'content': 0.0
        }
    
    # Extract signals
    cf_scores = [rec.get('score', 0) for rec in recommendations]
    popularity_scores = [rec.get('num_sold_time', 0) or 0 for rec in recommendations]
    quality_scores = [rec.get('avg_star', 3.0) or 3.0 for rec in recommendations]
    
    # Normalize signals
    if score_range:
        cf_normalized = robust_normalize(
            cf_scores,
            p01=score_range.get('p01'),
            p99=score_range.get('p99')
        )
    else:
        cf_normalized = min_max_normalize(cf_scores)
    
    popularity_normalized = min_max_normalize(popularity_scores)
    
    # Quality scores are already in [1, 5], normalize to [0, 1]
    quality_normalized = [(q - 1) / 4 for q in quality_scores]
    
    # Content similarity (optional)
    content_scores = [0.0] * len(recommendations)
    
    if weights.get('content', 0) > 0 and user_id is not None and phobert_loader is not None:
        try:
            # Get user profile from loader
            from service.recommender.loader import get_loader
            loader = get_loader()
            user_history = list(loader.get_user_history(user_id))
            
            if user_history:
                user_profile = phobert_loader.compute_user_profile(user_history)
                
                if user_profile is not None:
                    for i, rec in enumerate(recommendations):
                        emb = phobert_loader.get_embedding_normalized(rec['product_id'])
                        if emb is not None:
                            sim = float(np.dot(user_profile, emb) / (
                                np.linalg.norm(user_profile) + 1e-9
                            ))
                            content_scores[i] = max(0, sim)  # Clip negatives
        except Exception as e:
            logger.warning(f"Content similarity computation failed: {e}")
    
    content_normalized = min_max_normalize(content_scores) if any(content_scores) else content_scores
    
    # Compute final scores
    for i, rec in enumerate(recommendations):
        final_score = (
            weights.get('cf', 0) * cf_normalized[i] +
            weights.get('popularity', 0) * popularity_normalized[i] +
            weights.get('quality', 0) * quality_normalized[i] +
            weights.get('content', 0) * content_normalized[i]
        )
        
        rec['final_score'] = final_score
        rec['cf_score_normalized'] = cf_normalized[i]
        rec['popularity_score_normalized'] = popularity_normalized[i]
        rec['quality_score_normalized'] = quality_normalized[i]
        
        if weights.get('content', 0) > 0:
            rec['content_score_normalized'] = content_normalized[i]
    
    # Sort by final score (descending)
    recommendations.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Update ranks
    for i, rec in enumerate(recommendations):
        rec['rank'] = i + 1
    
    return recommendations


def rerank_cold_start(
    recommendations: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Rerank cold-start recommendations (content + popularity mix).
    
    Args:
        recommendations: Fallback recommendations
        weights: Optional weights for content and popularity
    
    Returns:
        Reranked recommendations
    """
    if not recommendations:
        return []
    
    if weights is None:
        weights = {
            'content': 0.6,
            'popularity': 0.3,
            'quality': 0.1
        }
    
    # Extract signals
    content_scores = [rec.get('content_score', rec.get('score', 0)) for rec in recommendations]
    popularity_scores = [rec.get('popularity_score', 0) for rec in recommendations]
    quality_scores = [rec.get('avg_star', 3.0) or 3.0 for rec in recommendations]
    
    # Normalize
    content_normalized = min_max_normalize(content_scores)
    popularity_normalized = min_max_normalize(popularity_scores)
    quality_normalized = [(q - 1) / 4 for q in quality_scores]
    
    # Compute final scores
    for i, rec in enumerate(recommendations):
        final_score = (
            weights.get('content', 0.6) * content_normalized[i] +
            weights.get('popularity', 0.3) * popularity_normalized[i] +
            weights.get('quality', 0.1) * quality_normalized[i]
        )
        
        rec['final_score'] = final_score
    
    # Sort and update ranks
    recommendations.sort(key=lambda x: x['final_score'], reverse=True)
    
    for i, rec in enumerate(recommendations):
        rec['rank'] = i + 1
    
    return recommendations


def diversify_recommendations(
    recommendations: List[Dict[str, Any]],
    diversity_key: str = 'brand',
    max_per_key: int = 3
) -> List[Dict[str, Any]]:
    """
    Diversify recommendations by limiting items per category/brand.
    
    Args:
        recommendations: Ranked recommendations
        diversity_key: Column to diversify on (e.g., 'brand')
        max_per_key: Maximum items per key value
    
    Returns:
        Diversified recommendations
    """
    if not recommendations:
        return []
    
    key_counts: Dict[Any, int] = {}
    diversified = []
    deferred = []
    
    for rec in recommendations:
        key_value = rec.get(diversity_key)
        
        if key_value is None:
            diversified.append(rec)
            continue
        
        current_count = key_counts.get(key_value, 0)
        
        if current_count < max_per_key:
            diversified.append(rec)
            key_counts[key_value] = current_count + 1
        else:
            deferred.append(rec)
    
    # Append deferred items at the end
    result = diversified + deferred
    
    # Update ranks
    for i, rec in enumerate(result):
        rec['rank'] = i + 1
    
    return result
