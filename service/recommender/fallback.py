"""
Cold-Start Fallback Strategies for Recommendation.

This module provides fallback strategies for users with insufficient
data for CF recommendations (cold-start users).

Strategies:
1. Popularity-based: Return top-selling products
2. Item-similarity: Content-based using PhoBERT embeddings
3. Hybrid: Mix of content similarity and popularity

Optimizations:
- LRU caching for user profiles and similarities
- Pre-computed popular items with metadata
- Cached fallback results for repeat requests

Example:
    >>> from service.recommender.fallback import FallbackRecommender
    >>> fallback = FallbackRecommender(loader, phobert_loader)
    >>> recs = fallback.recommend_for_cold_start(user_id=123, topk=10)
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)


def _convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif pd.isna(obj):
        return None
    return obj


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FallbackRecommendation:
    """Single fallback recommendation."""
    product_id: int
    score: float
    rank: int
    fallback_method: str
    content_score: Optional[float] = None
    popularity_score: Optional[float] = None


# ============================================================================
# FallbackRecommender
# ============================================================================

class FallbackRecommender:
    """
    Fallback recommendation strategies for cold-start users.
    
    Provides content-based and popularity-based recommendations
    for users without sufficient CF data.
    
    Optimizations for ~91% cold-start traffic:
    - LRU caching for user profiles
    - Pre-computed popular items
    - Cached fallback results
    
    Example:
        >>> fallback = FallbackRecommender(loader, phobert_loader)
        >>> recs = fallback.recommend(user_history, topk=10)
    """
    
    def __init__(
        self,
        cf_loader: 'CFModelLoader',
        phobert_loader: Optional['PhoBERTEmbeddingLoader'] = None,
        default_content_weight: float = 0.7,
        default_popularity_weight: float = 0.3,
        enable_cache: bool = True
    ):
        """
        Initialize FallbackRecommender.
        
        Args:
            cf_loader: CFModelLoader instance for data access
            phobert_loader: Optional PhoBERTEmbeddingLoader for content similarity
            default_content_weight: Default weight for content score in hybrid
            default_popularity_weight: Default weight for popularity in hybrid
            enable_cache: Enable LRU caching (default True)
        """
        self.loader = cf_loader
        self.phobert_loader = phobert_loader
        self.default_content_weight = default_content_weight
        self.default_popularity_weight = default_popularity_weight
        self.enable_cache = enable_cache
        
        # Cache popularity scores
        self._item_popularity: Optional[Dict[int, float]] = None
        
        # Cache manager reference (lazy-loaded)
        self._cache_manager = None
    
    @property
    def cache(self):
        """Get cache manager (lazy initialization)."""
        if self._cache_manager is None and self.enable_cache:
            try:
                from .cache import get_cache_manager
                self._cache_manager = get_cache_manager()
            except Exception as e:
                logger.debug(f"Cache manager not available: {e}")
        return self._cache_manager
    
    def _ensure_phobert(self) -> bool:
        """Ensure PhoBERT loader is available."""
        if self.phobert_loader is not None:
            return self.phobert_loader.is_loaded()
        
        # Try to lazy-load
        try:
            from service.recommender.phobert_loader import PhoBERTEmbeddingLoader
            self.phobert_loader = PhoBERTEmbeddingLoader()
            return self.phobert_loader.is_loaded()
        except Exception as e:
            logger.warning(f"Could not load PhoBERT embeddings: {e}")
            return False
    
    def _get_item_popularity(self) -> Dict[int, float]:
        """Get item popularity scores from metadata."""
        if self._item_popularity is not None:
            return self._item_popularity
        
        # Load from item metadata
        if self.loader.item_metadata is None:
            self.loader.load_item_metadata()
        
        metadata = self.loader.item_metadata
        if metadata is None:
            return {}
        
        # Use num_sold_time or popularity_score column
        pop_col = None
        for col in ['popularity_score', 'num_sold_time', 'total_sold']:
            if col in metadata.columns:
                pop_col = col
                break
        
        if pop_col is None:
            logger.warning("No popularity column found in metadata")
            return {}
        
        # Build popularity dict
        self._item_popularity = {}
        for _, row in metadata.iterrows():
            pid = row['product_id']
            pop = row[pop_col]
            if pd.notna(pop):
                self._item_popularity[pid] = float(pop)
        
        # Normalize to [0, 1]
        if self._item_popularity:
            max_pop = max(self._item_popularity.values())
            if max_pop > 0:
                self._item_popularity = {
                    k: v / max_pop for k, v in self._item_popularity.items()
                }
        
        return self._item_popularity
    
    def fallback_popularity(
        self,
        topk: int = 10,
        exclude_ids: Optional[Set[int]] = None,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Pure popularity-based recommendations for truly new users.
        
        Optimized: Uses pre-computed popular items from cache.
        
        Args:
            topk: Number of recommendations
            exclude_ids: Product IDs to exclude
            filter_params: Optional attribute filters
        
        Returns:
            List of recommendation dicts
        """
        start_time = time.perf_counter()
        exclude_ids = exclude_ids or set()
        
        # Try to use pre-computed enriched popular items from cache
        if self.cache is not None and filter_params is None:
            cached_popular = self.cache.get_popular_items_enriched()
            if cached_popular:
                # Filter and return quickly
                result = []
                for rec in cached_popular:
                    if len(result) >= topk:
                        break
                    if rec['product_id'] not in exclude_ids:
                        result.append({**rec, 'rank': len(result) + 1})
                
                if result:
                    logger.debug(
                        f"Popularity fallback from cache: {len(result)} items, "
                        f"latency={(time.perf_counter() - start_time)*1000:.1f}ms"
                    )
                    return result
        
        # Get top-K popular item indices
        popular_indices = self.loader.get_popular_items(topk=topk * 2)
        
        if not popular_indices:
            logger.warning("No popular items available")
            return []
        
        # Get item metadata for filtering
        metadata = self.loader.item_metadata
        if metadata is None:
            self.loader.load_item_metadata()
            metadata = self.loader.item_metadata
        
        # Get mappings
        mappings = self.loader.mappings
        if mappings is None:
            self.loader.load_mappings()
            mappings = self.loader.mappings
        
        idx_to_item = mappings.get('idx_to_item', {})
        popularity_scores = self._get_item_popularity()
        
        recommendations = []
        rank = 1
        
        for i_idx in popular_indices:
            if rank > topk:
                break
            
            # Convert index to product_id
            product_id = idx_to_item.get(str(i_idx))
            if product_id is None:
                continue
            
            product_id = int(product_id)
            
            # Skip excluded
            if product_id in exclude_ids:
                continue
            
            # Apply filters
            if filter_params and metadata is not None:
                product_row = metadata[metadata['product_id'] == product_id]
                if product_row.empty:
                    continue
                
                skip = False
                for key, value in filter_params.items():
                    if key in product_row.columns:
                        if product_row[key].iloc[0] != value:
                            skip = True
                            break
                if skip:
                    continue
            
            # Get product info
            pop_score = popularity_scores.get(product_id, 0.5)
            
            rec = {
                'product_id': int(product_id),
                'score': float(pop_score),
                'rank': rank,
                'fallback': True,
                'fallback_method': 'popularity',
            }
            
            # Enrich with metadata
            if metadata is not None:
                product_row = metadata[metadata['product_id'] == product_id]
                if not product_row.empty:
                    for col in ['product_name', 'brand', 'price', 'avg_star']:
                        if col in product_row.columns:
                            val = product_row[col].iloc[0]
                            if pd.notna(val):
                                rec[col] = _convert_numpy_types(val)
            
            recommendations.append(rec)
            rank += 1
        
        return recommendations
    
    def fallback_item_similarity(
        self,
        user_history: List[int],
        topk: int = 10,
        exclude_ids: Optional[Set[int]] = None,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Content-based recommendations using PhoBERT item similarity.
        
        Optimized: Uses cached user profiles and pre-computed similarities.
        
        Args:
            user_history: List of product_ids user has interacted with
            topk: Number of recommendations
            exclude_ids: Product IDs to exclude
            filter_params: Optional attribute filters
        
        Returns:
            List of recommendation dicts
        """
        start_time = time.perf_counter()
        
        if not user_history:
            # No history → pure popularity
            return self.fallback_popularity(topk, exclude_ids, filter_params)
        
        if not self._ensure_phobert():
            logger.warning("PhoBERT not available, falling back to popularity")
            return self.fallback_popularity(topk, exclude_ids, filter_params)
        
        exclude_ids = exclude_ids or set()
        exclude_ids.update(user_history)  # Exclude already purchased
        
        # Try to use cached user profile
        user_profile = None
        user_id_hash = hash(tuple(sorted(user_history)))  # Hash for cache key
        
        if self.cache is not None:
            user_profile = self.cache.get_user_profile(user_id_hash)
        
        # Compute user profile if not cached
        if user_profile is None:
            user_profile = self.phobert_loader.compute_user_profile(
                user_history_items=user_history,
                strategy='mean'
            )
            
            # Cache for future use
            if user_profile is not None and self.cache is not None:
                self.cache.set_user_profile(user_id_hash, user_profile)
        
        if user_profile is None:
            logger.warning("Could not compute user profile, falling back to popularity")
            return self.fallback_popularity(topk, exclude_ids, filter_params)
        
        # Find similar items to user profile
        similar_items = self.phobert_loader.find_similar_to_profile(
            user_profile=user_profile,
            topk=topk * 2,  # Get more for filtering
            exclude_ids=exclude_ids
        )
        
        # Get metadata for filtering and enrichment
        metadata = self.loader.item_metadata
        if metadata is None:
            self.loader.load_item_metadata()
            metadata = self.loader.item_metadata
        
        recommendations = []
        rank = 1
        
        for product_id, similarity in similar_items:
            if rank > topk:
                break
            
            # Apply filters
            if filter_params and metadata is not None:
                product_row = metadata[metadata['product_id'] == product_id]
                if product_row.empty:
                    continue
                
                skip = False
                for key, value in filter_params.items():
                    if key in product_row.columns:
                        if product_row[key].iloc[0] != value:
                            skip = True
                            break
                if skip:
                    continue
            
            rec = {
                'product_id': int(product_id),
                'score': float(similarity),
                'rank': rank,
                'fallback': True,
                'fallback_method': 'item_similarity',
                'content_score': float(similarity),
            }
            
            # Enrich with metadata
            if metadata is not None:
                product_row = metadata[metadata['product_id'] == product_id]
                if not product_row.empty:
                    for col in ['product_name', 'brand', 'price', 'avg_star']:
                        if col in product_row.columns:
                            val = product_row[col].iloc[0]
                            if pd.notna(val):
                                rec[col] = _convert_numpy_types(val)
            
            recommendations.append(rec)
            rank += 1
        
        return recommendations
    
    def hybrid_fallback(
        self,
        user_history: List[int],
        topk: int = 10,
        content_weight: Optional[float] = None,
        popularity_weight: Optional[float] = None,
        exclude_ids: Optional[Set[int]] = None,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid fallback combining content similarity and popularity.
        
        Args:
            user_history: User's purchase history
            topk: Number of recommendations
            content_weight: Weight for content similarity (default 0.7)
            popularity_weight: Weight for popularity (default 0.3)
            exclude_ids: Product IDs to exclude
            filter_params: Optional attribute filters
        
        Returns:
            List of recommendation dicts
        """
        content_weight = content_weight or self.default_content_weight
        popularity_weight = popularity_weight or self.default_popularity_weight
        
        exclude_ids = exclude_ids or set()
        if user_history:
            exclude_ids.update(user_history)
        
        # Get content-based candidates (2x topk for diversity)
        content_recs = self.fallback_item_similarity(
            user_history=user_history,
            topk=topk * 2,
            exclude_ids=exclude_ids,
            filter_params=filter_params
        )
        
        if not content_recs:
            # Fallback to pure popularity
            return self.fallback_popularity(topk, exclude_ids, filter_params)
        
        # Get popularity scores
        popularity_scores = self._get_item_popularity()
        
        # Combine scores
        combined = []
        for rec in content_recs:
            pid = rec['product_id']
            content_score = rec.get('content_score', rec['score'])
            pop_score = popularity_scores.get(pid, 0.5)
            
            # Weighted combination
            final_score = (
                content_weight * content_score + 
                popularity_weight * pop_score
            )
            
            combined.append({
                **rec,
                'score': float(final_score),
                'content_score': float(content_score),
                'popularity_score': float(pop_score),
                'fallback_method': 'hybrid',
            })
        
        # Sort by final score
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top-k and update ranks
        recommendations = combined[:topk]
        for i, rec in enumerate(recommendations):
            rec['rank'] = i + 1
        
        return recommendations
    
    def recommend(
        self,
        user_id: Optional[int] = None,
        user_history: Optional[List[int]] = None,
        topk: int = 10,
        strategy: str = 'hybrid',
        exclude_ids: Optional[Set[int]] = None,
        filter_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate fallback recommendations for cold-start user.
        
        Args:
            user_id: User ID (to fetch history if not provided)
            user_history: User's purchase history (overrides user_id lookup)
            topk: Number of recommendations
            strategy: 'popularity', 'item_similarity', or 'hybrid'
            exclude_ids: Product IDs to exclude
            filter_params: Attribute filters
        
        Returns:
            List of recommendation dicts
        """
        # Get user history if not provided
        if user_history is None and user_id is not None:
            history_set = self.loader.get_user_history(user_id)
            user_history = list(history_set) if history_set else []
        
        user_history = user_history or []
        exclude_ids = exclude_ids or set()
        
        # Add history to exclusions
        if user_history:
            exclude_ids.update(user_history)
        
        # Select strategy
        if strategy == 'popularity':
            return self.fallback_popularity(topk, exclude_ids, filter_params)
        elif strategy == 'item_similarity':
            return self.fallback_item_similarity(
                user_history, topk, exclude_ids, filter_params
            )
        elif strategy == 'hybrid':
            return self.hybrid_fallback(
                user_history, topk,
                content_weight=kwargs.get('content_weight'),
                popularity_weight=kwargs.get('popularity_weight'),
                exclude_ids=exclude_ids,
                filter_params=filter_params
            )
        else:
            raise ValueError(f"Unknown fallback strategy: {strategy}")
