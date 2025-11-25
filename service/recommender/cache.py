"""
Cache Manager for Recommendation Service.

This module provides LRU caching and warm-up strategies
for optimizing cold-start path latency (~91% of traffic).

Key Features:
- LRU caches for user profiles, item similarities, fallback results
- Pre-computation of popular items and their similarities
- Warm-up strategies for cold-start recommendations
- Cache invalidation hooks for model updates

Usage:
    from service.recommender.cache import CacheManager, get_cache_manager
    cache = get_cache_manager()
    cache.warmup()
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import numpy as np
import threading
import logging
import time
import json

logger = logging.getLogger(__name__)


# ============================================================================
# LRU Cache Implementation
# ============================================================================

class LRUCache:
    """
    Thread-safe LRU cache with TTL support.
    
    Features:
    - O(1) get/put operations
    - Optional TTL for entries
    - Max size enforcement
    - Hit/miss statistics
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        ttl_seconds: Optional[float] = None,
        name: str = "cache"
    ):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            ttl_seconds: Optional time-to-live in seconds
            name: Cache name for logging
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.name = name
        
        self._cache: OrderedDict[Any, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            value, timestamp = self._cache[key]
            
            # Check TTL
            if self.ttl_seconds is not None:
                if time.time() - timestamp > self.ttl_seconds:
                    del self._cache[key]
                    self.misses += 1
                    return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            
            return value
    
    def put(self, key: Any, value: Any) -> None:
        """
        Put value into cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            timestamp = time.time()
            
            if key in self._cache:
                # Update existing
                self._cache[key] = (value, timestamp)
                self._cache.move_to_end(key)
            else:
                # Add new
                self._cache[key] = (value, timestamp)
                
                # Evict if over size
                while len(self._cache) > self.max_size:
                    self._cache.popitem(last=False)
                    self.evictions += 1
    
    def delete(self, key: Any) -> bool:
        """Remove key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'name': self.name,
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions
        }


# ============================================================================
# Cache Manager
# ============================================================================

@dataclass
class CacheConfig:
    """Configuration for cache manager."""
    
    # User profile cache (BERT aggregation)
    user_profile_max_size: int = 50000
    user_profile_ttl_seconds: float = 3600  # 1 hour
    
    # Item similarity cache (top-K similar items)
    item_similarity_max_size: int = 5000
    item_similarity_ttl_seconds: float = 86400  # 24 hours
    
    # Fallback recommendation cache
    fallback_max_size: int = 10000
    fallback_ttl_seconds: float = 1800  # 30 min
    
    # Popular items - refreshed less frequently
    popular_items_ttl_seconds: float = 3600  # 1 hour
    
    # Pre-warm settings
    warmup_num_popular_items: int = 200
    warmup_num_user_profiles: int = 1000
    
    # Cold-start optimization
    precompute_popular_similarities: bool = True


class CacheManager:
    """
    Centralized cache manager for recommendation service.
    
    Manages multiple LRU caches for different components:
    - User BERT profiles
    - Item-item similarities
    - Fallback recommendations
    - Popular items
    
    Example:
        >>> cache = CacheManager()
        >>> cache.warmup()
        >>> profile = cache.get_user_profile(user_id)
    """
    
    _instance: Optional['CacheManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager."""
        if self._initialized:
            return
        
        self.config = config or CacheConfig()
        
        # Initialize caches
        self.user_profile_cache = LRUCache(
            max_size=self.config.user_profile_max_size,
            ttl_seconds=self.config.user_profile_ttl_seconds,
            name="user_profile"
        )
        
        self.item_similarity_cache = LRUCache(
            max_size=self.config.item_similarity_max_size,
            ttl_seconds=self.config.item_similarity_ttl_seconds,
            name="item_similarity"
        )
        
        self.fallback_cache = LRUCache(
            max_size=self.config.fallback_max_size,
            ttl_seconds=self.config.fallback_ttl_seconds,
            name="fallback"
        )
        
        # Pre-computed data
        self._popular_items: Optional[List[int]] = None
        self._popular_items_enriched: Optional[List[Dict]] = None
        self._popular_similarities: Optional[Dict[int, List[Tuple[int, float]]]] = None
        self._popular_items_timestamp: float = 0
        
        # Warm-up status
        self._warmed_up = False
        
        self._initialized = True
        
        logger.info("CacheManager initialized")
    
    # ========================================================================
    # User Profile Cache
    # ========================================================================
    
    def get_user_profile(self, user_id: int) -> Optional[np.ndarray]:
        """Get cached user BERT profile."""
        return self.user_profile_cache.get(user_id)
    
    def set_user_profile(self, user_id: int, profile: np.ndarray) -> None:
        """Cache user BERT profile."""
        self.user_profile_cache.put(user_id, profile)
    
    def compute_and_cache_user_profile(
        self,
        user_id: int,
        user_history: List[int],
        phobert_loader: 'PhoBERTEmbeddingLoader'
    ) -> Optional[np.ndarray]:
        """
        Compute user profile and cache it.
        
        Args:
            user_id: User ID
            user_history: List of product IDs in history
            phobert_loader: PhoBERT embedding loader
        
        Returns:
            User profile embedding or None
        """
        # Check cache first
        cached = self.get_user_profile(user_id)
        if cached is not None:
            return cached
        
        # Compute profile
        if not user_history:
            return None
        
        profile = phobert_loader.compute_user_profile(
            user_history,
            strategy='weighted_mean'
        )
        
        if profile is not None:
            self.set_user_profile(user_id, profile)
        
        return profile
    
    # ========================================================================
    # Item Similarity Cache
    # ========================================================================
    
    def get_similar_items(self, product_id: int) -> Optional[List[Tuple[int, float]]]:
        """Get cached similar items for product."""
        return self.item_similarity_cache.get(product_id)
    
    def set_similar_items(
        self,
        product_id: int,
        similar: List[Tuple[int, float]]
    ) -> None:
        """Cache similar items for product."""
        self.item_similarity_cache.put(product_id, similar)
    
    # ========================================================================
    # Fallback Cache
    # ========================================================================
    
    def get_fallback_recs(
        self,
        user_id: int,
        strategy: str = 'hybrid'
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached fallback recommendations."""
        cache_key = (user_id, strategy)
        return self.fallback_cache.get(cache_key)
    
    def set_fallback_recs(
        self,
        user_id: int,
        recommendations: List[Dict[str, Any]],
        strategy: str = 'hybrid'
    ) -> None:
        """Cache fallback recommendations."""
        cache_key = (user_id, strategy)
        self.fallback_cache.put(cache_key, recommendations)
    
    # ========================================================================
    # Popular Items (Pre-computed)
    # ========================================================================
    
    def get_popular_items(self) -> List[int]:
        """Get popular item IDs."""
        if self._popular_items is None:
            self._load_popular_items()
        return self._popular_items or []
    
    def get_popular_items_enriched(self) -> List[Dict[str, Any]]:
        """Get enriched popular items with metadata."""
        if self._popular_items_enriched is None:
            self._load_popular_items_enriched()
        return self._popular_items_enriched or []
    
    def get_popular_item_similar(
        self,
        product_id: int
    ) -> Optional[List[Tuple[int, float]]]:
        """Get pre-computed similar items for popular products."""
        if self._popular_similarities is not None:
            return self._popular_similarities.get(product_id)
        return None
    
    def _load_popular_items(self) -> None:
        """Load popular items from data."""
        try:
            from service.recommender.loader import get_loader
            loader = get_loader()
            
            self._popular_items = loader.get_popular_items(
                topk=self.config.warmup_num_popular_items
            )
            self._popular_items_timestamp = time.time()
            
            logger.info(f"Loaded {len(self._popular_items)} popular items")
        except Exception as e:
            logger.warning(f"Failed to load popular items: {e}")
            self._popular_items = []
    
    def _load_popular_items_enriched(self) -> None:
        """Load and enrich popular items with metadata."""
        try:
            from service.recommender.loader import get_loader
            import pandas as pd
            
            loader = get_loader()
            popular_indices = loader.get_popular_items(
                topk=self.config.warmup_num_popular_items
            )
            
            if not popular_indices:
                self._popular_items_enriched = []
                return
            
            # Get metadata
            metadata = loader.item_metadata
            if metadata is None:
                loader.load_item_metadata()
                metadata = loader.item_metadata
            
            # Convert indices to product IDs
            idx_to_item = loader.mappings.get('idx_to_item', {})
            
            enriched = []
            for rank, i_idx in enumerate(popular_indices, 1):
                pid = idx_to_item.get(str(i_idx))
                if pid is None:
                    continue
                
                pid = int(pid)
                rec = {
                    'product_id': pid,
                    'rank': rank,
                    'score': 1.0 - (rank - 1) / len(popular_indices),
                    'fallback': True,
                    'fallback_method': 'popularity'
                }
                
                # Add metadata
                if metadata is not None:
                    product_row = metadata[metadata['product_id'] == pid]
                    if not product_row.empty:
                        row = product_row.iloc[0]
                        for col in ['product_name', 'brand', 'price', 'avg_star', 'num_sold_time']:
                            if col in product_row.columns and pd.notna(row[col]):
                                rec[col] = row[col]
                
                enriched.append(rec)
            
            self._popular_items_enriched = enriched
            logger.info(f"Enriched {len(enriched)} popular items")
            
        except Exception as e:
            logger.warning(f"Failed to enrich popular items: {e}")
            self._popular_items_enriched = []
    
    # ========================================================================
    # Warm-up
    # ========================================================================
    
    def warmup(
        self,
        force: bool = False,
        include_similarities: bool = True
    ) -> Dict[str, Any]:
        """
        Pre-warm caches for optimal cold-start performance.
        
        This is CRITICAL for cold-start path (~91% traffic).
        
        Args:
            force: Force re-warmup even if already done
            include_similarities: Pre-compute item similarities
        
        Returns:
            Warmup statistics
        """
        if self._warmed_up and not force:
            logger.info("Cache already warmed up, skipping")
            return {'status': 'already_warmed', 'skipped': True}
        
        start_time = time.perf_counter()
        stats = {
            'status': 'warmup_complete',
            'skipped': False,
            'popular_items': 0,
            'popular_items_enriched': 0,
            'popular_similarities': 0,
            'warmup_duration_ms': 0
        }
        
        logger.info("Starting cache warmup...")
        
        # 1. Load popular items
        self._load_popular_items()
        stats['popular_items'] = len(self._popular_items or [])
        
        # 2. Enrich popular items with metadata
        self._load_popular_items_enriched()
        stats['popular_items_enriched'] = len(self._popular_items_enriched or [])
        
        # 3. Pre-compute similarities for popular items (expensive but critical)
        if include_similarities and self.config.precompute_popular_similarities:
            self._precompute_popular_similarities()
            stats['popular_similarities'] = len(self._popular_similarities or {})
        
        stats['warmup_duration_ms'] = (time.perf_counter() - start_time) * 1000
        self._warmed_up = True
        
        logger.info(
            f"Cache warmup complete: {stats['popular_items']} popular items, "
            f"{stats['popular_similarities']} similarities, "
            f"duration={stats['warmup_duration_ms']:.1f}ms"
        )
        
        return stats
    
    def _precompute_popular_similarities(self) -> None:
        """Pre-compute similar items for popular products."""
        try:
            from service.recommender.phobert_loader import get_phobert_loader
            
            phobert = get_phobert_loader()
            if not phobert.is_loaded():
                logger.warning("PhoBERT not loaded, skipping similarity precomputation")
                return
            
            popular_items = self.get_popular_items()
            if not popular_items:
                return
            
            # Pre-compute top-K similar for each popular item
            self._popular_similarities = {}
            topk = 50
            
            for pid in popular_items[:self.config.warmup_num_popular_items]:
                # Convert index to product_id if needed
                try:
                    from service.recommender.loader import get_loader
                    loader = get_loader()
                    idx_to_item = loader.mappings.get('idx_to_item', {})
                    
                    # pid might be an index, need actual product_id
                    actual_pid = int(idx_to_item.get(str(pid), pid))
                    
                    similar = phobert.find_similar_items(
                        product_id=actual_pid,
                        topk=topk,
                        exclude_self=True
                    )
                    
                    if similar:
                        self._popular_similarities[actual_pid] = similar
                        # Also cache in LRU
                        self.set_similar_items(actual_pid, similar)
                except Exception as e:
                    logger.debug(f"Failed to compute similarity for {pid}: {e}")
            
            logger.info(
                f"Pre-computed similarities for {len(self._popular_similarities)} popular items"
            )
            
        except Exception as e:
            logger.warning(f"Failed to precompute similarities: {e}")
            self._popular_similarities = {}
    
    # ========================================================================
    # Cache Invalidation
    # ========================================================================
    
    def invalidate_user(self, user_id: int) -> None:
        """Invalidate all caches for a user."""
        self.user_profile_cache.delete(user_id)
        
        # Invalidate fallback for all strategies
        for strategy in ['popularity', 'item_similarity', 'hybrid']:
            self.fallback_cache.delete((user_id, strategy))
    
    def invalidate_item(self, product_id: int) -> None:
        """Invalidate caches for an item."""
        self.item_similarity_cache.delete(product_id)
        
        # Clear popular if item is popular
        if self._popular_items and product_id in self._popular_items:
            self._popular_items = None
            self._popular_items_enriched = None
            self._popular_similarities = None
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.user_profile_cache.clear()
        self.item_similarity_cache.clear()
        self.fallback_cache.clear()
        
        self._popular_items = None
        self._popular_items_enriched = None
        self._popular_similarities = None
        self._warmed_up = False
        
        logger.info("All caches cleared")
    
    def on_model_update(self) -> None:
        """Handle model update - clear relevant caches."""
        # Clear item similarities (CF-based)
        self.item_similarity_cache.clear()
        
        # Keep user profiles and fallback (content-based, still valid)
        # Re-warm popular items
        self._load_popular_items()
        self._load_popular_items_enriched()
        
        logger.info("Caches updated for new model")
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'warmed_up': self._warmed_up,
            'caches': {
                'user_profile': self.user_profile_cache.stats(),
                'item_similarity': self.item_similarity_cache.stats(),
                'fallback': self.fallback_cache.stats()
            },
            'precomputed': {
                'popular_items': len(self._popular_items or []),
                'popular_items_enriched': len(self._popular_items_enriched or []),
                'popular_similarities': len(self._popular_similarities or {})
            }
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get singleton cache manager."""
    return CacheManager(config)


def reset_cache_manager() -> None:
    """Reset singleton cache manager."""
    with CacheManager._lock:
        if CacheManager._instance is not None:
            CacheManager._instance.clear_all()
        CacheManager._instance = None


# ============================================================================
# Async Warmup Task
# ============================================================================

async def async_warmup(cache: Optional[CacheManager] = None) -> Dict[str, Any]:
    """
    Async cache warmup for use in FastAPI lifespan.
    
    Args:
        cache: CacheManager instance (uses singleton if None)
    
    Returns:
        Warmup statistics
    """
    import asyncio
    
    if cache is None:
        cache = get_cache_manager()
    
    # Run warmup in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(
        None,
        lambda: cache.warmup(force=False, include_similarities=True)
    )
    
    return stats
