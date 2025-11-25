"""
Recommender Service Package.

This package provides the core recommendation engine and utilities
for serving CF recommendations.

Main components:
- CFModelLoader: Load models and data from registry
- CFRecommender: Main recommendation engine
- FallbackRecommender: Cold-start handling
- PhoBERTEmbeddingLoader: Content-based embeddings
- HybridReranker: Hybrid reranking with multiple signals
- CacheManager: LRU caching and warm-up for cold-start optimization

Example:
    >>> from service.recommender import CFRecommender
    >>> recommender = CFRecommender()
    >>> result = recommender.recommend(user_id=12345, topk=10)
"""

from .loader import CFModelLoader, get_loader, reset_loader
from .recommender import CFRecommender, RecommendationResult
from .fallback import FallbackRecommender
from .phobert_loader import PhoBERTEmbeddingLoader, get_phobert_loader, reset_phobert_loader
from .rerank import (
    HybridReranker, 
    get_reranker, 
    RerankerConfig, 
    RerankedResult,
    rerank_with_signals, 
    rerank_cold_start, 
    diversify_recommendations
)
from .filters import (
    apply_filters, 
    filter_by_brand, 
    filter_by_skin_type, 
    filter_by_price_range,
    boost_by_attributes,
    boost_by_user_preferences,
    infer_user_preferences,
    filter_and_boost
)
from .cache import (
    CacheManager,
    CacheConfig,
    LRUCache,
    get_cache_manager,
    reset_cache_manager,
    async_warmup
)

__all__ = [
    # Loaders
    'CFModelLoader',
    'get_loader',
    'reset_loader',
    'PhoBERTEmbeddingLoader',
    'get_phobert_loader',
    'reset_phobert_loader',
    
    # Core
    'CFRecommender',
    'RecommendationResult',
    'FallbackRecommender',
    
    # Hybrid Reranking (Task 08)
    'HybridReranker',
    'get_reranker',
    'RerankerConfig',
    'RerankedResult',
    
    # Legacy Reranking
    'rerank_with_signals',
    'rerank_cold_start',
    'diversify_recommendations',
    
    # Filtering & Boosting
    'apply_filters',
    'filter_by_brand',
    'filter_by_skin_type',
    'filter_by_price_range',
    'boost_by_attributes',
    'boost_by_user_preferences',
    'infer_user_preferences',
    'filter_and_boost',
    
    # Caching & Warm-up
    'CacheManager',
    'CacheConfig',
    'LRUCache',
    'get_cache_manager',
    'reset_cache_manager',
    'async_warmup',
]
