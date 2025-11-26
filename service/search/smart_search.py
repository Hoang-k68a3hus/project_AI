"""
Smart Search Service.

Main service class for semantic product search using PhoBERT embeddings.
Provides text-to-product discovery, similar items search, and user profile-based search.

Example:
    >>> from service.search import get_search_service
    >>> service = get_search_service()
    >>> results = service.search("kem dưỡng da cho da dầu", topk=10)
    >>> similar = service.search_similar(product_id=123, topk=10)
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import time
import threading
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SearchResult:
    """Single search result."""
    product_id: int
    product_name: str
    semantic_score: float
    final_score: float
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    avg_rating: Optional[float] = None
    num_sold: Optional[int] = None
    signals: Optional[Dict[str, float]] = None
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SearchResponse:
    """Search response container."""
    query: str
    results: List[SearchResult]
    count: int
    latency_ms: float
    method: str  # 'semantic', 'hybrid', 'similar_items', 'user_profile', 'popular'
    filters_applied: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with results as dicts."""
        return {
            'query': self.query,
            'results': [r.to_dict() for r in self.results],
            'count': self.count,
            'latency_ms': self.latency_ms,
            'method': self.method,
            'filters_applied': self.filters_applied
        }


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'default_topk': 10,
    'max_topk': 100,
    'min_semantic_score': 0.25,  # Minimum score to include in results
    'enable_rerank': True,
    'candidate_multiplier': 3,   # Fetch 3x candidates for reranking
    
    # Reranking weights
    'rerank_weights': {
        'semantic': 0.50,
        'popularity': 0.25,
        'quality': 0.15,
        'recency': 0.10
    },
    
    # Normalization config for signals
    'normalization': {
        'popularity': {
            'method': 'log',      # 'log', 'minmax', 'percentile'
            'max_value': 100000,  # Cap for log normalization
        },
        'quality': {
            'method': 'linear',
            'min_value': 1.0,
            'max_value': 5.0
        }
    },
    
    # User profile config
    'user_profile': {
        'strategy': 'weighted_mean',  # 'mean', 'weighted_mean', 'max'
        'max_history_items': 50       # Limit history items for profile
    }
}


# ============================================================================
# Smart Search Service
# ============================================================================

class SmartSearchService:
    """
    Smart Search Service for semantic product discovery.
    
    Features:
    - Text-to-product semantic search using PhoBERT
    - Item-to-item similarity search
    - User profile-based recommendations
    - Hybrid search with attribute filters
    - Multi-signal reranking (semantic, popularity, quality)
    
    Integration Points:
    - QueryEncoder: Encode text queries to embeddings
    - SearchIndex: Fast similarity search index
    - PhoBERTEmbeddingLoader: Product embeddings
    - Product metadata: Attribute filtering & enrichment
    
    Example:
        >>> service = SmartSearchService()
        >>> results = service.search("kem dưỡng da cho da dầu", topk=10)
        >>> similar = service.search_similar(product_id=123, topk=10)
    """
    
    def __init__(
        self,
        query_encoder=None,
        search_index=None,
        phobert_loader=None,
        product_metadata: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SmartSearchService.
        
        Args:
            query_encoder: QueryEncoder instance (lazy loaded if None)
            search_index: SearchIndex instance (lazy loaded if None)
            phobert_loader: PhoBERTEmbeddingLoader instance (lazy loaded if None)
            product_metadata: DataFrame with product info
            config: Service configuration (merged with defaults)
        """
        self.query_encoder = query_encoder
        self.search_index = search_index
        self.phobert_loader = phobert_loader
        self.product_metadata = product_metadata
        
        # Merge config with defaults
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._deep_update(self.config, config)
        
        # State
        self._initialized = False
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'searches_performed': 0,
            'similar_searches': 0,
            'profile_searches': 0,
            'total_latency_ms': 0.0,
            'errors': 0
        }
        
        # Product metadata lookup cache
        self._metadata_cache: Dict[int, Dict[str, Any]] = {}
    
    def _deep_update(self, base: dict, update: dict) -> None:
        """Deep update dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def initialize(self) -> None:
        """
        Initialize all components.
        
        This is called automatically on first use, but can be called
        explicitly for eager initialization.
        """
        with self._lock:
            if self._initialized:
                return
            
            self._initialize_internal()
    
    def _initialize_internal(self) -> None:
        """Internal initialization logic."""
        start = time.perf_counter()
        
        logger.info("Initializing SmartSearchService...")
        
        # Initialize PhoBERT loader (shared with other components)
        if self.phobert_loader is None:
            from service.recommender.phobert_loader import get_phobert_loader
            self.phobert_loader = get_phobert_loader()
            logger.info("PhoBERT loader initialized")
        
        # Initialize query encoder
        if self.query_encoder is None:
            from service.search.query_encoder import get_query_encoder
            self.query_encoder = get_query_encoder()
            logger.info("Query encoder initialized")
        
        # Initialize search index
        if self.search_index is None:
            from service.search.search_index import SearchIndex
            self.search_index = SearchIndex(
                phobert_loader=self.phobert_loader,
                product_metadata=self.product_metadata,
                use_faiss=False  # Use exact search for <10K products
            )
        
        # Build search index
        if not self.search_index.is_initialized:
            self.search_index.build_index()
            logger.info("Search index built")
        
        # Load product metadata if not provided
        if self.product_metadata is None:
            self._load_product_metadata()
        
        # Build metadata cache for fast lookup
        if self.product_metadata is not None:
            self._build_metadata_cache()
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"SmartSearchService initialized in {elapsed:.1f}ms")
        
        self._initialized = True
    
    def _load_product_metadata(self) -> None:
        """Load product metadata from default paths."""
        metadata_paths = [
            Path("data/processed/product_attributes_enriched.parquet"),
            Path("data/published_data/data_product.csv"),
        ]
        
        for path in metadata_paths:
            if path.exists():
                try:
                    if path.suffix == '.parquet':
                        self.product_metadata = pd.read_parquet(path)
                    else:
                        self.product_metadata = pd.read_csv(path, encoding='utf-8')
                    
                    logger.info(f"Loaded product metadata from {path}: {len(self.product_metadata)} products")
                    
                    # Update search index with metadata
                    if self.search_index is not None:
                        self.search_index.product_metadata = self.product_metadata
                    
                    return
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {path}: {e}")
        
        logger.warning("No product metadata found. Search enrichment disabled.")
    
    def _build_metadata_cache(self) -> None:
        """Build fast lookup cache for product metadata."""
        if self.product_metadata is None:
            return
        
        for _, row in self.product_metadata.iterrows():
            pid = row.get('product_id')
            if pid is None:
                continue
            
            pid = int(pid)
            self._metadata_cache[pid] = {
                'product_name': str(row.get('product_name', row.get('name', ''))),
                'brand': row.get('brand'),
                'category': row.get('type', row.get('category')),
                'price': row.get('price', row.get('price_sale')),
                'avg_rating': row.get('avg_star', row.get('avg_rating')),
                'num_sold': row.get('num_sold_time', row.get('num_sold')),
                'popularity_score': row.get('popularity_score'),
                'quality_score': row.get('quality_score'),
            }
        
        logger.debug(f"Metadata cache built for {len(self._metadata_cache)} products")
    
    def search(
        self,
        query: str,
        topk: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        exclude_ids: Optional[Set[int]] = None,
        rerank: bool = True
    ) -> SearchResponse:
        """
        Semantic search for products.
        
        Args:
            query: Text query in Vietnamese (e.g., "kem dưỡng da cho da dầu")
            topk: Number of results to return
            filters: Attribute filters:
                - 'brand': Brand name (string)
                - 'category': Category name (string)
                - 'min_price': Minimum price (float)
                - 'max_price': Maximum price (float)
            exclude_ids: Product IDs to exclude from results
            rerank: Apply multi-signal reranking
        
        Returns:
            SearchResponse with ranked results
        
        Example:
            >>> results = service.search("kem dưỡng ẩm cho da khô", topk=10)
            >>> results = service.search("sữa rửa mặt", filters={'brand': 'innisfree'})
        """
        self.initialize()
        
        start = time.perf_counter()
        
        try:
            # Validate topk
            topk = min(max(1, topk), self.config['max_topk'])
            
            # Encode query to embedding
            query_embedding = self.query_encoder.encode(query, normalize=True)
            
            # Determine candidate count for reranking
            candidate_k = topk * self.config['candidate_multiplier'] if rerank else topk
            
            # Search
            if filters:
                raw_results = self.search_index.search_with_filter(
                    query_embedding, candidate_k, filters, exclude_ids
                )
            else:
                raw_results = self.search_index.search(
                    query_embedding, candidate_k, exclude_ids
                )
            
            # Filter by minimum semantic score
            min_score = self.config['min_semantic_score']
            raw_results = [(pid, score) for pid, score in raw_results if score >= min_score]
            
            # Rerank if enabled and have results
            if rerank and raw_results:
                results = self._rerank_results(raw_results, topk)
            else:
                results = self._create_results(raw_results[:topk])
            
            # Assign ranks
            for i, result in enumerate(results):
                result.rank = i + 1
            
            latency = (time.perf_counter() - start) * 1000
            
            # Update stats
            self._stats['searches_performed'] += 1
            self._stats['total_latency_ms'] += latency
            
            method = 'hybrid' if filters else 'semantic'
            
            return SearchResponse(
                query=query,
                results=results,
                count=len(results),
                latency_ms=latency,
                method=method,
                filters_applied=filters
            )
        
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            self._stats['errors'] += 1
            
            latency = (time.perf_counter() - start) * 1000
            return SearchResponse(
                query=query,
                results=[],
                count=0,
                latency_ms=latency,
                method='error',
                filters_applied=filters
            )
    
    def search_similar(
        self,
        product_id: int,
        topk: int = 10,
        exclude_self: bool = True,
        exclude_ids: Optional[Set[int]] = None
    ) -> SearchResponse:
        """
        Find products similar to a given product.
        
        Uses PhoBERT embeddings for semantic similarity.
        
        Args:
            product_id: Source product ID
            topk: Number of similar products to return
            exclude_self: Exclude source product from results
            exclude_ids: Additional IDs to exclude
        
        Returns:
            SearchResponse with similar products
        
        Example:
            >>> similar = service.search_similar(product_id=123, topk=10)
        """
        self.initialize()
        
        start = time.perf_counter()
        query_str = f"similar_to:{product_id}"
        
        try:
            # Get product embedding
            product_emb = self.phobert_loader.get_embedding_normalized(product_id)
            
            if product_emb is None:
                logger.warning(f"Product {product_id} not found in embeddings")
                return SearchResponse(
                    query=query_str,
                    results=[],
                    count=0,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    method='similar_items'
                )
            
            # Build exclusion set
            exclusions = set(exclude_ids or [])
            if exclude_self:
                exclusions.add(product_id)
            
            # Search
            raw_results = self.search_index.search(product_emb, topk, exclusions)
            
            # Create results
            results = self._create_results(raw_results)
            
            # Assign ranks
            for i, result in enumerate(results):
                result.rank = i + 1
            
            latency = (time.perf_counter() - start) * 1000
            
            # Update stats
            self._stats['similar_searches'] += 1
            self._stats['total_latency_ms'] += latency
            
            return SearchResponse(
                query=query_str,
                results=results,
                count=len(results),
                latency_ms=latency,
                method='similar_items'
            )
        
        except Exception as e:
            logger.error(f"Similar search error for product {product_id}: {e}")
            self._stats['errors'] += 1
            
            return SearchResponse(
                query=query_str,
                results=[],
                count=0,
                latency_ms=(time.perf_counter() - start) * 1000,
                method='error'
            )
    
    def search_by_user_profile(
        self,
        user_history: List[int],
        topk: int = 10,
        exclude_history: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        weights: Optional[List[float]] = None
    ) -> SearchResponse:
        """
        Search products similar to user's interaction history.
        
        Computes a user profile embedding from history and finds similar products.
        Useful for cold-start personalization based on browsing history.
        
        Args:
            user_history: List of product IDs user has interacted with
            topk: Number of results
            exclude_history: Exclude products from history in results
            filters: Attribute filters
            weights: Optional weights for each history item (e.g., recency, rating)
        
        Returns:
            SearchResponse with personalized recommendations
        
        Example:
            >>> results = service.search_by_user_profile(
            ...     user_history=[123, 456, 789],
            ...     topk=10,
            ...     exclude_history=True
            ... )
        """
        self.initialize()
        
        start = time.perf_counter()
        query_str = f"user_profile:{len(user_history)}_items"
        
        try:
            if not user_history:
                # Return popular items as fallback
                return self._get_popular_items(topk, filters)
            
            # Limit history items
            max_items = self.config['user_profile']['max_history_items']
            if len(user_history) > max_items:
                user_history = user_history[-max_items:]  # Keep most recent
            
            # Compute user profile embedding
            strategy = self.config['user_profile']['strategy']
            profile_emb = self.phobert_loader.compute_user_profile(
                user_history,
                weights=weights,
                strategy=strategy
            )
            
            if profile_emb is None:
                logger.warning("Could not compute user profile embedding")
                return self._get_popular_items(topk, filters)
            
            # Normalize profile embedding
            norm = np.linalg.norm(profile_emb)
            if norm > 0:
                profile_emb = profile_emb / norm
            
            # Build exclusions
            exclusions = set(user_history) if exclude_history else None
            
            # Search
            if filters:
                raw_results = self.search_index.search_with_filter(
                    profile_emb, topk, filters, exclusions
                )
            else:
                raw_results = self.search_index.search(
                    profile_emb, topk, exclusions
                )
            
            # Create results
            results = self._create_results(raw_results)
            
            # Assign ranks
            for i, result in enumerate(results):
                result.rank = i + 1
            
            latency = (time.perf_counter() - start) * 1000
            
            # Update stats
            self._stats['profile_searches'] += 1
            self._stats['total_latency_ms'] += latency
            
            return SearchResponse(
                query=query_str,
                results=results,
                count=len(results),
                latency_ms=latency,
                method='user_profile',
                filters_applied=filters
            )
        
        except Exception as e:
            logger.error(f"User profile search error: {e}")
            self._stats['errors'] += 1
            
            return SearchResponse(
                query=query_str,
                results=[],
                count=0,
                latency_ms=(time.perf_counter() - start) * 1000,
                method='error',
                filters_applied=filters
            )
    
    def _rerank_results(
        self,
        raw_results: List[Tuple[int, float]],
        topk: int
    ) -> List[SearchResult]:
        """
        Rerank results using multiple signals.
        
        Signals:
        - semantic: Embedding similarity (from search)
        - popularity: num_sold_time or popularity_score
        - quality: avg_rating or quality_score
        - recency: Product freshness (placeholder)
        """
        weights = self.config['rerank_weights']
        norm_config = self.config['normalization']
        
        results_with_scores = []
        
        for pid, semantic_score in raw_results:
            signals = {'semantic': semantic_score}
            
            # Get metadata
            meta = self._metadata_cache.get(pid, {})
            
            # Popularity signal
            popularity = meta.get('popularity_score') or meta.get('num_sold', 0)
            if popularity and popularity > 0:
                # Log normalization for popularity
                max_pop = norm_config['popularity']['max_value']
                if max_pop > 0:
                    signals['popularity'] = min(np.log1p(float(popularity)) / np.log1p(max_pop), 1.0)
                else:
                    signals['popularity'] = 0.0
            else:
                signals['popularity'] = 0.0
            
            # Quality signal
            quality = meta.get('quality_score') or meta.get('avg_rating')
            if quality is not None and not (isinstance(quality, float) and np.isnan(quality)):
                # Linear normalization for ratings (1-5 → 0-1)
                min_q = norm_config['quality']['min_value']
                max_q = norm_config['quality']['max_value']
                # Avoid division by zero
                if max_q > min_q:
                    signals['quality'] = (float(quality) - min_q) / (max_q - min_q)
                else:
                    signals['quality'] = 0.5  # Default if min == max
            else:
                signals['quality'] = 0.5  # Default middle value
            
            # Recency signal (placeholder - needs actual product launch date)
            signals['recency'] = 0.5
            
            # Compute weighted final score
            final_score = sum(
                weights.get(signal, 0) * value
                for signal, value in signals.items()
            )
            
            # Create result
            result = self._create_result(pid, semantic_score, final_score, signals)
            results_with_scores.append(result)
        
        # Sort by final score descending
        results_with_scores.sort(key=lambda x: x.final_score, reverse=True)
        
        return results_with_scores[:topk]
    
    def _create_results(
        self,
        raw_results: List[Tuple[int, float]]
    ) -> List[SearchResult]:
        """Create SearchResult objects from raw (product_id, score) tuples."""
        results = []
        for pid, score in raw_results:
            result = self._create_result(pid, score, score, {'semantic': score})
            results.append(result)
        return results
    
    def _create_result(
        self,
        product_id: int,
        semantic_score: float,
        final_score: float,
        signals: Dict[str, float]
    ) -> SearchResult:
        """Create a single SearchResult with metadata enrichment."""
        meta = self._metadata_cache.get(product_id, {})
        
        # Handle NaN values
        price = meta.get('price')
        avg_rating = meta.get('avg_rating')
        num_sold = meta.get('num_sold')
        
        if isinstance(price, float) and np.isnan(price):
            price = None
        if isinstance(avg_rating, float) and np.isnan(avg_rating):
            avg_rating = None
        if isinstance(num_sold, float) and np.isnan(num_sold):
            num_sold = None
        
        return SearchResult(
            product_id=product_id,
            product_name=meta.get('product_name', ''),
            semantic_score=semantic_score,
            final_score=final_score,
            brand=meta.get('brand'),
            category=meta.get('category'),
            price=float(price) if price is not None else None,
            avg_rating=float(avg_rating) if avg_rating is not None else None,
            num_sold=int(num_sold) if num_sold is not None else None,
            signals=signals
        )
    
    def _get_popular_items(
        self,
        topk: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """Get popular items as fallback when no semantic search possible."""
        start = time.perf_counter()
        
        if self.product_metadata is None or self.product_metadata.empty:
            return SearchResponse(
                query="popular",
                results=[],
                count=0,
                latency_ms=(time.perf_counter() - start) * 1000,
                method='fallback_popular',
                filters_applied=filters
            )
        
        # Sort by popularity
        df = self.product_metadata.copy()
        
        # Apply filters
        if filters:
            if 'brand' in filters and filters['brand']:
                # Handle NaN values in brand column
                if 'brand' in df.columns:
                    df = df[df['brand'].notna() & (df['brand'].astype(str).str.lower() == filters['brand'].lower())]
            if 'category' in filters and filters['category']:
                cat_col = 'type' if 'type' in df.columns else 'category'
                if cat_col in df.columns:
                    df = df[df[cat_col].notna() & (df[cat_col].astype(str).str.lower() == filters['category'].lower())]
            if 'min_price' in filters:
                price_col = 'price' if 'price' in df.columns else 'price_sale'
                df = df[df[price_col] >= filters['min_price']]
            if 'max_price' in filters:
                price_col = 'price' if 'price' in df.columns else 'price_sale'
                df = df[df[price_col] <= filters['max_price']]
        
        # Sort by popularity columns
        sort_cols = []
        if 'popularity_score' in df.columns:
            sort_cols.append('popularity_score')
        if 'num_sold_time' in df.columns:
            sort_cols.append('num_sold_time')
        if 'avg_star' in df.columns:
            sort_cols.append('avg_star')
        
        if sort_cols:
            df = df.sort_values(by=sort_cols, ascending=False)
        
        # Get top-K
        results = []
        for i, (_, row) in enumerate(df.head(topk).iterrows()):
            pid = int(row['product_id'])
            
            # Handle NaN values
            price = row.get('price', row.get('price_sale'))
            avg_rating = row.get('avg_star', row.get('avg_rating'))
            num_sold = row.get('num_sold_time', row.get('num_sold'))
            
            if pd.isna(price):
                price = None
            if pd.isna(avg_rating):
                avg_rating = None
            if pd.isna(num_sold):
                num_sold = None
            
            results.append(SearchResult(
                product_id=pid,
                product_name=str(row.get('product_name', row.get('name', ''))),
                semantic_score=0.0,
                final_score=float(row.get('popularity_score', i)),
                brand=row.get('brand'),
                category=row.get('type', row.get('category')),
                price=float(price) if price is not None else None,
                avg_rating=float(avg_rating) if avg_rating is not None else None,
                num_sold=int(num_sold) if num_sold is not None else None,
                signals={'popularity': 1.0},
                rank=i + 1
            ))
        
        latency = (time.perf_counter() - start) * 1000
        
        return SearchResponse(
            query="popular",
            results=results,
            count=len(results),
            latency_ms=latency,
            method='fallback_popular',
            filters_applied=filters
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = self._stats.copy()
        
        # Compute averages
        total_searches = (
            stats['searches_performed'] + 
            stats['similar_searches'] + 
            stats['profile_searches']
        )
        
        if total_searches > 0:
            stats['avg_latency_ms'] = stats['total_latency_ms'] / total_searches
        else:
            stats['avg_latency_ms'] = 0.0
        
        stats['total_searches'] = total_searches
        stats['initialized'] = self._initialized
        
        # Add component stats
        if self.search_index is not None:
            stats['index'] = self.search_index.get_stats()
        if self.query_encoder is not None:
            stats['encoder'] = self.query_encoder.get_stats()
        
        return stats
    
    def get_available_filters(self) -> Dict[str, Any]:
        """Get available filter options."""
        if not self._initialized:
            self.initialize()
        
        return {
            'brands': self.search_index.get_available_brands(),
            'categories': self.search_index.get_available_categories(),
            'price_range': self.search_index.get_price_range()
        }
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized


# ============================================================================
# Singleton Access Functions
# ============================================================================

_search_service_instance: Optional[SmartSearchService] = None
_search_service_lock = threading.Lock()


def get_search_service(**kwargs) -> SmartSearchService:
    """
    Get or create SmartSearchService singleton instance.
    
    Args:
        **kwargs: Arguments passed to SmartSearchService constructor
    
    Returns:
        SmartSearchService instance
    """
    global _search_service_instance
    
    with _search_service_lock:
        if _search_service_instance is None:
            _search_service_instance = SmartSearchService(**kwargs)
        return _search_service_instance


def reset_search_service() -> None:
    """Reset SmartSearchService singleton (for testing)."""
    global _search_service_instance
    
    with _search_service_lock:
        _search_service_instance = None
    
    logger.info("SmartSearchService singleton reset")
