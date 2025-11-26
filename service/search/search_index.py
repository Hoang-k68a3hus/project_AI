"""
Search Index for Smart Search.

Manages product embeddings index for fast similarity search.
Supports exact search and Approximate Nearest Neighbor (ANN) with FAISS.

Example:
    >>> from service.search.search_index import SearchIndex
    >>> index = SearchIndex()
    >>> index.build_index()
    >>> results = index.search(query_embedding, topk=10)
"""

from typing import List, Tuple, Optional, Dict, Any, Set
from pathlib import Path
import numpy as np
import logging
import time
import threading

logger = logging.getLogger(__name__)


class SearchIndex:
    """
    Search index for semantic product search.
    
    Features:
    - Exact cosine similarity search (for small catalogs)
    - FAISS ANN search (optional, for large catalogs >5K items)
    - Metadata filtering (brand, category, price range)
    - Thread-safe operations
    - Integration with PhoBERTEmbeddingLoader
    
    Example:
        >>> index = SearchIndex()
        >>> index.build_index()
        >>> results = index.search(query_embedding, topk=10)
        >>> results = index.search_with_filter(query_embedding, filters={'brand': 'Innisfree'})
    """
    
    def __init__(
        self,
        phobert_loader=None,
        product_metadata=None,
        use_faiss: bool = False,
        faiss_index_type: str = "flat",
        auto_build: bool = False
    ):
        """
        Initialize SearchIndex.
        
        Args:
            phobert_loader: PhoBERTEmbeddingLoader instance
            product_metadata: DataFrame with product info (product_id, brand, type, price, etc.)
            use_faiss: Use FAISS for ANN search (faster for large catalogs)
            faiss_index_type: FAISS index type ('flat', 'ivf', 'hnsw')
            auto_build: Automatically build index on init
        """
        self.phobert_loader = phobert_loader
        self.product_metadata = product_metadata
        self.use_faiss = use_faiss
        self.faiss_index_type = faiss_index_type
        
        # FAISS index (if enabled)
        self._faiss_index = None
        
        # Ordered product IDs (index position → product_id)
        self._product_ids: List[int] = []
        
        # Reverse mapping (product_id → index position)
        self._pid_to_idx: Dict[int, int] = {}
        
        # Metadata inverted indices for filtering
        self._brand_index: Dict[str, Set[int]] = {}     # brand → set of product_ids
        self._category_index: Dict[str, Set[int]] = {}  # category → set of product_ids
        self._price_data: Dict[int, float] = {}         # product_id → price
        
        # Index state
        self._initialized = False
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'num_products': 0,
            'num_brands': 0,
            'num_categories': 0,
            'index_build_time_ms': 0.0,
            'searches_performed': 0,
            'filtered_searches': 0
        }
        
        if auto_build:
            self.build_index()
    
    def build_index(self) -> None:
        """
        Build search index from embeddings.
        
        Loads embeddings from PhoBERTEmbeddingLoader and builds
        necessary indices for fast similarity search.
        """
        with self._lock:
            if self._initialized:
                logger.info("Index already built. Use rebuild_index() to force rebuild.")
                return
            
            self._build_index_internal()
    
    def rebuild_index(self) -> None:
        """Force rebuild the search index."""
        with self._lock:
            self._initialized = False
            self._faiss_index = None
            self._product_ids.clear()
            self._pid_to_idx.clear()
            self._brand_index.clear()
            self._category_index.clear()
            self._price_data.clear()
            
            self._build_index_internal()
    
    def _build_index_internal(self) -> None:
        """Internal method to build index."""
        start = time.perf_counter()
        
        # Load PhoBERT embeddings if not provided
        if self.phobert_loader is None:
            from service.recommender.phobert_loader import get_phobert_loader
            self.phobert_loader = get_phobert_loader()
        
        if not self.phobert_loader.is_loaded():
            logger.info("PhoBERT embeddings not loaded. Loading now...")
            self.phobert_loader._load_embeddings()
        
        # Verify embeddings are loaded
        if not self.phobert_loader.is_loaded() or not self.phobert_loader.product_id_to_idx:
            logger.error("Failed to load PhoBERT embeddings. Cannot build index.")
            return
        
        # Get ordered product IDs from embedding loader
        self._product_ids = list(self.phobert_loader.product_id_to_idx.keys())
        self._pid_to_idx = {pid: idx for idx, pid in enumerate(self._product_ids)}
        
        logger.info(f"Building index for {len(self._product_ids)} products...")
        
        # Build FAISS index if enabled
        if self.use_faiss:
            self._build_faiss_index()
        
        # Build metadata indices
        if self.product_metadata is not None:
            self._build_metadata_indices()
        
        elapsed = (time.perf_counter() - start) * 1000
        
        # Update stats
        self._stats['num_products'] = len(self._product_ids)
        self._stats['num_brands'] = len(self._brand_index)
        self._stats['num_categories'] = len(self._category_index)
        self._stats['index_build_time_ms'] = elapsed
        
        self._initialized = True
        
        logger.info(
            f"Search index built in {elapsed:.1f}ms: "
            f"{self._stats['num_products']} products, "
            f"{self._stats['num_brands']} brands, "
            f"{self._stats['num_categories']} categories"
        )
    
    def _build_faiss_index(self) -> None:
        """Build FAISS index for approximate nearest neighbor search."""
        try:
            import faiss
        except ImportError:
            logger.warning(
                "FAISS not installed. Falling back to exact search. "
                "Install with: pip install faiss-cpu"
            )
            self.use_faiss = False
            return
        
        # Check if embeddings are loaded
        if self.phobert_loader.embeddings_norm is None:
            logger.error("Embeddings not loaded. Cannot build FAISS index.")
            self.use_faiss = False
            return
        
        embeddings = self.phobert_loader.embeddings_norm  # Pre-normalized
        if embeddings.size == 0:
            logger.error("Empty embeddings. Cannot build FAISS index.")
            self.use_faiss = False
            return
        
        dim = embeddings.shape[1]
        n_items = len(self._product_ids)
        
        logger.info(f"Building FAISS index (type: {self.faiss_index_type}, dim: {dim}, n: {n_items})")
        
        # Build appropriate index type
        if self.faiss_index_type == "flat":
            # Exact search (brute force) - good for <10K items
            # Using Inner Product = Cosine similarity for normalized vectors
            self._faiss_index = faiss.IndexFlatIP(dim)
        
        elif self.faiss_index_type == "ivf":
            # IVF index - good for 10K-1M items
            # Number of clusters: sqrt(n) is a good starting point
            nlist = max(10, min(100, int(np.sqrt(n_items))))
            quantizer = faiss.IndexFlatIP(dim)
            self._faiss_index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            # IVF needs training
            self._faiss_index.train(embeddings.astype('float32'))
            # Set nprobe for search (higher = more accurate but slower)
            self._faiss_index.nprobe = min(10, nlist)
        
        elif self.faiss_index_type == "hnsw":
            # HNSW - fast approximate search
            # M = number of connections (higher = better recall, more memory)
            M = 32
            self._faiss_index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            # efConstruction: higher = better index quality, slower build
            self._faiss_index.hnsw.efConstruction = 40
            # efSearch: higher = better recall, slower search
            self._faiss_index.hnsw.efSearch = 16
        
        else:
            logger.warning(f"Unknown FAISS index type: {self.faiss_index_type}. Using flat.")
            self._faiss_index = faiss.IndexFlatIP(dim)
        
        # Add vectors to index
        self._faiss_index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {self._faiss_index.ntotal} vectors")
    
    def _build_metadata_indices(self) -> None:
        """Build inverted indices for metadata filtering."""
        import pandas as pd
        
        if self.product_metadata is None or self.product_metadata.empty:
            logger.warning("No product metadata provided. Filtering disabled.")
            return
        
        # Get valid product IDs (those with embeddings)
        valid_pids = set(self._product_ids)
        
        for _, row in self.product_metadata.iterrows():
            pid = row.get('product_id')
            if pid is None or int(pid) not in valid_pids:
                continue
            
            pid = int(pid)
            
            # Brand index
            brand = row.get('brand', '')
            if pd.notna(brand) and brand:
                brand_key = str(brand).lower().strip()
                if brand_key:
                    if brand_key not in self._brand_index:
                        self._brand_index[brand_key] = set()
                    self._brand_index[brand_key].add(pid)
            
            # Category index (try 'type' first, then 'category')
            category = row.get('type', row.get('category', ''))
            if pd.notna(category) and category:
                cat_key = str(category).lower().strip()
                if cat_key:
                    if cat_key not in self._category_index:
                        self._category_index[cat_key] = set()
                    self._category_index[cat_key].add(pid)
            
            # Price data
            price = row.get('price', row.get('price_sale'))
            if pd.notna(price):
                try:
                    self._price_data[pid] = float(price)
                except (ValueError, TypeError):
                    pass
        
        logger.info(
            f"Metadata indices built: {len(self._brand_index)} brands, "
            f"{len(self._category_index)} categories, "
            f"{len(self._price_data)} price entries"
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        topk: int = 10,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Search for similar products.
        
        Args:
            query_embedding: Query embedding vector (should be normalized for cosine sim)
            topk: Number of results to return
            exclude_ids: Product IDs to exclude from results
        
        Returns:
            List of (product_id, similarity_score) tuples, sorted by score descending
        """
        if not self._initialized:
            self.build_index()
        
        self._stats['searches_performed'] += 1
        
        if self.use_faiss and self._faiss_index is not None:
            return self._search_faiss(query_embedding, topk, exclude_ids)
        else:
            return self._search_exact(query_embedding, topk, exclude_ids)
    
    def _search_exact(
        self,
        query_embedding: np.ndarray,
        topk: int,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Exact cosine similarity search using numpy."""
        # Ensure query is normalized
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Check if embeddings are loaded
        if self.phobert_loader.embeddings_norm is None:
            logger.error("Embeddings not loaded. Cannot perform search.")
            return []
        
        # Compute all similarities (dot product = cosine for normalized vectors)
        similarities = self.phobert_loader.embeddings_norm @ query_embedding
        
        # Apply exclusions
        if exclude_ids:
            for pid in exclude_ids:
                idx = self.phobert_loader.product_id_to_idx.get(pid)
                if idx is not None:
                    similarities[idx] = -np.inf
        
        # Get top-K indices
        if topk >= len(similarities):
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Partial sort for efficiency
            top_indices = np.argpartition(similarities, -topk)[-topk:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        # Build results
        results = []
        for idx in top_indices:
            if similarities[idx] == -np.inf:
                continue
            pid = self.phobert_loader.idx_to_product_id[idx]
            results.append((pid, float(similarities[idx])))
            if len(results) >= topk:
                break
        
        return results
    
    def _search_faiss(
        self,
        query_embedding: np.ndarray,
        topk: int,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Search using FAISS index."""
        # Request extra results if excluding
        request_k = topk * 2 if exclude_ids else topk
        request_k = min(request_k, len(self._product_ids))
        
        # FAISS search
        query = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self._faiss_index.search(query, request_k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for unfilled slots
                continue
            
            pid = self._product_ids[idx]
            
            if exclude_ids and pid in exclude_ids:
                continue
            
            results.append((pid, float(score)))
            
            if len(results) >= topk:
                break
        
        return results
    
    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        topk: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Search with metadata filtering.
        
        Args:
            query_embedding: Query embedding
            topk: Number of results
            filters: Metadata filters:
                - 'brand': Brand name (string, case-insensitive)
                - 'category': Category/type name (string, case-insensitive)
                - 'min_price': Minimum price (float)
                - 'max_price': Maximum price (float)
            exclude_ids: IDs to exclude
        
        Returns:
            Filtered and ranked results as list of (product_id, score) tuples
        """
        if not self._initialized:
            self.build_index()
        
        self._stats['filtered_searches'] += 1
        
        # Get candidate IDs from filter
        candidate_ids = self._apply_filters(filters)
        
        if candidate_ids is None:
            # No filter → search all
            return self.search(query_embedding, topk, exclude_ids)
        
        if not candidate_ids:
            # Filter matches nothing
            return []
        
        # Exclude IDs
        if exclude_ids:
            candidate_ids = candidate_ids - exclude_ids
        
        if not candidate_ids:
            return []
        
        # Check if phobert_loader is properly initialized
        if not hasattr(self.phobert_loader, 'product_id_to_idx') or not self.phobert_loader.product_id_to_idx:
            logger.error("PhoBERT loader not properly initialized. Cannot perform filtered search.")
            return []
        
        # Get indices for candidates
        candidate_indices = []
        valid_pids = []
        for pid in candidate_ids:
            idx = self.phobert_loader.product_id_to_idx.get(pid)
            if idx is not None:
                candidate_indices.append(idx)
                valid_pids.append(pid)
        
        if not candidate_indices:
            return []
        
        # Ensure query is normalized
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Check if embeddings are loaded
        if self.phobert_loader.embeddings_norm is None:
            logger.error("Embeddings not loaded. Cannot perform search.")
            return []
        
        # Compute similarities only for candidates
        candidate_embeddings = self.phobert_loader.embeddings_norm[candidate_indices]
        similarities = candidate_embeddings @ query_embedding
        
        # Sort and return top-K
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for i in sorted_indices[:topk]:
            results.append((valid_pids[i], float(similarities[i])))
        
        return results
    
    def _apply_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Set[int]]:
        """
        Apply metadata filters and return candidate IDs.
        
        Returns:
            Set of product IDs matching filters, or None if no filters applied
        """
        if not filters:
            return None
        
        candidate_sets = []
        
        # Brand filter
        if 'brand' in filters and filters['brand']:
            brand = str(filters['brand']).lower().strip()
            if brand in self._brand_index:
                candidate_sets.append(self._brand_index[brand])
            else:
                # Brand not found → no matches
                logger.debug(f"Brand filter '{brand}' not found in index")
                return set()
        
        # Category filter
        if 'category' in filters and filters['category']:
            category = str(filters['category']).lower().strip()
            if category in self._category_index:
                candidate_sets.append(self._category_index[category])
            else:
                # Category not found → no matches
                logger.debug(f"Category filter '{category}' not found in index")
                return set()
        
        # Price range filter
        min_price = filters.get('min_price')
        max_price = filters.get('max_price')
        
        if min_price is not None or max_price is not None:
            min_p = float(min_price) if min_price is not None else 0
            max_p = float(max_price) if max_price is not None else float('inf')
            
            price_matches = set()
            for pid, price in self._price_data.items():
                if min_p <= price <= max_p:
                    price_matches.add(pid)
            
            if price_matches:
                candidate_sets.append(price_matches)
            else:
                return set()
        
        if not candidate_sets:
            return None
        
        # Intersection of all filter sets
        return set.intersection(*candidate_sets)
    
    def get_available_brands(self) -> List[str]:
        """Get list of available brands for filtering."""
        return sorted(self._brand_index.keys())
    
    def get_available_categories(self) -> List[str]:
        """Get list of available categories for filtering."""
        return sorted(self._category_index.keys())
    
    def get_price_range(self) -> Tuple[float, float]:
        """Get min/max price range."""
        if not self._price_data:
            return (0.0, 0.0)
        prices = list(self._price_data.values())
        return (min(prices), max(prices))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = self._stats.copy()
        stats['initialized'] = self._initialized
        stats['use_faiss'] = self.use_faiss
        stats['faiss_index_type'] = self.faiss_index_type if self.use_faiss else None
        stats['available_brands'] = len(self._brand_index)
        stats['available_categories'] = len(self._category_index)
        stats['products_with_price'] = len(self._price_data)
        return stats
    
    @property
    def num_products(self) -> int:
        """Number of indexed products."""
        return len(self._product_ids)
    
    @property
    def is_initialized(self) -> bool:
        """Check if index is built."""
        return self._initialized


# ============================================================================
# Singleton Access
# ============================================================================

_search_index_instance: Optional[SearchIndex] = None
_search_index_lock = threading.Lock()


def get_search_index(**kwargs) -> SearchIndex:
    """
    Get or create SearchIndex singleton instance.
    
    Args:
        **kwargs: Arguments passed to SearchIndex constructor
    
    Returns:
        SearchIndex instance
    """
    global _search_index_instance
    
    with _search_index_lock:
        if _search_index_instance is None:
            _search_index_instance = SearchIndex(**kwargs)
        return _search_index_instance


def reset_search_index() -> None:
    """Reset SearchIndex singleton (for testing)."""
    global _search_index_instance
    
    with _search_index_lock:
        _search_index_instance = None
    
    logger.info("SearchIndex singleton reset")
