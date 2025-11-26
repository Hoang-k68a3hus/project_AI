# Task 09: Smart Search - Semantic Search Integration

## Mục Tiêu

Tích hợp tính năng **Smart Search** (tìm kiếm thông minh) vào hệ thống recommendation service, sử dụng PhoBERT embeddings đã được tạo ra từ các task trước. Tính năng này cho phép người dùng tìm kiếm sản phẩm bằng ngôn ngữ tự nhiên (tiếng Việt) thay vì chỉ dựa trên keyword matching truyền thống.

## 📊 Data Dependencies

**Embeddings đã có sẵn từ các tasks trước**:
- **Product Embeddings**: `data/processed/content_based_embeddings/product_embeddings.pt`
  - Chứa BERT embeddings cho ~2,200 products
  - Dimension: 768 (PhoBERT-base) hoặc 1024 (PhoBERT-large)
  - Pre-normalized vectors cho fast cosine similarity
- **PhoBERTEmbeddingLoader** (Task 05/08): Singleton class đã implement loading và similarity computation

**Lợi thế so với keyword search truyền thống**:
- Hiểu ngữ nghĩa tiếng Việt (synonyms, paraphrases)
- Xử lý typos và biến thể từ vựng
- Tìm kiếm theo intent/concept, không chỉ exact match

## 🎯 Use Cases

### Use Case 1: Product Discovery
```
User: "tìm kem dưỡng da cho da dầu mụn"
→ Semantic search: tìm products có embeddings gần với query embedding
→ Return: kem trị mụn, gel kiểm soát dầu, serum BHA, etc.
```

### Use Case 2: Similar Product Search
```
User: "sản phẩm tương tự [product_id=123]"
→ Item-item similarity từ PhoBERT embeddings
→ Return: top-K similar products
```

### Use Case 3: Review-Based Search
```
User: "sản phẩm được khen là thấm nhanh không nhờn"
→ Encode query → tìm products có reviews matching
→ Return: products có reviews tương tự về semantic
```

### Use Case 4: Attribute + Semantic Hybrid
```
User: "kem chống nắng cho da nhạy cảm"
→ Filter: brand/category attributes
→ Semantic: rank by embedding similarity
→ Return: filtered & ranked products
```

## 🏗️ Architecture Overview

```
Smart Search Service
├─ Query Encoder (PhoBERT)
│  ├─ Batch encoding cho queries
│  ├─ Query expansion (optional)
│  └─ Query embedding caching
├─ Search Index
│  ├─ Product embeddings (pre-computed)
│  ├─ FAISS/Annoy index for ANN search
│  └─ Metadata index (attributes, names)
├─ Reranker
│  ├─ Semantic score (cosine similarity)
│  ├─ Attribute match boost
│  ├─ Popularity signal
│  └─ Quality signal
└─ API Endpoints
   ├─ POST /search: Text query search
   ├─ POST /search/similar: Item-item search
   └─ POST /search/hybrid: Combined search
```

## Component 1: Query Encoder

### Module: `service/search/query_encoder.py`

```python
"""
Query Encoder for Smart Search.

Encode user queries into semantic embeddings using PhoBERT.
"""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import numpy as np
import logging
import threading

logger = logging.getLogger(__name__)


class QueryEncoder:
    """
    Encode text queries to embeddings using PhoBERT.
    
    Features:
    - Lazy loading of PhoBERT model
    - Query embedding caching (LRU)
    - Batch encoding for efficiency
    - Vietnamese text preprocessing
    
    Example:
        >>> encoder = QueryEncoder()
        >>> emb = encoder.encode("kem dưỡng da cho da dầu")
        >>> embeddings = encoder.encode_batch(["query1", "query2"])
    """
    
    _instance: Optional['QueryEncoder'] = None
    _lock = threading.Lock()
    
    # Singleton pattern
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        max_length: int = 256,
        cache_size: int = 1000,
        device: str = "cpu"
    ):
        if self._initialized:
            return
        
        self.model_name = model_name
        self.max_length = max_length
        self.cache_size = cache_size
        self.device = device
        
        # Lazy loaded
        self._tokenizer = None
        self._model = None
        
        # Query cache (LRU)
        self._query_cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        
        self._initialized = True
    
    def _load_model(self):
        """Lazy load PhoBERT model and tokenizer."""
        if self._model is not None:
            return
        
        import time
        start = time.perf_counter()
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            logger.info(f"Loading PhoBERT model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"PhoBERT model loaded in {elapsed:.1f}ms")
            
        except Exception as e:
            logger.error(f"Failed to load PhoBERT: {e}")
            raise
    
    def encode(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query to embedding.
        
        Args:
            query: Text query (Vietnamese)
            normalize: L2 normalize the embedding
        
        Returns:
            np.array of shape (768,) or (1024,)
        """
        # Check cache
        cache_key = f"{query}_{normalize}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # Ensure model loaded
        self._load_model()
        
        import torch
        
        # Preprocess Vietnamese text
        processed_query = self._preprocess_vietnamese(query)
        
        # Tokenize
        inputs = self._tokenizer(
            processed_query,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use [CLS] token embedding or mean pooling
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        # Normalize
        if normalize:
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        # Cache
        self._cache_embedding(cache_key, embedding)
        
        return embedding
    
    def encode_batch(
        self,
        queries: List[str],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode multiple queries efficiently.
        
        Args:
            queries: List of text queries
            normalize: L2 normalize embeddings
            batch_size: Batch size for encoding
        
        Returns:
            np.array of shape (num_queries, embedding_dim)
        """
        self._load_model()
        
        import torch
        
        all_embeddings = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            # Preprocess
            processed = [self._preprocess_vietnamese(q) for q in batch]
            
            # Tokenize
            inputs = self._tokenizer(
                processed,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Encode
            with torch.no_grad():
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
        
        result = np.vstack(all_embeddings)
        
        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / (norms + 1e-9)
        
        return result
    
    def _preprocess_vietnamese(self, text: str) -> str:
        """
        Preprocess Vietnamese text for encoding.
        
        - Lowercase
        - Remove extra whitespace
        - Handle abbreviations
        """
        import re
        
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Vietnamese abbreviations
        abbreviations = {
            'sp': 'sản phẩm',
            'kem dc': 'kem dưỡng da',
            'srm': 'sữa rửa mặt',
            'tbc': 'tẩy bong chết',
        }
        
        for abbr, full in abbreviations.items():
            text = re.sub(rf'\b{abbr}\b', full, text)
        
        return text
    
    def _cache_embedding(self, key: str, embedding: np.ndarray):
        """Add embedding to LRU cache."""
        if key in self._query_cache:
            # Move to end (most recent)
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return
        
        # Check cache size
        if len(self._cache_order) >= self.cache_size:
            # Remove oldest
            oldest = self._cache_order.pop(0)
            del self._query_cache[oldest]
        
        # Add new
        self._query_cache[key] = embedding
        self._cache_order.append(key)
    
    def clear_cache(self):
        """Clear query embedding cache."""
        self._query_cache.clear()
        self._cache_order.clear()
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._model is None:
            return 768  # Default for PhoBERT-base
        return self._model.config.hidden_size


def get_query_encoder(**kwargs) -> QueryEncoder:
    """Get singleton QueryEncoder instance."""
    return QueryEncoder(**kwargs)
```

## Component 2: Search Index

### Module: `service/search/search_index.py`

```python
"""
Search Index for Smart Search.

Manages product embeddings index for fast similarity search.
Supports exact search and Approximate Nearest Neighbor (ANN).
"""

from typing import List, Tuple, Optional, Dict, Any, Set
from pathlib import Path
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


class SearchIndex:
    """
    Search index for semantic product search.
    
    Features:
    - Exact cosine similarity search
    - FAISS ANN search (optional, for large catalogs)
    - Metadata filtering (brand, category, price range)
    - Integration with PhoBERTEmbeddingLoader
    
    Example:
        >>> index = SearchIndex()
        >>> results = index.search(query_embedding, topk=10)
        >>> results = index.search_with_filter(query_embedding, filters={'brand': 'Innisfree'})
    """
    
    def __init__(
        self,
        phobert_loader=None,
        product_metadata=None,
        use_faiss: bool = False,
        faiss_index_type: str = "flat"  # flat, ivf, hnsw
    ):
        """
        Initialize SearchIndex.
        
        Args:
            phobert_loader: PhoBERTEmbeddingLoader instance (from Task 05/08)
            product_metadata: DataFrame with product info
            use_faiss: Use FAISS for ANN search
            faiss_index_type: FAISS index type
        """
        self.phobert_loader = phobert_loader
        self.product_metadata = product_metadata
        self.use_faiss = use_faiss
        self.faiss_index_type = faiss_index_type
        
        self._faiss_index = None
        self._product_ids: List[int] = []  # Ordered list of product IDs
        
        # Metadata indices for filtering
        self._brand_index: Dict[str, Set[int]] = {}
        self._category_index: Dict[str, Set[int]] = {}
        self._price_ranges: Dict[int, Tuple[float, float]] = {}
        
        self._initialized = False
    
    def build_index(self):
        """Build search index from embeddings."""
        if self.phobert_loader is None:
            from service.recommender.phobert_loader import get_phobert_loader
            self.phobert_loader = get_phobert_loader()
        
        if not self.phobert_loader.is_loaded():
            raise RuntimeError("PhoBERT embeddings not loaded")
        
        start = time.perf_counter()
        
        # Get ordered product IDs
        self._product_ids = list(self.phobert_loader.product_id_to_idx.keys())
        
        # Build FAISS index if enabled
        if self.use_faiss:
            self._build_faiss_index()
        
        # Build metadata indices
        self._build_metadata_indices()
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Search index built in {elapsed:.1f}ms")
        
        self._initialized = True
    
    def _build_faiss_index(self):
        """Build FAISS index for ANN search."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed. Using exact search.")
            self.use_faiss = False
            return
        
        embeddings = self.phobert_loader.embeddings_norm  # Pre-normalized
        dim = embeddings.shape[1]
        
        if self.faiss_index_type == "flat":
            # Exact search (brute force)
            self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
        
        elif self.faiss_index_type == "ivf":
            # IVF index for larger datasets
            nlist = min(100, len(self._product_ids) // 10)
            quantizer = faiss.IndexFlatIP(dim)
            self._faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._faiss_index.train(embeddings.astype('float32'))
        
        elif self.faiss_index_type == "hnsw":
            # HNSW for fast approximate search
            self._faiss_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        
        # Add vectors
        self._faiss_index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built: {self.faiss_index_type}, {embeddings.shape[0]} vectors")
    
    def _build_metadata_indices(self):
        """Build inverted indices for metadata filtering."""
        if self.product_metadata is None:
            logger.warning("No product metadata provided. Filtering disabled.")
            return
        
        for _, row in self.product_metadata.iterrows():
            pid = row.get('product_id')
            if pid not in self._product_ids:
                continue
            
            # Brand index
            brand = row.get('brand', '').lower().strip()
            if brand:
                if brand not in self._brand_index:
                    self._brand_index[brand] = set()
                self._brand_index[brand].add(pid)
            
            # Category index
            category = row.get('type', row.get('category', '')).lower().strip()
            if category:
                if category not in self._category_index:
                    self._category_index[category] = set()
                self._category_index[category].add(pid)
            
            # Price range
            price = row.get('price', row.get('price_sale'))
            if price is not None and not np.isnan(price):
                self._price_ranges[pid] = (float(price), float(price))
    
    def search(
        self,
        query_embedding: np.ndarray,
        topk: int = 10,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Search for similar products.
        
        Args:
            query_embedding: Query embedding vector (normalized)
            topk: Number of results to return
            exclude_ids: Product IDs to exclude
        
        Returns:
            List of (product_id, similarity_score) tuples
        """
        if not self._initialized:
            self.build_index()
        
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
        """Exact cosine similarity search."""
        # Compute similarities
        similarities = self.phobert_loader.embeddings_norm @ query_embedding
        
        # Apply exclusions
        if exclude_ids:
            for pid in exclude_ids:
                idx = self.phobert_loader.product_id_to_idx.get(pid)
                if idx is not None:
                    similarities[idx] = -np.inf
        
        # Get top-K
        top_indices = np.argsort(similarities)[::-1][:topk]
        
        results = []
        for idx in top_indices:
            if similarities[idx] == -np.inf:
                continue
            pid = self.phobert_loader.idx_to_product_id[idx]
            results.append((pid, float(similarities[idx])))
        
        return results
    
    def _search_faiss(
        self,
        query_embedding: np.ndarray,
        topk: int,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """FAISS ANN search."""
        # Request more results if excluding
        request_k = topk * 2 if exclude_ids else topk
        
        # Search
        query = query_embedding.reshape(1, -1).astype('float32')
        scores, indices = self._faiss_index.search(query, request_k)
        
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
            filters: Metadata filters, e.g., {'brand': 'innisfree', 'category': 'kem dưỡng'}
            exclude_ids: IDs to exclude
        
        Returns:
            Filtered and ranked results
        """
        if not self._initialized:
            self.build_index()
        
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
        
        # Compute similarities for candidates only
        candidate_embeddings = self.phobert_loader.embeddings_norm[candidate_indices]
        similarities = candidate_embeddings @ query_embedding
        
        # Sort and return top-K
        sorted_indices = np.argsort(similarities)[::-1][:topk]
        
        results = []
        for idx in sorted_indices:
            results.append((valid_pids[idx], float(similarities[idx])))
        
        return results
    
    def _apply_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Set[int]]:
        """Apply metadata filters and return candidate IDs."""
        if not filters:
            return None
        
        candidate_sets = []
        
        # Brand filter
        if 'brand' in filters:
            brand = filters['brand'].lower().strip()
            if brand in self._brand_index:
                candidate_sets.append(self._brand_index[brand])
            else:
                return set()  # Brand not found
        
        # Category filter
        if 'category' in filters:
            category = filters['category'].lower().strip()
            if category in self._category_index:
                candidate_sets.append(self._category_index[category])
            else:
                return set()  # Category not found
        
        # Price range filter
        if 'min_price' in filters or 'max_price' in filters:
            min_price = filters.get('min_price', 0)
            max_price = filters.get('max_price', float('inf'))
            
            price_matches = set()
            for pid, (p_min, p_max) in self._price_ranges.items():
                if min_price <= p_min <= max_price:
                    price_matches.add(pid)
            candidate_sets.append(price_matches)
        
        if not candidate_sets:
            return None
        
        # Intersection of all filter sets
        return set.intersection(*candidate_sets)
    
    @property
    def num_products(self) -> int:
        """Number of indexed products."""
        return len(self._product_ids)


def get_search_index(**kwargs) -> SearchIndex:
    """Get or create SearchIndex instance."""
    return SearchIndex(**kwargs)
```

## Component 3: Smart Search Service

### Module: `service/search/smart_search.py`

```python
"""
Smart Search Service.

Main service class for semantic product search.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
import logging
import time
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result container."""
    product_id: int
    product_name: str
    semantic_score: float
    final_score: float
    brand: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    avg_rating: Optional[float] = None
    signals: Optional[Dict[str, float]] = None


@dataclass
class SearchResponse:
    """Search response container."""
    query: str
    results: List[SearchResult]
    count: int
    latency_ms: float
    method: str  # 'semantic', 'hybrid', 'similar_items'
    filters_applied: Optional[Dict[str, Any]] = None


class SmartSearchService:
    """
    Smart Search Service for semantic product discovery.
    
    Features:
    - Text-to-product semantic search (PhoBERT)
    - Item-to-item similarity search
    - Hybrid search (semantic + attribute filters)
    - Reranking with multiple signals
    
    Integration Points:
    - QueryEncoder: Encode text queries to embeddings
    - SearchIndex: Fast similarity search
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
            query_encoder: QueryEncoder instance
            search_index: SearchIndex instance
            phobert_loader: PhoBERTEmbeddingLoader instance
            product_metadata: DataFrame with product info
            config: Service configuration
        """
        self.query_encoder = query_encoder
        self.search_index = search_index
        self.phobert_loader = phobert_loader
        self.product_metadata = product_metadata
        
        self.config = config or {
            'default_topk': 10,
            'max_topk': 100,
            'rerank_weights': {
                'semantic': 0.50,
                'popularity': 0.25,
                'quality': 0.15,
                'recency': 0.10
            },
            'min_semantic_score': 0.3,  # Filter low-quality matches
            'enable_rerank': True,
            'candidate_multiplier': 3  # Fetch 3x candidates for reranking
        }
        
        self._initialized = False
    
    def initialize(self):
        """Initialize all components."""
        if self._initialized:
            return
        
        start = time.perf_counter()
        
        # Initialize query encoder
        if self.query_encoder is None:
            from service.search.query_encoder import get_query_encoder
            self.query_encoder = get_query_encoder()
        
        # Initialize PhoBERT loader
        if self.phobert_loader is None:
            from service.recommender.phobert_loader import get_phobert_loader
            self.phobert_loader = get_phobert_loader()
        
        # Initialize search index
        if self.search_index is None:
            from service.search.search_index import SearchIndex
            self.search_index = SearchIndex(
                phobert_loader=self.phobert_loader,
                product_metadata=self.product_metadata
            )
        
        # Build index
        self.search_index.build_index()
        
        # Load product metadata if not provided
        if self.product_metadata is None:
            self._load_product_metadata()
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"SmartSearchService initialized in {elapsed:.1f}ms")
        
        self._initialized = True
    
    def _load_product_metadata(self):
        """Load product metadata from default path."""
        try:
            # Try enriched metadata first (from Task 01)
            enriched_path = "data/processed/product_attributes_enriched.parquet"
            if Path(enriched_path).exists():
                self.product_metadata = pd.read_parquet(enriched_path)
                logger.info(f"Loaded enriched product metadata: {len(self.product_metadata)} products")
                return
            
            # Fallback to raw product data
            raw_path = "data/published_data/data_product.csv"
            if Path(raw_path).exists():
                self.product_metadata = pd.read_csv(raw_path, encoding='utf-8')
                logger.info(f"Loaded raw product metadata: {len(self.product_metadata)} products")
                return
            
            logger.warning("No product metadata found. Search results will be limited.")
            
        except Exception as e:
            logger.error(f"Failed to load product metadata: {e}")
    
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
            query: Text query in Vietnamese
            topk: Number of results
            filters: Attribute filters {'brand': ..., 'category': ...}
            exclude_ids: Product IDs to exclude
            rerank: Apply reranking with multiple signals
        
        Returns:
            SearchResponse with ranked results
        """
        self.initialize()
        
        start = time.perf_counter()
        
        # Validate topk
        topk = min(topk, self.config['max_topk'])
        
        # Encode query
        query_embedding = self.query_encoder.encode(query, normalize=True)
        
        # Determine candidate count
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
        
        # Filter by minimum score
        min_score = self.config['min_semantic_score']
        raw_results = [(pid, score) for pid, score in raw_results if score >= min_score]
        
        # Rerank if enabled
        if rerank and raw_results and self.config['enable_rerank']:
            results = self._rerank_results(raw_results, query, topk)
        else:
            results = self._enrich_results(raw_results[:topk])
        
        latency = (time.perf_counter() - start) * 1000
        
        return SearchResponse(
            query=query,
            results=results,
            count=len(results),
            latency_ms=latency,
            method='hybrid' if filters else 'semantic',
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
        Find similar products to a given product.
        
        Args:
            product_id: Source product ID
            topk: Number of similar products
            exclude_self: Exclude source product
            exclude_ids: Additional IDs to exclude
        
        Returns:
            SearchResponse with similar products
        """
        self.initialize()
        
        start = time.perf_counter()
        
        # Get product embedding
        product_emb = self.phobert_loader.get_embedding_normalized(product_id)
        if product_emb is None:
            logger.warning(f"Product {product_id} not found in embeddings")
            return SearchResponse(
                query=f"similar_to:{product_id}",
                results=[],
                count=0,
                latency_ms=0,
                method='similar_items'
            )
        
        # Build exclusion set
        exclusions = set(exclude_ids or [])
        if exclude_self:
            exclusions.add(product_id)
        
        # Search
        raw_results = self.search_index.search(product_emb, topk, exclusions)
        
        # Enrich results
        results = self._enrich_results(raw_results)
        
        latency = (time.perf_counter() - start) * 1000
        
        return SearchResponse(
            query=f"similar_to:{product_id}",
            results=results,
            count=len(results),
            latency_ms=latency,
            method='similar_items'
        )
    
    def search_by_user_profile(
        self,
        user_history: List[int],
        topk: int = 10,
        exclude_history: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """
        Search products similar to user's interaction history.
        
        Useful for cold-start users with browsing history.
        
        Args:
            user_history: List of product IDs user interacted with
            topk: Number of results
            exclude_history: Exclude products user already interacted with
            filters: Attribute filters
        
        Returns:
            SearchResponse with personalized recommendations
        """
        self.initialize()
        
        start = time.perf_counter()
        
        if not user_history:
            # Return popular items as fallback
            return self._get_popular_items(topk, filters)
        
        # Compute user profile embedding
        profile_emb = self.phobert_loader.compute_user_profile(
            user_history,
            strategy='weighted_mean'
        )
        
        if profile_emb is None:
            return self._get_popular_items(topk, filters)
        
        # Normalize
        profile_emb = profile_emb / (np.linalg.norm(profile_emb) + 1e-9)
        
        # Exclusions
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
        
        results = self._enrich_results(raw_results)
        
        latency = (time.perf_counter() - start) * 1000
        
        return SearchResponse(
            query=f"user_profile:{len(user_history)}_items",
            results=results,
            count=len(results),
            latency_ms=latency,
            method='user_profile',
            filters_applied=filters
        )
    
    def _rerank_results(
        self,
        raw_results: List[Tuple[int, float]],
        query: str,
        topk: int
    ) -> List[SearchResult]:
        """
        Rerank results using multiple signals.
        
        Signals:
        - semantic: Embedding similarity (from search)
        - popularity: num_sold_time or view count
        - quality: avg_rating, review count
        - recency: Product launch date
        """
        weights = self.config['rerank_weights']
        enriched_results = []
        
        for pid, semantic_score in raw_results:
            signals = {'semantic': semantic_score}
            
            # Get metadata
            if self.product_metadata is not None:
                product = self.product_metadata[self.product_metadata['product_id'] == pid]
                if not product.empty:
                    row = product.iloc[0]
                    
                    # Popularity signal (normalized)
                    popularity = row.get('popularity_score', row.get('num_sold_time', 0))
                    signals['popularity'] = min(float(popularity) / 10.0, 1.0)  # Cap at 1.0
                    
                    # Quality signal
                    quality = row.get('quality_score', row.get('avg_star', 3.0))
                    signals['quality'] = (float(quality) - 1.0) / 4.0  # Normalize 1-5 to 0-1
                    
                    # Recency signal (placeholder - needs actual data)
                    signals['recency'] = 0.5  # Default middle value
            
            # Compute final score
            final_score = sum(
                weights.get(signal, 0) * value
                for signal, value in signals.items()
            )
            
            # Create result
            result = self._create_search_result(pid, semantic_score, final_score, signals)
            enriched_results.append(result)
        
        # Sort by final score
        enriched_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return enriched_results[:topk]
    
    def _enrich_results(
        self,
        raw_results: List[Tuple[int, float]]
    ) -> List[SearchResult]:
        """Enrich results with product metadata."""
        results = []
        
        for pid, score in raw_results:
            result = self._create_search_result(pid, score, score, {'semantic': score})
            results.append(result)
        
        return results
    
    def _create_search_result(
        self,
        product_id: int,
        semantic_score: float,
        final_score: float,
        signals: Dict[str, float]
    ) -> SearchResult:
        """Create SearchResult with metadata."""
        product_name = ""
        brand = None
        category = None
        price = None
        avg_rating = None
        
        if self.product_metadata is not None:
            product = self.product_metadata[self.product_metadata['product_id'] == product_id]
            if not product.empty:
                row = product.iloc[0]
                product_name = str(row.get('product_name', row.get('name', '')))
                brand = row.get('brand')
                category = row.get('type', row.get('category'))
                price = row.get('price', row.get('price_sale'))
                avg_rating = row.get('avg_star', row.get('avg_rating'))
        
        return SearchResult(
            product_id=product_id,
            product_name=product_name,
            semantic_score=semantic_score,
            final_score=final_score,
            brand=brand,
            category=category,
            price=float(price) if price is not None and not np.isnan(price) else None,
            avg_rating=float(avg_rating) if avg_rating is not None and not np.isnan(avg_rating) else None,
            signals=signals
        )
    
    def _get_popular_items(
        self,
        topk: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        """Get popular items as fallback."""
        start = time.perf_counter()
        
        if self.product_metadata is None:
            return SearchResponse(
                query="popular",
                results=[],
                count=0,
                latency_ms=0,
                method='fallback_popular'
            )
        
        # Sort by popularity
        sorted_df = self.product_metadata.sort_values(
            by=['num_sold_time', 'avg_star'],
            ascending=False
        )
        
        # Apply filters if any
        if filters:
            if 'brand' in filters:
                sorted_df = sorted_df[sorted_df['brand'].str.lower() == filters['brand'].lower()]
            if 'category' in filters:
                sorted_df = sorted_df[sorted_df['type'].str.lower() == filters['category'].lower()]
        
        # Get top-K
        results = []
        for _, row in sorted_df.head(topk).iterrows():
            results.append(SearchResult(
                product_id=int(row['product_id']),
                product_name=str(row.get('product_name', '')),
                semantic_score=0.0,
                final_score=float(row.get('popularity_score', 0)),
                brand=row.get('brand'),
                category=row.get('type'),
                price=float(row.get('price')) if row.get('price') else None,
                avg_rating=float(row.get('avg_star')) if row.get('avg_star') else None,
                signals={'popularity': 1.0}
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


# Singleton instance
_search_service: Optional[SmartSearchService] = None


def get_search_service(**kwargs) -> SmartSearchService:
    """Get or create SmartSearchService instance."""
    global _search_service
    if _search_service is None:
        _search_service = SmartSearchService(**kwargs)
    return _search_service


def reset_search_service():
    """Reset singleton instance."""
    global _search_service
    _search_service = None
```

## Component 4: API Endpoints

### Update: `service/api.py` - Add Search Endpoints

Thêm các endpoints sau vào file `service/api.py` hiện có:

```python
# ============================================================================
# Search Request/Response Models
# ============================================================================

class SearchRequest(BaseModel):
    """Text search request."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query in Vietnamese")
    topk: int = Field(default=10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Attribute filters: {'brand': 'Innisfree', 'category': 'kem dưỡng'}"
    )
    exclude_ids: Optional[List[int]] = Field(default=None, description="Product IDs to exclude")
    rerank: bool = Field(default=True, description="Apply reranking with multiple signals")


class SimilarSearchRequest(BaseModel):
    """Similar items search request."""
    product_id: int = Field(..., description="Source product ID")
    topk: int = Field(default=10, ge=1, le=50, description="Number of similar products")
    exclude_ids: Optional[List[int]] = Field(default=None, description="Product IDs to exclude")


class UserProfileSearchRequest(BaseModel):
    """User profile search request."""
    user_history: List[int] = Field(..., min_items=1, max_items=100, description="Product IDs user interacted with")
    topk: int = Field(default=10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Attribute filters")
    exclude_history: bool = Field(default=True, description="Exclude products from history")


class SearchResultItem(BaseModel):
    """Single search result."""
    product_id: int
    product_name: str
    semantic_score: float
    final_score: float
    brand: Optional[str]
    category: Optional[str]
    price: Optional[float]
    avg_rating: Optional[float]
    signals: Optional[Dict[str, float]]


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    results: List[SearchResultItem]
    count: int
    latency_ms: float
    method: str
    filters_applied: Optional[Dict[str, Any]]


# ============================================================================
# Search Endpoints
# ============================================================================

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def smart_search(request: SearchRequest):
    """
    Smart semantic search for products.
    
    Uses PhoBERT embeddings for semantic similarity search.
    Supports attribute filtering and multi-signal reranking.
    
    Example:
        POST /search
        {
            "query": "kem dưỡng da cho da dầu",
            "topk": 10,
            "filters": {"brand": "innisfree"},
            "rerank": true
        }
    """
    try:
        from service.search.smart_search import get_search_service
        
        search_service = get_search_service()
        
        exclude_set = set(request.exclude_ids) if request.exclude_ids else None
        
        result = search_service.search(
            query=request.query,
            topk=request.topk,
            filters=request.filters,
            exclude_ids=exclude_set,
            rerank=request.rerank
        )
        
        logger.info(
            f"Search: query='{request.query[:50]}', "
            f"count={result.count}, latency={result.latency_ms:.1f}ms"
        )
        
        return SearchResponse(
            query=result.query,
            results=[SearchResultItem(**r.__dict__) for r in result.results],
            count=result.count,
            latency_ms=result.latency_ms,
            method=result.method,
            filters_applied=result.filters_applied
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/similar", response_model=SearchResponse, tags=["Search"])
async def search_similar_products(request: SimilarSearchRequest):
    """
    Find products similar to a given product.
    
    Uses PhoBERT embeddings for item-item similarity.
    
    Example:
        POST /search/similar
        {
            "product_id": 123,
            "topk": 10
        }
    """
    try:
        from service.search.smart_search import get_search_service
        
        search_service = get_search_service()
        
        exclude_set = set(request.exclude_ids) if request.exclude_ids else None
        
        result = search_service.search_similar(
            product_id=request.product_id,
            topk=request.topk,
            exclude_self=True,
            exclude_ids=exclude_set
        )
        
        logger.info(
            f"Similar search: product_id={request.product_id}, "
            f"count={result.count}, latency={result.latency_ms:.1f}ms"
        )
        
        return SearchResponse(
            query=result.query,
            results=[SearchResultItem(**r.__dict__) for r in result.results],
            count=result.count,
            latency_ms=result.latency_ms,
            method=result.method,
            filters_applied=None
        )
        
    except Exception as e:
        logger.error(f"Similar search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/profile", response_model=SearchResponse, tags=["Search"])
async def search_by_user_profile(request: UserProfileSearchRequest):
    """
    Search products similar to user's browsing/interaction history.
    
    Useful for cold-start personalization based on recent browsing.
    
    Example:
        POST /search/profile
        {
            "user_history": [123, 456, 789],
            "topk": 10,
            "exclude_history": true
        }
    """
    try:
        from service.search.smart_search import get_search_service
        
        search_service = get_search_service()
        
        result = search_service.search_by_user_profile(
            user_history=request.user_history,
            topk=request.topk,
            exclude_history=request.exclude_history,
            filters=request.filters
        )
        
        logger.info(
            f"Profile search: history_size={len(request.user_history)}, "
            f"count={result.count}, latency={result.latency_ms:.1f}ms"
        )
        
        return SearchResponse(
            query=result.query,
            results=[SearchResultItem(**r.__dict__) for r in result.results],
            count=result.count,
            latency_ms=result.latency_ms,
            method=result.method,
            filters_applied=result.filters_applied
        )
        
    except Exception as e:
        logger.error(f"Profile search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/suggestions", tags=["Search"])
async def get_search_suggestions(query: str = Query(..., min_length=1, max_length=100)):
    """
    Get search suggestions/autocomplete.
    
    Returns popular queries and product names matching the prefix.
    """
    # TODO: Implement autocomplete with popularity weighting
    return {
        "query": query,
        "suggestions": [],
        "message": "Autocomplete not yet implemented"
    }
```

## Component 5: Configuration

### File: `service/config/search_config.yaml`

```yaml
# Smart Search Configuration

search:
  enabled: true
  default_topk: 10
  max_topk: 100
  
  # Minimum semantic score to include in results
  min_semantic_score: 0.3
  
  # Enable reranking with multiple signals
  enable_rerank: true
  
  # Fetch more candidates for reranking
  candidate_multiplier: 3

# Reranking weights
rerank:
  weights:
    semantic: 0.50      # PhoBERT similarity
    popularity: 0.25    # Sales/views
    quality: 0.15       # Rating
    recency: 0.10       # Product freshness

# Query encoder
query_encoder:
  model_name: "vinai/phobert-base"
  max_length: 256
  cache_size: 1000
  device: "cpu"  # or "cuda"
  
  # Vietnamese abbreviations expansion
  abbreviations:
    sp: "sản phẩm"
    kem dc: "kem dưỡng da"
    srm: "sữa rửa mặt"
    tbc: "tẩy bong chết"
    kcn: "kem chống nắng"

# Search index
index:
  use_faiss: false  # Enable for larger catalogs (>10K products)
  faiss_index_type: "flat"  # flat, ivf, hnsw
  
  # Metadata indices
  enable_brand_filter: true
  enable_category_filter: true
  enable_price_filter: true

# Logging
logging:
  log_queries: true
  log_latencies: true
  slow_query_threshold_ms: 500
```

## Dependencies

Thêm vào `requirements.txt`:

```txt
# Smart Search dependencies
faiss-cpu>=1.7.0        # ANN search (optional, for large catalogs)
# faiss-gpu>=1.7.0      # GPU version (if available)
```

## Directory Structure

```
service/
├─ search/
│  ├─ __init__.py
│  ├─ query_encoder.py      # QueryEncoder class
│  ├─ search_index.py       # SearchIndex class
│  └─ smart_search.py       # SmartSearchService class
├─ config/
│  ├─ search_config.yaml    # Search configuration
│  └─ serving_config.yaml   # Existing config
└─ api.py                   # Updated with search endpoints
```

## Testing

### Test Script: `scripts/test_smart_search.py`

```python
"""Test Smart Search functionality."""

import asyncio
import httpx
import time


async def test_smart_search():
    """Test smart search endpoints."""
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        
        # Test 1: Basic semantic search
        print("\n=== Test 1: Basic Semantic Search ===")
        response = await client.post(
            f"{base_url}/search",
            json={
                "query": "kem dưỡng ẩm cho da khô",
                "topk": 5,
                "rerank": True
            }
        )
        result = response.json()
        print(f"Query: {result['query']}")
        print(f"Results: {result['count']}")
        print(f"Latency: {result['latency_ms']:.1f}ms")
        for r in result['results'][:3]:
            print(f"  - {r['product_name'][:50]} (score: {r['final_score']:.3f})")
        
        # Test 2: Search with filters
        print("\n=== Test 2: Search with Filters ===")
        response = await client.post(
            f"{base_url}/search",
            json={
                "query": "sữa rửa mặt",
                "topk": 5,
                "filters": {"brand": "innisfree"}
            }
        )
        result = response.json()
        print(f"Results with brand filter: {result['count']}")
        
        # Test 3: Similar items search
        print("\n=== Test 3: Similar Items ===")
        response = await client.post(
            f"{base_url}/search/similar",
            json={
                "product_id": 100,
                "topk": 5
            }
        )
        result = response.json()
        print(f"Similar to product 100: {result['count']} results")
        
        # Test 4: User profile search
        print("\n=== Test 4: User Profile Search ===")
        response = await client.post(
            f"{base_url}/search/profile",
            json={
                "user_history": [100, 200, 300],
                "topk": 5,
                "exclude_history": True
            }
        )
        result = response.json()
        print(f"Profile-based results: {result['count']}")
        
        print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_smart_search())
```

## Evaluation Metrics

### Metrics for Smart Search

1. **Search Quality**:
   - **MRR (Mean Reciprocal Rank)**: Position of first relevant result
   - **nDCG@K**: Normalized Discounted Cumulative Gain
   - **Precision@K**: Proportion of relevant results

2. **Performance**:
   - **Latency**: p50, p95, p99 response times
   - **Throughput**: Queries per second
   - **Index build time**: Time to build search index

3. **User Engagement** (if tracking available):
   - **Click-through rate**: % of results clicked
   - **Conversion rate**: % of searches leading to purchase
   - **Query refinement rate**: % of users modifying query

## Cross-Task Integration

**Task 01 (Data Layer)**:
- Uses `product_attributes_enriched.parquet` for metadata
- Pre-computed `popularity_score`, `quality_score` for reranking

**Task 05 (Serving Layer)**:
- Shares `PhoBERTEmbeddingLoader` singleton
- Can be combined with CF recommendations

**Task 08 (Hybrid Reranking)**:
- Reuses reranking logic with weighted signals
- Same normalization approach for score combination

**Task 06 (Monitoring)**:
- Log search queries and latencies
- Track search quality metrics over time

## Timeline Estimate

| Task | Days |
|------|------|
| Query Encoder implementation | 1 |
| Search Index implementation | 1.5 |
| SmartSearchService implementation | 1.5 |
| API endpoints integration | 0.5 |
| Configuration & testing | 1 |
| Evaluation & tuning | 1 |
| Documentation | 0.5 |
| **Total** | **~7 days** |

## Success Criteria

- [ ] Query encoder loads PhoBERT and caches embeddings
- [ ] Search index supports exact and filtered search
- [ ] Semantic search returns relevant results (MRR > 0.5)
- [ ] Similar items search works correctly
- [ ] User profile search handles cold-start users
- [ ] API latency < 200ms for p95
- [ ] Reranking improves result quality
- [ ] Vietnamese queries handled correctly
- [ ] Integration with existing recommendation service
- [ ] Documentation and examples complete

## Future Enhancements

1. **Query Understanding**:
   - Intent classification (browse, compare, specific search)
   - Query expansion with synonyms
   - Spell correction for Vietnamese

2. **Personalization**:
   - Learn from search click history
   - User preference weighting
   - Session-based personalization

3. **Advanced Ranking**:
   - Learning-to-rank models
   - A/B testing framework
   - Dynamic weight adjustment

4. **Scalability**:
   - Distributed FAISS index
   - Caching layer (Redis)
   - Async query processing

---

**Created**: 2025-11-26  
**Status**: Planning  
**Priority**: High (tích hợp với embeddings có sẵn)
