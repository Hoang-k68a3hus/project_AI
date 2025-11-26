# Task 05: Serving Layer

## Mục Tiêu

Xây dựng service layer để serve recommendations trong production, bao gồm model loading, recommendation generation, **user segmentation routing**, cold-start handling, filtering logic, và optional API endpoints. Service phải đảm bảo latency thấp, reliability cao, và dễ dàng integration.

## 🔄 Updated Serving Strategy (November 2025)

### Context: High Sparsity Data (~1.23 interactions/user, Updated ≥2 Threshold)
- **Trainable users** (≥2 interactions): ~26,000 users (~8.6% of total) → Serve with CF (ALS/BPR) + reranking
- **Cold-start users** (1 interaction or new): ~274,000 users (~91.4%) → Serve with content-based + popularity
- **Routing decision**: Load `user_metadata.pkl` to check `is_trainable_user` flag
- **Key insight**: 90%+ traffic will use content-based; CF is for the minority with ≥2 interactions

### Serving Flow by User Type:

```
User Request
    ↓
Check user_metadata
    ↓
├─ Trainable User? (≥2 interactions + ≥1 positive)
│  ├─ CF Recommender (ALS/BPR)
│  ├─ Generate Top-K candidates
│  ├─ Hybrid Reranking (CF + Content + Popularity)
│  └─ Return personalized results (~8.6% of traffic)
│
└─ Cold-Start User? (1 interaction or new user)
   ├─ Skip CF (no reliable user embedding)
   ├─ Item-Item Similarity (PhoBERT)
   │  └─ Find similar products to user's purchase history
   ├─ Mix with Popularity (Top sellers)
   └─ Return content-based + popular results (~91.4% of traffic)
```

**Benefits**:
- Don't waste CF computation on users with insufficient data
- Content-based provides better recommendations for sparse users than weak CF embeddings
- Clear separation of concerns for monitoring and A/B testing
- **Traffic optimization**: ~91.4% traffic uses fast content-based path; only ~8.6% uses CF

## Architecture Overview

```
service/
├── recommender/
│   ├── __init__.py          # Package exports
│   ├── loader.py            # CFModelLoader (singleton)
│   ├── recommender.py       # CFRecommender (main engine)
│   ├── fallback.py          # FallbackRecommender (cold-start)
│   ├── phobert_loader.py    # PhoBERTEmbeddingLoader (singleton)
│   ├── rerank.py            # HybridReranker (hybrid reranking)
│   ├── filters.py           # Attribute filtering & boosting
│   └── cache.py             # CacheManager (LRU caching & warm-up)
├── search/                  # Smart Search (Task 09)
│   ├── query_encoder.py
│   ├── search_index.py
│   └── smart_search.py
├── api.py                   # FastAPI REST API
├── dashboard.py             # Monitoring dashboard
└── config/
    ├── serving_config.yaml
    ├── rerank_config.yaml
    └── search_config.yaml
```

### Package Exports

```python
from service.recommender import (
    # Loaders
    CFModelLoader, get_loader,
    PhoBERTEmbeddingLoader, get_phobert_loader,
    
    # Core
    CFRecommender, RecommendationResult,
    FallbackRecommender,
    
    # Hybrid Reranking
    HybridReranker, get_reranker, RerankedResult,
    
    # Filtering
    apply_filters, boost_by_attributes,
    
    # Caching
    CacheManager, get_cache_manager
)
```

## Component 1: Model Loader

### Module: `service/recommender/loader.py`

#### Class: `CFModelLoader`

##### Purpose
Singleton class for loading CF models, mappings, metadata, and trainable user routing information. Handles hot-reload when registry updates.

##### Initialization

```python
from service.recommender.loader import CFModelLoader, get_loader

# Option 1: Direct instantiation (singleton pattern)
loader = CFModelLoader(
    registry_path='artifacts/cf/registry.json',
    data_dir='data/processed',
    published_dir='data/published_data',
    auto_load=False  # Set True to auto-load on init
)

# Option 2: Singleton getter (recommended)
loader = get_loader()  # Returns singleton instance
```

##### Attributes

```python
class CFModelLoader:
    # Cached state
    current_model: Optional[Dict[str, Any]]  # Loaded model dict
    current_model_id: Optional[str]
    mappings: Optional[Dict[str, Any]]  # User/item ID mappings
    trainable_user_mapping: Optional[Dict[int, int]]  # u_idx -> u_idx_cf
    trainable_user_set: Optional[Set[int]]  # Set of trainable u_idx
    item_metadata: Optional[pd.DataFrame]  # Product metadata
    user_history_cache: Optional[Dict[int, Set[int]]]  # user_id -> {product_ids}
    top_k_popular_items: Optional[List[int]]  # Pre-computed popular items
    data_stats: Optional[Dict[str, Any]]  # Data statistics
```

##### Method 1: `load_model(model_id=None)`

```python
model = loader.load_model(model_id=None)

# Returns:
# {
#     'model_id': 'als_v2_20250116_141500',
#     'model_type': 'als',
#     'U': np.ndarray (num_trainable_users, factors),
#     'V': np.ndarray (num_items, factors),
#     'params': dict,
#     'metadata': dict,
#     'score_range': {'min': 0.0, 'max': 1.5, 'p01': ..., 'p99': ...},
#     'loaded_at': '2025-01-16T14:30:00'
# }

# Features:
# - Auto-detects current_best if model_id=None
# - Handles U/V matrix swap detection (from Colab training)
# - Loads score_range for normalization
# - Caches model in memory
```

##### Method 2: `load_mappings(data_version=None)`

```python
mappings = loader.load_mappings(data_version=None)

# Returns:
# {
#     'user_to_idx': {user_id: u_idx},
#     'idx_to_user': {u_idx: user_id},
#     'item_to_idx': {product_id: i_idx},
#     'idx_to_item': {i_idx: product_id},
#     'metadata': {
#         'num_users': 300000,
#         'num_items': 2244,
#         'num_trainable_users': 26000,
#         'data_hash': 'abc123...'
#     }
# }

# Also automatically loads:
# - trainable_user_mapping (u_idx -> u_idx_cf)
# - top_k_popular_items
# - data_stats
```

##### Method 3: `load_item_metadata()`

```python
metadata = loader.load_item_metadata()

# Returns: pd.DataFrame with product info
# Tries enriched_products.parquet first, falls back to raw CSVs
# Columns: product_id, product_name, brand, price, avg_star, num_sold_time, ...
```

##### Method 4: `load_user_histories()`

```python
histories = loader.load_user_histories()

# Returns: Dict[int, Set[int]] - {user_id: {product_ids}}
# IMPORTANT: Only loads TRAIN split to avoid data leakage
# Cached in memory for fast seen-item filtering
```

##### Method 5: `is_trainable_user(user_id)`

```python
is_trainable = loader.is_trainable_user(user_id=12345)

# Returns: bool
# True if user has ≥2 interactions AND ≥1 positive rating
# Used for routing: CF vs content-based
```

##### Method 6: `get_cf_user_index(user_id)`

```python
u_idx_cf = loader.get_cf_user_index(user_id=12345)

# Returns: int or None
# CF matrix row index (u_idx_cf) for trainable users
# None if user is not trainable
```

##### Method 7: `get_user_history(user_id)`

```python
history = loader.get_user_history(user_id=12345)

# Returns: Set[int] - Set of product_ids user has interacted with
# Uses cached user_history_cache (train split only)
```

##### Method 8: `reload_if_updated()`

```python
reloaded = loader.reload_if_updated()

# Returns: bool - True if model was reloaded
# Checks registry for new current_best and reloads if changed
```

##### Method 9: `get_popular_items(topk=50)`

```python
popular_indices = loader.get_popular_items(topk=50)

# Returns: List[int] - Top-K popular item indices
# Uses pre-computed top_k_popular_items from data processing
```

## Component 2: Core Recommender

### Module: `service/recommender/recommender.py`

#### Class: `CFRecommender`

##### Purpose
Main recommendation engine with user segmentation routing, CF scoring, hybrid reranking, and fallback handling.

##### Initialization

```python
from service.recommender import CFRecommender

recommender = CFRecommender(
    loader=None,  # Uses get_loader() singleton if None
    phobert_loader=None,  # Lazy-loaded if None
    auto_load=True,  # Auto-load models and data on init
    enable_reranking=True,  # Enable hybrid reranking by default
    rerank_config_path=None  # Path to rerank config YAML
)
```

##### Data Class: `RecommendationResult`

```python
@dataclass
class RecommendationResult:
    user_id: int
    recommendations: List[Dict[str, Any]]  # Enriched recommendations
    count: int
    is_fallback: bool  # True if used fallback (cold-start)
    fallback_method: Optional[str]  # 'popularity', 'item_similarity', 'hybrid'
    latency_ms: float
    model_id: Optional[str]  # CF model ID (None for fallback)
```

##### Method 1: `recommend()`

```python
result = recommender.recommend(
    user_id=12345,
    topk=10,
    exclude_seen=True,
    filter_params={'brand': 'Innisfree'},  # Optional
    normalize_scores=False,  # Normalize CF scores to [0, 1]
    rerank=None  # Override default reranking (None = use default)
)

# Returns: RecommendationResult

# Workflow:
# 1. Check if user is trainable (is_trainable_user)
# 2. If trainable:
#    - Get CF user index (get_cf_user_index)
#    - Compute CF scores: U[u_idx_cf] @ V.T
#    - Exclude seen items
#    - Apply attribute filters
#    - Get top-K candidates (5x if reranking)
#    - Apply hybrid reranking if enabled
# 3. If cold-start:
#    - Use FallbackRecommender
#    - Strategy: 'hybrid' (content + popularity)
#    - Apply reranking to fallback results
```

##### Method 2: `batch_recommend()`

```python
results = recommender.batch_recommend(
    user_ids=[12345, 67890, 11111],
    topk=10,
    exclude_seen=True
)

# Returns: Dict[int, RecommendationResult]
# Uses vectorized CF scoring for efficiency
# Separates trainable vs cold-start users
# Batch matrix multiplication: U[u_indices] @ V.T
```

##### Method 3: `similar_items()`

```python
similar = recommender.similar_items(
    product_id=123,
    topk=10,
    use_cf=True  # If False, uses PhoBERT embeddings
)

# Returns: List[Dict[str, Any]] - Similar items with metadata
# If use_cf=True: Uses V @ V.T for CF-based similarity
# If use_cf=False: Uses PhoBERT embeddings for content similarity
```

##### Method 4: `reload_model()`

```python
reloaded = recommender.reload_model()

# Returns: bool - True if model was reloaded
# Checks registry for updates and reloads if current_best changed
# Updates U, V, model_id, score_range
```

##### Method 5: `get_model_info()`

```python
info = recommender.get_model_info()

# Returns:
# {
#     'model_id': 'als_v2_20250116_141500',
#     'model_type': 'als',
#     'num_users': 26000,
#     'num_items': 2244,
#     'factors': 128,
#     'trainable_users': 26000,
#     'reranking_enabled': True,
#     ...
# }
```

##### Properties

```python
# Lazy-loaded fallback recommender
fallback = recommender.fallback  # Returns FallbackRecommender instance

# Lazy-loaded hybrid reranker
reranker = recommender.reranker  # Returns HybridReranker instance
```

##### Method 2: `_get_user_history(user_id)`
```python
def _get_user_history(self, user_id):
    """
    Retrieve items user đã interact.
    
    Args:
        user_id: Original user ID
    
    Returns:
        set: Set of product_ids
    """
    cached = self.user_history_cache.get(user_id)
    return set(cached) if cached else set()
```

##### Method 3: `_apply_filters(filter_params)`
```python
def _apply_filters(self, filter_params):
    """
    Apply attribute filters.
    
    Args:
        filter_params: Dict như {'brand': 'Innisfree', 'skin_type': 'oily'}
    
    Returns:
        np.array: Indices of valid items
    """
    mask = pd.Series([True] * len(self.item_metadata))
    
    for key, value in filter_params.items():
        if key in self.item_metadata.columns:
            mask &= self.item_metadata[key] == value
    
    valid_product_ids = self.item_metadata[mask]['product_id'].values
    valid_indices = [self.mappings['item_to_idx'][str(pid)] for pid in valid_product_ids]
    
    return np.array(valid_indices)
```

##### Method 4: `batch_recommend(user_ids, topk=10, exclude_seen=True)`
```python
def batch_recommend(self, user_ids, topk=10, exclude_seen=True):
    """
    Batch recommendation cho nhiều users (efficient).
    
    Args:
        user_ids: List of user IDs
        topk: Number of recommendations per user
        exclude_seen: Filter seen items
    
    Returns:
        dict: {user_id: [recommendations]}
    """
    results = {}
    
    # Separate cold-start users
    known_users = [uid for uid in user_ids if uid in self.mappings['user_to_idx']]
    cold_users = [uid for uid in user_ids if uid not in self.mappings['user_to_idx']]
    
    # Batch scoring cho known users
    if known_users:
        u_indices = [self.mappings['user_to_idx'][uid] for uid in known_users]
        scores_batch = self.U[u_indices] @ self.V.T  # (len(known_users), num_items)
        
        for i, uid in enumerate(known_users):
            scores = scores_batch[i]
            
            if exclude_seen:
                seen = self._get_user_history(uid)
                seen_indices = [self.mappings['item_to_idx'][str(pid)] for pid in seen]
                scores[seen_indices] = -np.inf
            
            top_k_indices = np.argsort(scores)[::-1][:topk]
            product_ids = [self.mappings['idx_to_item'][str(i)] for i in top_k_indices]
            
            # Enrich (simplified)
            results[uid] = [{'product_id': pid, 'score': float(scores[i])} for pid, i in zip(product_ids, top_k_indices)]
    
    # Fallback cho cold-start
    for uid in cold_users:
        results[uid] = self._fallback_recommendations(topk)
    
    return results
```

## Component 3: Cold-Start Fallback

### Module: `service/recommender/fallback.py`

#### Class: `FallbackRecommender`

##### Purpose
Handles cold-start users (~91.4% of traffic) with content-based and popularity-based recommendations. Optimized with LRU caching for low latency.

##### Initialization

```python
from service.recommender.fallback import FallbackRecommender

fallback = FallbackRecommender(
    cf_loader=loader,  # CFModelLoader instance
    phobert_loader=None,  # Lazy-loaded if None
    default_content_weight=0.7,
    default_popularity_weight=0.3,
    enable_cache=True  # Enable LRU caching
)
```

##### Strategy Overview
For users with <2 interactions (cold-start), skip CF and use:
1. **Item-Item Similarity** (PhoBERT embeddings)
2. **Popularity** (Top-selling products)
3. **Hybrid** (Weighted combination of both)

##### Method 1: `recommend()`

```python
recs = fallback.recommend(
    user_id=12345,  # Optional, to fetch history
    user_history=[100, 200, 300],  # Overrides user_id lookup
    topk=10,
    strategy='hybrid',  # 'popularity', 'item_similarity', or 'hybrid'
    exclude_ids={400, 500},  # Product IDs to exclude
    filter_params={'brand': 'Innisfree'}  # Optional filters
)

# Returns: List[Dict[str, Any]] - Recommendations with metadata
```

##### Method 2: `fallback_popularity()`

```python
recs = fallback.fallback_popularity(
    topk=10,
    exclude_ids=None,
    filter_params=None
)

# Returns: List of popular products
# Uses pre-computed top_k_popular_items from loader
# Optimized: Uses cached enriched popular items if available
```

##### Method 3: `fallback_item_similarity()`

```python
recs = fallback.fallback_item_similarity(
    user_history=[100, 200, 300],
    topk=10,
    exclude_ids=None,
    filter_params=None
)

# Returns: List of content-similar products
# Uses PhoBERTEmbeddingLoader.compute_user_profile()
# Caches user profiles for performance
# Falls back to popularity if PhoBERT unavailable
```

##### Method 4: `hybrid_fallback()`

```python
recs = fallback.hybrid_fallback(
    user_history=[100, 200, 300],
    topk=10,
    content_weight=0.7,  # Weight for content similarity
    popularity_weight=0.3,  # Weight for popularity
    exclude_ids=None,
    filter_params=None
)

# Returns: List of hybrid recommendations
# Combines content similarity and popularity scores
# Weighted combination: final_score = content_weight * content + popularity_weight * pop
```

## Component 4: PhoBERT Embedding Loader

### Module: `service/recommender/phobert_loader.py`

#### Class: `PhoBERTEmbeddingLoader`

##### Purpose
Singleton class for loading and using PhoBERT product embeddings for content-based recommendations. Pre-normalizes embeddings for fast cosine similarity.

##### Initialization

```python
from service.recommender.phobert_loader import PhoBERTEmbeddingLoader, get_phobert_loader

# Option 1: Direct instantiation (singleton)
phobert = PhoBERTEmbeddingLoader(
    embeddings_path='data/processed/content_based_embeddings/product_embeddings.pt',
    auto_load=True
)

# Option 2: Singleton getter (recommended)
phobert = get_phobert_loader()
```

##### Method 1: `get_embedding(product_id)`

```python
emb = phobert.get_embedding(product_id=123)

# Returns: np.ndarray (768,) or (1024,) - Raw embedding
# None if product not found
```

##### Method 2: `get_embedding_normalized(product_id)`

```python
emb_norm = phobert.get_embedding_normalized(product_id=123)

# Returns: L2-normalized embedding for fast cosine similarity
```

##### Method 3: `compute_user_profile()`

```python
profile = phobert.compute_user_profile(
    user_history_items=[100, 200, 300],
    weights=[1.0, 1.5, 1.0],  # Optional weights (e.g., ratings)
    strategy='weighted_mean'  # 'mean', 'weighted_mean', or 'max'
)

# Returns: np.ndarray (768,) - User profile embedding
# Aggregates history items into single embedding
```

##### Method 4: `find_similar_items()`

```python
similar = phobert.find_similar_items(
    product_id=123,
    topk=10,
    exclude_self=True,
    exclude_ids={400, 500}
)

# Returns: List[Tuple[int, float]] - [(product_id, similarity_score), ...]
# Uses pre-normalized embeddings for fast cosine similarity
```

##### Method 5: `find_similar_to_profile()`

```python
similar = phobert.find_similar_to_profile(
    user_profile=profile_embedding,
    topk=10,
    exclude_ids={100, 200, 300}
)

# Returns: List[Tuple[int, float]] - Similar items to user profile
```

##### Method 6: `precompute_item_similarity()`

```python
phobert.precompute_item_similarity(max_items=3000)

# Precomputes V @ V.T similarity matrix for small catalogs
# Speeds up repeated similar item queries
```

## Component 5: Hybrid Reranking

### Module: `service/recommender/rerank.py`

#### Class: `HybridReranker`

##### Purpose
Hybrid reranker combining CF, content (PhoBERT), popularity, and quality signals. Uses global normalization for consistent scoring across requests.

##### Initialization

```python
from service.recommender.rerank import HybridReranker, get_reranker, RerankerConfig

# Option 1: Direct instantiation
reranker = HybridReranker(
    phobert_loader=phobert_loader,
    item_metadata=item_metadata,
    config=None,  # Uses default RerankerConfig if None
    config_path='service/config/rerank_config.yaml'  # Optional YAML config
)

# Option 2: Singleton getter (recommended)
reranker = get_reranker(
    phobert_loader=phobert_loader,
    item_metadata=item_metadata,
    config_path='service/config/rerank_config.yaml'
)
```

##### Data Class: `RerankerConfig`

```python
@dataclass
class RerankerConfig:
    # Weights for trainable users (≥2 interactions)
    weights_trainable: Dict[str, float] = {
        'cf': 0.30,         # SECONDARY - Collaborative signal
        'content': 0.40,    # PRIMARY - PhoBERT semantic similarity
        'popularity': 0.20, # TERTIARY - Trending items
        'quality': 0.10     # BONUS - High-rated products
    }
    
    # Weights for cold-start users (<2 interactions)
    weights_cold_start: Dict[str, float] = {
        'content': 0.60,    # DOMINANT - Only reliable signal
        'popularity': 0.30, # Social proof
        'quality': 0.10     # Bonus
    }
    
    # Diversity settings
    diversity_enabled: bool = True
    diversity_penalty: float = 0.1
    diversity_threshold: float = 0.85  # BERT similarity threshold
    
    # Normalization ranges (global, not local)
    cf_score_min: float = 0.0
    cf_score_max: float = 1.5
    content_score_min: float = -1.0
    content_score_max: float = 1.0
    quality_min: float = 1.0
    quality_max: float = 5.0
```

##### Data Class: `RerankedResult`

```python
@dataclass
class RerankedResult:
    recommendations: List[Dict[str, Any]]
    latency_ms: float
    diversity_score: float
    weights_used: Dict[str, float]
    num_candidates: int
    num_output: int
```

##### Method 1: `rerank()`

```python
result = reranker.rerank(
    cf_recommendations=cf_recs,  # List from CFRecommender
    user_id=12345,
    user_history=[100, 200, 300],
    topk=10,
    is_cold_start=False  # True for cold-start users
)

# Returns: RerankedResult
# Workflow:
# 1. Compute signals: CF, content, popularity, quality
# 2. Normalize signals using global ranges (not local min/max)
# 3. Combine with weights (trainable vs cold-start)
# 4. Apply diversity penalty if enabled
# 5. Re-sort and return top-K
```

##### Method 2: `rerank_cold_start()`

```python
result = reranker.rerank_cold_start(
    recommendations=fallback_recs,
    user_history=[100, 200],
    topk=10
)

# Returns: RerankedResult
# Uses weights_cold_start (no CF signal)
```

##### Method 3: `update_config()`

```python
reranker.update_config(
    weights_trainable={'cf': 0.35, 'content': 0.35, 'popularity': 0.20, 'quality': 0.10},
    weights_cold_start={'content': 0.65, 'popularity': 0.25, 'quality': 0.10}
)

# Dynamically update reranking weights
```

##### Convenience Functions

```python
from service.recommender.rerank import (
    rerank_with_signals,  # Legacy function
    rerank_cold_start,    # Legacy function
    diversify_recommendations  # Diversity-only function
)
```

## Component 6: Cache Manager

### Module: `service/recommender/cache.py`

#### Class: `CacheManager`

##### Purpose
LRU caching and warm-up strategies for optimizing cold-start path latency (~91% of traffic).

##### Features
- LRU caches for user profiles, item similarities, fallback results
- Pre-computation of popular items and their similarities
- Warm-up strategies for cold-start recommendations
- Cache invalidation hooks for model updates

##### Initialization

```python
from service.recommender.cache import CacheManager, get_cache_manager, CacheConfig

# Option 1: Direct instantiation
cache = CacheManager(
    config=CacheConfig(
        max_user_profiles=10000,
        max_popular_items=1000,
        ttl_seconds=3600
    )
)

# Option 2: Singleton getter (recommended)
cache = get_cache_manager()
```

##### Method: `warmup()`

```python
stats = cache.warmup()

# Pre-computes:
# - Popular items with enriched metadata
# - Popular item similarities
# - Common user profiles
# Returns warmup statistics
```

##### Method: `get_popular_items_enriched()`

```python
popular = cache.get_popular_items_enriched()

# Returns: List[Dict] - Pre-computed popular items with metadata
# Used by FallbackRecommender for fast popularity fallback
```

##### Method: `get_user_profile(user_id_hash)`

```python
profile = cache.get_user_profile(user_id_hash)

# Returns: Cached user profile embedding or None
# Used by FallbackRecommender to avoid recomputing profiles
```

##### Method: `get_stats()`

```python
stats = cache.get_stats()

# Returns: Dict with cache hit rates, sizes, etc.
```

## Component 7: API Layer (FastAPI)

### Module: `service/api.py`

#### FastAPI Application

```python
from fastapi import FastAPI
from service.recommender import CFRecommender

app = FastAPI(
    title="CF Recommendation Service",
    description="Collaborative Filtering recommendation API for Vietnamese cosmetics",
    version="1.0.0",
    lifespan=lifespan  # Startup/shutdown handlers
)
```

#### Endpoints

##### 1. Health Check

```python
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check service health and model status.
    
    Returns:
        HealthResponse with model information
    """
    # Returns: status, model_id, model_type, num_users, num_items, trainable_users
```

##### 2. Single User Recommendation

```python
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Get recommendations for a single user.
    
    Request:
        {
            "user_id": 12345,
            "topk": 10,
            "exclude_seen": true,
            "filter_params": {"brand": "Innisfree"},
            "rerank": false,
            "rerank_weights": {...}  # Optional override
        }
    
    Returns:
        RecommendResponse with recommendations and metadata
    """
```

##### 3. Batch Recommendation

```python
@app.post("/batch_recommend", response_model=BatchResponse)
async def batch_recommend(request: BatchRequest):
    """
    Get recommendations for multiple users.
    
    Returns:
        BatchResponse with results for all users
        Includes: cf_users, fallback_users counts
    """
```

##### 4. Similar Items

```python
@app.post("/similar_items", response_model=SimilarItemsResponse)
async def similar_items(request: SimilarItemsRequest):
    """
    Find similar items to a given product.
    
    Request:
        {
            "product_id": 123,
            "topk": 10,
            "use_cf": true  # If false, uses PhoBERT
        }
    """
```

##### 5. Model Reload

```python
@app.post("/reload_model", response_model=ReloadResponse)
async def reload_model():
    """
    Hot-reload model from registry.
    
    Returns:
        ReloadResponse with reload status
    """
```

##### 6. Service Statistics

```python
@app.get("/stats")
async def service_stats():
    """
    Get service statistics.
    
    Returns:
        Model info, user counts, cache stats
    """
```

##### 7. Cache Management

```python
@app.get("/cache_stats")
async def cache_stats():
    """Get detailed cache statistics."""

@app.post("/cache_warmup")
async def trigger_warmup(force: bool = False):
    """Trigger cache warmup."""

@app.post("/cache_clear")
async def clear_cache():
    """Clear all caches."""
```

##### 8. Smart Search Endpoints (Task 09)

```python
@app.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """Semantic search for products using Vietnamese query."""

@app.post("/search/similar", response_model=SearchResponse)
async def search_similar_products(request: SearchSimilarRequest):
    """Find products similar to a given product."""

@app.post("/search/profile", response_model=SearchResponse)
async def search_by_profile(request: SearchByProfileRequest):
    """Search based on user profile/history."""
```

## Configuration

### File: `service/config/serving_config.yaml`

```yaml
model:
  registry_path: "artifacts/cf/registry.json"
  data_dir: "data/processed"
  published_dir: "data/published_data"

serving:
  default_topk: 10
  max_topk: 100
  exclude_seen_default: true
  enable_reranking: true

fallback:
  default_content_weight: 0.7
  default_popularity_weight: 0.3
  enable_cache: true

cache:
  max_user_profiles: 10000
  max_popular_items: 1000
  ttl_seconds: 3600

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  log_level: "info"
```

### File: `service/config/rerank_config.yaml`

```yaml
reranking:
  # Weights for trainable users (≥2 interactions)
  weights_trainable:
    cf: 0.30
    content: 0.40
    popularity: 0.20
    quality: 0.10
  
  # Weights for cold-start users
  weights_cold_start:
    content: 0.60
    popularity: 0.30
    quality: 0.10
  
  diversity:
    enabled: true
    penalty: 0.1
    threshold: 0.85
  
  normalization:
    cf_score_min: 0.0
    cf_score_max: 1.5
    content_score_min: -1.0
    content_score_max: 1.0
    quality_min: 1.0
    quality_max: 5.0
```

## Logging & Monitoring

### Request Logging

The API automatically logs all requests to:
- **Console**: Structured logging with user_id, topk, latency, fallback status
- **Metrics DB**: SQLite database (`logs/service_metrics.db`) for aggregation

```python
# Automatic logging in API endpoints
logger.info(
    f"user_id={request.user_id}, topk={request.topk}, "
    f"count={result.count}, fallback={result.is_fallback}, "
    f"latency={latency:.1f}ms"
)

# Background logging to metrics DB
log_request_metrics(
    user_id=request.user_id,
    topk=request.topk,
    latency_ms=latency,
    num_recommendations=result.count,
    fallback=result.is_fallback,
    fallback_method=result.fallback_method,
    rerank_enabled=request.rerank,
    error=None
)
```

### Metrics to Track
- **Latency**: p50, p95, p99 response time (logged per request)
- **Throughput**: Requests per second (aggregated hourly)
- **Fallback rate**: % requests using fallback (~91.4% expected)
- **Error rate**: % failed requests
- **Cache hit rate**: From CacheManager stats
- **Reranking overhead**: Latency difference with/without reranking

### Background Tasks

```python
# Periodic health aggregation (every minute)
async def periodic_health_aggregation():
    # Aggregates metrics from requests table
    # Updates service_health table
    # Runs in background task
```

## Performance Optimizations

### 1. Singleton Pattern
- **CFModelLoader**: Single instance shared across requests
- **PhoBERTEmbeddingLoader**: Single instance, embeddings loaded once
- **HybridReranker**: Single instance, config cached
- **CacheManager**: Single instance, shared LRU caches

### 2. Pre-normalized Embeddings
- PhoBERT embeddings pre-normalized on load
- Fast cosine similarity: `embeddings_norm @ query_norm` (no per-request normalization)

### 3. Pre-computed Popular Items
- Top-K popular items loaded once from `top_k_popular_items.json`
- Cached enriched popular items in CacheManager
- Fast fallback for truly new users

### 4. User History Caching
- User histories loaded once (train split only)
- Cached in `CFModelLoader.user_history_cache`
- Fast seen-item filtering

### 5. Batch Inference
- `batch_recommend()` uses vectorized CF scoring
- Batch matrix multiplication: `U[u_indices] @ V.T`
- Amortizes overhead across multiple users

### 6. LRU Caching (CacheManager)
- User profiles cached (avoid recomputing from history)
- Popular items enriched and cached
- Similarity results cached for repeated queries

### 7. Lazy Loading
- PhoBERT loader initialized only when needed
- Fallback recommender created on first cold-start request
- Reranker initialized only when reranking enabled

### 8. Hot Reload
- Model reload without service restart
- Registry checked periodically
- Seamless transition to new best model

## Deployment

### Docker Container
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY service/ service/
COPY artifacts/ artifacts/
COPY data/processed/ data/processed/

CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Run Locally
```bash
# Install dependencies
pip install fastapi uvicorn

# Start service
uvicorn service.api:app --reload --port 8000
```

### Test Endpoint
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "topk": 10,
    "exclude_seen": true
  }'
```

## Component 8: Attribute Filtering

### Module: `service/recommender/filters.py`

#### Function: `apply_filters()`

```python
from service.recommender.filters import apply_filters

filtered = apply_filters(
    recommendations=recs,
    filter_params={
        'brand': 'Innisfree',
        'skin_type': ['oily', 'acne'],  # List support
        'price_min': 100000,
        'price_max': 500000,
        'min_rating': 4.0
    }
)

# Returns: Filtered recommendations with updated ranks
```

#### Function: `boost_by_attributes()`

```python
from service.recommender.filters import boost_by_attributes

boosted = boost_by_attributes(
    recommendations=recs,
    boost_config={
        'brand': {'Innisfree': 1.2, 'Cetaphil': 1.1},
        'skin_type': {'oily': 1.1}
    },
    metadata=item_metadata
)

# Returns: Recommendations with boosted scores
```

#### Function: `infer_user_preferences()`

```python
from service.recommender.filters import infer_user_preferences

prefs = infer_user_preferences(
    user_history=[100, 200, 300],
    metadata=item_metadata
)

# Returns: Dict with inferred brand, skin_type preferences
# Used for personalized boosting
```

## Timeline Estimate

- **Loader + Recommender**: 2 days
- **Fallback logic**: 0.5 day
- **API endpoints**: 1 day
- **BERT integration + Reranking**: 2 days
- **Testing**: 1 day
- **Deployment setup**: 0.5 day
- **Total**: ~7 days

## Integration Points

### Task 01 (Data Layer)
- Uses `user_item_mappings.json` for ID mappings
- Uses `trainable_user_mapping.json` for routing
- Uses `top_k_popular_items.json` for fallback
- Uses `enriched_products.parquet` for metadata
- Uses `interactions.parquet` (train split) for user histories

### Task 02 (Training Pipelines)
- Loads models from registry (Task 04)
- Uses `score_range` from model metadata for normalization

### Task 03 (Evaluation Metrics)
- Uses evaluation metrics for model comparison
- Can integrate with evaluation pipeline

### Task 04 (Model Registry)
- Uses `ModelLoader` from registry for model loading
- Hot-reload when registry updates

### Task 06 (Monitoring)
- Logs requests to `ServiceMetricsDB`
- Tracks latency, fallback rate, error rate

### Task 08 (Hybrid Reranking)
- Uses `HybridReranker` for weighted signal combination
- Uses `RerankerConfig` for weight management

### Task 09 (Smart Search)
- Integrated search endpoints in API
- Shares `PhoBERTEmbeddingLoader` with recommender

## Success Criteria

- [x] Load model từ registry (<1 second)
- [x] Generate recommendations (<100ms per user CF-only)
- [x] Two-stage reranking (<200ms with BERT)
- [x] BERT embeddings loaded and cached
- [x] Cold-start fallback works (popularity, content, hybrid)
- [x] API endpoints functional (recommend, batch, similar, search)
- [x] Hot-reload model without downtime
- [x] Logging tracks latency, fallback rate, rerank metrics
- [x] Cache manager with warm-up for cold-start optimization
- [x] User segmentation routing (trainable vs cold-start)
- [x] Thread-safe singleton loaders
- [x] Docker deployment ready
