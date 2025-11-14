# Task 05: Serving Layer

## Mục Tiêu

Xây dựng service layer để serve recommendations trong production, bao gồm model loading, recommendation generation, cold-start handling, filtering logic, và optional API endpoints. Service phải đảm bảo latency thấp, reliability cao, và dễ dàng integration.

## Architecture Overview

```
service/
├── recommender/
│   ├── __init__.py
│   ├── loader.py           # Load models từ registry
│   ├── recommender.py      # Core recommendation logic
│   ├── rerank.py          # Hybrid reranking (optional)
│   ├── filters.py         # Attribute filtering
│   └── fallback.py        # Cold-start handling
├── api.py                 # REST API (FastAPI)
└── config/
    └── serving_config.yaml
```

## Component 1: Model Loader

### Module: `service/recommender/loader.py`

#### Class: `CFModelLoader`

##### Purpose
Singleton class quản lý model loading, caching, và hot-reload khi registry updates.

##### Attributes
```python
class CFModelLoader:
    def __init__(self, registry_path='artifacts/cf/registry.json'):
        self.registry_path = registry_path
        self.current_model = None  # Cached model
        self.current_model_id = None
        self.mappings = None  # User/item mappings
        self.item_metadata = None  # Product info
```

##### Method 1: `load_model(model_id=None)`
```python
def load_model(self, model_id=None):
    """
    Load CF model từ registry.
    
    Args:
        model_id: Optional model ID. Nếu None → load current_best
    
    Returns:
        dict: {
            'model_id': str,
            'model_type': 'als' | 'bpr',
            'U': np.array (num_users, factors),
            'V': np.array (num_items, factors),
            'params': dict,
            'metadata': dict
        }
    
    Raises:
        FileNotFoundError: Model artifacts không tồn tại
        ValueError: Invalid model_id
    """
    # Load registry
    registry = load_registry_json(self.registry_path)
    
    # Determine model to load
    if model_id is None:
        model_id = registry['current_best']['model_id']
    
    # Get model info
    model_info = registry['models'][model_id]
    model_path = model_info['path']
    
    # Load embeddings
    U = np.load(f"{model_path}/{model_info['model_type']}_U.npy")
    V = np.load(f"{model_path}/{model_info['model_type']}_V.npy")
    
    # Load params
    # (đảm bảo module đã import os)
    param_pattern = os.path.join(model_path, f"{model_info['model_type']}_params.json")
    with open(param_pattern) as f:
        params = json.load(f)
    
    # Cache
    self.current_model = {
        'model_id': model_id,
        'model_type': model_info['model_type'],
        'U': U,
        'V': V,
        'params': params,
        'metadata': model_info
    }
    self.current_model_id = model_id
    
    return self.current_model
```

##### Method 2: `load_mappings(data_version=None)`
```python
def load_mappings(self, data_version=None):
    """
    Load user/item ID mappings.
    
    Args:
        data_version: Optional hash. Nếu None → load latest
    
    Returns:
        dict: {
            'user_to_idx': {user_id: u_idx},
            'idx_to_user': {u_idx: user_id},
            'item_to_idx': {product_id: i_idx},
            'idx_to_item': {i_idx: product_id},
            'metadata': {...}
        }
    """
    path = 'data/processed/user_item_mappings.json'
    
    with open(path) as f:
        mappings = json.load(f)
    
    # Validate data version nếu có
    if data_version and mappings['metadata']['data_hash'] != data_version:
        warnings.warn(f"Data version mismatch: {data_version} vs {mappings['metadata']['data_hash']}")
    
    self.mappings = mappings
    return mappings
```

##### Method 3: `load_item_metadata()`
```python
def load_item_metadata(self):
    """
    Load product metadata cho enrichment.
    
    Returns:
        pd.DataFrame: Products với columns [product_id, product_name, brand, ...]
    """
    products = pd.read_csv('data/published_data/data_product.csv', encoding='utf-8')
    attributes = pd.read_csv('data/published_data/data_product_attribute.csv', encoding='utf-8')
    
    # Merge
    metadata = products.merge(attributes, on='product_id', how='left')
    
    # Cache
    self.item_metadata = metadata
    return metadata
```

##### Method 4: `reload_if_updated()`
```python
def reload_if_updated(self):
    """
    Check registry for updates và reload nếu current_best changed.
    
    Returns:
        bool: True nếu reloaded, False otherwise
    """
    registry = load_registry_json(self.registry_path)
    new_best_id = registry['current_best']['model_id']
    
    if new_best_id != self.current_model_id:
        logger.info(f"Registry updated: {self.current_model_id} → {new_best_id}")
        self.load_model(new_best_id)
        return True
    
    return False
```

## Component 2: Core Recommender

### Module: `service/recommender/recommender.py`

#### Class: `CFRecommender`

##### Purpose
Main recommendation engine với scoring, filtering, ranking logic.

##### Initialization
```python
class CFRecommender:
    def __init__(self, model_loader: CFModelLoader):
        self.loader = model_loader
        
        # Load artifacts
        self.model = model_loader.load_model()
        self.mappings = model_loader.load_mappings(
            data_version=self.model['metadata'].get('data_version')
        )
        self.item_metadata = model_loader.load_item_metadata()
        
        # Precompute
        self.U = self.model['U']
        self.V = self.model['V']
        self.num_items = self.V.shape[0]
        self.user_history_cache = self._load_user_histories()
```

##### Helper: `_load_user_histories()`
```python
def _load_user_histories(self):
    """
    Preload user → product interactions once để phục vụ low-latency.
    """
    interactions = pd.read_parquet(
        'data/processed/interactions.parquet',
        columns=['user_id', 'product_id']
    )
    history = (
        interactions.groupby('user_id')['product_id']
        .apply(lambda s: set(s.tolist()))
        .to_dict()
    )
    return history
```

##### Method 1: `recommend(user_id, topk=10, exclude_seen=True, filter_params=None)`
```python
def recommend(self, user_id, topk=10, exclude_seen=True, filter_params=None):
    """
    Generate top-K recommendations cho user.
    
    Args:
        user_id: Original user ID (int)
        topk: Number of recommendations (default 10)
        exclude_seen: Nếu True, loại bỏ items user đã tương tác
        filter_params: Dict với attribute filters (e.g., {'brand': 'Innisfree'})
    
    Returns:
        list of dict: [
            {
                'product_id': int,
                'score': float,
                'product_name': str,
                'brand': str,
                'price': float,
                ...
            },
            ...
        ]
    
    Raises:
        KeyError: User ID không tồn tại (cold-start)
    """
    # Check user exists
    if user_id not in self.mappings['user_to_idx']:
        # Cold-start fallback
        return self._fallback_recommendations(topk, filter_params)
    
    # Map user_id → u_idx
    u_idx = self.mappings['user_to_idx'][user_id]
    
    # Compute scores
    scores = self.U[u_idx] @ self.V.T  # Shape: (num_items,)
    
    # Exclude seen items
    if exclude_seen:
        seen_items = self._get_user_history(user_id)
        seen_indices = [self.mappings['item_to_idx'][pid] for pid in seen_items if pid in self.mappings['item_to_idx']]
        scores[seen_indices] = -np.inf
    
    # Apply attribute filters
    if filter_params:
        valid_indices = self._apply_filters(filter_params)
        mask = np.ones(self.num_items, dtype=bool)
        mask[valid_indices] = False
        scores[mask] = -np.inf
    
    # Top-K
    top_k_indices = np.argsort(scores)[::-1][:topk]
    
    # Map i_idx → product_id
    product_ids = [self.mappings['idx_to_item'][str(i)] for i in top_k_indices]
    
    # Enrich với metadata
    recommendations = []
    for i, pid in enumerate(product_ids):
        product_info = self.item_metadata[self.item_metadata['product_id'] == pid].iloc[0].to_dict()
        recommendations.append({
            'product_id': pid,
            'score': float(scores[top_k_indices[i]]),
            'rank': i + 1,
            **product_info  # product_name, brand, price, ...
        })
    
    return recommendations
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

#### Function: `_fallback_recommendations(topk, filter_params=None)`

##### Strategy 1: Popularity-Based
```python
def _fallback_recommendations(self, topk=10, filter_params=None):
    """
    Fallback tới popularity ranking cho cold-start users.
    
    Args:
        topk: Number of recommendations
        filter_params: Optional attribute filters
    
    Returns:
        list of dict: Popular products
    """
    # Sort by num_sold_time (popularity signal)
    popular = self.item_metadata.sort_values('num_sold_time', ascending=False)
    
    # Apply filters nếu có
    if filter_params:
        for key, value in filter_params.items():
            if key in popular.columns:
                popular = popular[popular[key] == value]
    
    # Top-K
    top_popular = popular.head(topk)
    
    # Format output
    recommendations = []
    for i, row in top_popular.iterrows():
        recommendations.append({
            'product_id': row['product_id'],
            'score': row['num_sold_time'],  # Popularity score
            'rank': len(recommendations) + 1,
            'fallback': True,  # Flag fallback
            **row.to_dict()
        })
    
    return recommendations
```

##### Strategy 2: Hybrid với PhoBERT (Advanced)
```python
def _fallback_phobert(self, user_query, topk=10):
    """
    Fallback tới PhoBERT semantic search nếu có user query.
    
    Args:
        user_query: Vietnamese text query
        topk: Number of recommendations
    
    Returns:
        list of dict: Semantically similar products
    """
    from model.phobert_recommendation import PhoBERTRecommender
    
    # Initialize PhoBERT (cache instance)
    if not hasattr(self, '_phobert'):
        self._phobert = PhoBERTRecommender(...)
    
    # Search by query
    results = self._phobert.get_recommendations_by_name(user_query, topk)
    
    # Format
    recommendations = []
    for r in results:
        recommendations.append({
            'product_id': r['product_id'],
            'score': r['similarity_score'],
            'rank': len(recommendations) + 1,
            'fallback': True,
            'fallback_method': 'phobert',
            **r
        })
    
    return recommendations
```

## Component 4: Hybrid Reranking (Optional)

### Module: `service/recommender/rerank.py`

#### Function: `rerank_with_signals(recommendations, user_id, weights=None)`

##### Purpose
Combine CF scores với additional signals (popularity, quality, content similarity).

##### Formula
```
final_score = α * CF_score + β * popularity + γ * quality + δ * content_similarity
```

##### Implementation
```python
def rerank_with_signals(recommendations, user_id, weights=None):
    """
    Rerank recommendations bằng weighted combination of signals.
    
    Args:
        recommendations: List từ CFRecommender.recommend()
        user_id: For personalized content similarity (optional)
        weights: Dict {'cf': 0.5, 'popularity': 0.2, 'quality': 0.2, 'content': 0.1}
    
    Returns:
        list: Reranked recommendations
    """
    if weights is None:
        weights = {'cf': 0.6, 'popularity': 0.2, 'quality': 0.2, 'content': 0.0}
    
    # Normalize signals
    cf_scores = normalize([r['score'] for r in recommendations])
    popularity_scores = normalize([r.get('num_sold_time', 0) for r in recommendations])
    quality_scores = normalize([r.get('avg_star', 3.0) for r in recommendations])
    
    # Content similarity (nếu có PhoBERT embeddings)
    if weights['content'] > 0:
        content_scores = compute_content_similarity(user_id, [r['product_id'] for r in recommendations])
    else:
        content_scores = [0] * len(recommendations)
    
    # Combine
    for i, rec in enumerate(recommendations):
        rec['final_score'] = (
            weights['cf'] * cf_scores[i] +
            weights['popularity'] * popularity_scores[i] +
            weights['quality'] * quality_scores[i] +
            weights['content'] * content_scores[i]
        )
    
    # Re-sort
    recommendations.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Update ranks
    for i, rec in enumerate(recommendations):
        rec['rank'] = i + 1
    
    return recommendations
```

## Component 5: API Layer (FastAPI)

### Module: `service/api.py`

#### Endpoints

##### 1. Health Check
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="CF Recommendation Service")

@app.get("/health")
def health_check():
    """Check service status."""
    return {
        "status": "healthy",
        "model_id": recommender.model['model_id'],
        "model_type": recommender.model['model_type']
    }
```

##### 2. Single User Recommendation
```python
class RecommendRequest(BaseModel):
    user_id: int
    topk: int = 10
    exclude_seen: bool = True
    filter_params: dict = None

@app.post("/recommend")
def recommend(request: RecommendRequest):
    """
    Get recommendations cho single user.
    
    Example:
        POST /recommend
        {
            "user_id": 12345,
            "topk": 10,
            "exclude_seen": true,
            "filter_params": {"brand": "Innisfree"}
        }
    """
    try:
        recommendations = recommender.recommend(
            user_id=request.user_id,
            topk=request.topk,
            exclude_seen=request.exclude_seen,
            filter_params=request.filter_params
        )
        
        return {
            "user_id": request.user_id,
            "recommendations": recommendations,
            "count": len(recommendations),
            "fallback": recommendations[0].get('fallback', False) if recommendations else False
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

##### 3. Batch Recommendation
```python
class BatchRequest(BaseModel):
    user_ids: list[int]
    topk: int = 10
    exclude_seen: bool = True

@app.post("/batch_recommend")
def batch_recommend(request: BatchRequest):
    """
    Batch recommendations cho multiple users.
    """
    results = recommender.batch_recommend(
        user_ids=request.user_ids,
        topk=request.topk,
        exclude_seen=request.exclude_seen
    )
    
    return {
        "results": results,
        "num_users": len(request.user_ids)
    }
```

##### 4. Model Reload
```python
@app.post("/reload_model")
def reload_model():
    """
    Reload model từ registry (hot-reload).
    """
    updated = loader.reload_if_updated()
    
    if updated:
        # Reinitialize recommender với new model
        global recommender
        recommender = CFRecommender(loader)
        
        return {
            "status": "reloaded",
            "new_model_id": recommender.model['model_id']
        }
    else:
        return {
            "status": "no_update",
            "current_model_id": recommender.model['model_id']
        }
```

## Configuration

### File: `service/config/serving_config.yaml`
```yaml
model:
  registry_path: "artifacts/cf/registry.json"
  auto_reload: true  # Periodically check registry
  reload_interval_seconds: 300  # 5 minutes

serving:
  default_topk: 10
  max_topk: 100
  exclude_seen_default: true
  
fallback:
  strategy: "popularity"  # or "phobert"
  popularity_metric: "num_sold_time"  # or "avg_star"

reranking:
  enabled: false
  weights:
    cf: 0.6
    popularity: 0.2
    quality: 0.2
    content: 0.0

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4  # Uvicorn workers
  log_level: "info"
```

## Logging & Monitoring

### Request Logging
```python
import logging
from datetime import datetime

logger = logging.getLogger("cf_service")

@app.post("/recommend")
def recommend(request: RecommendRequest):
    start_time = datetime.now()
    
    try:
        recommendations = recommender.recommend(...)
        
        latency = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"user_id={request.user_id}, topk={request.topk}, latency={latency:.3f}s, fallback={recommendations[0].get('fallback', False)}")
        
        return {...}
    except Exception as e:
        logger.error(f"user_id={request.user_id}, error={str(e)}")
        raise
```

### Metrics to Track
- **Latency**: p50, p95, p99 response time
- **Throughput**: Requests per second
- **Fallback rate**: % requests sử dụng fallback
- **Error rate**: % failed requests
- **Cache hit rate**: Nếu có caching layer

## Performance Optimization

### 1. Precompute Item-Item Similarity (Optional)
```python
# For "similar items" recommendations
self.V_dot_V = self.V @ self.V.T  # (num_items, num_items)

def similar_items(self, product_id, topk=10):
    i_idx = self.mappings['item_to_idx'][str(product_id)]
    scores = self.V_dot_V[i_idx]
    top_k = np.argsort(scores)[::-1][1:topk+1]  # Exclude self
    return [self.mappings['idx_to_item'][str(i)] for i in top_k]
```

### 2. Caching User History
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def _get_user_history_cached(self, user_id):
    return self._get_user_history(user_id)
```

### 3. Batch Inference
- Process multiple users simultaneously → amortize overhead
- Use NumPy vectorization

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

## Component 6: BERT Embeddings Integration

### Module: `service/recommender/phobert_loader.py`

#### Class: `PhoBERTEmbeddingLoader`

```python
import numpy as np
import torch
from functools import lru_cache

class PhoBERTEmbeddingLoader:
    """
    Load và cache BERT embeddings cho serving.
    """
    
    def __init__(self, embeddings_path='data/processed/content_based_embeddings/product_embeddings.pt'):
        self.embeddings_path = embeddings_path
        self.embeddings = None  # (num_products, 768)
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load BERT embeddings và create mappings."""
        bert_data = torch.load(self.embeddings_path, map_location='cpu')
        
        self.embeddings = bert_data['embeddings'].numpy()  # (N, 768)
        product_ids = bert_data['product_ids']
        
        # Create mappings
        for idx, pid in enumerate(product_ids):
            self.product_id_to_idx[int(pid)] = idx
            self.idx_to_product_id[idx] = int(pid)
        
        # Normalize for fast cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_norm = self.embeddings / norms
        
        print(f"Loaded {len(self.product_id_to_idx)} BERT embeddings (dim={self.embeddings.shape[1]})")
    
    def get_embedding(self, product_id):
        """Get embedding for single product."""
        idx = self.product_id_to_idx.get(product_id)
        if idx is not None:
            return self.embeddings[idx]
        return None
    
    def compute_user_profile(self, user_history_items, strategy='weighted_mean'):
        """
        Compute user profile embedding từ interaction history.
        
        Args:
            user_history_items: List[(product_id, weight)] or List[product_id]
            strategy: 'mean', 'weighted_mean', 'max'
        
        Returns:
            np.array: (768,) user profile embedding
        """
        # Parse history
        if isinstance(user_history_items[0], tuple):
            product_ids = [pid for pid, _ in user_history_items]
            weights = [w for _, w in user_history_items]
        else:
            product_ids = user_history_items
            weights = [1.0] * len(product_ids)
        
        # Get embeddings
        history_embeddings = []
        history_weights = []
        
        for pid, weight in zip(product_ids, weights):
            emb = self.get_embedding(pid)
            if emb is not None:
                history_embeddings.append(emb)
                history_weights.append(weight)
        
        if not history_embeddings:
            # Fallback: return mean embedding
            return np.zeros(self.embeddings.shape[1])
        
        history_embeddings = np.array(history_embeddings)
        history_weights = np.array(history_weights).reshape(-1, 1)
        
        # Aggregate
        if strategy == 'mean':
            profile = np.mean(history_embeddings, axis=0)
        elif strategy == 'weighted_mean':
            weights_norm = history_weights / history_weights.sum()
            profile = (history_embeddings * weights_norm).sum(axis=0)
        elif strategy == 'max':
            profile = np.max(history_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return profile
```

## Timeline Estimate

- **Loader + Recommender**: 2 days
- **Fallback logic**: 0.5 day
- **API endpoints**: 1 day
- **BERT integration + Reranking**: 2 days
- **Testing**: 1 day
- **Deployment setup**: 0.5 day
- **Total**: ~7 days

## Success Criteria

- [ ] Load model từ registry (<1 second)
- [ ] Generate recommendations (<100ms per user CF-only)
- [ ] Two-stage reranking (<200ms with BERT)
- [ ] BERT embeddings loaded and cached
- [ ] Cold-start fallback works
- [ ] API endpoints functional
- [ ] Hot-reload model without downtime
- [ ] Logging tracks latency, fallback rate, rerank metrics
- [ ] Docker deployment tested
