# Task 08: Hybrid Reranking & Integration

## Mục Tiêu

Kết hợp Collaborative Filtering (ALS/BPR) với PhoBERT embeddings và product attributes để tạo hệ thống hybrid recommendation. Mục tiêu là tăng diversity, personalization, và handle cold-start tốt hơn bằng cách combine multiple signals.

## Hybrid Strategy Overview

```
User Request
    ↓
CF Recommender (ALS/BPR)
    ↓
Top-K Candidates (e.g., K=50)
    ↓
Reranking Layer
    ↓
├─ Content Similarity (PhoBERT)
├─ Popularity Signal (num_sold_time)
├─ Quality Signal (avg_star)
├─ Attribute Match (brand, skin_type)
└─ Diversity Boost
    ↓
Weighted Combination
    ↓
Final Top-K (e.g., K=10)
```

## Component 1: PhoBERT Integration

### Load PhoBERT Embeddings

#### Module: `service/recommender/phobert_loader.py`

##### Class: `PhoBERTEmbeddings`
```python
import numpy as np
import torch
from pathlib import Path

class PhoBERTEmbeddings:
    """
    Load và manage PhoBERT embeddings cho content-based similarity.
    """
    
    def __init__(self, embeddings_path='model/data/published_data/content_based_embeddings'):
        self.embeddings_path = Path(embeddings_path)
        self.product_embeddings = None  # Shape: (num_products, 768)
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load PhoBERT embeddings từ .pt file."""
        # Load product embeddings
        emb_file = self.embeddings_path / 'product_embeddings.pt'
        
        if emb_file.exists():
            embeddings_dict = torch.load(emb_file, map_location='cpu')
            
            # Convert to numpy
            self.product_embeddings = embeddings_dict['embeddings'].numpy()  # (N, 768)
            product_ids = embeddings_dict['product_ids']  # List hoặc tensor
            
            # Create mappings
            for idx, pid in enumerate(product_ids):
                self.product_id_to_idx[int(pid)] = idx
                self.idx_to_product_id[idx] = int(pid)
            
            print(f"Loaded {len(self.product_id_to_idx)} product embeddings")
        else:
            raise FileNotFoundError(f"Embeddings file not found: {emb_file}")
    
    def get_embedding(self, product_id):
        """
        Get embedding cho single product.
        
        Args:
            product_id: Integer product ID
        
        Returns:
            np.array: Shape (768,) hoặc None nếu không tồn tại
        """
        idx = self.product_id_to_idx.get(product_id)
        if idx is not None:
            return self.product_embeddings[idx]
        return None
    
    def compute_similarity(self, product_id, candidate_ids):
        """
        Compute cosine similarity giữa product và candidates.
        
        Args:
            product_id: Source product ID
            candidate_ids: List of candidate product IDs
        
        Returns:
            dict: {candidate_id: similarity_score}
        """
        source_emb = self.get_embedding(product_id)
        if source_emb is None:
            return {cid: 0.0 for cid in candidate_ids}
        
        # Normalize source
        source_norm = source_emb / np.linalg.norm(source_emb)
        
        similarities = {}
        for cid in candidate_ids:
            cand_emb = self.get_embedding(cid)
            if cand_emb is not None:
                cand_norm = cand_emb / np.linalg.norm(cand_emb)
                similarities[cid] = float(np.dot(source_norm, cand_norm))
            else:
                similarities[cid] = 0.0
        
        return similarities
    
    def compute_user_profile_similarity(self, user_history, candidate_ids, aggregation='mean'):
        """
        Compute similarity giữa user profile (aggregated history) và candidates.
        
        Args:
            user_history: List of product IDs user đã interact
            candidate_ids: List of candidate product IDs
            aggregation: 'mean' hoặc 'max' để aggregate user history
        
        Returns:
            dict: {candidate_id: similarity_score}
        """
        # Get embeddings của user history
        history_embeddings = []
        for pid in user_history:
            emb = self.get_embedding(pid)
            if emb is not None:
                history_embeddings.append(emb)
        
        if not history_embeddings:
            return {cid: 0.0 for cid in candidate_ids}
        
        # Aggregate user profile
        history_embeddings = np.array(history_embeddings)
        if aggregation == 'mean':
            user_profile = np.mean(history_embeddings, axis=0)
        elif aggregation == 'max':
            user_profile = np.max(history_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Normalize
        user_profile_norm = user_profile / np.linalg.norm(user_profile)
        
        # Compute similarities
        similarities = {}
        for cid in candidate_ids:
            cand_emb = self.get_embedding(cid)
            if cand_emb is not None:
                cand_norm = cand_emb / np.linalg.norm(cand_emb)
                similarities[cid] = float(np.dot(user_profile_norm, cand_norm))
            else:
                similarities[cid] = 0.0
        
        return similarities
```

## Component 2: Reranking Module

### Module: `service/recommender/rerank.py`

#### Class: `HybridReranker`

##### Initialization
```python
class HybridReranker:
    """
    Hybrid reranker combining CF, content, popularity, quality signals.
    """
    
    def __init__(self, phobert_embeddings, item_metadata, config=None):
        """
        Args:
            phobert_embeddings: PhoBERTEmbeddings instance
            item_metadata: DataFrame với product metadata
            config: Dict với weights và thresholds
        """
        self.phobert = phobert_embeddings
        self.metadata = item_metadata
        
        # Default weights
        self.config = config or {
            'weights': {
                'cf': 0.5,
                'content': 0.2,
                'popularity': 0.15,
                'quality': 0.15
            },
            'diversity': {
                'enabled': True,
                'penalty': 0.1,  # Penalty cho similar items
                'threshold': 0.9  # Similarity threshold
            }
        }
```

##### Method 1: `rerank(cf_recommendations, user_id, user_history=None)`
```python
def rerank(self, cf_recommendations, user_id, user_history=None):
    """
    Rerank CF recommendations với hybrid signals.
    
    Args:
        cf_recommendations: List of dicts từ CFRecommender
            [{product_id, score, product_name, ...}, ...]
        user_id: User ID for personalization
        user_history: Optional list of product IDs user interacted with
    
    Returns:
        list: Reranked recommendations
    """
    if not cf_recommendations:
        return []
    
    # Extract product IDs
    candidate_ids = [rec['product_id'] for rec in cf_recommendations]
    
    # Compute signals
    signals = self._compute_signals(candidate_ids, user_history)
    
    # Normalize signals
    normalized = self._normalize_signals(signals)
    
    # Combine với weights
    final_scores = self._combine_scores(normalized, self.config['weights'])
    
    # Apply diversity penalty (optional)
    if self.config['diversity']['enabled']:
        final_scores = self._apply_diversity_penalty(final_scores, candidate_ids)
    
    # Update recommendations với final scores
    for rec in cf_recommendations:
        rec['cf_score'] = rec['score']  # Original CF score
        rec['final_score'] = final_scores[rec['product_id']]
        rec['signals'] = {
            signal_name: signals[signal_name][rec['product_id']]
            for signal_name in signals.keys()
        }
    
    # Re-sort
    cf_recommendations.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Update ranks
    for i, rec in enumerate(cf_recommendations):
        rec['rank'] = i + 1
    
    return cf_recommendations
```

##### Method 2: `_compute_signals(candidate_ids, user_history)`
```python
def _compute_signals(self, candidate_ids, user_history=None):
    """
    Compute all signals cho candidates.
    
    Returns:
        dict: {
            'cf': {product_id: score},
            'content': {product_id: score},
            'popularity': {product_id: score},
            'quality': {product_id: score}
        }
    """
    signals = {}
    
    # Content similarity (PhoBERT)
    if user_history:
        signals['content'] = self.phobert.compute_user_profile_similarity(
            user_history, candidate_ids, aggregation='mean'
        )
    else:
        # Fallback: no content signal
        signals['content'] = {cid: 0.0 for cid in candidate_ids}
    
    # Popularity
    signals['popularity'] = {}
    for cid in candidate_ids:
        product = self.metadata[self.metadata['product_id'] == cid]
        if not product.empty:
            signals['popularity'][cid] = float(product.iloc[0]['num_sold_time'])
        else:
            signals['popularity'][cid] = 0.0
    
    # Quality (avg_star)
    signals['quality'] = {}
    for cid in candidate_ids:
        product = self.metadata[self.metadata['product_id'] == cid]
        if not product.empty:
            signals['quality'][cid] = float(product.iloc[0].get('avg_star', 3.0))
        else:
            signals['quality'][cid] = 3.0
    
    return signals
```

##### Method 3: `_normalize_signals(signals)`
```python
def _normalize_signals(self, signals):
    """
    Normalize all signals to [0, 1] range.
    
    Args:
        signals: Dict of signal dicts
    
    Returns:
        dict: Normalized signals (same structure)
    """
    normalized = {}
    
    for signal_name, scores in signals.items():
        values = np.array(list(scores.values()))
        
        # Min-max normalization
        min_val = values.min()
        max_val = values.max()
        
        if max_val > min_val:
            norm_values = (values - min_val) / (max_val - min_val)
        else:
            norm_values = np.zeros_like(values)
        
        # Map back to product_ids
        normalized[signal_name] = {
            pid: float(norm_values[i])
            for i, pid in enumerate(scores.keys())
        }
    
    return normalized
```

##### Method 4: `_combine_scores(normalized_signals, weights)`
```python
def _combine_scores(self, normalized_signals, weights):
    """
    Combine normalized signals với weighted sum.
    
    Args:
        normalized_signals: Dict of normalized signal dicts
        weights: Dict {'cf': w1, 'content': w2, ...}
    
    Returns:
        dict: {product_id: final_score}
    """
    # Extract all product IDs
    product_ids = list(next(iter(normalized_signals.values())).keys())
    
    final_scores = {}
    for pid in product_ids:
        score = 0.0
        for signal_name, weight in weights.items():
            if signal_name in normalized_signals:
                score += weight * normalized_signals[signal_name][pid]
        
        final_scores[pid] = score
    
    return final_scores
```

##### Method 5: `_apply_diversity_penalty(scores, candidate_ids)`
```python
def _apply_diversity_penalty(self, scores, candidate_ids):
    """
    Apply diversity penalty để giảm similar items trong top-K.
    
    Args:
        scores: Dict {product_id: score}
        candidate_ids: List of product IDs
    
    Returns:
        dict: Penalized scores
    """
    diversity_config = self.config['diversity']
    penalty = diversity_config['penalty']
    threshold = diversity_config['threshold']
    
    # Compute pairwise similarities
    penalized_scores = scores.copy()
    
    # Sort by current scores
    sorted_ids = sorted(candidate_ids, key=lambda x: scores[x], reverse=True)
    
    selected = []
    for pid in sorted_ids:
        # Check similarity với items đã selected
        if selected:
            similarities = self.phobert.compute_similarity(pid, selected)
            max_sim = max(similarities.values())
            
            # Apply penalty nếu too similar
            if max_sim > threshold:
                penalized_scores[pid] *= (1 - penalty)
        
        selected.append(pid)
    
    return penalized_scores
```

## Component 3: Attribute Filtering

### Module: `service/recommender/filters.py`

#### Function: `apply_attribute_filters(recommendations, filters, metadata)`
```python
def apply_attribute_filters(recommendations, filters, metadata):
    """
    Filter recommendations theo product attributes.
    
    Args:
        recommendations: List of recommendation dicts
        filters: Dict {'brand': 'Innisfree', 'skin_type': 'oily', ...}
        metadata: DataFrame với product attributes
    
    Returns:
        list: Filtered recommendations
    """
    if not filters:
        return recommendations
    
    filtered = []
    for rec in recommendations:
        product_id = rec['product_id']
        product = metadata[metadata['product_id'] == product_id]
        
        if product.empty:
            continue
        
        # Check all filter conditions
        match = True
        for attr, value in filters.items():
            if attr in product.columns:
                if product.iloc[0][attr] != value:
                    match = False
                    break
        
        if match:
            filtered.append(rec)
    
    return filtered
```

#### Function: `boost_by_attributes(recommendations, boost_config, metadata)`
```python
def boost_by_attributes(recommendations, boost_config, metadata):
    """
    Boost scores cho products matching desired attributes.
    
    Args:
        recommendations: List of dicts
        boost_config: Dict {'brand': {'Innisfree': 1.2, 'Cetaphil': 1.1}, ...}
        metadata: DataFrame
    
    Returns:
        list: Recommendations với boosted scores
    """
    for rec in recommendations:
        product_id = rec['product_id']
        product = metadata[metadata['product_id'] == product_id]
        
        if product.empty:
            continue
        
        boost_factor = 1.0
        for attr, boost_values in boost_config.items():
            if attr in product.columns:
                attr_value = product.iloc[0][attr]
                if attr_value in boost_values:
                    boost_factor *= boost_values[attr_value]
        
        rec['final_score'] *= boost_factor
        rec['boost_factor'] = boost_factor
    
    # Re-sort
    recommendations.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Update ranks
    for i, rec in enumerate(recommendations):
        rec['rank'] = i + 1
    
    return recommendations
```

## Component 4: Integration với CFRecommender

### Updated Method: `recommend()` với Reranking

#### File: `service/recommender/recommender.py` (Modified)

```python
class CFRecommender:
    def __init__(self, model_loader, enable_reranking=True):
        # ... existing init ...
        
        # Reranking components
        self.enable_reranking = enable_reranking
        if enable_reranking:
            self.phobert_embeddings = PhoBERTEmbeddings()
            self.reranker = HybridReranker(
                phobert_embeddings=self.phobert_embeddings,
                item_metadata=self.item_metadata
            )
    
    def recommend(self, user_id, topk=10, exclude_seen=True, 
                  filter_params=None, rerank=None):
        """
        Generate recommendations với optional reranking.
        
        Args:
            ...existing args...
            rerank: Bool to override default reranking setting
        """
        # Determine rerank setting
        use_rerank = rerank if rerank is not None else self.enable_reranking
        
        # Generate more candidates nếu reranking enabled
        candidate_k = topk * 5 if use_rerank else topk
        
        # Get CF recommendations (existing logic)
        cf_recommendations = self._generate_cf_recommendations(
            user_id, candidate_k, exclude_seen, filter_params
        )
        
        # Rerank nếu enabled
        if use_rerank and cf_recommendations:
            user_history = self._get_user_history(user_id)
            cf_recommendations = self.reranker.rerank(
                cf_recommendations, user_id, user_history
            )
        
        # Return top-K after reranking
        return cf_recommendations[:topk]
```

## Component 5: Configuration Management

### File: `service/config/rerank_config.yaml`
```yaml
reranking:
  enabled: true
  candidate_multiplier: 5  # Generate 5x candidates for reranking
  
  weights:
    cf: 0.5
    content: 0.2
    popularity: 0.15
    quality: 0.15
  
  diversity:
    enabled: true
    penalty: 0.1
    threshold: 0.9
  
  content_similarity:
    aggregation: "mean"  # mean or max
    min_history_length: 3  # Min items in history to use content
  
  attribute_boost:
    brand:
      Innisfree: 1.2
      Cetaphil: 1.1
      CeraVe: 1.15
    skin_type:
      oily: 1.1  # Boost cho oily skin products (nếu user oily skin)
```

## Component 6: Evaluation

### Metric: Diversity

#### Function: `compute_diversity(recommendations)`
```python
def compute_diversity(recommendations, phobert_embeddings):
    """
    Compute diversity trong recommendation list.
    
    Args:
        recommendations: List of product IDs
        phobert_embeddings: PhoBERTEmbeddings instance
    
    Returns:
        float: Average pairwise distance (higher = more diverse)
    """
    if len(recommendations) < 2:
        return 0.0
    
    # Compute pairwise similarities
    similarities = []
    for i in range(len(recommendations)):
        for j in range(i+1, len(recommendations)):
            sim = phobert_embeddings.compute_similarity(
                recommendations[i], [recommendations[j]]
            )[recommendations[j]]
            similarities.append(sim)
    
    # Diversity = 1 - avg_similarity
    avg_similarity = np.mean(similarities)
    diversity = 1 - avg_similarity
    
    return diversity
```

### Metric: Coverage by Category

#### Function: `compute_category_coverage(recommendations, metadata)`
```python
def compute_category_coverage(recommendations, metadata):
    """
    Compute % unique categories trong recommendations.
    
    Args:
        recommendations: List of product IDs
        metadata: DataFrame với category column
    
    Returns:
        float: % unique categories
    """
    # Get categories
    categories = []
    for pid in recommendations:
        product = metadata[metadata['product_id'] == pid]
        if not product.empty:
            categories.append(product.iloc[0]['type'])  # or 'brand'
    
    # Compute coverage
    unique_categories = len(set(categories))
    total_categories = len(categories)
    
    return unique_categories / total_categories if total_categories > 0 else 0.0
```

### Compare CF vs Hybrid

#### Script: `scripts/evaluate_hybrid.py`
```python
"""
Evaluate hybrid reranking vs pure CF.
"""

def main():
    # Load models
    cf_recommender = CFRecommender(model_loader, enable_reranking=False)
    hybrid_recommender = CFRecommender(model_loader, enable_reranking=True)
    
    # Load test users
    test_data = pd.read_parquet('data/processed/interactions.parquet')
    test_users = test_data[test_data['split'] == 'test']['user_id'].unique()
    
    # Evaluate both
    cf_metrics = evaluate_recommendations(cf_recommender, test_users)
    hybrid_metrics = evaluate_recommendations(hybrid_recommender, test_users)
    
    # Compare
    print("CF Metrics:", cf_metrics)
    print("Hybrid Metrics:", hybrid_metrics)
    
    # Diversity comparison
    cf_diversity = compute_avg_diversity(cf_recommender, test_users[:100])
    hybrid_diversity = compute_avg_diversity(hybrid_recommender, test_users[:100])
    
    print(f"CF Diversity: {cf_diversity:.3f}")
    print(f"Hybrid Diversity: {hybrid_diversity:.3f}")
```

## Component 7: Use Cases

### Use Case 1: Personalized Recommendations
```python
# Pure CF
recs_cf = recommender.recommend(user_id=12345, topk=10, rerank=False)

# Hybrid (CF + content + popularity)
recs_hybrid = recommender.recommend(user_id=12345, topk=10, rerank=True)
```

### Use Case 2: Cold-Start User với Attribute Filtering
```python
# Cold-start user → fallback popularity
# But filter by brand preference
recs = recommender.recommend(
    user_id=99999,  # New user
    topk=10,
    filter_params={'brand': 'Innisfree'}
)
```

### Use Case 3: Similar Items (Content-Based)
```python
# Recommend similar products to a given item
source_product = 123
candidates = recommender.phobert_embeddings.compute_similarity(
    source_product,
    candidate_ids=list(range(2244))  # All products
)

# Sort by similarity
similar_items = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:10]
```

### Use Case 4: Diverse Recommendations
```python
# Enable diversity penalty
recommender.reranker.config['diversity']['enabled'] = True
recommender.reranker.config['diversity']['penalty'] = 0.2  # Stronger penalty

recs = recommender.recommend(user_id=12345, topk=10, rerank=True)
```

## Dependencies

```python
# requirements_hybrid.txt
torch>=1.13.0  # PhoBERT embeddings
transformers>=4.25.0  # PhoBERT model (nếu cần encode mới)
```

## Timeline Estimate

- **PhoBERT integration**: 1 day
- **Reranker implementation**: 2 days
- **Attribute filtering**: 1 day
- **Configuration**: 0.5 day
- **Evaluation**: 1 day
- **Testing**: 1 day
- **Total**: ~6.5 days

## Success Criteria

- [ ] Hybrid reranking combines CF + content + popularity
- [ ] Diversity metric improves vs pure CF
- [ ] Cold-start performance improves
- [ ] Attribute filtering works
- [ ] Configuration flexible (tunable weights)
- [ ] Latency acceptable (<200ms per request)
- [ ] Evaluation shows improvement in diversity + coverage
