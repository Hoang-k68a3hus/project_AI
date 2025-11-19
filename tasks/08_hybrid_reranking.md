# Task 08: Hybrid Reranking & Integration

## Mục Tiêu

Kết hợp Collaborative Filtering (ALS/BPR) với PhoBERT embeddings và product attributes để tạo hệ thống hybrid recommendation. Mục tiêu là tăng diversity, personalization, và handle cold-start tốt hơn bằng cách combine multiple signals.

## 📊 Data Dependencies (Updated November 2025)

**Critical**: This task depends on enriched metadata from Task 01:
- **Input File**: `data/processed/product_attributes_enriched.parquet`
- **Pre-computed Signals**:
  - `popularity_score`: Log-transformed `num_sold_time` → ready for normalization
  - `quality_score`: Derived from review ratings (avg_rating, pct_5star)
  - `skin_type_standardized`: List of standardized tags for hard filtering
- **BERT Embeddings**: `data/processed/content_based_embeddings/product_embeddings.pt` with rich text context

**Benefit**: Không cần compute log-transform hoặc aggregate ratings on-the-fly, tất cả đã được chuẩn bị sẵn trong pipeline Task 01.

## 🎯 Content-First Hybrid Strategy (Updated for Sparsity)

### Context
- **Data sparsity**: ~1.23 interactions/user → CF has limited collaborative signal
- **Trainable users**: ~26,000 (8.6%) with ≥2 interactions; ~274,000 (91.4%) cold-start
- **Solution**: Shift weight from CF to content-based (PhoBERT) for more reliable recommendations

### Recommended Weight Distribution
```yaml
# For Trainable Users (≥2 interactions, ~8.6% traffic)
weights_trainable:
  content: 0.40    # PRIMARY - Semantic similarity (PhoBERT)
  cf: 0.30         # SECONDARY - Collaborative signal (ALS/BPR with BERT init)
  popularity: 0.20 # TERTIARY - Trending/popular items
  quality: 0.10    # BONUS - High-rated products

# For Cold-Start Users (1 interaction or new, ~91.4% traffic)
weights_cold_start:
  content: 0.60    # DOMINANT - Only reliable signal
  popularity: 0.30 # Social proof
  quality: 0.10    # High-rated products
  # cf: 0.0        # Skipped - no user embedding
```

**Rationale**:
- **Trainable users**: Content (40%) most reliable despite ≥2 threshold; CF (30%) still valuable with BERT init + higher regularization (λ=0.1)
- **Cold-start users** (majority): Content (60%) dominant; CF unusable
- Popularity (20-30%): Social proof, handles trending items
- Quality (10%): Bonus for highly-rated products

**Critical**: With 91.4% traffic using cold-start path, content-based infrastructure must be optimized for latency.

## Hybrid Strategy Overview

```
User Request
    ↓
CF Recommender (ALS/BPR)
    ↓
Top-K Candidates (e.g., K=50)
    ↓
Reranking Layer (PhoBERTEmbeddingLoader from Task 05)
    ↓
├─ Content Similarity (PhoBERT cosine similarity)
├─ Popularity Signal (log-transformed popularity_score from Task 01)
├─ Quality Signal (quality_score / avg_rating from Task 01)
├─ Attribute Match (brand, skin_type_standardized)
└─ Diversity Boost (intra-list diversity)
    ↓
Weighted Combination
    ↓
Final Top-K (e.g., K=10)
```

**Key Integration Points**:
- **PhoBERTEmbeddingLoader** (Task 05): Handles BERT embedding loading, user profile computation, and cosine similarity scoring
- **Enriched Metadata** (Task 01): Uses `product_attributes_enriched.parquet` with standardized attributes and pre-computed auxiliary signals
- **Diversity Metrics** (Task 03): `compute_diversity_bert()` evaluates intra-list diversity using BERT embeddings
- **Semantic Alignment** (Task 03): `compute_semantic_alignment()` measures user profile → recommendation relevance
- **Model Registry** (Task 04): Tracks BERT embedding versions and compatibility with CF models

## Component 1: PhoBERT Integration

### Load PhoBERT Embeddings

#### Module: `service/recommender/phobert_loader.py`

**NOTE**: This module is now superseded by `PhoBERTEmbeddingLoader` in Task 05 (`serving_layer.md`). The class below provides basic embedding loading, while the Task 05 implementation includes advanced features like user profile computation, LRU caching, and normalized embeddings.

##### Class: `PhoBERTEmbeddings` (Basic Version)
```python
import numpy as np
import torch
from pathlib import Path

class PhoBERTEmbeddings:
    """
    Load và manage PhoBERT embeddings cho content-based similarity.
    
    For production use, prefer PhoBERTEmbeddingLoader from Task 05 which includes:
    - User profile computation (interaction-weighted, TF-IDF)
    - LRU caching for performance
    - Normalized embeddings for cosine similarity
    """
    
    def __init__(self, embeddings_path='data/processed/content_based_embeddings'):
        self.embeddings_path = Path(embeddings_path)
        self.product_embeddings = None  # Shape: (num_products, 768)
        self.product_id_to_idx = {}
        self.idx_to_product_id = {}
        
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load PhoBERT embeddings từ .pt file."""
        # Load product embeddings (standardized path from Task 01)
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
        
        # Default weights (UPDATED for sparse data - content-first strategy)
        self.config = config or {
            'weights': {
                'content': 0.40,    # PRIMARY - PhoBERT semantic similarity
                'cf': 0.30,         # SECONDARY - Collaborative filtering score
                'popularity': 0.20, # TERTIARY - Log-transformed popularity
                'quality': 0.10     # BONUS - Quality score from reviews
            },
            'diversity': {
                'enabled': True,
                'penalty': 0.1,      # Penalty cho similar items
                'threshold': 0.9     # BERT similarity threshold
            },
            'user_profile_strategy': 'weighted_mean',  # From Task 01: weighted_mean, tf_idf, recency
            'adaptive_weights': True,  # Adjust weights based on user interaction count
            
            # CRITICAL: Global normalization ranges (NOT local per-request)
            'normalization': {
                'cf': {
                    'method': 'clip',  # 'clip' or 'minmax'
                    'min': 0.0,
                    'max': 1.5,  # ALS/BPR scores typically in [0, 1.5] after U@V.T
                    'clip': True  # Hard clip to prevent outliers
                },
                'content': {
                    'method': 'clip',
                    'min': -1.0,  # Cosine similarity range
                    'max': 1.0,
                    'clip': True
                },
                'popularity': {
                    'method': 'percentile',  # Use pre-computed percentiles from Task 01
                    'p99': None,  # Will be loaded from metadata
                    'p01': None
                },
                'quality': {
                    'method': 'fixed',
                    'min': 1.0,  # Rating range
                    'max': 5.0,
                    'clip': True
                }
            }
        }
        
        # Load global statistics for normalization (from Task 01 processed data)
        self._load_global_stats()

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
        list: Reranked recommendations với augmented scores
    
    Note:
        - Uses PhoBERTEmbeddingLoader.compute_user_profile() for content scoring
        - Applies diversity penalty using Task 03 diversity metric logic
        - Final scores stored in 'final_score' key
    """
    if not cf_recommendations:
        return []
    
    # Extract product IDs
    candidate_ids = [rec['product_id'] for rec in cf_recommendations]
    
    # Compute user profile (Task 05: PhoBERTEmbeddingLoader)
    user_profile_emb = None
    if user_history:
        user_profile_emb = self.phobert.compute_user_profile(
            user_history,
            strategy=self.config['user_profile_strategy']
        )
    
    # Compute signals
    signals = self._compute_signals(candidate_ids, user_profile_emb)
    
    # Normalize signals
    normalized = self._normalize_signals(signals)
    
    # Combine với weights
    final_scores = self._combine_scores(normalized, self.config['weights'])
    
    # Apply diversity penalty (Task 03: BERT-based diversity)
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
    
    # Popularity (log-transformed from Task 01)
    signals['popularity'] = {}
    for cid in candidate_ids:
        product = self.metadata[self.metadata['product_id'] == cid]
        if not product.empty:
            # Use pre-computed log-transformed popularity from Task 01
            signals['popularity'][cid] = float(product.iloc[0].get('popularity_score', 0.0))
        else:
            signals['popularity'][cid] = 0.0
    
    # Quality (pre-computed from Task 01: quality_score, avg_rating, pct_5star)
    signals['quality'] = {}
    for cid in candidate_ids:
        product = self.metadata[self.metadata['product_id'] == cid]
        if not product.empty:
            # Prefer quality_score from Task 01, fallback to avg_rating
            signals['quality'][cid] = float(product.iloc[0].get('quality_score', 
                                              product.iloc[0].get('avg_rating', 3.0)))
        else:
            signals['quality'][cid] = 3.0
    
    return signals
```

##### Method 0: `_load_global_stats()` (NEW - Critical for Global Normalization)
```python
def _load_global_stats(self):
    """
    Load global statistics for normalization from Task 01 processed data.
    
    This prevents local normalization issues where different requests
    get incomparable normalized scores.
    """
    # Load data_stats.json from Task 01
    try:
        with open('data/processed/data_stats.json', 'r') as f:
            stats = json.load(f)
        
        # Extract global ranges for normalization
        if 'popularity' in stats:
            self.config['normalization']['popularity']['p99'] = stats['popularity'].get('p99', 10.0)
            self.config['normalization']['popularity']['p01'] = stats['popularity'].get('p01', 0.0)
        
        # CF score range (optional - can be computed from model validation)
        if 'cf_score_range' in stats:
            self.config['normalization']['cf']['max'] = stats['cf_score_range']['max']
            self.config['normalization']['cf']['min'] = stats['cf_score_range']['min']
        
        print("Loaded global normalization statistics")
        
    except FileNotFoundError:
        print("Warning: data_stats.json not found. Using default normalization ranges.")
        # Use hardcoded defaults from config
        pass
```

##### Method 3: `_normalize_signals(signals)` (UPDATED - Global Normalization)
```python
def _normalize_signals(self, signals):
    """
    Normalize all signals to [0, 1] range using GLOBAL statistics.
    
    CRITICAL FIX: Uses pre-defined global ranges instead of local min/max
    to ensure consistent normalization across different requests.
    
    Problem with old approach:
    - User A gets [0.91, ..., 0.99] → normalized to [0.0, ..., 1.0]
    - User B gets [0.11, ..., 0.19] → normalized to [0.0, ..., 1.0]
    - Both look equally good despite User A's scores being much higher!
    
    New approach:
    - Use global ranges (e.g., CF scores in [0, 1.5])
    - User A: [0.91/1.5, ..., 0.99/1.5] = [0.61, ..., 0.66]
    - User B: [0.11/1.5, ..., 0.19/1.5] = [0.07, ..., 0.13]
    - Now correctly reflects quality difference!
    
    Args:
        signals: Dict of signal dicts
    
    Returns:
        dict: Normalized signals (same structure)
    """
    normalized = {}
    
    for signal_name, scores in signals.items():
        if signal_name not in self.config['normalization']:
            # No normalization config → skip or use identity
            normalized[signal_name] = scores
            continue
        
        norm_config = self.config['normalization'][signal_name]
        method = norm_config['method']
        
        values = np.array(list(scores.values()))
        
        if method == 'clip':
            # Hard clip to global range
            min_val = norm_config['min']
            max_val = norm_config['max']
            
            # Clip outliers
            if norm_config.get('clip', True):
                values = np.clip(values, min_val, max_val)
            
            # Normalize to [0, 1]
            if max_val > min_val:
                norm_values = (values - min_val) / (max_val - min_val)
            else:
                norm_values = np.zeros_like(values)
        
        elif method == 'percentile':
            # Use pre-computed percentiles (e.g., for popularity)
            p01 = norm_config.get('p01', 0.0)
            p99 = norm_config.get('p99', 1.0)
            
            if p99 > p01:
                # Clip to percentile range and normalize
                values_clipped = np.clip(values, p01, p99)
                norm_values = (values_clipped - p01) / (p99 - p01)
            else:
                norm_values = np.zeros_like(values)
        
        elif method == 'fixed':
            # Fixed range (e.g., ratings 1-5)
            min_val = norm_config['min']
            max_val = norm_config['max']
            
            if norm_config.get('clip', True):
                values = np.clip(values, min_val, max_val)
            
            norm_values = (values - min_val) / (max_val - min_val)
        
        else:
            # Fallback: identity (no normalization)
            norm_values = values
        
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
Evaluate hybrid reranking vs pure CF using Task 03 metrics.
"""

def main():
    # Load models
    cf_recommender = CFRecommender(model_loader, enable_reranking=False)
    hybrid_recommender = CFRecommender(model_loader, enable_reranking=True)
    
    # Load test users
    test_data = pd.read_parquet('data/processed/interactions.parquet')
    test_users = test_data[test_data['split'] == 'test']['user_id'].unique()
    
    # Evaluate both (Task 03: evaluate_hybrid_model)
    cf_metrics = evaluate_recommendations(cf_recommender, test_users)
    hybrid_metrics = evaluate_recommendations(hybrid_recommender, test_users)
    
    # Compare
    print("CF Metrics:", cf_metrics)
    print("Hybrid Metrics:", hybrid_metrics)
    
    # Diversity comparison (Task 03: compute_diversity_bert)
    cf_diversity = compute_diversity_bert(
        cf_recommender, 
        test_users[:100],
        bert_embeddings=phobert_loader
    )
    hybrid_diversity = compute_diversity_bert(
        hybrid_recommender, 
        test_users[:100],
        bert_embeddings=phobert_loader
    )
    
    print(f"CF Diversity: {cf_diversity:.3f}")
    print(f"Hybrid Diversity: {hybrid_diversity:.3f}")
    
    # Semantic alignment (Task 03)
    hybrid_alignment = compute_semantic_alignment(
        hybrid_recommender,
        test_users[:100],
        bert_embeddings=phobert_loader
    )
    print(f"Semantic Alignment: {hybrid_alignment:.3f}")
```

## Component 7: Use Cases

### Use Case 1: Personalized Recommendations
```python
# Pure CF
recs_cf = recommender.recommend(user_id=12345, topk=10, rerank=False)

# Hybrid (CF + content + popularity)
recs_hybrid = recommender.recommend(user_id=12345, topk=10, rerank=True)

# Compare diversity
print(f"CF Diversity: {compute_diversity_bert(recs_cf):.3f}")
print(f"Hybrid Diversity: {compute_diversity_bert(recs_hybrid):.3f}")
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

# For completely new users, use BERT-based content recommendations
# (Task 05: PhoBERTEmbeddingLoader can generate user profiles from browsing history)
```

### Use Case 3: Similar Items (Content-Based with BERT)
```python
# Recommend similar products to a given item (Task 05 implementation)
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
torch>=1.13.0  # BERT embeddings (Task 01)
transformers>=4.25.0  # PhoBERT model
numpy>=1.22.0
pandas>=1.4.0
scipy>=1.8.0
scikit-learn>=1.1.0  # Normalization
```

## Configuration Examples

### Config 1: Content-Heavy (Cold-Start Focused)
```python
config = {
    'weights': {
        'cf': 0.3,
        'content': 0.4,  # Higher weight on BERT similarity
        'popularity': 0.2,
        'quality': 0.1
    },
    'diversity': {
        'enabled': True,
        'penalty': 0.15,
        'threshold': 0.85  # More aggressive diversity
    },
    'user_profile_strategy': 'tf_idf'  # Task 01: emphasize unique preferences
}
```

### Config 2: CF-Heavy (Collaborative Signal Focused)
```python
config = {
    'weights': {
        'cf': 0.7,
        'content': 0.1,
        'popularity': 0.1,
        'quality': 0.1
    },
    'diversity': {
        'enabled': False  # Pure CF ranking
    },
    'user_profile_strategy': 'weighted_mean'  # Task 01: balanced averaging
}
```

### Config 3: Balanced Hybrid (Recommended)
```python
config = {
    'weights': {
        'cf': 0.5,
        'content': 0.2,
        'popularity': 0.15,
        'quality': 0.15
    },
    'diversity': {
        'enabled': True,
        'penalty': 0.1,
        'threshold': 0.9  # Moderate diversity boost
    },
    'user_profile_strategy': 'weighted_mean'  # Task 01: default strategy
}
```

## Cross-Task Integration Summary

**Task 01 (Data Layer)**:
- `BERTEmbeddingGenerator`: Generates product_embeddings.pt
- User profile strategies: weighted_mean, tf_idf, recency
- Embedding versioning with metadata

**Task 02 (Training Pipelines)**:
- `BERTEnhancedALS`: Initialize item factors from BERT embeddings via SVD
- Multi-task learning: Dual loss (CF + content alignment)

**Task 03 (Evaluation Metrics)**:
- `compute_diversity_bert()`: Intra-list diversity using BERT similarities
- `compute_semantic_alignment()`: User profile → recommendations relevance
- `evaluate_hybrid_model()`: CF vs CF+BERT comparison framework

**Task 04 (Model Registry)**:
- BERT embedding artifact tracking with version compatibility
- `load_model_with_embeddings()`: Atomic loading of CF model + BERT embeddings

**Task 05 (Serving Layer)**:
- `PhoBERTEmbeddingLoader`: Production-ready BERT loader with LRU caching
- Two-stage recommendation: CF candidates → BERT reranking
- `compute_user_profile()`: Aggregate user history into BERT space

**Task 06 (Monitoring)**:
- `check_embedding_freshness()`: Alert if BERT embeddings stale
- `detect_semantic_drift()`: Compare embedding versions via cosine similarity

**Task 07 (Automation)**:
- `refresh_bert_embeddings.py`: Weekly regeneration of product embeddings
- Cron job: Tuesday 3am BERT refresh, aligned with model training schedule

**Task 08 (This Task)**:
- `HybridReranker`: Weighted combination of CF + BERT + metadata signals
- Diversity penalty using BERT similarity thresholds
- End-to-end integration of all previous components

## Timeline Estimate

- **PhoBERT loader refactoring**: 0.5 day (already in Task 05)
- **HybridReranker implementation**: 1 day
- **Signal computation methods**: 1 day
- **Diversity penalty logic**: 0.5 day
- **Configuration tuning**: 1 day (using Task 03 metrics)
- **Testing và validation**: 1 day
- **Documentation updates**: 0.5 day
- **Total**: ~5.5 days (excluding Task 05 work already done)

## Success Criteria

- [ ] HybridReranker integrated with PhoBERTEmbeddingLoader (Task 05)
- [ ] Weighted score combination implemented
- [ ] Diversity penalty reduces BERT similarity clustering
- [ ] Semantic alignment (Task 03) > 0.7 for hybrid recommendations
- [ ] Diversity (Task 03) improves by ≥10% over pure CF
- [ ] Cold-start coverage (Task 03) increases with content signals
- [ ] Configuration examples tested across user segments
- [ ] Integration with monitoring (Task 06) for embedding freshness
- [ ] Automation (Task 07) ensures weekly BERT updates
- [ ] Documentation complete with cross-task references
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
