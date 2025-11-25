# End-to-End Evaluation Report: CF vs Hybrid Reranking

**Date:** 2025-11-25  
**Model:** bert_als_20251125_061805  
**Dataset:** Vietnamese Cosmetics Recommendations

## Executive Summary

End-to-end testing of the recommendation pipeline from **ingest → serving** completed successfully. The hybrid reranking system shows improved **diversity (+6.9%)** at the cost of reduced **recall/NDCG (-12-18%)**, representing a typical exploration-exploitation tradeoff.

## Data Pipeline Verification

| Artifact | Path | Status |
|----------|------|--------|
| Interactions | data/processed/interactions.parquet | ✅ 56,338 rows |
| User Mappings | data/processed/user_item_mappings.json | ✅ 26,050 trainable users |
| CF Model (U) | artifacts/cf/bert_als/20251125_061805/U.npy | ✅ (26050, 64) |
| CF Model (V) | artifacts/cf/bert_als/20251125_061805/V.npy | ✅ (1423, 64) |
| PhoBERT Embeddings | data/processed/content_based_embeddings/product_embeddings.pt | ✅ 2,244 products, 1024-dim |

### Fixes Applied During Testing

1. **U/V Matrix Swap**: Model files had U/V swapped. Added auto-detection in `loader.py` to fix at load time.
2. **User History Data Leakage**: `load_user_histories()` was loading all interactions including test set. Fixed to load train split only.
3. **PhoBERT API Mismatch**: `rerank.py` and `evaluate_hybrid.py` were using wrong method signatures for BERT similarity. Fixed to use `get_embedding_normalized()` + dot product.

## Evaluation Results (N=500 users)

### Pure CF vs Hybrid Comparison

| Metric | Pure CF | Hybrid | Change |
|--------|---------|--------|--------|
| **Recall@5** | 0.0900 | 0.0800 | -11.1% |
| **Recall@10** | 0.1280 | 0.1060 | **-17.2%** |
| **Recall@20** | 0.1880 | 0.1380 | -26.6% |
| **NDCG@5** | 0.0653 | 0.0593 | -9.2% |
| **NDCG@10** | 0.0774 | 0.0674 | **-12.9%** |
| **NDCG@20** | 0.0924 | 0.0756 | -18.2% |
| **Diversity** | 0.1850 | 0.1978 | **+6.9%** |
| **Semantic Alignment** | 0.8280 | 0.8181 | -1.2% |
| **Latency (ms)** | 9.73 | 61.63 | +533% |

### Key Observations

1. **Pure CF Performance**
   - Recall@10 = 12.8% is reasonable for extremely sparse data (~1.23 interactions/user)
   - Baseline confirms the BERT-enhanced ALS model is functional

2. **Hybrid Reranking Trade-offs**
   - **Diversity improved**: +6.9% (0.185 → 0.198) as intended by diversity penalty
   - **Recall/NDCG decreased**: -12-18% because reranking promotes diverse but less relevant items
   - **Latency increased significantly**: 10ms → 62ms due to BERT similarity computations

3. **Semantic Alignment**
   - Both methods have high alignment (>0.82) indicating recommendations match user profiles
   - Slight decrease with hybrid is expected as diversity pushes items further from user center

## Recommendations

### For Production Deployment

1. **Use Pure CF for conversion-focused scenarios** where accuracy matters most
2. **Use Hybrid for discovery/exploration scenarios** where variety improves user experience
3. **Consider adaptive weights** based on user engagement patterns

### Hybrid Weight Tuning Suggestions

Current weights (trainable users):
```yaml
weights_trainable:
  cf: 0.30      # Consider increasing to 0.40-0.50
  content: 0.40  # Consider reducing to 0.30
  popularity: 0.20
  quality: 0.10
```

### Performance Optimization

1. **Pre-compute BERT similarities** for top-K candidates to reduce latency
2. **Batch inference** for user profile computation
3. **Cache popular item embeddings** in memory

## Files Modified

| File | Change |
|------|--------|
| `service/recommender/loader.py` | Added U/V swap detection; Fixed user history to load train only |
| `service/recommender/rerank.py` | Fixed `_apply_diversity_penalty()` to use correct BERT API |
| `scripts/evaluate_hybrid.py` | Fixed `compute_diversity_bert()` and `compute_semantic_alignment()` |

## Next Steps

1. Tune hybrid weights to balance diversity/accuracy
2. Implement weight personalization per user segment
3. Add A/B testing framework for production comparison
4. Profile and optimize BERT similarity computation latency
