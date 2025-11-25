# Deployment Optimization Guide

## Overview

This document describes the deployment optimization strategies for the Vietnamese Cosmetics Recommendation System, with a focus on achieving **<200ms latency** for the cold-start path that handles **~91% of traffic**.

## Key Metrics & Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| P99 Latency | <200ms | TBD | ⏳ |
| P90 Latency | <100ms | TBD | ⏳ |
| Cold-Start Latency | <200ms | TBD | ⏳ |
| CF Path Latency | <100ms | TBD | ⏳ |
| Cache Hit Rate | >70% | TBD | ⏳ |

## Architecture Overview

```
Request → Router → Is Trainable User?
                   ├─ Yes (8.6%) → CF Scoring → Hybrid Rerank → Response
                   └─ No (91.4%) → Fallback Path:
                                   ├─ User Profile (PhoBERT) [CACHED]
                                   ├─ Similar Items [CACHED]
                                   ├─ Popularity Items [PRE-COMPUTED]
                                   └─ Hybrid Mix → Response
```

## Cache Strategy

### 1. LRU Caches

Three LRU caches optimized for the cold-start path:

| Cache | Max Size | TTL | Purpose |
|-------|----------|-----|---------|
| User Profile | 50,000 | 1 hour | BERT user embeddings |
| Item Similarity | 5,000 | 24 hours | Pre-computed similar items |
| Fallback Results | 10,000 | 30 min | Complete fallback responses |

### 2. Pre-computed Data

For cold-start optimization:

- **Popular Items (200)**: Pre-enriched with metadata
- **Popular Item Similarities**: Top-50 similar items for each popular product
- **Item-Item Similarity Matrix**: V @ V.T for PhoBERT (if <3K items)

## Warm-up Strategy

### Critical: Warm-up MUST run before accepting traffic

```python
# During service startup (api.py lifespan)
from service.recommender.cache import get_cache_manager, async_warmup

cache_manager = get_cache_manager()
await async_warmup(cache_manager)
```

### Warm-up Steps

1. **Load Popular Items** (~10ms)
   - Top 200 items by `num_sold_time`
   - Enriched with product metadata

2. **Pre-compute Similarities** (~500-2000ms)
   - PhoBERT similarity for each popular item
   - Top-50 similar items cached

3. **Optional: Pre-warm User Profiles**
   - For frequent/VIP users
   - Compute and cache BERT profiles

## Cold-Start Path Optimization

### Before Optimization (Estimated)
```
Cold-Start Request → Fallback → Compute User Profile → Find Similar → Enrich → Return
                                ↓ 50-100ms            ↓ 100-200ms    ↓ 20-50ms
                                Total: 170-350ms ❌
```

### After Optimization
```
Cold-Start Request → Check Cache → [HIT] → Return (5-10ms) ✓
                   ↓ [MISS]
                   → Cached Popular → Quick Similarity Lookup → Return
                   ↓ 10ms          ↓ 20-50ms (cached)
                   Total: 30-60ms ✓
```

## API Endpoints

### Health & Monitoring

```bash
# Service health
GET /health

# Detailed statistics (includes cache stats)
GET /stats

# Cache-specific stats
GET /cache_stats
```

### Cache Management

```bash
# Trigger cache warmup
POST /cache_warmup?force=false

# Clear all caches (use with caution)
POST /cache_clear
```

## Load Testing

### Running Load Tests

```powershell
# Start server first
uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 4

# Run load test (mixed traffic)
python scripts/load_test.py --host localhost --port 8000 --total 500 --concurrency 20

# Test cold-start specifically
python scripts/load_test.py --test-type coldstart --total 500

# Test with reranking enabled
python scripts/load_test.py --rerank --total 500
```

### Expected Results

| Test Type | Target P99 | Expected P99 |
|-----------|------------|--------------|
| CF Only | <100ms | ~50-80ms |
| Cold-Start Only | <200ms | ~80-150ms |
| Mixed (Natural) | <200ms | ~100-180ms |
| With Reranking | <200ms | +20-50ms overhead |

## Deployment Analyzer

Run the optimization analyzer to check current status:

```powershell
# Full analysis
python scripts/deployment_optimizer.py --analyze

# Latency profiling only
python scripts/deployment_optimizer.py --profile --samples 50

# Cache analysis only
python scripts/deployment_optimizer.py --cache
```

### Sample Output

```
📊 DEPLOYMENT OPTIMIZATION REPORT
==================================
Status: GOOD
Critical Issues: 0

⏱️ LATENCY PROFILES
-------------------
✓ cf_path:
   Mean: 45.2ms | P99: 89.3ms
✓ coldstart_path:
   Mean: 78.5ms | P99: 156.2ms
✓ phobert_similarity:
   Mean: 12.3ms | P99: 28.7ms

💾 CACHE ANALYSIS
-----------------
Warmed Up: Yes ✓
✓ user_profile: 1234/50000 entries, 72.3% hit rate
✓ item_similarity: 200/5000 entries, 85.1% hit rate
✓ fallback: 567/10000 entries, 68.4% hit rate

💡 RECOMMENDATIONS
------------------
✅ All optimizations look good! Latency targets are met.
```

## Configuration

Edit `config/serving_config.yaml` to tune:

- Cache sizes and TTLs
- Warm-up settings
- Reranking weights
- Performance targets

## Troubleshooting

### High Cold-Start Latency (>200ms)

1. **Check cache warmup**: `GET /cache_stats` - Is `warmed_up: true`?
2. **Check PhoBERT**: Is it loaded? Check startup logs
3. **Check popular items**: Are they pre-computed?

### Low Cache Hit Rate (<50%)

1. Increase cache sizes in config
2. Extend TTL (trade freshness for performance)
3. Ensure warmup runs on startup

### Memory Issues

1. Reduce cache sizes
2. Disable item-item similarity matrix pre-computation
3. Use approximate nearest neighbor (FAISS) for large catalogs

## Monitoring Checklist

- [ ] Cache warmup completes on startup
- [ ] P99 latency <200ms in load tests
- [ ] Cache hit rates >70% after warm period
- [ ] No OOM errors under load
- [ ] Cold-start path specifically tested

## Future Optimizations

1. **FAISS Integration**: For faster similarity search with large catalogs
2. **Redis Caching**: Distributed cache for multi-instance deployment
3. **Async Pre-computation**: Background job to refresh similarities
4. **CDN/Edge Caching**: Cache popular item responses at CDN level
