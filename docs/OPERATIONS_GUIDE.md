# Operations Guide

## Table of Contents

1. [System Overview](#system-overview)
2. [Service Management](#service-management)
3. [Configuration](#configuration)
4. [Monitoring & Alerting](#monitoring--alerting)
5. [Cache Management](#cache-management)
6. [Model Management](#model-management)
7. [Troubleshooting](#troubleshooting)
8. [Rollback Procedures](#rollback-procedures)
9. [Emergency Contacts](#emergency-contacts)

---

## System Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Load Balancer / API Gateway                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  Worker 1 │   │  Worker 2 │   │  Worker N │
            │  (FastAPI)│   │  (FastAPI)│   │  (FastAPI)│
            └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                  │               │               │
                  └───────────────┼───────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
            ┌───────────────┐           ┌───────────────┐
            │ Model Registry│           │  Data Layer   │
            │ (artifacts/cf)│           │(data/processed)│
            └───────────────┘           └───────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| API Server | `service/api.py` | FastAPI recommendation endpoints |
| Recommender | `service/recommender/` | Core recommendation logic |
| Cache Manager | `service/recommender/cache.py` | LRU caching |
| Model Loader | `service/recommender/loader.py` | Model loading & hot-reload |
| Model Registry | `artifacts/cf/registry.json` | Model versioning |
| Data Artifacts | `data/processed/` | Processed data files |

### Traffic Pattern

- **Cold-Start Users (91.4%)**: Content-based + popularity fallback
- **Trainable Users (8.6%)**: CF scoring + hybrid reranking

---

## Service Management

### Starting the Service

```powershell
# Development (single worker, auto-reload)
uvicorn service.api:app --host 0.0.0.0 --port 8000 --reload

# Production (multiple workers)
uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn (recommended for production)
gunicorn service.api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Stopping the Service

```powershell
# Graceful shutdown (SIGTERM)
kill -SIGTERM <pid>

# Find process
Get-Process -Name python | Where-Object {$_.CommandLine -like "*uvicorn*"}
```

### Health Check

```powershell
# Basic health
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "model_id": "bert_als_20251108_v1",
  "model_type": "bert_als",
  "num_users": 26127,
  "num_items": 2187,
  "trainable_users": 26127,
  "timestamp": "2025-11-25T10:30:00"
}
```

### Service Statistics

```powershell
# Detailed stats
curl http://localhost:8000/stats

# Cache stats specifically
curl http://localhost:8000/cache_stats
```

---

## Configuration

### Main Configuration Files

| File | Purpose |
|------|---------|
| `config/serving_config.yaml` | Serving layer settings |
| `config/als_config.yaml` | ALS training config |
| `config/alerts_config.yaml` | Alert thresholds |

### Key Configuration Parameters

```yaml
# config/serving_config.yaml

cache:
  user_profile:
    max_size: 50000      # Increase for more cache capacity
    ttl_seconds: 3600    # Decrease for fresher data
    
reranking:
  enabled: true          # Set to false to disable reranking
  weights_trainable:
    cf: 0.30
    content: 0.40
    popularity: 0.20
    quality: 0.10
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RECSYS_LOG_LEVEL` | `INFO` | Logging level |
| `RECSYS_WORKERS` | `4` | Number of workers |
| `RECSYS_PORT` | `8000` | API port |
| `RECSYS_MODEL_PATH` | `artifacts/cf` | Model directory |

---

## Monitoring & Alerting

### Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| P99 Latency | <200ms | >300ms |
| P50 Latency | <50ms | >100ms |
| Error Rate | <1% | >5% |
| Cache Hit Rate | >70% | <50% |
| Fallback Rate | ~91% | >95% |

### Log Locations

```
logs/
├── cf/               # CF model logs
├── service/          # API service logs
│   └── api.log
├── pipelines/        # Pipeline execution logs
└── test/             # Test execution logs
```

### Monitoring Endpoints

```powershell
# Health check
GET /health

# Service statistics
GET /stats

# Cache statistics
GET /cache_stats

# Model information
GET /model_info
```

### Alert Conditions

1. **High Latency** (P99 > 300ms for 5 minutes)
   - Check cache warmup status
   - Check PhoBERT embedding loading
   - Check system resources

2. **High Error Rate** (>5% for 5 minutes)
   - Check model loading
   - Check data file availability
   - Check memory usage

3. **Low Cache Hit Rate** (<50% for 10 minutes)
   - Trigger cache warmup
   - Check cache configuration
   - Verify TTL settings

---

## Cache Management

### Cache Warmup

```powershell
# Trigger warmup via API
curl -X POST http://localhost:8000/cache_warmup

# Force re-warmup
curl -X POST "http://localhost:8000/cache_warmup?force=true"
```

### Cache Statistics

```powershell
curl http://localhost:8000/cache_stats

# Response example
{
  "warmed_up": true,
  "caches": {
    "user_profile": {
      "size": 12345,
      "max_size": 50000,
      "hit_rate": 0.75,
      "hits": 98765,
      "misses": 32100
    },
    "item_similarity": {
      "size": 200,
      "max_size": 5000,
      "hit_rate": 0.92
    },
    "fallback": {
      "size": 5678,
      "max_size": 10000,
      "hit_rate": 0.68
    }
  },
  "precomputed": {
    "popular_items": 200,
    "popular_similarities": 200
  }
}
```

### Cache Clear

```powershell
# Clear all caches (use with caution!)
curl -X POST http://localhost:8000/cache_clear

# After clearing, warmup again
curl -X POST http://localhost:8000/cache_warmup
```

---

## Model Management

### Current Model Check

```powershell
curl http://localhost:8000/model_info

# Response
{
  "model_id": "bert_als_20251108_v1",
  "model_type": "bert_als",
  "num_users": 26127,
  "num_items": 2187,
  "factors": 64,
  "loaded_at": "2025-11-25T10:30:00",
  "score_range": {"min": -0.5, "max": 1.5, "p01": 0.1, "p99": 1.2}
}
```

### Model Hot-Reload

```powershell
# Check for new model and reload if available
curl -X POST http://localhost:8000/reload_model

# Response
{
  "status": "reloaded",  # or "no_update"
  "previous_model_id": "bert_als_20251107_v1",
  "new_model_id": "bert_als_20251108_v1",
  "reloaded": true
}
```

### Model Registry

Location: `artifacts/cf/registry.json`

```json
{
  "current_best": {
    "model_id": "bert_als_20251108_v1",
    "model_type": "bert_als",
    "registered_at": "2025-11-08T15:30:00"
  },
  "models": {
    "bert_als_20251108_v1": {
      "path": "artifacts/cf/bert_als/20251108_v1",
      "model_type": "bert_als",
      "metrics": {"recall_10": 0.285, "ndcg_10": 0.182}
    }
  }
}
```

### Switching Models

```powershell
# 1. Update registry.json to change current_best
# 2. Trigger reload
curl -X POST http://localhost:8000/reload_model
```

---

## Troubleshooting

### Common Issues

#### 1. High Latency

**Symptoms**: P99 > 200ms

**Diagnosis**:
```powershell
# Check cache status
curl http://localhost:8000/cache_stats

# Check if warmed up
# Look for "warmed_up": true
```

**Resolution**:
```powershell
# Trigger warmup
curl -X POST http://localhost:8000/cache_warmup

# If persists, check PhoBERT loading in logs
tail -f logs/service/api.log | grep -i phobert
```

#### 2. Service Won't Start

**Symptoms**: uvicorn fails to start

**Diagnosis**:
```powershell
# Check model files exist
ls artifacts/cf/registry.json
ls data/processed/user_item_mappings.json

# Check for port conflict
netstat -ano | findstr :8000
```

**Resolution**:
```powershell
# Kill conflicting process
Stop-Process -Id <pid>

# Or use different port
uvicorn service.api:app --port 8001
```

#### 3. Model Load Failure

**Symptoms**: 503 "Service not initialized"

**Diagnosis**:
```powershell
# Check registry
cat artifacts/cf/registry.json

# Check model files
ls artifacts/cf/bert_als/*/
```

**Resolution**:
- Verify model files exist (U.npy, V.npy, params.json, metadata.json)
- Check file permissions
- Verify registry.json points to valid model

#### 4. Out of Memory

**Symptoms**: Process killed, high memory usage

**Diagnosis**:
```powershell
# Check memory usage
Get-Process python | Select-Object Name, WorkingSet64
```

**Resolution**:
```yaml
# Reduce cache sizes in config/serving_config.yaml
cache:
  user_profile:
    max_size: 25000  # Reduced from 50000
  item_similarity:
    max_size: 2000   # Reduced from 5000
```

#### 5. Low Recommendation Quality

**Symptoms**: Low Recall@10, poor user feedback

**Diagnosis**:
```powershell
# Run evaluation
python scripts/evaluate_hybrid.py --output reports/eval_check.csv
```

**Resolution**:
- Check if correct model is loaded
- Verify data pipeline ran successfully
- Consider retraining with updated data

---

## Rollback Procedures

### Scenario 1: Hybrid Reranker Issues

**Symptoms**: Degraded recommendation quality, high latency in reranking

**Immediate Mitigation** (Disable Reranking):

```powershell
# Option A: API-level disable (temporary)
# Edit requests to include rerank=false

# Option B: Config-level disable
# Edit config/serving_config.yaml
```

```yaml
# config/serving_config.yaml
reranking:
  enabled: false  # Disable reranking
```

```powershell
# Restart service to apply
# Service will use CF scores directly without reranking
```

**Full Rollback** (Restore Previous Weights):

```powershell
# 1. Backup current config
cp config/serving_config.yaml config/serving_config.yaml.backup

# 2. Restore previous weights or defaults
# Edit config/serving_config.yaml:
```

```yaml
reranking:
  enabled: true
  weights_trainable:
    cf: 0.60       # Increase CF weight
    content: 0.20  # Decrease content
    popularity: 0.15
    quality: 0.05
  weights_cold_start:
    content: 0.50
    popularity: 0.40
    quality: 0.10
```

### Scenario 2: Model Rollback

**Symptoms**: New model performs poorly

**Procedure**:

```powershell
# 1. Check available models
cat artifacts/cf/registry.json

# 2. Edit registry to point to previous model
# Change current_best to previous model_id

# 3. Reload model
curl -X POST http://localhost:8000/reload_model

# 4. Verify
curl http://localhost:8000/model_info
```

### Scenario 3: Full Service Rollback

**Symptoms**: Critical failures, need to restore previous version

**Procedure**:

```powershell
# 1. Stop current service
Stop-Process -Name python -Force

# 2. Restore from backup
# (Assuming you have version control or backup)
git checkout <previous-tag> -- service/

# 3. Restore model artifacts
# Copy from backup location

# 4. Restart service
uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 4

# 5. Warmup caches
curl -X POST http://localhost:8000/cache_warmup

# 6. Verify health
curl http://localhost:8000/health
```

### Scenario 4: Cache Corruption

**Symptoms**: Inconsistent results, stale recommendations

**Procedure**:

```powershell
# 1. Clear all caches
curl -X POST http://localhost:8000/cache_clear

# 2. Re-warmup
curl -X POST "http://localhost:8000/cache_warmup?force=true"

# 3. Verify cache stats
curl http://localhost:8000/cache_stats
```

### Rollback Verification Checklist

After any rollback:

- [ ] Health check passes
- [ ] Model info shows expected model_id
- [ ] Cache is warmed up
- [ ] Sample recommendations return valid results
- [ ] Latency is within targets (<200ms P99)
- [ ] Error rate is normal (<1%)

---

## Emergency Contacts

| Role | Name | Contact |
|------|------|---------|
| On-Call Engineer | TBD | TBD |
| Team Lead | TBD | TBD |
| DevOps | TBD | TBD |

### Escalation Path

1. **Level 1** (0-15 min): On-call engineer
2. **Level 2** (15-30 min): Team lead
3. **Level 3** (30+ min): DevOps + Management

---

## Appendix

### Useful Commands

```powershell
# Quick health check
curl -s http://localhost:8000/health | jq .status

# Check latency with timing
Measure-Command { curl -s http://localhost:8000/health } | Select TotalMilliseconds

# Test recommendation
curl -X POST http://localhost:8000/recommend `
  -H "Content-Type: application/json" `
  -d '{"user_id": 12345, "topk": 10}'

# Run load test
python scripts/load_test.py --host localhost --port 8000 --total 100

# Run optimization analyzer
python scripts/deployment_optimizer.py --analyze
```

### Log Analysis

```powershell
# Find errors
Select-String -Path "logs/service/api.log" -Pattern "ERROR"

# Find slow requests
Select-String -Path "logs/service/api.log" -Pattern "latency=[2-9]\d{2,}ms"

# Tail logs
Get-Content -Path "logs/service/api.log" -Tail 100 -Wait
```
