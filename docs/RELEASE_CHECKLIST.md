# Release Checklist

## Vietnamese Cosmetics Recommendation System - Go-Live Checklist

**Version**: 1.0.0  
**Release Date**: 2025-11-25  
**Release Manager**: _________________

---

## Pre-Release (T-7 days)

### Code Freeze

- [ ] All feature branches merged to main
- [ ] Code review completed for all changes
- [ ] No pending critical bugs
- [ ] CHANGELOG.md updated

### Testing

- [ ] Unit tests passing (`pytest tests/`)
- [ ] Integration tests passing
- [ ] Load tests completed with acceptable results
  - [ ] P99 latency < 200ms
  - [ ] Throughput > 100 req/s
- [ ] Cold-start path tested specifically
- [ ] Hybrid reranker tested with various weight configs

### Documentation

- [ ] CHANGELOG.md finalized
- [ ] OPERATIONS_GUIDE.md reviewed
- [ ] DEPLOYMENT_OPTIMIZATION.md current
- [ ] API documentation updated

---

## Staging Deployment (T-3 days)

### Environment Preparation

- [ ] Staging environment matches production specs
- [ ] All dependencies installed
- [ ] Python version: _______ (required: 3.9+)
- [ ] Available disk space: _______ GB (required: 10+ GB)
- [ ] Available memory: _______ GB (required: 8+ GB)

### Artifact Sync

```powershell
# Sync model artifacts to staging
robocopy artifacts/cf staging-server:/app/artifacts/cf /E /Z

# Sync data files
robocopy data/processed staging-server:/app/data/processed /E /Z

# Sync configuration
robocopy config staging-server:/app/config /E /Z
```

- [ ] Model registry synced (`artifacts/cf/registry.json`)
- [ ] Model files synced (U.npy, V.npy, params.json, metadata.json)
- [ ] Data artifacts synced:
  - [ ] `user_item_mappings.json`
  - [ ] `trainable_user_mapping.json`
  - [ ] `top_k_popular_items.json`
  - [ ] `interactions.parquet`
  - [ ] `enriched_products.parquet`
- [ ] PhoBERT embeddings synced (`product_embeddings.pt`)
- [ ] Configuration files synced

### Configuration Migration

- [ ] Review `config/serving_config.yaml` for staging
- [ ] Verify model paths are correct
- [ ] Verify data paths are correct
- [ ] Update environment-specific settings:

```yaml
# Staging-specific overrides
cache:
  user_profile:
    max_size: 25000  # Reduced for staging
    
logging:
  level: "DEBUG"  # More verbose for staging
```

- [ ] Config files committed to staging environment

### Staging Deployment

```powershell
# Deploy to staging
ssh staging-server
cd /app

# Start service
uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 2

# Verify startup
curl http://localhost:8000/health
```

- [ ] Service starts without errors
- [ ] Health check passes
- [ ] Model loaded successfully
- [ ] Cache warmup completed

### Staging Smoke Tests

```powershell
# Run smoke tests
python scripts/smoke_test.py --host staging-server --port 8000
```

- [ ] Health endpoint responds
- [ ] Single recommendation works
- [ ] Batch recommendation works
- [ ] CF path works (trainable user)
- [ ] Cold-start path works
- [ ] Reranking works
- [ ] Similar items works
- [ ] Cache stats endpoint works

### Staging Load Tests

```powershell
# Run load test on staging
python scripts/load_test.py --host staging-server --port 8000 --total 500 --concurrency 20
```

- [ ] P99 latency < 200ms
- [ ] Error rate < 1%
- [ ] No memory leaks (monitor during test)
- [ ] Cache hit rate > 50% after warmup

### Staging Sign-off

- [ ] QA sign-off: _______________ Date: ___________
- [ ] Dev sign-off: _______________ Date: ___________
- [ ] Product sign-off: _______________ Date: ___________

---

## Production Deployment (T-0)

### Pre-Deployment Checks

- [ ] Staging tests passed
- [ ] All sign-offs obtained
- [ ] Rollback plan reviewed
- [ ] On-call engineer notified
- [ ] Deployment window confirmed
- [ ] Monitoring dashboards ready

### Production Artifact Sync

```powershell
# Sync to production (use secure transfer)
robocopy artifacts/cf prod-server:/app/artifacts/cf /E /Z

# Sync data files
robocopy data/processed prod-server:/app/data/processed /E /Z

# Sync configuration
robocopy config prod-server:/app/config /E /Z
```

- [ ] Model registry synced
- [ ] Model files synced
- [ ] Data artifacts synced
- [ ] PhoBERT embeddings synced
- [ ] Configuration synced

### Production Configuration

- [ ] Review `config/serving_config.yaml` for production
- [ ] Verify all paths correct
- [ ] Verify cache sizes appropriate for production load
- [ ] Logging level set to INFO (not DEBUG)

```yaml
# Production configuration
cache:
  user_profile:
    max_size: 50000
  item_similarity:
    max_size: 5000
    
logging:
  level: "INFO"
```

### Deployment Steps

#### Step 1: Pre-flight

```powershell
# On production server
ssh prod-server
cd /app

# Verify artifacts
ls -la artifacts/cf/registry.json
ls -la data/processed/user_item_mappings.json
ls -la data/processed/content_based_embeddings/product_embeddings.pt
```

- [ ] All required files present

#### Step 2: Stop Current Service (if upgrading)

```powershell
# Graceful shutdown
kill -SIGTERM $(pgrep -f "uvicorn.*api:app")

# Wait for graceful shutdown
sleep 10

# Verify stopped
pgrep -f "uvicorn.*api:app"  # Should return nothing
```

- [ ] Previous service stopped gracefully

#### Step 3: Start New Service

```powershell
# Start service
uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 4

# Alternative with gunicorn
gunicorn service.api:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --daemon
```

- [ ] Service started
- [ ] No startup errors in logs

#### Step 4: Verify Startup

```powershell
# Health check
curl http://localhost:8000/health

# Check model loaded
curl http://localhost:8000/model_info

# Check cache warmup
curl http://localhost:8000/cache_stats
```

- [ ] Health check passes
- [ ] Correct model loaded
- [ ] Cache warmed up

### Production Smoke Tests

```powershell
# Run smoke tests
python scripts/smoke_test.py --host localhost --port 8000
```

- [ ] Health endpoint responds
- [ ] Recommendation for trainable user works
- [ ] Recommendation for cold-start user works
- [ ] Latency within targets

### Post-Deployment Monitoring

**First 15 minutes**:
- [ ] Error rate normal (<1%)
- [ ] Latency normal (P99 < 200ms)
- [ ] No memory spikes
- [ ] Cache hit rate improving

**First hour**:
- [ ] Sustained normal operation
- [ ] No error spikes
- [ ] Resource usage stable
- [ ] User feedback (if available)

### Rollback Trigger Conditions

Initiate rollback if ANY of the following occur:
- [ ] Error rate > 5% for 5 minutes
- [ ] P99 latency > 500ms for 5 minutes
- [ ] Service unavailable > 2 minutes
- [ ] Critical bug discovered

### Rollback Procedure

If rollback needed:

```powershell
# 1. Stop new service
kill -SIGTERM $(pgrep -f "uvicorn.*api:app")

# 2. Restore previous artifacts
cp -r artifacts/cf.backup/* artifacts/cf/

# 3. Start previous version
# (restore previous code if needed)
uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 4

# 4. Verify rollback
curl http://localhost:8000/health
```

- [ ] Rollback executed (if needed)
- [ ] Service restored to previous state

---

## Post-Release (T+1 day)

### Validation

- [ ] 24-hour operation stable
- [ ] No critical issues reported
- [ ] Metrics within expected ranges
- [ ] User feedback positive (if available)

### Documentation Update

- [ ] Deployment notes added
- [ ] Any issues documented
- [ ] Lessons learned recorded

### Cleanup

- [ ] Remove old backups (after 7 days)
- [ ] Archive deployment logs
- [ ] Close deployment ticket

---

## Appendix: Smoke Test Script

Create `scripts/smoke_test.py`:

```python
"""
Smoke Test Script for Production Deployment.

Usage:
    python scripts/smoke_test.py --host localhost --port 8000
"""

import requests
import sys
import argparse


def run_smoke_tests(base_url: str) -> bool:
    """Run smoke tests against the API."""
    tests_passed = 0
    tests_failed = 0
    
    tests = [
        ("Health Check", "GET", "/health", None),
        ("Model Info", "GET", "/model_info", None),
        ("Cache Stats", "GET", "/cache_stats", None),
        ("Stats", "GET", "/stats", None),
        ("Recommend (trainable)", "POST", "/recommend", {"user_id": 1, "topk": 10}),
        ("Recommend (cold-start)", "POST", "/recommend", {"user_id": 999999, "topk": 10}),
        ("Batch Recommend", "POST", "/batch_recommend", {"user_ids": [1, 2, 3], "topk": 5}),
    ]
    
    for name, method, endpoint, payload in tests:
        try:
            url = f"{base_url}{endpoint}"
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                print(f"✓ {name}: PASSED")
                tests_passed += 1
            else:
                print(f"✗ {name}: FAILED (status {response.status_code})")
                tests_failed += 1
        except Exception as e:
            print(f"✗ {name}: ERROR ({e})")
            tests_failed += 1
    
    print(f"\n{'='*40}")
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print(f"{'='*40}")
    
    return tests_failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    print(f"Running smoke tests against {base_url}\n")
    
    success = run_smoke_tests(base_url)
    sys.exit(0 if success else 1)
```

---

## Sign-off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Release Manager | | | |
| QA Lead | | | |
| Dev Lead | | | |
| Ops Lead | | | |

---

**Release Status**: ⬜ Not Started | ⬜ In Progress | ⬜ Completed | ⬜ Rolled Back
