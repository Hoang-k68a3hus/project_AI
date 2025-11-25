"""
Test script for Monitoring & Logging System (Task 06).

This script tests:
1. Training metrics database
2. Service metrics database  
3. Drift detection
4. Alerting system

Usage:
    python scripts/test_monitoring.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_logging_utils():
    """Test logging utilities."""
    print("\n" + "="*60)
    print("Testing Logging Utilities...")
    print("="*60)
    
    from recsys.cf.logging_utils import (
        setup_training_logger,
        setup_service_logger,
        TrainingMetricsDB,
        ServiceMetricsDB,
        generate_run_id,
        format_params,
        format_metrics
    )
    
    # Test logger setup
    run_id = generate_run_id('test')
    logger = setup_training_logger('test', run_id, console=False)
    logger.info("Test log message")
    print(f"✓ Training logger created: {run_id}")
    
    service_logger = setup_service_logger('test', console=False)
    service_logger.info("Test service log")
    print("✓ Service logger created")
    
    # Test format helpers
    params = {'factors': 64, 'reg': 0.01, 'alpha': 40}
    formatted = format_params(params)
    print(f"✓ format_params: {formatted}")
    
    metrics = {'recall@10': 0.1457, 'ndcg@10': 0.0892}
    formatted = format_metrics(metrics)
    print(f"✓ format_metrics: {formatted}")
    
    return True


def test_training_metrics_db():
    """Test training metrics database."""
    print("\n" + "="*60)
    print("Testing Training Metrics Database...")
    print("="*60)
    
    from recsys.cf.logging_utils import TrainingMetricsDB, generate_run_id
    
    db = TrainingMetricsDB()
    print("✓ Database initialized")
    
    # Create test run
    run_id = generate_run_id('test')
    params = {
        'factors': 64,
        'regularization': 0.01,
        'iterations': 15
    }
    
    # Log training start
    db.log_training_start(run_id, 'test', params, data_version='test_v1')
    print(f"✓ Logged training start: {run_id}")
    
    # Log iterations
    for i in range(1, 4):
        db.log_iteration(
            run_id=run_id,
            iteration=i,
            loss=1.0 / i,
            validation_recall=0.1 * i,
            validation_ndcg=0.08 * i,
            wall_time_seconds=1.5
        )
    print("✓ Logged 3 iterations")
    
    # Log completion
    metrics = {
        'recall@10': 0.15,
        'ndcg@10': 0.12,
        'coverage': 0.85
    }
    db.log_training_complete(
        run_id=run_id,
        metrics=metrics,
        artifacts_path='artifacts/cf/test',
        training_time_seconds=45.0
    )
    print("✓ Logged training completion")
    
    # Query run
    run = db.get_run(run_id)
    print(f"✓ Retrieved run: status={run['status']}, recall@10={run['recall_at_10']}")
    
    # Query iterations
    iterations = db.get_iteration_metrics(run_id)
    print(f"✓ Retrieved {len(iterations)} iteration records")
    
    # Get recent runs
    recent = db.get_recent_runs(limit=5)
    print(f"✓ Retrieved {len(recent)} recent runs")
    
    return True


def test_service_metrics_db():
    """Test service metrics database."""
    print("\n" + "="*60)
    print("Testing Service Metrics Database...")
    print("="*60)
    
    from recsys.cf.logging_utils import ServiceMetricsDB
    
    db = ServiceMetricsDB()
    print("✓ Database initialized")
    
    # Log some requests
    for i in range(10):
        db.log_request(
            user_id=1000 + i,
            topk=10,
            latency_ms=50 + i * 5,
            num_recommendations=10,
            fallback=(i % 3 == 0),
            model_id='test_model',
            fallback_method='popularity' if i % 3 == 0 else None
        )
    print("✓ Logged 10 test requests")
    
    # Aggregate health
    db.aggregate_health_metrics('test_model')
    print("✓ Aggregated health metrics")
    
    # Query stats
    stats = db.get_request_stats(minutes=60)
    print(f"✓ Request stats: {stats}")
    
    # Query health
    health = db.get_recent_health(minutes=60)
    print(f"✓ Retrieved {len(health)} health records")
    
    return True


def test_drift_detection():
    """Test drift detection."""
    print("\n" + "="*60)
    print("Testing Drift Detection...")
    print("="*60)
    
    from recsys.cf.drift_detection import (
        detect_rating_drift,
        detect_popularity_shift,
        detect_user_activity_drift,
        detect_embedding_drift,
        check_embedding_freshness,
        should_retrain
    )
    
    # Test rating drift - no drift case
    hist_ratings = np.array([5, 5, 5, 4, 5, 4, 5, 5, 4, 5])
    new_ratings = np.array([5, 4, 5, 5, 4, 5, 5, 4, 5, 5])
    
    result = detect_rating_drift(hist_ratings, new_ratings)
    print(f"✓ Rating drift (similar): drift={result['drift_detected']}, p={result['p_value']:.4f}")
    
    # Test rating drift - drift case
    new_ratings_drift = np.array([3, 2, 3, 4, 2, 3, 2, 3, 4, 3])
    result = detect_rating_drift(hist_ratings, new_ratings_drift)
    print(f"✓ Rating drift (different): drift={result['drift_detected']}, p={result['p_value']:.4f}")
    
    # Test popularity shift
    old_pop = np.array([100, 80, 60, 40, 20])
    new_pop = np.array([95, 85, 55, 45, 22])  # Similar ranking
    
    result = detect_popularity_shift(old_pop, new_pop)
    print(f"✓ Popularity shift (stable): shift={result['shift_detected']}, corr={result['correlation']:.3f}")
    
    # Significant shift
    new_pop_shift = np.array([20, 100, 40, 80, 60])  # Changed ranking
    result = detect_popularity_shift(old_pop, new_pop_shift)
    print(f"✓ Popularity shift (changed): shift={result['shift_detected']}, corr={result['correlation']:.3f}")
    
    # Test embedding drift
    old_emb = np.random.randn(100, 64)
    new_emb = old_emb + np.random.randn(100, 64) * 0.01  # Small noise
    
    result = detect_embedding_drift(old_emb, new_emb)
    print(f"✓ Embedding drift (small): drift={result['drift_detected']}, sim={result['mean_similarity']:.4f}")
    
    # Test embedding freshness
    emb_path = "data/processed/content_based_embeddings/product_embeddings.pt"
    if Path(emb_path).exists():
        result = check_embedding_freshness(emb_path)
        print(f"✓ Embedding freshness: exists={result['exists']}, age={result.get('age_days', 'N/A')} days")
    else:
        print("✓ Embedding freshness: file not found (expected in test)")
    
    # Test retrain decision
    result = should_retrain(data_age_days=45)
    print(f"✓ Should retrain (old data): {result['should_retrain']}, reasons: {result['reasons']}")
    
    return True


def test_alerting():
    """Test alerting system."""
    print("\n" + "="*60)
    print("Testing Alerting System...")
    print("="*60)
    
    from recsys.cf.alerting import (
        AlertManager,
        send_alert,
        alert_high_latency,
        alert_data_drift
    )
    
    # Create alert manager
    mgr = AlertManager()
    print("✓ AlertManager created")
    
    # Send test alert (log only)
    result = mgr.send_alert(
        subject="Test Alert",
        message="This is a test alert message",
        severity="info"
    )
    print(f"✓ Test alert sent: {result}")
    
    # Check alert log file
    log_path = Path("logs/service/alerts.log")
    if log_path.exists():
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"✓ Alert logged: {lines[-1][:80]}...")
    
    # Test alert conditions
    mgr.check_alert_conditions({
        'avg_latency_ms': 100,  # Below threshold
        'error_rate': 0.01      # Below threshold
    })
    print("✓ Alert conditions checked (no triggers expected)")
    
    return True


def test_integration():
    """Test integration of all monitoring components."""
    print("\n" + "="*60)
    print("Testing Integration...")
    print("="*60)
    
    # Import all modules
    from recsys.cf.logging_utils import (
        TrainingMetricsDB,
        ServiceMetricsDB,
        setup_training_logger,
        generate_run_id
    )
    from recsys.cf.drift_detection import run_drift_detection
    from recsys.cf.alerting import AlertManager
    
    # Simulate a training workflow
    run_id = generate_run_id('integration_test')
    logger = setup_training_logger('integration_test', run_id, console=False)
    db = TrainingMetricsDB()
    
    # Log training
    logger.info(f"Training started | run_id={run_id}")
    db.log_training_start(run_id, 'als', {'factors': 64})
    
    for i in range(1, 6):
        db.log_iteration(run_id, i, loss=1.0/i, validation_recall=0.1*i)
        logger.info(f"Iteration {i}/5 | loss={1.0/i:.4f}")
    
    db.log_training_complete(run_id, {'recall@10': 0.15, 'ndcg@10': 0.12}, training_time_seconds=30)
    logger.info("Training completed")
    
    print(f"✓ Training workflow simulated: {run_id}")
    
    # Verify
    run = db.get_run(run_id)
    assert run['status'] == 'completed'
    assert run['recall_at_10'] == 0.15
    print("✓ Training run verified in database")
    
    return True


def main():
    """Run all monitoring tests."""
    print("="*60)
    print("Monitoring & Logging System Tests (Task 06)")
    print("="*60)
    
    tests = [
        ("Logging Utilities", test_logging_utils),
        ("Training Metrics DB", test_training_metrics_db),
        ("Service Metrics DB", test_service_metrics_db),
        ("Drift Detection", test_drift_detection),
        ("Alerting System", test_alerting),
        ("Integration", test_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            import traceback
            print(f"\n✗ {name} failed: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ All monitoring tests passed!")
        print("\nTo view the dashboard, run:")
        print("  streamlit run service/dashboard.py")
        print("\nLog files created:")
        print("  - logs/cf/test.log")
        print("  - logs/service/test.log")
        print("  - logs/service/alerts.log")
        print("  - logs/training_metrics.db")
        print("  - logs/service_metrics.db")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
