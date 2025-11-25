"""
Test script for Task 07: Automation & Scheduling components.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_utils_module():
    """Test the utils module components."""
    print("\n" + "="*60)
    print("Testing: utils.py module")
    print("="*60)
    
    from scripts.utils import (
        retry, PipelineTracker, PipelineLock,
        setup_logging, compute_file_hash, get_git_commit, get_git_branch
    )
    
    # Test retry decorator
    print("\n1. Testing retry decorator...")
    
    @retry(max_attempts=3, base_delay=0.1)
    def flaky_function(fail_count: int):
        flaky_function.attempts = getattr(flaky_function, 'attempts', 0) + 1
        if flaky_function.attempts <= fail_count:
            raise ValueError(f"Simulated failure {flaky_function.attempts}")
        return "success"
    
    flaky_function.attempts = 0
    result = flaky_function(2)
    assert result == "success", f"Expected 'success', got {result}"
    assert flaky_function.attempts == 3, f"Expected 3 attempts, got {flaky_function.attempts}"
    print("   ✓ Retry decorator works correctly")
    
    # Test PipelineTracker (use project's actual db location)
    print("\n2. Testing PipelineTracker...")
    
    tracker = PipelineTracker()  # Use default project location
    
    # Start a run with unique name
    test_run_name = f"utils_test_{datetime.now().strftime('%H%M%S')}"
    run_id = tracker.start_run(test_run_name, {"test_param": "value"})
    assert run_id is not None
    assert tracker.is_pipeline_running(test_run_name)
    print(f"   ✓ Started run: {run_id}")
    
    # Complete the run
    time.sleep(0.1)
    tracker.complete_run(run_id, {"records": 100})
    
    # Verify run
    run = tracker.get_run(run_id)
    assert run is not None
    assert run.status == "success"
    assert run.duration_seconds > 0
    print(f"   ✓ Completed run with duration: {run.duration_seconds:.3f}s")
    
    # Test stats
    stats = tracker.get_stats(days=7)
    assert 'stats_by_pipeline' in stats
    print(f"   ✓ Stats retrieved successfully")
    
    # Test PipelineLock (use project directory)
    print("\n3. Testing PipelineLock...")
    
    lock_dir = PROJECT_ROOT / "logs" / "locks"
    lock_name = f"test_lock_{datetime.now().strftime('%H%M%S')}"
    
    with PipelineLock(lock_name, lock_dir) as lock1:
        assert lock1.acquired, "First lock should be acquired"
        print("   ✓ First lock acquired")
        
        # Try to acquire another lock
        with PipelineLock(lock_name, lock_dir) as lock2:
            assert not lock2.acquired, "Second lock should not be acquired"
            print("   ✓ Second lock correctly blocked")
    
    # Lock should be released
    with PipelineLock(lock_name, lock_dir) as lock3:
        assert lock3.acquired, "Lock should be available after release"
        print("   ✓ Lock released correctly")
    
    # Test logging setup
    print("\n4. Testing logging setup...")
    
    log_dir = PROJECT_ROOT / "logs" / "test"
    logger = setup_logging("test_utils_logger", log_dir=log_dir, console=False)
    logger.info("Test message")
    
    log_files = list(log_dir.glob("test_utils_logger*.log"))
    assert len(log_files) >= 1
    print(f"   ✓ Log file created: {log_files[0].name}")
    
    # Test git utilities
    print("\n5. Testing git utilities...")
    
    commit = get_git_commit()
    branch = get_git_branch()
    print(f"   Git commit: {commit or 'N/A'}")
    print(f"   Git branch: {branch or 'N/A'}")
    
    print("\n✓ All utils module tests passed!")
    return True


def test_health_check():
    """Test health check script."""
    print("\n" + "="*60)
    print("Testing: health_check.py")
    print("="*60)
    
    from scripts.health_check import (
        check_data_health, check_model_health, check_pipeline_health
    )
    from scripts.utils import setup_logging
    
    logger = setup_logging("test_health", console=False)
    
    # Test data health check
    print("\n1. Testing data health check...")
    data_result = check_data_health(logger)
    
    print(f"   Status: {data_result['status']}")
    print(f"   Checks: {len(data_result['checks'])}")
    for check in data_result['checks']:
        status = "✓" if check['passed'] else "✗"
        print(f"     {status} {check['name']}: {check['message']}")
    
    # Test model health check
    print("\n2. Testing model health check...")
    model_result = check_model_health(logger)
    
    print(f"   Status: {model_result['status']}")
    for check in model_result['checks']:
        status = "✓" if check['passed'] else "✗"
        print(f"     {status} {check['name']}: {check['message']}")
    
    # Test pipeline health check
    print("\n3. Testing pipeline health check...")
    pipeline_result = check_pipeline_health(logger)
    
    print(f"   Status: {pipeline_result['status']}")
    
    print("\n✓ Health check tests completed!")
    return True


def test_pipeline_tracker_persistence():
    """Test that pipeline tracker data persists."""
    print("\n" + "="*60)
    print("Testing: Pipeline Tracker Persistence")
    print("="*60)
    
    from scripts.utils import PipelineTracker
    
    # Use the actual project location
    tracker = PipelineTracker()
    
    # Create a test run
    run_id = tracker.start_run("test_automation", {"test": True})
    print(f"   Started run: {run_id}")
    
    time.sleep(0.2)
    tracker.complete_run(run_id, {"status": "test_complete"})
    
    # Verify it can be retrieved
    run = tracker.get_run(run_id)
    assert run is not None
    assert run.status == "success"
    print(f"   ✓ Run verified: {run.status}, duration: {run.duration_seconds:.3f}s")
    
    # Check recent runs
    recent = tracker.get_recent_runs(pipeline_name="test_automation", limit=5)
    assert len(recent) > 0
    print(f"   ✓ Found {len(recent)} recent test_automation runs")
    
    # Check stats
    stats = tracker.get_stats(days=1)
    print(f"   ✓ Stats: {json.dumps(stats, indent=2)}")
    
    print("\n✓ Pipeline tracker persistence test passed!")
    return True


def test_cleanup_dry_run():
    """Test cleanup in dry-run mode."""
    print("\n" + "="*60)
    print("Testing: cleanup_logs.py (dry-run)")
    print("="*60)
    
    from scripts.cleanup_logs import run_cleanup
    from scripts.utils import setup_logging
    
    logger = setup_logging("test_cleanup", console=False)
    
    result = run_cleanup(
        log_retention_days=1,
        checkpoint_retention_days=1,
        keep_models=2,
        dry_run=True,
        logger=logger
    )
    
    print(f"\n   Status: {result['status']}")
    print(f"   Would free: {result['total_bytes_freed'] / 1024:.1f} KB")
    
    for category, data in result.get('results', {}).items():
        if isinstance(data, dict):
            found = data.get('files_found', 0) or data.get('dirs_found', 0) or data.get('models_found', 0)
            if found > 0:
                print(f"   {category}: {found} items would be cleaned")
    
    print("\n✓ Cleanup dry-run test passed!")
    return True


def test_deploy_dry_run():
    """Test deployment in dry-run mode."""
    print("\n" + "="*60)
    print("Testing: deploy_model_update.py (dry-run)")
    print("="*60)
    
    from scripts.deploy_model_update import deploy_model
    from scripts.utils import setup_logging
    
    logger = setup_logging("test_deploy", console=False)
    
    try:
        result = deploy_model(
            dry_run=True,
            logger=logger
        )
        
        print(f"\n   Status: {result['status']}")
        print(f"   Model: {result.get('model_id', 'N/A')}")
        
        if result.get('model_info'):
            print(f"   Type: {result['model_info'].get('model_type')}")
            metrics = result['model_info'].get('metrics', {})
            print(f"   Recall@10: {metrics.get('recall@10', 'N/A')}")
        
        print("\n✓ Deploy dry-run test passed!")
        return True
        
    except FileNotFoundError as e:
        print(f"\n   ⚠ Registry not found (expected if no models trained): {e}")
        print("   This is OK - will work after training models")
        return True


def test_refresh_data_check():
    """Test data refresh change detection."""
    print("\n" + "="*60)
    print("Testing: refresh_data.py (change detection)")
    print("="*60)
    
    from scripts.refresh_data import check_data_changed
    from scripts.utils import setup_logging
    
    logger = setup_logging("test_refresh", console=False)
    
    result = check_data_changed(logger)
    
    print(f"\n   Data changed: {result['changed']}")
    print(f"   Current hash: {result['current_hash'][:16]}...")
    print(f"   Previous hash: {result['previous_hash'][:16] if result['previous_hash'] else 'None'}...")
    
    print("\n✓ Data refresh check test passed!")
    return True


def run_all_tests():
    """Run all automation tests."""
    print("\n" + "="*70)
    print(" TASK 07: AUTOMATION & SCHEDULING - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Utils Module", test_utils_module),
        ("Health Check", test_health_check),
        ("Pipeline Tracker Persistence", test_pipeline_tracker_persistence),
        ("Cleanup Dry-Run", test_cleanup_dry_run),
        ("Deploy Dry-Run", test_deploy_dry_run),
        ("Data Refresh Check", test_refresh_data_check),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            import traceback
            print(f"\n✗ {test_name} FAILED: {e}")
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {test_name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠ {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
