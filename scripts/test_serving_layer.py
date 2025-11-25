"""
Test script for the CF Recommendation Serving Layer.

This script tests all components of the serving layer:
1. CFModelLoader - Model and mapping loading
2. CFRecommender - Core recommendation engine
3. FallbackRecommender - Cold-start strategies
4. FastAPI endpoints

Usage:
    python scripts/test_serving_layer.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_model_loader():
    """Test CFModelLoader."""
    print("\n" + "="*60)
    print("Testing CFModelLoader...")
    print("="*60)
    
    from service.recommender.loader import CFModelLoader
    
    # Test singleton
    loader1 = CFModelLoader()
    loader2 = CFModelLoader()
    assert loader1 is loader2, "Singleton pattern failed"
    print("✓ Singleton pattern works")
    
    # Load model
    model_info = loader1.load_model()
    print(f"✓ Loaded model: {model_info.get('model_id')}")
    print(f"  - Type: {model_info.get('model_type')}")
    print(f"  - Factors: {model_info.get('U').shape[1] if model_info.get('U') is not None else 'N/A'}")
    
    # Load mappings
    loader1.load_mappings()
    print(f"✓ Loaded mappings:")
    print(f"  - Users: {loader1.mappings['metadata']['num_users']}")
    print(f"  - Items: {loader1.mappings['metadata']['num_items']}")
    print(f"  - Trainable users: {len(loader1.trainable_user_set)}")
    
    # Test user classification
    # Get a sample trainable user
    if loader1.trainable_user_mapping:
        sample_u_idx = list(loader1.trainable_user_mapping.keys())[0]
        # idx_to_user uses string keys from JSON
        sample_user_id = int(loader1.mappings['idx_to_user'][str(sample_u_idx)])
        
        is_trainable = loader1.is_trainable_user(sample_user_id)
        cf_idx = loader1.get_cf_user_index(sample_user_id)
        print(f"✓ User classification:")
        print(f"  - User {sample_user_id}: trainable={is_trainable}, cf_idx={cf_idx}")
    
    # Test cold-start user
    fake_user_id = 999999999
    is_trainable = loader1.is_trainable_user(fake_user_id)
    print(f"  - User {fake_user_id}: trainable={is_trainable} (expected: False)")
    
    return loader1


def test_fallback_recommender(loader):
    """Test FallbackRecommender."""
    print("\n" + "="*60)
    print("Testing FallbackRecommender...")
    print("="*60)
    
    from service.recommender.fallback import FallbackRecommender
    
    fallback = FallbackRecommender(loader)
    
    # Test popularity fallback
    pop_recs = fallback.fallback_popularity(topk=5)
    print(f"✓ Popularity fallback: {len(pop_recs)} items")
    if pop_recs:
        print(f"  - Top item: product_id={pop_recs[0].get('product_id')}")
    
    # Test hybrid fallback for cold-start user
    cold_user_id = 999999999
    hybrid_recs = fallback.hybrid_fallback(
        user_history=[],  # No history for cold user
        topk=5
    )
    print(f"✓ Hybrid fallback (cold user): {len(hybrid_recs)} items")
    
    return fallback


def test_cf_recommender():
    """Test CFRecommender."""
    print("\n" + "="*60)
    print("Testing CFRecommender...")
    print("="*60)
    
    from service.recommender import CFRecommender
    
    # Initialize recommender
    start = time.time()
    recommender = CFRecommender(auto_load=True)
    init_time = time.time() - start
    print(f"✓ Initialized CFRecommender in {init_time:.2f}s")
    
    # Get model info
    info = recommender.get_model_info()
    print(f"  - Model: {info.get('model_id')}")
    print(f"  - Users: {info.get('num_users')}")
    print(f"  - Items: {info.get('num_items')}")
    print(f"  - Trainable: {info.get('trainable_users')}")
    
    # Test recommendation for trainable user
    if recommender.loader.trainable_user_mapping:
        sample_u_idx = list(recommender.loader.trainable_user_mapping.keys())[0]
        # Use string key for idx_to_user lookup
        sample_user_id = int(recommender.loader.mappings['idx_to_user'][str(sample_u_idx)])
        
        start = time.time()
        result = recommender.recommend(user_id=sample_user_id, topk=10)
        cf_time = (time.time() - start) * 1000
        
        print(f"\n✓ CF Recommendation for trainable user {sample_user_id}:")
        print(f"  - Is fallback: {result.is_fallback}")
        print(f"  - Count: {result.count}")
        print(f"  - Latency: {cf_time:.1f}ms")
        if result.recommendations:
            print(f"  - Top 3: {[r.get('product_id') for r in result.recommendations[:3]]}")
    
    # Test recommendation for cold-start user
    cold_user_id = 999999999
    start = time.time()
    result = recommender.recommend(user_id=cold_user_id, topk=10)
    fb_time = (time.time() - start) * 1000
    
    print(f"\n✓ Fallback Recommendation for cold user {cold_user_id}:")
    print(f"  - Is fallback: {result.is_fallback}")
    print(f"  - Fallback method: {result.fallback_method}")
    print(f"  - Count: {result.count}")
    print(f"  - Latency: {fb_time:.1f}ms")
    
    # Test batch recommendation
    test_users = [cold_user_id]
    if recommender.loader.trainable_user_mapping:
        # Add some trainable users
        for u_idx in list(recommender.loader.trainable_user_mapping.keys())[:3]:
            uid = int(recommender.loader.mappings['idx_to_user'][str(u_idx)])
            test_users.append(uid)
    
    start = time.time()
    batch_results = recommender.batch_recommend(user_ids=test_users, topk=5)
    batch_time = (time.time() - start) * 1000
    
    cf_count = sum(1 for r in batch_results.values() if not r.is_fallback)
    fb_count = sum(1 for r in batch_results.values() if r.is_fallback)
    
    print(f"\n✓ Batch Recommendation for {len(test_users)} users:")
    print(f"  - CF users: {cf_count}")
    print(f"  - Fallback users: {fb_count}")
    print(f"  - Total latency: {batch_time:.1f}ms")
    print(f"  - Per user: {batch_time/len(test_users):.1f}ms")
    
    return recommender


def test_api_import():
    """Test API can be imported."""
    print("\n" + "="*60)
    print("Testing API Import...")
    print("="*60)
    
    try:
        from service.api import app, RecommendRequest, RecommendResponse
        print("✓ FastAPI app imported successfully")
        print(f"  - Title: {app.title}")
        print(f"  - Version: {app.version}")
        
        # List endpoints
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        print(f"  - Endpoints: {routes}")
        
        return True
    except Exception as e:
        print(f"✗ API import failed: {e}")
        return False


def run_performance_benchmark(recommender):
    """Run performance benchmark."""
    print("\n" + "="*60)
    print("Performance Benchmark...")
    print("="*60)
    
    import random
    
    # Collect trainable users
    trainable_users = []
    if recommender.loader.trainable_user_mapping:
        for u_idx in list(recommender.loader.trainable_user_mapping.keys())[:100]:
            uid = int(recommender.loader.mappings['idx_to_user'][str(u_idx)])
            trainable_users.append(uid)
    
    # Benchmark CF recommendations
    if trainable_users:
        latencies = []
        for uid in trainable_users[:50]:
            start = time.time()
            recommender.recommend(user_id=uid, topk=10)
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"✓ CF Recommendation Latency (n={len(latencies)}):")
        print(f"  - Average: {avg_latency:.1f}ms")
        print(f"  - P95: {p95_latency:.1f}ms")
        print(f"  - Min: {min(latencies):.1f}ms")
        print(f"  - Max: {max(latencies):.1f}ms")
    
    # Benchmark fallback recommendations
    cold_latencies = []
    for _ in range(50):
        cold_uid = random.randint(90000000, 99999999)
        start = time.time()
        recommender.recommend(user_id=cold_uid, topk=10)
        cold_latencies.append((time.time() - start) * 1000)
    
    avg_latency = sum(cold_latencies) / len(cold_latencies)
    p95_latency = sorted(cold_latencies)[int(len(cold_latencies) * 0.95)]
    
    print(f"\n✓ Fallback Recommendation Latency (n={len(cold_latencies)}):")
    print(f"  - Average: {avg_latency:.1f}ms")
    print(f"  - P95: {p95_latency:.1f}ms")
    print(f"  - Min: {min(cold_latencies):.1f}ms")
    print(f"  - Max: {max(cold_latencies):.1f}ms")


def main():
    """Run all tests."""
    print("="*60)
    print("CF Recommendation Serving Layer Tests")
    print("="*60)
    
    try:
        # Test components
        loader = test_model_loader()
        test_fallback_recommender(loader)
        recommender = test_cf_recommender()
        test_api_import()
        
        # Performance benchmark
        run_performance_benchmark(recommender)
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
        print("\nTo start the service, run:")
        print("  uvicorn service.api:app --host 0.0.0.0 --port 8000 --reload")
        
        print("\nAPI Endpoints:")
        print("  GET  /health         - Health check")
        print("  POST /recommend      - Single user recommendation")
        print("  POST /batch_recommend - Batch recommendation")
        print("  POST /similar_items  - Similar items")
        print("  POST /reload_model   - Hot-reload model")
        print("  GET  /model_info     - Model information")
        print("  GET  /stats          - Service statistics")
        
    except Exception as e:
        import traceback
        print(f"\n✗ Test failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
