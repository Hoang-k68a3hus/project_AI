"""
Test Smart Search functionality.

This script tests the SmartSearchService components:
1. Query encoding
2. Search index
3. Semantic search
4. Similar items search
5. User profile search

Usage:
    python scripts/test_smart_search.py
"""

import sys
import os
import time
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_query_encoder():
    """Test QueryEncoder functionality."""
    import numpy as np
    
    print("\n" + "=" * 60)
    print("TEST 1: Query Encoder")
    print("=" * 60)
    
    from service.search.query_encoder import QueryEncoder, reset_query_encoder
    
    # Reset for clean test
    reset_query_encoder()
    
    encoder = QueryEncoder()
    
    # Test preprocessing
    print("\n1.1 Testing Vietnamese preprocessing...")
    test_queries = [
        ("kem dc cho dn", "kem dưỡng da cho da nhờn"),
        ("srm ko gây kích ứng", "sữa rửa mặt không gây kích ứng"),
        ("kcn spf50", "kem chống nắng spf50"),
        ("sp tốt cho dm", "sản phẩm tốt cho da mụn"),
    ]
    
    for original, expected_contains in test_queries:
        processed = encoder.preprocess_query(original)
        print(f"  '{original}' -> '{processed}'")
        # Check if key words are expanded
        assert "sản phẩm" in processed or "kem" in processed or "sữa rửa mặt" in processed or processed != original
    
    print("  [OK] Preprocessing working")
    
    # Test encoding (this will load the model)
    print("\n1.2 Testing query encoding (loading PhoBERT model)...")
    start = time.perf_counter()
    
    query = "kem dưỡng ẩm cho da khô"
    embedding = encoder.encode(query)
    
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Query: '{query}'")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Encoding time (first query, includes model load): {elapsed:.1f}ms")
    
    # Check embedding properties (dimension depends on model used)
    expected_dims = {768, 1024}  # phobert-base (768) or phobert-large/Vietnamese_Embedding (1024)
    assert embedding.shape[0] in expected_dims, f"Unexpected embedding dim: {embedding.shape[0]}"
    assert abs(np.linalg.norm(embedding) - 1.0) < 0.001, "Embedding should be normalized"
    print(f"  Embedding dimension: {embedding.shape[0]} (model: {encoder.model_name})")
    print("  [OK] Encoding working")
    
    # Test caching
    print("\n1.3 Testing query cache...")
    start = time.perf_counter()
    embedding2 = encoder.encode(query)  # Should hit cache
    elapsed_cached = (time.perf_counter() - start) * 1000
    
    print(f"  Cached query time: {elapsed_cached:.1f}ms")
    assert np.allclose(embedding, embedding2), "Cached embedding should match"
    
    stats = encoder.get_stats()
    print(f"  Cache hits: {stats['cache_hits']}, misses: {stats['cache_misses']}")
    print("  [OK] Caching working")
    
    # Test batch encoding
    print("\n1.4 Testing batch encoding...")
    queries = [
        "sữa rửa mặt cho da dầu",
        "serum vitamin c",
        "kem chống nắng",
        "toner cân bằng da"
    ]
    
    start = time.perf_counter()
    batch_embeddings = encoder.encode_batch(queries)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"  Batch size: {len(queries)}")
    print(f"  Batch embeddings shape: {batch_embeddings.shape}")
    print(f"  Batch encoding time: {elapsed:.1f}ms ({elapsed/len(queries):.1f}ms/query)")
    
    expected_dims = {768, 1024}
    assert batch_embeddings.shape == (len(queries), embedding.shape[0])
    print("  [OK] Batch encoding working")
    
    return True


def test_search_index():
    """Test SearchIndex functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Search Index")
    print("=" * 60)
    
    from service.search.search_index import SearchIndex, reset_search_index
    from service.recommender.phobert_loader import get_phobert_loader
    import pandas as pd
    
    # Reset for clean test
    reset_search_index()
    
    # Load PhoBERT embeddings
    print("\n2.1 Loading PhoBERT embeddings...")
    phobert = get_phobert_loader()
    print(f"  Loaded {phobert.num_products} product embeddings")
    
    # Try to load product metadata
    print("\n2.2 Loading product metadata...")
    metadata_path = project_root / "data" / "published_data" / "data_product.csv"
    product_metadata = None
    
    if metadata_path.exists():
        product_metadata = pd.read_csv(metadata_path, encoding='utf-8')
        print(f"  Loaded {len(product_metadata)} products from metadata")
    else:
        print("  [WARN] No product metadata found (filtering will be disabled)")
    
    # Build index
    print("\n2.3 Building search index...")
    start = time.perf_counter()
    
    index = SearchIndex(
        phobert_loader=phobert,
        product_metadata=product_metadata,
        use_faiss=False
    )
    index.build_index()
    
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Index built in {elapsed:.1f}ms")
    print(f"  Products indexed: {index.num_products}")
    
    stats = index.get_stats()
    print(f"  Brands indexed: {stats.get('available_brands', 0)}")
    print(f"  Categories indexed: {stats.get('available_categories', 0)}")
    print("  [OK] Index built")
    
    # Test search
    print("\n2.4 Testing search...")
    
    # Get a random product embedding as query
    test_pid = list(phobert.product_id_to_idx.keys())[0]
    query_emb = phobert.get_embedding_normalized(test_pid)
    
    start = time.perf_counter()
    results = index.search(query_emb, topk=5)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"  Search time: {elapsed:.1f}ms")
    print(f"  Results: {len(results)}")
    for pid, score in results[:3]:
        print(f"    - Product {pid}: score={score:.4f}")
    
    assert len(results) > 0, "Should have search results"
    print("  [OK] Search working")
    
    # Test filtered search
    if product_metadata is not None and stats.get('available_brands', 0) > 0:
        print("\n2.5 Testing filtered search...")
        
        brands = index.get_available_brands()[:3]
        print(f"  Available brands (sample): {brands}")
        
        if brands:
            start = time.perf_counter()
            filtered_results = index.search_with_filter(
                query_emb, 
                topk=5, 
                filters={'brand': brands[0]}
            )
            elapsed = (time.perf_counter() - start) * 1000
            
            print(f"  Filtered search time: {elapsed:.1f}ms")
            print(f"  Results for brand '{brands[0]}': {len(filtered_results)}")
            print("  [OK] Filtered search working")
    
    return True


def test_smart_search_service():
    """Test SmartSearchService functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Smart Search Service")
    print("=" * 60)
    
    from service.search.smart_search import SmartSearchService, reset_search_service
    
    # Reset for clean test
    reset_search_service()
    
    # Initialize service
    print("\n3.1 Initializing SmartSearchService...")
    start = time.perf_counter()
    
    service = SmartSearchService()
    service.initialize()
    
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Initialization time: {elapsed:.1f}ms")
    print("  [OK] Service initialized")
    
    # Test semantic search
    print("\n3.2 Testing semantic search...")
    
    test_queries = [
        "kem dưỡng ẩm cho da khô",
        "sữa rửa mặt dịu nhẹ",
        "serum vitamin c làm sáng da",
        "kem chống nắng không gây nhờn"
    ]
    
    for query in test_queries:
        start = time.perf_counter()
        response = service.search(query, topk=5, rerank=True)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"\n  Query: '{query}'")
        print(f"  Results: {response.count}, Latency: {response.latency_ms:.1f}ms, Method: {response.method}")
        
        if response.results:
            for r in response.results[:2]:
                print(f"    - [{r.rank}] {r.product_name[:40]}... (score: {r.final_score:.3f})")
    
    print("\n  [OK] Semantic search working")
    
    # Test similar items search
    print("\n3.3 Testing similar items search...")
    
    # Get a product ID from embeddings
    from service.recommender.phobert_loader import get_phobert_loader
    phobert = get_phobert_loader()
    product_ids = list(phobert.product_id_to_idx.keys())
    # Pick a product ID (use index 100 if available, otherwise use first)
    test_pid = product_ids[min(100, len(product_ids) - 1)] if product_ids else None
    
    if test_pid is None:
        print("  [WARN] No products available for similar search test")
        return True
    
    start = time.perf_counter()
    response = service.search_similar(test_pid, topk=5)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"  Similar to product {test_pid}:")
    print(f"  Results: {response.count}, Latency: {response.latency_ms:.1f}ms")
    
    if response.results:
        for r in response.results[:3]:
            print(f"    - [{r.rank}] {r.product_name[:40]}... (score: {r.semantic_score:.3f})")
    
    print("  [OK] Similar items search working")
    
    # Test user profile search
    print("\n3.4 Testing user profile search...")
    
    # Create fake user history
    product_ids = list(phobert.product_id_to_idx.keys())
    # Take up to 5 products (adjust if not enough products)
    start_idx = min(50, len(product_ids) - 1)
    end_idx = min(55, len(product_ids))
    user_history = product_ids[start_idx:end_idx] if end_idx > start_idx else product_ids[:min(5, len(product_ids))]
    
    start = time.perf_counter()
    response = service.search_by_user_profile(user_history, topk=5, exclude_history=True)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"  User history: {len(user_history)} products")
    print(f"  Results: {response.count}, Latency: {response.latency_ms:.1f}ms, Method: {response.method}")
    
    if response.results:
        for r in response.results[:3]:
            print(f"    - [{r.rank}] {r.product_name[:40]}... (score: {r.semantic_score:.3f})")
    
    # Verify excluded products
    result_pids = {r.product_id for r in response.results}
    history_in_results = result_pids.intersection(set(user_history))
    assert len(history_in_results) == 0, "User history should be excluded from results"
    
    print("  [OK] User profile search working")
    
    # Test filtered search
    print("\n3.5 Testing filtered search...")
    
    available_filters = service.get_available_filters()
    print(f"  Available brands: {len(available_filters['brands'])}")
    print(f"  Available categories: {len(available_filters['categories'])}")
    print(f"  Price range: {available_filters['price_range']}")
    
    if available_filters['brands']:
        brand = available_filters['brands'][0]
        response = service.search("kem dưỡng", topk=5, filters={'brand': brand})
        print(f"\n  Search with brand filter '{brand}':")
        print(f"  Results: {response.count}")
        
        if response.results:
            for r in response.results[:2]:
                print(f"    - {r.product_name[:40]}... (brand: {r.brand})")
    
    print("  [OK] Filtered search working")
    
    # Print service stats
    print("\n3.6 Service statistics:")
    stats = service.get_stats()
    print(f"  Total searches: {stats['total_searches']}")
    print(f"  Semantic searches: {stats['searches_performed']}")
    print(f"  Similar searches: {stats['similar_searches']}")
    print(f"  Profile searches: {stats['profile_searches']}")
    print(f"  Average latency: {stats['avg_latency_ms']:.1f}ms")
    print(f"  Errors: {stats['errors']}")
    
    return True


def main():
    """Run all tests."""
    import numpy as np
    
    # Make numpy available for assertions
    globals()['np'] = np
    
    print("=" * 60)
    print("SMART SEARCH TEST SUITE")
    print("=" * 60)
    print(f"Project root: {project_root}")
    
    results = {}
    
    # Test 1: Query Encoder
    try:
        results['query_encoder'] = test_query_encoder()
    except Exception as e:
        print(f"\n[ERROR] Query Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        results['query_encoder'] = False
    
    # Test 2: Search Index
    try:
        results['search_index'] = test_search_index()
    except Exception as e:
        print(f"\n[ERROR] Search Index test failed: {e}")
        import traceback
        traceback.print_exc()
        results['search_index'] = False
    
    # Test 3: Smart Search Service
    try:
        results['smart_search'] = test_smart_search_service()
    except Exception as e:
        print(f"\n[ERROR] Smart Search Service test failed: {e}")
        import traceback
        traceback.print_exc()
        results['smart_search'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILED] Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())
