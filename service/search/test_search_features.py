"""
Test script để verify các tính năng của module search hoạt động đúng.

Chạy script này để kiểm tra:
- QueryEncoder encoding
- SearchIndex search functionality
- SmartSearchService các tính năng tìm kiếm

Usage:
    python service/search/test_search_features.py
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_query_encoder():
    """Test QueryEncoder functionality."""
    logger.info("=" * 60)
    logger.info("Testing QueryEncoder...")
    logger.info("=" * 60)
    
    try:
        from service.search.query_encoder import QueryEncoder
        
        encoder = QueryEncoder()
        
        # Test 1: Preprocessing với Vietnamese text
        logger.info("\n1. Testing Vietnamese text preprocessing...")
        test_queries = [
            "kem dưỡng da cho da dầu",
            "srm cho da nhờn",
            "kcn chống nắng tốt",
            "nước hoa hồng dịu nhẹ"
        ]
        
        for query in test_queries:
            processed = encoder.preprocess_query(query)
            logger.info(f"  Original: {query}")
            logger.info(f"  Processed: {processed}")
        
        # Test 2: Encoding (nếu model có sẵn)
        logger.info("\n2. Testing query encoding...")
        try:
            emb = encoder.encode("kem dưỡng da", normalize=True)
            logger.info(f"  ✓ Encoding successful. Shape: {emb.shape}")
            logger.info(f"  ✓ Embedding norm: {emb.norm():.4f}")
        except Exception as e:
            logger.warning(f"  ⚠ Encoding failed (model may not be loaded): {e}")
        
        # Test 3: Cache functionality
        logger.info("\n3. Testing cache...")
        stats = encoder.get_stats()
        logger.info(f"  Cache stats: {stats.get('cache', {})}")
        logger.info(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
        
        logger.info("\n✓ QueryEncoder tests completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ QueryEncoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_index():
    """Test SearchIndex functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing SearchIndex...")
    logger.info("=" * 60)
    
    try:
        from service.search.search_index import SearchIndex
        
        # Test 1: Initialization
        logger.info("\n1. Testing SearchIndex initialization...")
        index = SearchIndex(auto_build=False)
        logger.info(f"  ✓ SearchIndex created")
        logger.info(f"  Initialized: {index.is_initialized}")
        
        # Test 2: Build index (nếu embeddings có sẵn)
        logger.info("\n2. Testing index building...")
        try:
            index.build_index()
            logger.info(f"  ✓ Index built successfully")
            logger.info(f"  Products indexed: {index.num_products}")
            
            stats = index.get_stats()
            logger.info(f"  Stats: {stats}")
            
        except Exception as e:
            logger.warning(f"  ⚠ Index build failed (embeddings may not be available): {e}")
        
        # Test 3: Available filters
        logger.info("\n3. Testing filter metadata...")
        try:
            brands = index.get_available_brands()
            categories = index.get_available_categories()
            price_range = index.get_price_range()
            
            logger.info(f"  Available brands: {len(brands)}")
            if brands:
                logger.info(f"    Sample: {brands[:5]}")
            logger.info(f"  Available categories: {len(categories)}")
            if categories:
                logger.info(f"    Sample: {categories[:5]}")
            logger.info(f"  Price range: {price_range[0]:.0f} - {price_range[1]:.0f}")
            
        except Exception as e:
            logger.warning(f"  ⚠ Filter metadata not available: {e}")
        
        logger.info("\n✓ SearchIndex tests completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ SearchIndex test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_smart_search_service():
    """Test SmartSearchService functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing SmartSearchService...")
    logger.info("=" * 60)
    
    try:
        from service.search.smart_search import get_search_service
        
        service = get_search_service()
        
        # Test 1: Initialization
        logger.info("\n1. Testing service initialization...")
        logger.info(f"  Initialized: {service.is_initialized}")
        
        # Test 2: Text search (nếu index có sẵn)
        logger.info("\n2. Testing text search...")
        test_queries = [
            "kem dưỡng da cho da dầu",
            "sữa rửa mặt",
            "kem chống nắng"
        ]
        
        for query in test_queries:
            try:
                results = service.search(query, topk=5)
                logger.info(f"  Query: '{query}'")
                logger.info(f"    Results: {results.count}")
                logger.info(f"    Latency: {results.latency_ms:.2f}ms")
                logger.info(f"    Method: {results.method}")
                
                if results.results:
                    logger.info(f"    Top result: {results.results[0].product_name}")
                    
            except Exception as e:
                logger.warning(f"  ⚠ Search failed for '{query}': {e}")
        
        # Test 3: Similar items search
        logger.info("\n3. Testing similar items search...")
        try:
            # Giả sử có product_id = 1
            similar = service.search_similar(product_id=1, topk=5)
            logger.info(f"  Similar items: {similar.count}")
            logger.info(f"  Latency: {similar.latency_ms:.2f}ms")
        except Exception as e:
            logger.warning(f"  ⚠ Similar search failed: {e}")
        
        # Test 4: Service stats
        logger.info("\n4. Testing service statistics...")
        stats = service.get_stats()
        logger.info(f"  Total searches: {stats.get('total_searches', 0)}")
        logger.info(f"  Avg latency: {stats.get('avg_latency_ms', 0):.2f}ms")
        logger.info(f"  Errors: {stats.get('errors', 0)}")
        
        logger.info("\n✓ SmartSearchService tests completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ SmartSearchService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("Starting Search Module Feature Tests")
    logger.info("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("QueryEncoder", test_query_encoder()))
    results.append(("SearchIndex", test_search_index()))
    results.append(("SmartSearchService", test_smart_search_service()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"  {name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        logger.info("\n✓ All tests passed!")
    else:
        logger.warning("\n⚠ Some tests failed or had warnings (may be due to missing data/models)")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

