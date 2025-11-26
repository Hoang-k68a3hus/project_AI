"""
FastAPI Recommendation Service.

This module provides REST API endpoints for the CF recommendation service.

Endpoints:
- GET /health: Health check
- POST /recommend: Single user recommendation
- POST /batch_recommend: Batch recommendation
- POST /similar_items: Similar items
- POST /reload_model: Hot-reload model
- POST /search: Semantic search for products
- POST /search/similar: Find similar products by product ID
- POST /search/profile: Search based on user profile

Usage:
    uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 4
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager
import logging
import time
import os
import sys
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from service.recommender import CFRecommender, get_loader
from service.recommender.cache import get_cache_manager, async_warmup

# Import Smart Search service (lazy initialization)
search_service = None


def get_search_service():
    """Lazy initialization of search service."""
    global search_service
    if search_service is None:
        try:
            from service.search import SmartSearchService
            search_service = SmartSearchService()
            search_service.initialize()
            logger.info("Smart Search service initialized")
        except Exception as e:
            logger.warning(f"Could not initialize search service: {e}")
    return search_service

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cf_service")


# ============================================================================
# Global State
# ============================================================================

recommender: Optional[CFRecommender] = None
cache_manager = None  # Cache manager instance
service_metrics_db = None  # Lazy initialization


def get_service_metrics_db():
    """Get or create service metrics database."""
    global service_metrics_db
    if service_metrics_db is None:
        try:
            from recsys.cf.logging_utils import ServiceMetricsDB
            service_metrics_db = ServiceMetricsDB()
        except Exception as e:
            logger.warning(f"Could not initialize metrics DB: {e}")
    return service_metrics_db


# ============================================================================
# Lifespan Handler
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global recommender, cache_manager
    
    logger.info("Starting CF Recommendation Service...")
    
    try:
        recommender = CFRecommender(auto_load=True)
        model_info = recommender.get_model_info()
        logger.info(f"Loaded model: {model_info.get('model_id')}")
        logger.info(
            f"Users: {model_info.get('num_users')}, "
            f"Items: {model_info.get('num_items')}, "
            f"Trainable: {model_info.get('trainable_users')}"
        )
        
        # Initialize cache manager
        cache_manager = get_cache_manager()
        
        # Initialize metrics DB
        get_service_metrics_db()
        logger.info("Service metrics database initialized")
        
        # Warm up caches for cold-start optimization (~91% traffic)
        logger.info("Warming up caches for cold-start path...")
        warmup_stats = await async_warmup(cache_manager)
        logger.info(
            f"Cache warmup complete: {warmup_stats.get('popular_items', 0)} popular items, "
            f"{warmup_stats.get('popular_similarities', 0)} similarities precomputed, "
            f"duration={warmup_stats.get('warmup_duration_ms', 0):.1f}ms"
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")
        raise
    
    # Start background health aggregation task
    aggregation_task = asyncio.create_task(periodic_health_aggregation())
    
    yield
    
    # Cleanup
    aggregation_task.cancel()
    logger.info("Shutting down CF Recommendation Service...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="CF Recommendation Service",
    description="Collaborative Filtering recommendation API for Vietnamese cosmetics",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Background Tasks
# ============================================================================

async def periodic_health_aggregation():
    """Periodically aggregate health metrics."""
    while True:
        try:
            await asyncio.sleep(60)  # Every minute
            db = get_service_metrics_db()
            if db and recommender:
                db.aggregate_health_metrics(recommender.model_id)
                logger.debug("Health metrics aggregated")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health aggregation error: {e}")


def log_request_metrics(
    user_id: int,
    topk: int,
    latency_ms: float,
    num_recommendations: int,
    fallback: bool,
    fallback_method: Optional[str] = None,
    rerank_enabled: bool = False,
    error: Optional[str] = None
):
    """Log request metrics to database (in background)."""
    try:
        db = get_service_metrics_db()
        if db:
            db.log_request(
                user_id=user_id,
                topk=topk,
                latency_ms=latency_ms,
                num_recommendations=num_recommendations,
                fallback=fallback,
                model_id=recommender.model_id if recommender else None,
                fallback_method=fallback_method,
                rerank_enabled=rerank_enabled,
                error=error
            )
    except Exception as e:
        logger.warning(f"Failed to log request metrics: {e}")


# ============================================================================
# Request/Response Models
# ============================================================================

class RecommendRequest(BaseModel):
    """Single user recommendation request."""
    user_id: int = Field(..., description="User ID")
    topk: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    exclude_seen: bool = Field(default=True, description="Exclude items user has interacted with")
    filter_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Attribute filters, e.g., {'brand': 'Innisfree'}"
    )
    rerank: bool = Field(default=False, description="Apply hybrid reranking")
    rerank_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Reranking weights: {cf, popularity, quality, content}"
    )


class RecommendResponse(BaseModel):
    """Recommendation response."""
    user_id: int
    recommendations: List[Dict[str, Any]]
    count: int
    is_fallback: bool
    fallback_method: Optional[str]
    latency_ms: float
    model_id: Optional[str]


class BatchRequest(BaseModel):
    """Batch recommendation request."""
    user_ids: List[int] = Field(..., description="List of user IDs")
    topk: int = Field(default=10, ge=1, le=100)
    exclude_seen: bool = Field(default=True)


class BatchResponse(BaseModel):
    """Batch recommendation response."""
    results: Dict[int, Dict[str, Any]]
    num_users: int
    total_latency_ms: float
    cf_users: int
    fallback_users: int


class SimilarItemsRequest(BaseModel):
    """Similar items request."""
    product_id: int = Field(..., description="Query product ID")
    topk: int = Field(default=10, ge=1, le=50)
    use_cf: bool = Field(default=True, description="Use CF embeddings (else PhoBERT)")


class SimilarItemsResponse(BaseModel):
    """Similar items response."""
    product_id: int
    similar_items: List[Dict[str, Any]]
    count: int
    method: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_id: Optional[str]
    model_type: Optional[str]
    num_users: int
    num_items: int
    trainable_users: int
    timestamp: str


class ReloadResponse(BaseModel):
    """Model reload response."""
    status: str
    previous_model_id: Optional[str]
    new_model_id: Optional[str]
    reloaded: bool


# ============================================================================
# Smart Search Request/Response Models
# ============================================================================

class SearchRequest(BaseModel):
    """Semantic search request."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query in Vietnamese")
    topk: int = Field(default=10, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Attribute filters: {brand, category, min_price, max_price}"
    )
    rerank: bool = Field(default=True, description="Apply hybrid reranking")


class SearchSimilarRequest(BaseModel):
    """Similar products search request."""
    product_id: int = Field(..., description="Product ID to find similar products")
    topk: int = Field(default=10, ge=1, le=50, description="Number of similar products")
    exclude_self: bool = Field(default=True, description="Exclude the query product from results")


class SearchByProfileRequest(BaseModel):
    """Search based on user profile/history."""
    product_history: List[int] = Field(..., min_length=1, description="List of product IDs user has interacted with")
    topk: int = Field(default=10, ge=1, le=100, description="Number of results")
    exclude_history: bool = Field(default=True, description="Exclude products in history from results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Attribute filters")


class SearchResultItem(BaseModel):
    """Single search result item."""
    rank: int
    product_id: int
    product_name: str
    brand: Optional[str]
    category: Optional[str]
    price: Optional[float]
    avg_rating: Optional[float]
    num_sold: Optional[int]
    semantic_score: float
    final_score: float


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    results: List[SearchResultItem]
    count: int
    method: str
    latency_ms: float
    available_filters: Optional[Dict[str, Any]] = None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check service health and model status.
    
    Returns:
        Health status with model information
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    model_info = recommender.get_model_info()
    
    return HealthResponse(
        status="healthy",
        model_id=model_info.get('model_id'),
        model_type=model_info.get('model_type'),
        num_users=model_info.get('num_users', 0),
        num_items=model_info.get('num_items', 0),
        trainable_users=model_info.get('trainable_users', 0),
        timestamp=datetime.now().isoformat()
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Get recommendations for a single user.
    
    Args:
        request: RecommendRequest with user_id, topk, filters, etc.
    
    Returns:
        RecommendResponse with recommendations
    
    Example:
        POST /recommend
        {
            "user_id": 12345,
            "topk": 10,
            "exclude_seen": true,
            "filter_params": {"brand": "Innisfree"}
        }
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        start_time = time.perf_counter()
        
        result = recommender.recommend(
            user_id=request.user_id,
            topk=request.topk,
            exclude_seen=request.exclude_seen,
            filter_params=request.filter_params
        )
        
        # Apply reranking if requested
        if request.rerank and result.recommendations:
            from service.recommender.rerank import rerank_with_signals
            
            result.recommendations = rerank_with_signals(
                recommendations=result.recommendations,
                user_id=request.user_id,
                weights=request.rerank_weights,
                score_range=recommender.score_range
            )
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Log request to console
        logger.info(
            f"user_id={request.user_id}, topk={request.topk}, "
            f"count={result.count}, fallback={result.is_fallback}, "
            f"latency={latency:.1f}ms"
        )
        
        # Log to metrics DB (background)
        log_request_metrics(
            user_id=request.user_id,
            topk=request.topk,
            latency_ms=latency,
            num_recommendations=result.count,
            fallback=result.is_fallback,
            fallback_method=result.fallback_method,
            rerank_enabled=request.rerank
        )
        
        return RecommendResponse(
            user_id=result.user_id,
            recommendations=result.recommendations,
            count=result.count,
            is_fallback=result.is_fallback,
            fallback_method=result.fallback_method,
            latency_ms=latency,
            model_id=result.model_id
        )
    
    except Exception as e:
        logger.error(f"Recommendation error for user {request.user_id}: {e}")
        # Log error to metrics
        log_request_metrics(
            user_id=request.user_id,
            topk=request.topk,
            latency_ms=0,
            num_recommendations=0,
            fallback=False,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_recommend", response_model=BatchResponse)
async def batch_recommend(request: BatchRequest):
    """
    Get recommendations for multiple users.
    
    Args:
        request: BatchRequest with list of user_ids
    
    Returns:
        BatchResponse with results for all users
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        start_time = time.perf_counter()
        
        results = recommender.batch_recommend(
            user_ids=request.user_ids,
            topk=request.topk,
            exclude_seen=request.exclude_seen
        )
        
        total_latency = (time.perf_counter() - start_time) * 1000
        
        # Convert results to dict format
        results_dict = {}
        cf_count = 0
        fallback_count = 0
        
        for uid, result in results.items():
            results_dict[uid] = {
                'recommendations': result.recommendations,
                'count': result.count,
                'is_fallback': result.is_fallback,
                'fallback_method': result.fallback_method,
            }
            
            if result.is_fallback:
                fallback_count += 1
            else:
                cf_count += 1
        
        logger.info(
            f"Batch recommendation: {len(request.user_ids)} users, "
            f"cf={cf_count}, fallback={fallback_count}, "
            f"latency={total_latency:.1f}ms"
        )
        
        return BatchResponse(
            results=results_dict,
            num_users=len(request.user_ids),
            total_latency_ms=total_latency,
            cf_users=cf_count,
            fallback_users=fallback_count
        )
    
    except Exception as e:
        logger.error(f"Batch recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar_items", response_model=SimilarItemsResponse)
async def similar_items(request: SimilarItemsRequest):
    """
    Find similar items to a given product.
    
    Args:
        request: SimilarItemsRequest with product_id
    
    Returns:
        SimilarItemsResponse with similar items
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        similar = recommender.similar_items(
            product_id=request.product_id,
            topk=request.topk,
            use_cf=request.use_cf
        )
        
        method = "cf_embeddings" if request.use_cf else "phobert"
        
        return SimilarItemsResponse(
            product_id=request.product_id,
            similar_items=similar,
            count=len(similar),
            method=method
        )
    
    except Exception as e:
        logger.error(f"Similar items error for product {request.product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload_model", response_model=ReloadResponse)
async def reload_model():
    """
    Hot-reload model from registry.
    
    Checks if a new best model is available and reloads if so.
    
    Returns:
        ReloadResponse with reload status
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        previous_model_id = recommender.model_id
        reloaded = recommender.reload_model()
        new_model_id = recommender.model_id
        
        status = "reloaded" if reloaded else "no_update"
        
        logger.info(
            f"Model reload: {status}, "
            f"previous={previous_model_id}, new={new_model_id}"
        )
        
        return ReloadResponse(
            status=status,
            previous_model_id=previous_model_id,
            new_model_id=new_model_id,
            reloaded=reloaded
        )
    
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def model_info():
    """Get detailed model information."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return recommender.get_model_info()


@app.get("/stats")
async def service_stats():
    """Get service statistics."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    loader = recommender.loader
    
    # Get cache stats
    cache_stats = {}
    if cache_manager is not None:
        cache_stats = cache_manager.get_stats()
    
    return {
        "model_id": recommender.model_id,
        "total_users": loader.mappings['metadata']['num_users'] if loader.mappings else 0,
        "trainable_users": len(loader.trainable_user_set or set()),
        "cold_start_users": (
            loader.mappings['metadata']['num_users'] - len(loader.trainable_user_set or set())
            if loader.mappings else 0
        ),
        "trainable_percentage": (
            len(loader.trainable_user_set or set()) / 
            max(1, loader.mappings['metadata']['num_users']) * 100
            if loader.mappings else 0
        ),
        "num_items": loader.mappings['metadata']['num_items'] if loader.mappings else 0,
        "popular_items_cached": len(loader.top_k_popular_items or []),
        "user_histories_cached": len(loader.user_history_cache or {}),
        "cache": cache_stats
    }


@app.get("/cache_stats")
async def cache_stats():
    """Get detailed cache statistics."""
    if cache_manager is None:
        return {"status": "not_initialized"}
    
    return cache_manager.get_stats()


@app.post("/cache_warmup")
async def trigger_warmup(force: bool = False):
    """Trigger cache warmup."""
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    stats = await async_warmup(cache_manager)
    return stats


@app.post("/cache_clear")
async def clear_cache():
    """Clear all caches."""
    if cache_manager is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    
    cache_manager.clear_all()
    return {"status": "cleared"}


# ============================================================================
# Smart Search Endpoints
# ============================================================================

@app.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """
    Semantic search for products using Vietnamese query.
    
    This endpoint uses PhoBERT embeddings to find products semantically
    similar to the search query. Supports Vietnamese text with automatic
    abbreviation expansion (e.g., "srm" → "sữa rửa mặt").
    
    Args:
        request: SearchRequest with query, topk, filters, rerank
    
    Returns:
        SearchResponse with ranked results
    
    Example:
        POST /search
        {
            "query": "kem dưỡng ẩm cho da khô",
            "topk": 10,
            "filters": {"brand": "Innisfree"},
            "rerank": true
        }
    """
    service = get_search_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    try:
        start_time = time.perf_counter()
        
        response = service.search(
            query=request.query,
            topk=request.topk,
            filters=request.filters,
            rerank=request.rerank
        )
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Convert to response format
        results = [
            SearchResultItem(
                rank=r.rank,
                product_id=r.product_id,
                product_name=r.product_name,
                brand=r.brand,
                category=r.category,
                price=r.price,
                avg_rating=r.avg_rating,
                num_sold=r.num_sold,
                semantic_score=r.semantic_score,
                final_score=r.final_score
            )
            for r in response.results
        ]
        
        logger.info(
            f"Search: query='{request.query[:50]}...', "
            f"results={len(results)}, latency={latency:.1f}ms"
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            count=len(results),
            method=response.method,
            latency_ms=latency
        )
    
    except Exception as e:
        logger.error(f"Search error for query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/similar", response_model=SearchResponse)
async def search_similar_products(request: SearchSimilarRequest):
    """
    Find products similar to a given product.
    
    Uses PhoBERT semantic embeddings to find products with similar
    descriptions, ingredients, and features.
    
    Args:
        request: SearchSimilarRequest with product_id, topk
    
    Returns:
        SearchResponse with similar products
    
    Example:
        POST /search/similar
        {
            "product_id": 12345,
            "topk": 10,
            "exclude_self": true
        }
    """
    service = get_search_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    try:
        start_time = time.perf_counter()
        
        response = service.search_similar(
            product_id=request.product_id,
            topk=request.topk,
            exclude_self=request.exclude_self
        )
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Convert to response format
        results = [
            SearchResultItem(
                rank=r.rank,
                product_id=r.product_id,
                product_name=r.product_name,
                brand=r.brand,
                category=r.category,
                price=r.price,
                avg_rating=r.avg_rating,
                num_sold=r.num_sold,
                semantic_score=r.semantic_score,
                final_score=r.final_score
            )
            for r in response.results
        ]
        
        logger.info(
            f"Similar search: product_id={request.product_id}, "
            f"results={len(results)}, latency={latency:.1f}ms"
        )
        
        return SearchResponse(
            query=f"similar_to:{request.product_id}",
            results=results,
            count=len(results),
            method=response.method,
            latency_ms=latency
        )
    
    except Exception as e:
        logger.error(f"Similar search error for product {request.product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/profile", response_model=SearchResponse)
async def search_by_profile(request: SearchByProfileRequest):
    """
    Search based on user profile/history.
    
    Computes an average embedding from the user's product history
    and finds products semantically similar to their interests.
    
    Args:
        request: SearchByProfileRequest with product_history, topk
    
    Returns:
        SearchResponse with personalized recommendations
    
    Example:
        POST /search/profile
        {
            "product_history": [123, 456, 789],
            "topk": 10,
            "exclude_history": true
        }
    """
    service = get_search_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    try:
        start_time = time.perf_counter()
        
        response = service.search_by_user_profile(
            product_history=request.product_history,
            topk=request.topk,
            exclude_history=request.exclude_history,
            filters=request.filters
        )
        
        latency = (time.perf_counter() - start_time) * 1000
        
        # Convert to response format
        results = [
            SearchResultItem(
                rank=r.rank,
                product_id=r.product_id,
                product_name=r.product_name,
                brand=r.brand,
                category=r.category,
                price=r.price,
                avg_rating=r.avg_rating,
                num_sold=r.num_sold,
                semantic_score=r.semantic_score,
                final_score=r.final_score
            )
            for r in response.results
        ]
        
        logger.info(
            f"Profile search: history_size={len(request.product_history)}, "
            f"results={len(results)}, latency={latency:.1f}ms"
        )
        
        return SearchResponse(
            query=f"profile:{len(request.product_history)}_products",
            results=results,
            count=len(results),
            method=response.method,
            latency_ms=latency
        )
    
    except Exception as e:
        logger.error(f"Profile search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/filters")
async def get_search_filters():
    """
    Get available filter options for search.
    
    Returns:
        Dict with available brands, categories, and price range
    """
    service = get_search_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Search service not initialized")
    
    try:
        return service.get_available_filters()
    except Exception as e:
        logger.error(f"Get filters error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/stats")
async def get_search_stats():
    """
    Get search service statistics.
    
    Returns:
        Search performance metrics and cache statistics
    """
    service = get_search_service()
    if service is None:
        return {"status": "not_initialized"}
    
    return service.get_stats()


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "service.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
