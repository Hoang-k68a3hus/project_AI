"""
CF Recommendation Service Package.

This package provides the serving layer for CF recommendations.

Components:
- api: FastAPI application with REST endpoints
- recommender: Core recommendation engine
    - CFRecommender: Main recommendation class
    - CFModelLoader: Model loading and management
    - PhoBERTEmbeddingLoader: Content embeddings
    - FallbackRecommender: Cold-start strategies

Usage:
    # Start the service
    uvicorn service.api:app --host 0.0.0.0 --port 8000

    # Or programmatically
    from service.recommender import CFRecommender
    
    rec = CFRecommender(auto_load=True)
    result = rec.recommend(user_id=12345, topk=10)
"""

from service.recommender import (
    CFRecommender,
    CFModelLoader,
    PhoBERTEmbeddingLoader,
    FallbackRecommender,
    get_loader,
    get_phobert_loader,
    RecommendationResult
)

__all__ = [
    'CFRecommender',
    'CFModelLoader',
    'PhoBERTEmbeddingLoader',
    'FallbackRecommender',
    'get_loader',
    'get_phobert_loader',
    'RecommendationResult',
]

__version__ = "1.0.0"
