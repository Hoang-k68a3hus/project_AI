"""
Smart Search Module for Semantic Product Search.

This module provides semantic search capabilities using PhoBERT embeddings
for Vietnamese cosmetics product discovery.

Components:
- QueryEncoder: Encode text queries to embeddings
- SearchIndex: Fast similarity search index
- SmartSearchService: Main search service

Example:
    >>> from service.search import get_search_service
    >>> service = get_search_service()
    >>> results = service.search("kem dưỡng da cho da dầu", topk=10)
"""

from service.search.query_encoder import QueryEncoder, get_query_encoder
from service.search.search_index import SearchIndex, get_search_index
from service.search.smart_search import (
    SmartSearchService,
    get_search_service,
    reset_search_service,
    SearchResult,
    SearchResponse
)

__all__ = [
    # Query Encoder
    'QueryEncoder',
    'get_query_encoder',
    
    # Search Index
    'SearchIndex',
    'get_search_index',
    
    # Smart Search Service
    'SmartSearchService',
    'get_search_service',
    'reset_search_service',
    'SearchResult',
    'SearchResponse',
]
