"""
PhoBERT Embedding Loader for Content-Based Recommendations.

This module provides the PhoBERTEmbeddingLoader class for loading
and using PhoBERT product embeddings for content-based recommendations.

Example:
    >>> from service.recommender.phobert_loader import PhoBERTEmbeddingLoader
    >>> phobert = PhoBERTEmbeddingLoader()
    >>> emb = phobert.get_embedding(product_id=123)
    >>> similar = phobert.find_similar_items(product_id=123, topk=10)
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import numpy as np
import logging
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_EMBEDDINGS_PATH = Path(
    "data/processed/content_based_embeddings/product_embeddings.pt"
)
FALLBACK_EMBEDDING_PATHS = [
    Path("data/published_data/content_based_embeddings/product_embeddings.pt"),
    Path("data/published_data/content_based_embeddings/phobert_description_feature.pt"),
]


# ============================================================================
# PhoBERTEmbeddingLoader
# ============================================================================

class PhoBERTEmbeddingLoader:
    """
    Load and cache PhoBERT product embeddings for content-based recommendations.
    
    Features:
    - Load embeddings from PyTorch .pt file
    - Pre-normalize embeddings for fast cosine similarity
    - Compute user profiles from history
    - Find similar items efficiently
    
    Example:
        >>> loader = PhoBERTEmbeddingLoader()
        >>> emb = loader.get_embedding(123)
        >>> similar = loader.find_similar_items(123, topk=10)
    """
    
    _instance: Optional['PhoBERTEmbeddingLoader'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(
        self,
        embeddings_path: Optional[str] = None,
        fallback_paths: Optional[List[str]] = None,
        auto_load: bool = True
    ):
        """
        Initialize PhoBERTEmbeddingLoader.
        
        Args:
            embeddings_path: Path to product_embeddings.pt
            auto_load: Automatically load embeddings on init
        """
        if self._initialized:
            return
        
        primary_path = Path(embeddings_path) if embeddings_path else DEFAULT_EMBEDDINGS_PATH
        fallback_candidates = fallback_paths or FALLBACK_EMBEDDING_PATHS
        self._embedding_candidates: List[Path] = []
        seen: Set[Path] = set()
        
        for candidate in [primary_path] + list(fallback_candidates):
            path_obj = Path(candidate)
            if path_obj in seen:
                continue
            self._embedding_candidates.append(path_obj)
            seen.add(path_obj)
        
        self.embeddings_path = self._embedding_candidates[0]
        self.embeddings: Optional[np.ndarray] = None  # (num_products, 768 or 1024)
        self.embeddings_norm: Optional[np.ndarray] = None  # Pre-normalized
        self.product_id_to_idx: Dict[int, int] = {}
        self.idx_to_product_id: Dict[int, int] = {}
        
        # Item-item similarity cache (precomputed if small enough)
        self._similarity_matrix: Optional[np.ndarray] = None
        self._similarity_computed = False
        self._resolved_embeddings_path: Optional[Path] = None
        
        self._initialized = True
        
        if auto_load:
            try:
                self._load_embeddings()
            except Exception as e:
                logger.warning(f"Failed to auto-load embeddings: {e}")
    
    def _resolve_embeddings_path(self) -> Path:
        """Resolve first existing embeddings path from candidates."""
        if self._resolved_embeddings_path and self._resolved_embeddings_path.exists():
            return self._resolved_embeddings_path
        
        missing_paths = []
        for candidate in self._embedding_candidates:
            if candidate.exists():
                self._resolved_embeddings_path = candidate
                if candidate != self._embedding_candidates[0]:
                    logger.info(
                        "PhoBERTEmbeddingLoader using fallback embeddings at %s",
                        candidate
                    )
                return candidate
            missing_paths.append(str(candidate))
        
        raise FileNotFoundError(
            "PhoBERT embeddings not found in any of the expected paths: "
            + ", ".join(missing_paths)
        )
    
    def _load_embeddings(self) -> None:
        """Load BERT embeddings from file."""
        import time
        start = time.perf_counter()
        
        resolved_path = self._resolve_embeddings_path()
        self.embeddings_path = resolved_path
        
        # Load PyTorch file
        try:
            import torch
            bert_data = torch.load(resolved_path, map_location='cpu')
        except ImportError:
            logger.error("PyTorch not available, cannot load embeddings")
            raise
        
        # Extract embeddings and product IDs
        if isinstance(bert_data, dict):
            embeddings_tensor = bert_data.get('embeddings', bert_data.get('item_embeddings'))
            product_ids = bert_data.get('product_ids', bert_data.get('item_ids'))
        else:
            raise ValueError("Unexpected embeddings format")
        
        # Convert to numpy
        if hasattr(embeddings_tensor, 'numpy'):
            self.embeddings = embeddings_tensor.numpy()
        else:
            self.embeddings = np.array(embeddings_tensor)
        
        # Create mappings
        for idx, pid in enumerate(product_ids):
            pid_int = int(pid) if hasattr(pid, 'item') else int(pid)
            self.product_id_to_idx[pid_int] = idx
            self.idx_to_product_id[idx] = pid_int
        
        # Pre-normalize for fast cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
        self.embeddings_norm = self.embeddings / norms
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"Loaded {len(self.product_id_to_idx)} BERT embeddings "
            f"(dim={self.embeddings.shape[1]}) from {resolved_path} "
            f"in {elapsed:.1f}ms"
        )
    
    def is_loaded(self) -> bool:
        """Check if embeddings are loaded."""
        return self.embeddings is not None
    
    def get_embedding(self, product_id: int) -> Optional[np.ndarray]:
        """
        Get embedding for a single product.
        
        Args:
            product_id: Product ID
        
        Returns:
            np.array of shape (768,) or (1024,), or None if not found
        """
        if not self.is_loaded():
            self._load_embeddings()
        
        idx = self.product_id_to_idx.get(product_id)
        if idx is None:
            return None
        
        return self.embeddings[idx]
    
    def get_embedding_normalized(self, product_id: int) -> Optional[np.ndarray]:
        """Get L2-normalized embedding for a product."""
        if not self.is_loaded():
            self._load_embeddings()
        
        idx = self.product_id_to_idx.get(product_id)
        if idx is None:
            return None
        
        return self.embeddings_norm[idx]
    
    def compute_user_profile(
        self,
        user_history_items: List[int],
        weights: Optional[List[float]] = None,
        strategy: str = 'weighted_mean'
    ) -> Optional[np.ndarray]:
        """
        Compute user profile embedding from interaction history.
        
        Args:
            user_history_items: List of product_ids user has interacted with
            weights: Optional weights for each item (e.g., ratings)
            strategy: 'mean', 'weighted_mean', or 'max'
        
        Returns:
            np.array of shape (768,) representing user profile, or None
        """
        if not self.is_loaded():
            self._load_embeddings()
        
        if not user_history_items:
            return None
        
        # Collect embeddings for history items
        history_embeddings = []
        history_weights = []
        
        for i, pid in enumerate(user_history_items):
            emb = self.get_embedding(pid)
            if emb is not None:
                history_embeddings.append(emb)
                if weights is not None and i < len(weights):
                    history_weights.append(weights[i])
                else:
                    history_weights.append(1.0)
        
        if not history_embeddings:
            return None
        
        history_embeddings = np.array(history_embeddings)
        history_weights = np.array(history_weights).reshape(-1, 1)
        
        # Aggregate based on strategy
        if strategy == 'mean':
            profile = np.mean(history_embeddings, axis=0)
        elif strategy == 'weighted_mean':
            weights_norm = history_weights / history_weights.sum()
            profile = (history_embeddings * weights_norm).sum(axis=0)
        elif strategy == 'max':
            profile = np.max(history_embeddings, axis=0)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return profile
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and all/candidate items.
        
        Args:
            query_embedding: Query embedding vector
            candidate_indices: Optional list of item indices to compare against
        
        Returns:
            np.array of similarity scores
        """
        if not self.is_loaded():
            self._load_embeddings()
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        
        if candidate_indices is not None:
            # Compare against specific candidates
            candidates = self.embeddings_norm[candidate_indices]
            similarities = candidates @ query_norm
        else:
            # Compare against all items
            similarities = self.embeddings_norm @ query_norm
        
        return similarities
    
    def find_similar_items(
        self,
        product_id: int,
        topk: int = 10,
        exclude_self: bool = True,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Find top-K similar items to a given product.
        
        Args:
            product_id: Query product ID
            topk: Number of similar items to return
            exclude_self: Exclude the query product from results
            exclude_ids: Optional set of product IDs to exclude
        
        Returns:
            List of (product_id, similarity_score) tuples
        """
        query_emb = self.get_embedding_normalized(product_id)
        if query_emb is None:
            return []
        
        # Compute similarities
        similarities = self.embeddings_norm @ query_emb
        
        # Create mask for exclusions
        mask = np.ones(len(similarities), dtype=bool)
        
        if exclude_self:
            query_idx = self.product_id_to_idx.get(product_id)
            if query_idx is not None:
                mask[query_idx] = False
        
        if exclude_ids:
            for pid in exclude_ids:
                idx = self.product_id_to_idx.get(pid)
                if idx is not None:
                    mask[idx] = False
        
        # Apply mask
        similarities[~mask] = -np.inf
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[::-1][:topk]
        
        # Build results
        results = []
        for idx in top_indices:
            if similarities[idx] == -np.inf:
                continue
            pid = self.idx_to_product_id[idx]
            results.append((pid, float(similarities[idx])))
        
        return results
    
    def find_similar_to_profile(
        self,
        user_profile: np.ndarray,
        topk: int = 10,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Find top-K items similar to a user profile embedding.
        
        Args:
            user_profile: User profile embedding
            topk: Number of items to return
            exclude_ids: Product IDs to exclude (e.g., already purchased)
        
        Returns:
            List of (product_id, similarity_score) tuples
        """
        if not self.is_loaded():
            self._load_embeddings()
        
        # Normalize profile
        profile_norm = user_profile / (np.linalg.norm(user_profile) + 1e-9)
        
        # Compute similarities
        similarities = self.embeddings_norm @ profile_norm
        
        # Apply exclusions
        if exclude_ids:
            for pid in exclude_ids:
                idx = self.product_id_to_idx.get(pid)
                if idx is not None:
                    similarities[idx] = -np.inf
        
        # Get top-K
        top_indices = np.argsort(similarities)[::-1][:topk]
        
        results = []
        for idx in top_indices:
            if similarities[idx] == -np.inf:
                continue
            pid = self.idx_to_product_id[idx]
            results.append((pid, float(similarities[idx])))
        
        return results
    
    def precompute_item_similarity(self, max_items: int = 3000) -> None:
        """
        Precompute item-item similarity matrix (for small catalogs).
        
        Args:
            max_items: Maximum items to precompute for
        """
        if not self.is_loaded():
            self._load_embeddings()
        
        if len(self.product_id_to_idx) > max_items:
            logger.warning(
                f"Skipping precomputation: {len(self.product_id_to_idx)} items > {max_items} max"
            )
            return
        
        import time
        start = time.perf_counter()
        
        # V @ V.T for item-item similarity
        self._similarity_matrix = self.embeddings_norm @ self.embeddings_norm.T
        self._similarity_computed = True
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Precomputed item-item similarity matrix in {elapsed:.1f}ms")
    
    def get_precomputed_similar(
        self,
        product_id: int,
        topk: int = 10,
        exclude_ids: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Get similar items using precomputed matrix (faster if available).
        
        Falls back to compute_similar_items if not precomputed.
        """
        if not self._similarity_computed:
            return self.find_similar_items(product_id, topk, True, exclude_ids)
        
        query_idx = self.product_id_to_idx.get(product_id)
        if query_idx is None:
            return []
        
        similarities = self._similarity_matrix[query_idx].copy()
        
        # Exclude self
        similarities[query_idx] = -np.inf
        
        # Apply exclusions
        if exclude_ids:
            for pid in exclude_ids:
                idx = self.product_id_to_idx.get(pid)
                if idx is not None:
                    similarities[idx] = -np.inf
        
        # Get top-K
        top_indices = np.argsort(similarities)[::-1][:topk]
        
        results = []
        for idx in top_indices:
            if similarities[idx] == -np.inf:
                continue
            pid = self.idx_to_product_id[idx]
            results.append((pid, float(similarities[idx])))
        
        return results
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.embeddings is None:
            return 0
        return self.embeddings.shape[1]
    
    @property
    def num_products(self) -> int:
        """Get number of products with embeddings."""
        return len(self.product_id_to_idx)


# ============================================================================
# Convenience Functions
# ============================================================================

def get_phobert_loader(**kwargs) -> PhoBERTEmbeddingLoader:
    """Get singleton PhoBERT loader instance."""
    return PhoBERTEmbeddingLoader(**kwargs)


def reset_phobert_loader() -> None:
    """Reset singleton loader instance."""
    with PhoBERTEmbeddingLoader._lock:
        PhoBERTEmbeddingLoader._instance = None
