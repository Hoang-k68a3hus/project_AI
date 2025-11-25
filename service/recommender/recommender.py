"""
Core CF Recommender for Serving Layer.

This module provides the CFRecommender class which is the main
recommendation engine with scoring, filtering, reranking, and routing logic.

Example:
    >>> from service.recommender import CFRecommender
    >>> recommender = CFRecommender()
    >>> recs = recommender.recommend(user_id=12345, topk=10)
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging
import time

from .loader import CFModelLoader, get_loader
from .phobert_loader import PhoBERTEmbeddingLoader, get_phobert_loader
from .fallback import FallbackRecommender
from .rerank import HybridReranker, get_reranker, RerankedResult

logger = logging.getLogger(__name__)


def _convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif pd.isna(obj):
        return None
    return obj


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RecommendationResult:
    """Result of a recommendation request."""
    user_id: int
    recommendations: List[Dict[str, Any]]
    count: int
    is_fallback: bool
    fallback_method: Optional[str]
    latency_ms: float
    model_id: Optional[str]


# ============================================================================
# CFRecommender
# ============================================================================

class CFRecommender:
    """
    Main recommendation engine with CF scoring, reranking, and fallback handling.
    
    Features:
    - User segmentation routing (CF vs content-based)
    - CF scoring using U @ V.T
    - Hybrid reranking with content, popularity, quality signals
    - Seen-item filtering
    - Attribute-based filtering
    - Cold-start fallback to content-based + popularity
    - Score normalization for hybrid reranking
    
    Example:
        >>> recommender = CFRecommender()
        >>> recs = recommender.recommend(user_id=12345, topk=10)
    """
    
    def __init__(
        self,
        loader: Optional[CFModelLoader] = None,
        phobert_loader: Optional[PhoBERTEmbeddingLoader] = None,
        auto_load: bool = True,
        enable_reranking: bool = True,
        rerank_config_path: Optional[str] = None
    ):
        """
        Initialize CFRecommender.
        
        Args:
            loader: CFModelLoader instance (creates new if None)
            phobert_loader: PhoBERTEmbeddingLoader instance (lazy loads if None)
            auto_load: Auto-load models and data on init
            enable_reranking: Enable hybrid reranking (default True)
            rerank_config_path: Path to rerank config YAML
        """
        # Initialize loaders
        self.loader = loader or get_loader()
        self.phobert_loader = phobert_loader
        
        # Reranking configuration
        self.enable_reranking = enable_reranking
        self._reranker: Optional[HybridReranker] = None
        self._rerank_config_path = rerank_config_path
        
        # Lazy-initialized fallback recommender
        self._fallback_recommender: Optional[FallbackRecommender] = None
        
        # Cached references (updated on model load)
        self.U: Optional[np.ndarray] = None  # User embeddings
        self.V: Optional[np.ndarray] = None  # Item embeddings
        self.num_items: int = 0
        self.model_id: Optional[str] = None
        self.score_range: Dict[str, float] = {}
        
        # Auto-load
        if auto_load:
            self._initialize()
    
    def _initialize(self) -> None:
        """Load model and data."""
        try:
            # Load model
            model = self.loader.load_model()
            self.U = model['U']
            self.V = model['V']
            self.num_items = self.V.shape[0]
            self.model_id = model['model_id']
            self.score_range = model.get('score_range', {})
            
            # Load mappings
            self.loader.load_mappings()
            
            # Load item metadata
            self.loader.load_item_metadata()
            
            # Load user histories
            self.loader.load_user_histories()
            
            logger.info(
                f"CFRecommender initialized: model={self.model_id}, "
                f"users={self.U.shape[0]}, items={self.V.shape[0]}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize CFRecommender: {e}")
            raise
    
    @property
    def fallback(self) -> FallbackRecommender:
        """Get fallback recommender (lazy initialization)."""
        if self._fallback_recommender is None:
            # Lazy load PhoBERT
            if self.phobert_loader is None:
                try:
                    self.phobert_loader = get_phobert_loader()
                except Exception as e:
                    logger.warning(f"Could not load PhoBERT: {e}")
            
            self._fallback_recommender = FallbackRecommender(
                cf_loader=self.loader,
                phobert_loader=self.phobert_loader
            )
        
        return self._fallback_recommender
    
    @property
    def reranker(self) -> HybridReranker:
        """Get hybrid reranker (lazy initialization)."""
        if self._reranker is None:
            # Lazy load PhoBERT
            if self.phobert_loader is None:
                try:
                    self.phobert_loader = get_phobert_loader()
                except Exception as e:
                    logger.warning(f"Could not load PhoBERT for reranker: {e}")
            
            self._reranker = get_reranker(
                phobert_loader=self.phobert_loader,
                item_metadata=self.loader.item_metadata,
                config_path=self._rerank_config_path
            )
        
        return self._reranker
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize CF scores to [0, 1] range using score_range.
        
        Args:
            scores: Raw CF scores
        
        Returns:
            Normalized scores in [0, 1]
        """
        if not self.score_range:
            # Fallback to min-max normalization
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                return (scores - s_min) / (s_max - s_min)
            return np.zeros_like(scores)
        
        # Use precomputed p01/p99 for robust normalization
        p01 = self.score_range.get('p01', scores.min())
        p99 = self.score_range.get('p99', scores.max())
        
        if p99 > p01:
            normalized = (scores - p01) / (p99 - p01)
            return np.clip(normalized, 0, 1)
        
        return np.zeros_like(scores)
    
    def _get_seen_item_indices(self, user_id: int) -> Set[int]:
        """Get item indices user has interacted with."""
        history = self.loader.get_user_history(user_id)
        
        if not history:
            return set()
        
        mappings = self.loader.mappings
        item_to_idx = mappings.get('item_to_idx', {})
        
        seen_indices = set()
        for pid in history:
            idx = item_to_idx.get(str(pid))
            if idx is not None:
                seen_indices.add(int(idx))
        
        return seen_indices
    
    def _apply_attribute_filters(
        self,
        filter_params: Dict[str, Any]
    ) -> Set[int]:
        """
        Get valid item indices based on attribute filters.
        
        Args:
            filter_params: Dict like {'brand': 'Innisfree', 'skin_type': 'oily'}
        
        Returns:
            Set of valid item indices
        """
        metadata = self.loader.item_metadata
        if metadata is None:
            return set(range(self.num_items))
        
        mask = pd.Series([True] * len(metadata))
        
        for key, value in filter_params.items():
            if key in metadata.columns:
                # Handle list-type columns (e.g., skin_type_standardized)
                col = metadata[key]
                if isinstance(value, list):
                    mask &= col.apply(
                        lambda x: any(v in str(x) for v in value) if pd.notna(x) else False
                    )
                else:
                    mask &= col == value
        
        valid_pids = set(metadata[mask]['product_id'].values)
        
        # Convert to indices
        item_to_idx = self.loader.mappings.get('item_to_idx', {})
        valid_indices = set()
        for pid in valid_pids:
            idx = item_to_idx.get(str(pid))
            if idx is not None:
                valid_indices.add(int(idx))
        
        return valid_indices
    
    def _enrich_recommendations(
        self,
        product_ids: List[int],
        scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Enrich recommendations with product metadata.
        
        Args:
            product_ids: List of product IDs
            scores: List of scores
        
        Returns:
            List of enriched recommendation dicts
        """
        metadata = self.loader.item_metadata
        
        recommendations = []
        for rank, (pid, score) in enumerate(zip(product_ids, scores), 1):
            rec = {
                'product_id': int(pid),  # Ensure native Python int
                'score': float(score),
                'rank': rank,
                'fallback': False,
            }
            
            # Add metadata if available
            if metadata is not None:
                product_row = metadata[metadata['product_id'] == pid]
                if not product_row.empty:
                    row = product_row.iloc[0]
                    for col in ['product_name', 'brand', 'price', 'avg_star', 'num_sold_time']:
                        if col in product_row.columns:
                            val = row[col]
                            if pd.notna(val):
                                # Convert numpy types to native Python types
                                rec[col] = _convert_numpy_types(val)
            
            recommendations.append(rec)
        
        return recommendations
    
    def recommend(
        self,
        user_id: int,
        topk: int = 10,
        exclude_seen: bool = True,
        filter_params: Optional[Dict[str, Any]] = None,
        normalize_scores: bool = False,
        rerank: Optional[bool] = None
    ) -> RecommendationResult:
        """
        Generate top-K recommendations for user.
        
        Args:
            user_id: Original user ID (int)
            topk: Number of recommendations (default 10)
            exclude_seen: If True, exclude items user has interacted with
            filter_params: Dict with attribute filters (e.g., {'brand': 'Innisfree'})
            normalize_scores: If True, normalize CF scores to [0, 1]
            rerank: Override default reranking setting (None = use default)
        
        Returns:
            RecommendationResult with recommendations and metadata
        
        Raises:
            KeyError: User ID not found (handled via fallback)
        """
        start_time = time.perf_counter()
        
        # Determine if reranking should be used
        use_rerank = rerank if rerank is not None else self.enable_reranking
        
        # Generate more candidates for reranking
        candidate_k = topk * 5 if use_rerank else topk
        
        # Check if user is trainable (has enough interactions for CF)
        is_trainable = self.loader.is_trainable_user(user_id)
        
        if not is_trainable:
            # Cold-start user → use fallback
            logger.debug(f"User {user_id} is cold-start, using fallback")
            
            user_history = list(self.loader.get_user_history(user_id))
            
            fallback_recs = self.fallback.recommend(
                user_id=user_id,
                user_history=user_history,
                topk=candidate_k if use_rerank else topk,
                strategy='hybrid',
                filter_params=filter_params
            )
            
            # Apply reranking to fallback results
            if use_rerank and fallback_recs:
                rerank_result = self.reranker.rerank_cold_start(
                    recommendations=fallback_recs,
                    user_history=user_history,
                    topk=topk
                )
                fallback_recs = rerank_result.recommendations
            
            latency = (time.perf_counter() - start_time) * 1000
            
            fallback_method = 'hybrid'
            if fallback_recs:
                fallback_method = fallback_recs[0].get('fallback_method', 'hybrid')
            
            return RecommendationResult(
                user_id=user_id,
                recommendations=fallback_recs,
                count=len(fallback_recs),
                is_fallback=True,
                fallback_method=fallback_method,
                latency_ms=latency,
                model_id=None
            )
        
        # Trainable user → use CF
        u_idx_cf = self.loader.get_cf_user_index(user_id)
        
        if u_idx_cf is None:
            # Should not happen, but handle gracefully
            logger.warning(f"User {user_id} marked trainable but no CF index found")
            fallback_recs = self.fallback.recommend(
                user_id=user_id,
                topk=topk,
                strategy='popularity'
            )
            latency = (time.perf_counter() - start_time) * 1000
            return RecommendationResult(
                user_id=user_id,
                recommendations=fallback_recs,
                count=len(fallback_recs),
                is_fallback=True,
                fallback_method='popularity',
                latency_ms=latency,
                model_id=None
            )
        
        # Compute CF scores: U[u_idx_cf] @ V.T
        scores = self.U[u_idx_cf] @ self.V.T  # Shape: (num_items,)
        
        # Normalize if requested
        if normalize_scores:
            scores = self._normalize_scores(scores)
        
        # Exclude seen items
        if exclude_seen:
            seen_indices = self._get_seen_item_indices(user_id)
            for idx in seen_indices:
                if idx < len(scores):
                    scores[idx] = -np.inf
        
        # Apply attribute filters
        if filter_params:
            valid_indices = self._apply_attribute_filters(filter_params)
            invalid_mask = np.ones(self.num_items, dtype=bool)
            for idx in valid_indices:
                if idx < len(invalid_mask):
                    invalid_mask[idx] = False
            scores[invalid_mask] = -np.inf
        
        # Get top-K indices (more if reranking)
        top_k_indices = np.argsort(scores)[::-1][:candidate_k]
        
        # Filter out -inf scores
        valid_top_k = [(idx, scores[idx]) for idx in top_k_indices if scores[idx] > -np.inf]
        
        if not valid_top_k:
            # No valid items → fallback
            logger.warning(f"No valid CF items for user {user_id}, using fallback")
            fallback_recs = self.fallback.recommend(
                user_id=user_id,
                topk=topk,
                strategy='popularity'
            )
            latency = (time.perf_counter() - start_time) * 1000
            return RecommendationResult(
                user_id=user_id,
                recommendations=fallback_recs,
                count=len(fallback_recs),
                is_fallback=True,
                fallback_method='popularity',
                latency_ms=latency,
                model_id=self.model_id
            )
        
        # Map indices to product IDs
        idx_to_item = self.loader.mappings.get('idx_to_item', {})
        product_ids = []
        final_scores = []
        
        for idx, score in valid_top_k:
            pid = idx_to_item.get(str(idx))
            if pid is not None:
                product_ids.append(int(pid))
                final_scores.append(score)
        
        # Enrich with metadata
        recommendations = self._enrich_recommendations(product_ids, final_scores)
        
        # Apply hybrid reranking
        if use_rerank and recommendations:
            user_history = list(self.loader.get_user_history(user_id))
            rerank_result = self.reranker.rerank(
                cf_recommendations=recommendations,
                user_id=user_id,
                user_history=user_history,
                topk=topk,
                is_cold_start=False
            )
            recommendations = rerank_result.recommendations
            
            logger.debug(
                f"Reranked {rerank_result.num_candidates} candidates for user {user_id}: "
                f"diversity={rerank_result.diversity_score:.3f}"
            )
        else:
            # Truncate to topk if no reranking
            recommendations = recommendations[:topk]
        
        latency = (time.perf_counter() - start_time) * 1000
        
        logger.debug(
            f"CF recommendation for user {user_id}: {len(recommendations)} items, "
            f"latency={latency:.1f}ms, reranked={use_rerank}"
        )
        
        return RecommendationResult(
            user_id=user_id,
            recommendations=recommendations,
            count=len(recommendations),
            is_fallback=False,
            fallback_method=None,
            latency_ms=latency,
            model_id=self.model_id
        )
    
    def batch_recommend(
        self,
        user_ids: List[int],
        topk: int = 10,
        exclude_seen: bool = True
    ) -> Dict[int, RecommendationResult]:
        """
        Batch recommendations for multiple users.
        
        Uses vectorized CF scoring for efficiency.
        
        Args:
            user_ids: List of user IDs
            topk: Number of recommendations per user
            exclude_seen: Exclude seen items
        
        Returns:
            Dict mapping user_id to RecommendationResult
        """
        start_time = time.perf_counter()
        results = {}
        
        # Separate trainable vs cold-start users
        trainable_users = []
        cold_start_users = []
        
        for uid in user_ids:
            if self.loader.is_trainable_user(uid):
                u_idx_cf = self.loader.get_cf_user_index(uid)
                if u_idx_cf is not None:
                    trainable_users.append((uid, u_idx_cf))
                else:
                    cold_start_users.append(uid)
            else:
                cold_start_users.append(uid)
        
        # Batch CF scoring for trainable users
        if trainable_users:
            uids = [u[0] for u in trainable_users]
            u_indices = [u[1] for u in trainable_users]
            
            # Batch matrix multiplication
            scores_batch = self.U[u_indices] @ self.V.T  # (len(trainable_users), num_items)
            
            idx_to_item = self.loader.mappings.get('idx_to_item', {})
            
            for i, (uid, u_idx_cf) in enumerate(trainable_users):
                scores = scores_batch[i].copy()
                
                # Exclude seen
                if exclude_seen:
                    seen_indices = self._get_seen_item_indices(uid)
                    for idx in seen_indices:
                        if idx < len(scores):
                            scores[idx] = -np.inf
                
                # Top-K
                top_k_indices = np.argsort(scores)[::-1][:topk]
                
                product_ids = []
                final_scores = []
                for idx in top_k_indices:
                    if scores[idx] > -np.inf:
                        pid = idx_to_item.get(str(idx))
                        if pid is not None:
                            product_ids.append(int(pid))
                            final_scores.append(float(scores[idx]))
                
                recommendations = self._enrich_recommendations(product_ids, final_scores)
                
                results[uid] = RecommendationResult(
                    user_id=uid,
                    recommendations=recommendations,
                    count=len(recommendations),
                    is_fallback=False,
                    fallback_method=None,
                    latency_ms=0,  # Will be updated at end
                    model_id=self.model_id
                )
        
        # Handle cold-start users
        for uid in cold_start_users:
            fallback_recs = self.fallback.recommend(
                user_id=uid,
                topk=topk,
                strategy='hybrid'
            )
            
            fallback_method = 'hybrid'
            if fallback_recs:
                fallback_method = fallback_recs[0].get('fallback_method', 'hybrid')
            
            results[uid] = RecommendationResult(
                user_id=uid,
                recommendations=fallback_recs,
                count=len(fallback_recs),
                is_fallback=True,
                fallback_method=fallback_method,
                latency_ms=0,
                model_id=None
            )
        
        # Update latency for all results
        total_latency = (time.perf_counter() - start_time) * 1000
        per_user_latency = total_latency / max(len(user_ids), 1)
        
        for uid in results:
            results[uid].latency_ms = per_user_latency
        
        logger.info(
            f"Batch recommendation for {len(user_ids)} users: "
            f"{len(trainable_users)} CF, {len(cold_start_users)} fallback, "
            f"total_latency={total_latency:.1f}ms"
        )
        
        return results
    
    def similar_items(
        self,
        product_id: int,
        topk: int = 10,
        use_cf: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find similar items using CF embeddings or content embeddings.
        
        Args:
            product_id: Query product ID
            topk: Number of similar items
            use_cf: If True, use CF item embeddings; else use PhoBERT
        
        Returns:
            List of similar item dicts
        """
        if use_cf:
            # Use V @ V.T for CF-based similarity
            item_to_idx = self.loader.mappings.get('item_to_idx', {})
            idx = item_to_idx.get(str(product_id))
            
            if idx is None:
                logger.warning(f"Product {product_id} not in CF model")
                use_cf = False
            else:
                idx = int(idx)
                # Compute similarity
                query_v = self.V[idx]
                similarities = self.V @ query_v
                similarities[idx] = -np.inf  # Exclude self
                
                top_indices = np.argsort(similarities)[::-1][:topk]
                
                idx_to_item = self.loader.mappings.get('idx_to_item', {})
                product_ids = []
                scores = []
                
                for i in top_indices:
                    if similarities[i] > -np.inf:
                        pid = idx_to_item.get(str(i))
                        if pid is not None:
                            product_ids.append(int(pid))
                            scores.append(float(similarities[i]))
                
                return self._enrich_recommendations(product_ids, scores)
        
        # Use PhoBERT for content-based similarity
        if self.phobert_loader is None:
            try:
                self.phobert_loader = get_phobert_loader()
            except Exception:
                logger.warning("PhoBERT not available for similar items")
                return []
        
        similar = self.phobert_loader.find_similar_items(
            product_id=product_id,
            topk=topk,
            exclude_self=True
        )
        
        product_ids = [pid for pid, _ in similar]
        scores = [score for _, score in similar]
        
        return self._enrich_recommendations(product_ids, scores)
    
    def reload_model(self) -> bool:
        """
        Reload model from registry if updated.
        
        Returns:
            True if model was reloaded
        """
        updated = self.loader.reload_if_updated()
        
        if updated:
            model = self.loader.current_model
            self.U = model['U']
            self.V = model['V']
            self.num_items = self.V.shape[0]
            self.model_id = model['model_id']
            self.score_range = model.get('score_range', {})
            
            logger.info(f"Model reloaded: {self.model_id}")
        
        return updated
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get info about current model."""
        return {
            **self.loader.get_model_info(),
            'trainable_users': len(self.loader.trainable_user_set or set()),
            'reranking_enabled': self.enable_reranking,
        }
    
    def set_reranking(self, enabled: bool) -> None:
        """Enable or disable hybrid reranking."""
        self.enable_reranking = enabled
        logger.info(f"Reranking {'enabled' if enabled else 'disabled'}")
    
    def update_rerank_weights(
        self,
        weights_trainable: Optional[Dict[str, float]] = None,
        weights_cold_start: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update reranking weights dynamically.
        
        Args:
            weights_trainable: New weights for trainable users
            weights_cold_start: New weights for cold-start users
        """
        self.reranker.update_config(
            weights_trainable=weights_trainable,
            weights_cold_start=weights_cold_start
        )
        logger.info(f"Rerank weights updated")
