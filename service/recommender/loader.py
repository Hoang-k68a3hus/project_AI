"""
CF Model Loader for Serving Layer.

This module provides the CFModelLoader class for loading CF models,
mappings, and metadata for the recommendation service.

Example:
    >>> from service.recommender.loader import CFModelLoader
    >>> loader = CFModelLoader()
    >>> loader.load_model()
    >>> loader.load_mappings()
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import json
import logging
import threading
import pickle
import warnings

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_REGISTRY_PATH = "artifacts/cf/registry.json"
DEFAULT_DATA_DIR = "data/processed"
DEFAULT_PUBLISHED_DIR = "data/published_data"


# ============================================================================
# CFModelLoader
# ============================================================================

class CFModelLoader:
    """
    Singleton class for loading CF models, mappings, and metadata for serving.
    
    Features:
    - Load CF model embeddings (U, V) from registry
    - Load user/item ID mappings
    - Load trainable user mapping for routing
    - Load item metadata for enrichment
    - Hot-reload when registry updates
    - Thread-safe operations
    
    Example:
        >>> loader = CFModelLoader()
        >>> model = loader.load_model()
        >>> mappings = loader.load_mappings()
        >>> metadata = loader.load_item_metadata()
    """
    
    _instance: Optional['CFModelLoader'] = None
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
        registry_path: str = DEFAULT_REGISTRY_PATH,
        data_dir: str = DEFAULT_DATA_DIR,
        published_dir: str = DEFAULT_PUBLISHED_DIR,
        auto_load: bool = False
    ):
        """
        Initialize CFModelLoader.
        
        Args:
            registry_path: Path to registry.json
            data_dir: Path to processed data directory
            published_dir: Path to published data directory
            auto_load: Auto-load model and mappings on init
        """
        # Avoid re-initialization for singleton
        if self._initialized:
            return
        
        self.registry_path = Path(registry_path)
        self.data_dir = Path(data_dir)
        self.published_dir = Path(published_dir)
        
        # Cached state
        self.current_model: Optional[Dict[str, Any]] = None
        self.current_model_id: Optional[str] = None
        self.mappings: Optional[Dict[str, Any]] = None
        self.trainable_user_mapping: Optional[Dict[str, int]] = None
        self.trainable_user_set: Optional[Set[int]] = None
        self.item_metadata: Optional[pd.DataFrame] = None
        self.user_history_cache: Optional[Dict[int, Set[int]]] = None
        self.top_k_popular_items: Optional[List[int]] = None
        self.data_stats: Optional[Dict[str, Any]] = None
        
        self._registry_cache: Optional[Dict] = None
        self._last_registry_check: Optional[datetime] = None
        
        self._initialized = True
        
        # Auto-load if requested
        if auto_load:
            try:
                self.load_model()
                self.load_mappings()
                self.load_item_metadata()
            except Exception as e:
                logger.warning(f"Auto-load failed: {e}")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from JSON file."""
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Registry not found: {self.registry_path}")
        
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        self._registry_cache = registry
        self._last_registry_check = datetime.now()
        
        return registry
    
    def load_model(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load CF model from registry.
        
        Args:
            model_id: Optional model ID. If None → load current_best
        
        Returns:
            dict: {
                'model_id': str,
                'model_type': 'als' | 'bpr' | 'bert_als',
                'U': np.array (num_users, factors),
                'V': np.array (num_items, factors),
                'params': dict,
                'metadata': dict,
                'score_range': dict (for normalization)
            }
        
        Raises:
            FileNotFoundError: Model artifacts không tồn tại
            ValueError: Invalid model_id or no current_best
        """
        import time
        start_time = time.perf_counter()
        
        # Load registry
        registry = self._load_registry()
        
        # Determine model to load
        if model_id is None:
            current_best = registry.get('current_best')
            if not current_best:
                raise ValueError("No current_best model in registry")
            model_id = current_best['model_id']
        
        # Get model info
        model_info = registry['models'].get(model_id)
        if not model_info:
            raise ValueError(f"Model not found in registry: {model_id}")
        
        model_path = Path(model_info['path'])
        model_type = model_info['model_type']
        
        # Load embeddings
        U_raw = np.load(model_path / f"{model_type}_U.npy")
        V_raw = np.load(model_path / f"{model_type}_V.npy")
        
        # Load metadata to check dimensions
        metadata_path = model_path / f"{model_type}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            num_users_meta = meta.get('num_users', 0)
            num_items_meta = meta.get('num_items', 0)
        else:
            num_users_meta = 0
            num_items_meta = 0
        
        # FIX: Check if U/V are swapped (issue from Colab training)
        # Load mappings to check actual dimensions
        mappings_path = self.data_dir / 'user_item_mappings.json'
        if mappings_path.exists():
            with open(mappings_path, 'r', encoding='utf-8') as f:
                mappings_check = json.load(f)
            actual_num_items = mappings_check['metadata'].get('num_items', 0)
            actual_num_trainable = mappings_check['metadata'].get('num_trainable_users', 0)
            
            # Convention: U = (num_trainable_users, factors), V = (num_items, factors)
            # If U.shape[0] matches num_items, matrices are swapped
            if U_raw.shape[0] == actual_num_items and V_raw.shape[0] == actual_num_trainable:
                logger.warning(
                    f"U/V matrices appear swapped in model files. "
                    f"Raw U: {U_raw.shape}, Raw V: {V_raw.shape}. "
                    f"Expected: U=({actual_num_trainable}, factors), V=({actual_num_items}, factors). "
                    f"Swapping to fix."
                )
                U = V_raw  # V_raw contains user embeddings
                V = U_raw  # U_raw contains item embeddings
            else:
                U = U_raw
                V = V_raw
        else:
            # No mappings file, use raw
            U = U_raw
            V = V_raw
        
        # Load params
        with open(model_path / f"{model_type}_params.json", 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        # Load full metadata (already loaded partially above for dimension check)
        if metadata_path.exists():
            metadata = meta
        else:
            metadata = {}
        
        # Extract score_range for normalization
        score_range = metadata.get('score_range', {})
        if not score_range:
            # Compute from model info if not available
            score_range = {
                'min': -1.0,
                'max': 1.0,
                'mean': 0.0,
                'std': 0.1,
            }
        
        # Build model dict
        self.current_model = {
            'model_id': model_id,
            'model_type': model_type,
            'U': U,
            'V': V,
            'params': params,
            'metadata': metadata,
            'score_range': score_range,
            'loaded_at': datetime.now().isoformat(),
        }
        self.current_model_id = model_id
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(f"Loaded model {model_id} in {elapsed:.1f}ms (U: {U.shape}, V: {V.shape})")
        
        return self.current_model
    
    def load_mappings(self, data_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load user/item ID mappings.
        
        Args:
            data_version: Optional hash to validate against
        
        Returns:
            dict: {
                'user_to_idx': {user_id: u_idx},
                'idx_to_user': {u_idx: user_id},
                'item_to_idx': {product_id: i_idx},
                'idx_to_item': {i_idx: product_id},
                'metadata': {...}
            }
        """
        mappings_path = self.data_dir / 'user_item_mappings.json'
        
        with open(mappings_path, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        # Validate data version if provided
        if data_version:
            actual_hash = mappings.get('metadata', {}).get('data_hash')
            if actual_hash and actual_hash != data_version:
                warnings.warn(
                    f"Data version mismatch: expected {data_version}, got {actual_hash}"
                )
        
        self.mappings = mappings
        
        # Also load trainable user mapping
        self._load_trainable_user_mapping()
        
        # Load top-k popular items
        self._load_top_k_popular()
        
        # Load data stats
        self._load_data_stats()
        
        logger.info(
            f"Loaded mappings: {mappings['metadata']['num_users']} users, "
            f"{mappings['metadata']['num_items']} items, "
            f"{mappings['metadata']['num_trainable_users']} trainable"
        )
        
        return mappings
    
    def _load_trainable_user_mapping(self) -> None:
        """Load trainable user mapping (u_idx -> u_idx_cf)."""
        mapping_path = self.data_dir / 'trainable_user_mapping.json'
        
        if not mapping_path.exists():
            logger.warning(f"Trainable user mapping not found: {mapping_path}")
            self.trainable_user_mapping = {}
            self.trainable_user_set = set()
            return
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert string keys to int
        self.trainable_user_mapping = {
            int(k): int(v) for k, v in data['u_idx_to_u_idx_cf'].items()
        }
        self.trainable_user_set = set(self.trainable_user_mapping.keys())
        
        logger.info(f"Loaded {len(self.trainable_user_mapping)} trainable user mappings")
    
    def _load_top_k_popular(self) -> None:
        """Load top-K popular items."""
        popular_path = self.data_dir / 'top_k_popular_items.json'
        
        if not popular_path.exists():
            logger.warning(f"Top-K popular items not found: {popular_path}")
            self.top_k_popular_items = []
            return
        
        with open(popular_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.top_k_popular_items = data.get('top_k_items', [])
        logger.info(f"Loaded {len(self.top_k_popular_items)} popular items")
    
    def _load_data_stats(self) -> None:
        """Load data statistics."""
        stats_path = self.data_dir / 'data_stats.json'
        
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.data_stats = json.load(f)
    
    def load_item_metadata(self) -> pd.DataFrame:
        """
        Load product metadata for enrichment.
        
        Returns:
            pd.DataFrame: Products với columns [product_id, product_name, brand, ...]
        """
        # Try enriched products first
        enriched_path = self.data_dir / 'enriched_products.parquet'
        
        if enriched_path.exists():
            self.item_metadata = pd.read_parquet(enriched_path)
            logger.info(f"Loaded enriched product metadata: {len(self.item_metadata)} products")
            return self.item_metadata
        
        # Fallback to raw product data
        products_path = self.published_dir / 'data_product.csv'
        attributes_path = self.published_dir / 'data_product_attribute.csv'
        
        products = pd.read_csv(products_path, encoding='utf-8')
        
        if attributes_path.exists():
            attributes = pd.read_csv(attributes_path, encoding='utf-8')
            # Merge on product_id
            self.item_metadata = products.merge(
                attributes, on='product_id', how='left'
            )
        else:
            self.item_metadata = products
        
        logger.info(f"Loaded product metadata: {len(self.item_metadata)} products")
        return self.item_metadata
    
    def load_user_histories(self) -> Dict[int, Set[int]]:
        """
        Preload user → product interactions for seen-item filtering.
        
        IMPORTANT: Only loads TRAIN split to avoid data leakage.
        
        Returns:
            dict: {user_id: set(product_ids)}
        """
        if self.user_history_cache is not None:
            return self.user_history_cache
        
        interactions_path = self.data_dir / 'interactions.parquet'
        
        if not interactions_path.exists():
            logger.warning(f"Interactions file not found: {interactions_path}")
            self.user_history_cache = {}
            return self.user_history_cache
        
        import time
        start = time.perf_counter()
        
        interactions = pd.read_parquet(
            interactions_path,
            columns=['user_id', 'product_id', 'split']
        )
        
        # CRITICAL: Only use train split for history (avoid data leakage!)
        train_interactions = interactions[interactions['split'] == 'train']
        
        # Group by user_id
        self.user_history_cache = (
            train_interactions.groupby('user_id')['product_id']
            .apply(lambda x: set(x.tolist()))
            .to_dict()
        )
        
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Loaded user histories for {len(self.user_history_cache)} users in {elapsed:.1f}ms (train only)")
        
        return self.user_history_cache
    
    def is_trainable_user(self, user_id: int) -> bool:
        """
        Check if user is trainable (has ≥2 interactions + ≥1 positive).
        
        Args:
            user_id: Original user ID
        
        Returns:
            bool: True if user is trainable (can use CF)
        """
        if self.mappings is None:
            self.load_mappings()
        
        # Get u_idx from user_id
        u_idx = self.mappings['user_to_idx'].get(str(user_id))
        
        if u_idx is None:
            return False
        
        # Check if u_idx is in trainable set
        return int(u_idx) in self.trainable_user_set
    
    def get_cf_user_index(self, user_id: int) -> Optional[int]:
        """
        Get CF matrix row index for user.
        
        Args:
            user_id: Original user ID
        
        Returns:
            int or None: CF matrix row index (u_idx_cf)
        """
        if self.mappings is None:
            self.load_mappings()
        
        # Get u_idx from user_id
        u_idx = self.mappings['user_to_idx'].get(str(user_id))
        
        if u_idx is None:
            return None
        
        # Get u_idx_cf from trainable mapping
        return self.trainable_user_mapping.get(int(u_idx))
    
    def get_user_history(self, user_id: int) -> Set[int]:
        """
        Get items user has interacted with.
        
        Args:
            user_id: Original user ID
        
        Returns:
            set: Set of product_ids
        """
        if self.user_history_cache is None:
            self.load_user_histories()
        
        return self.user_history_cache.get(user_id, set())
    
    def reload_if_updated(self) -> bool:
        """
        Check registry for updates and reload if current_best changed.
        
        Returns:
            bool: True if model was reloaded
        """
        try:
            registry = self._load_registry()
        except FileNotFoundError:
            return False
        
        new_best = registry.get('current_best')
        if not new_best:
            return False
        
        new_best_id = new_best['model_id']
        
        if new_best_id != self.current_model_id:
            logger.info(f"Registry updated: {self.current_model_id} -> {new_best_id}")
            self.load_model(new_best_id)
            return True
        
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get info about current loaded model."""
        if not self.current_model:
            return {}
        
        return {
            'model_id': self.current_model['model_id'],
            'model_type': self.current_model['model_type'],
            'num_users': self.current_model['U'].shape[0],
            'num_items': self.current_model['V'].shape[0],
            'factors': self.current_model['U'].shape[1],
            'loaded_at': self.current_model['loaded_at'],
            'score_range': self.current_model['score_range'],
        }
    
    def get_popular_items(self, topk: int = 50) -> List[int]:
        """Get top-K popular item indices."""
        if self.top_k_popular_items is None:
            self._load_top_k_popular()
        
        return self.top_k_popular_items[:topk]


# ============================================================================
# Convenience Functions
# ============================================================================

def get_loader(**kwargs) -> CFModelLoader:
    """Get singleton loader instance."""
    return CFModelLoader(**kwargs)


def reset_loader() -> None:
    """Reset singleton loader instance."""
    with CFModelLoader._lock:
        CFModelLoader._instance = None
