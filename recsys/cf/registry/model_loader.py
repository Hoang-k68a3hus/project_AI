"""
Model Loader Module for CF Models.

This module provides functionality to load models from the registry for serving:
- Load model embeddings and metadata
- Hot-reload models without service restart
- Cache management for efficient loading
- Thread-safe loading with locking

Example:
    >>> from recsys.cf.registry import ModelLoader
    >>> loader = ModelLoader()
    >>> U, V, metadata = loader.load_current_best()
    >>> loader.reload_model()  # Hot reload
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import json
import os
import logging
import threading
import pickle
from dataclasses import dataclass, field

from .registry import ModelRegistry, DEFAULT_REGISTRY_PATH

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ModelState:
    """Current loaded model state."""
    model_id: str
    model_type: str
    version: str
    U: np.ndarray  # User embeddings
    V: np.ndarray  # Item embeddings
    params: Dict[str, Any]
    metadata: Dict[str, Any]
    loaded_at: str
    path: str


@dataclass
class LoaderStats:
    """Model loader statistics."""
    total_loads: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    reload_count: int = 0
    last_load_time_ms: float = 0
    last_reload_at: Optional[str] = None


# ============================================================================
# Model Loader
# ============================================================================

class ModelLoader:
    """
    Model loader with caching and hot-reload support.
    
    Features:
    - Load models from registry
    - Cache embeddings in memory
    - Hot-reload without restart
    - Thread-safe operations
    
    Example:
        >>> loader = ModelLoader()
        >>> U, V, metadata = loader.load_current_best()
        >>> loader.reload_model()  # Hot reload on signal
    """
    
    def __init__(
        self,
        registry_path: str = DEFAULT_REGISTRY_PATH,
        cache_enabled: bool = True,
        auto_load: bool = True
    ):
        """
        Initialize model loader.
        
        Args:
            registry_path: Path to registry.json
            cache_enabled: Enable embedding caching
            auto_load: Auto-load current best on init
        """
        self.registry_path = registry_path
        self.cache_enabled = cache_enabled
        
        # State
        self._current_model: Optional[ModelState] = None
        self._cache: Dict[str, ModelState] = {}
        self._stats = LoaderStats()
        self._lock = threading.RLock()
        
        # Load registry
        try:
            self._registry = ModelRegistry(registry_path, auto_create=False)
        except FileNotFoundError:
            logger.warning(f"Registry not found at {registry_path}, creating empty registry")
            self._registry = ModelRegistry(registry_path, auto_create=True)
        
        # Auto-load
        if auto_load:
            try:
                self.load_current_best()
            except (ValueError, KeyError) as e:
                logger.warning(f"Auto-load failed (no models in registry): {e}")
            except Exception as e:
                logger.warning(f"Auto-load failed: {e}")
    
    def _load_model_files(
        self,
        model_path: str,
        model_type: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """
        Load model files from disk.
        
        Args:
            model_path: Path to model folder
            model_type: Type of model
        
        Returns:
            Tuple of (U, V, params, metadata)
        
        Raises:
            FileNotFoundError: If required files are missing
        """
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        prefix = model_type
        
        # Load embeddings
        u_file = path / f"{prefix}_U.npy"
        v_file = path / f"{prefix}_V.npy"
        
        if not u_file.exists():
            raise FileNotFoundError(f"Missing file: {u_file}")
        if not v_file.exists():
            raise FileNotFoundError(f"Missing file: {v_file}")
        
        U = np.load(u_file)
        V = np.load(v_file)
        
        # Load params
        params_file = path / f"{prefix}_params.json"
        if not params_file.exists():
            raise FileNotFoundError(f"Missing file: {params_file}")
        
        with open(params_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        # Load metadata
        metadata_file = path / f"{prefix}_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Missing file: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return U, V, params, metadata
    
    def load_model(self, model_id: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load a specific model by ID.
        
        Args:
            model_id: Model identifier
        
        Returns:
            Tuple of (U, V, metadata)
        """
        import time
        start_time = time.perf_counter()
        
        with self._lock:
            # Check cache
            if self.cache_enabled and model_id in self._cache:
                self._stats.cache_hits += 1
                state = self._cache[model_id]
                logger.debug(f"Cache hit for {model_id}")
                return state.U, state.V, state.metadata
            
            self._stats.cache_misses += 1
            
            # Get model info from registry
            model_info = self._registry.get_model(model_id)
            if not model_info:
                raise KeyError(f"Model not found in registry: {model_id}")
            
            model_path = model_info['path']
            model_type = model_info['model_type']
            version = model_info['version']
            
            # Load files
            U, V, params, metadata = self._load_model_files(model_path, model_type)
            
            # Create state
            state = ModelState(
                model_id=model_id,
                model_type=model_type,
                version=version,
                U=U,
                V=V,
                params=params,
                metadata=metadata,
                loaded_at=datetime.now().isoformat(),
                path=model_path
            )
            
            # Update cache
            if self.cache_enabled:
                self._cache[model_id] = state
            
            self._current_model = state
            self._stats.total_loads += 1
            
            # Stats
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.last_load_time_ms = elapsed
            
            logger.info(f"Loaded model {model_id} in {elapsed:.1f}ms")
            
            return U, V, metadata
    
    def load_current_best(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load current best model from registry.
        
        Returns:
            Tuple of (U, V, metadata)
        """
        best = self._registry.get_current_best()
        if not best:
            raise ValueError("No current best model in registry")
        
        model_id = best['model_id']
        return self.load_model(model_id)
    
    def reload_model(self) -> bool:
        """
        Reload current model (hot reload).
        
        Useful for picking up new best model without restart.
        
        Returns:
            True if model changed
        """
        with self._lock:
            old_model_id = self._current_model.model_id if self._current_model else None
            
            # Clear cache
            self._cache.clear()
            
            # Reload registry
            self._registry = ModelRegistry(self.registry_path, auto_create=False)
            
            # Load current best
            self.load_current_best()
            
            new_model_id = self._current_model.model_id if self._current_model else None
            
            # Stats
            self._stats.reload_count += 1
            self._stats.last_reload_at = datetime.now().isoformat()
            
            model_changed = old_model_id != new_model_id
            
            if model_changed:
                logger.info(f"Model changed: {old_model_id} -> {new_model_id}")
            else:
                logger.info(f"Model reloaded (same version): {new_model_id}")
            
            return model_changed
    
    def get_current_model(self) -> Optional[ModelState]:
        """Get current loaded model state."""
        return self._current_model
    
    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current model embeddings.
        
        Returns:
            Tuple of (U, V) embeddings
        """
        if not self._current_model:
            raise ValueError("No model loaded")
        
        return self._current_model.U, self._current_model.V
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get info about current loaded model."""
        if not self._current_model:
            return {}
        
        return {
            'model_id': self._current_model.model_id,
            'model_type': self._current_model.model_type,
            'version': self._current_model.version,
            'num_users': self._current_model.U.shape[0],
            'num_items': self._current_model.V.shape[0],
            'factors': self._current_model.U.shape[1],
            'loaded_at': self._current_model.loaded_at,
            'path': self._current_model.path
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            'total_loads': self._stats.total_loads,
            'cache_hits': self._stats.cache_hits,
            'cache_misses': self._stats.cache_misses,
            'cache_hit_rate': (
                self._stats.cache_hits / 
                max(1, self._stats.cache_hits + self._stats.cache_misses)
            ),
            'reload_count': self._stats.reload_count,
            'last_load_time_ms': self._stats.last_load_time_ms,
            'last_reload_at': self._stats.last_reload_at,
            'cached_models': list(self._cache.keys())
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def preload_models(self, model_ids: List[str]) -> int:
        """
        Preload multiple models into cache.
        
        Args:
            model_ids: List of model IDs to preload
        
        Returns:
            Number of models loaded
        """
        loaded = 0
        for model_id in model_ids:
            try:
                self.load_model(model_id)
                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to preload {model_id}: {e}")
        
        logger.info(f"Preloaded {loaded}/{len(model_ids)} models")
        return loaded


# ============================================================================
# Singleton Pattern
# ============================================================================

_loader_instance: Optional[ModelLoader] = None
_loader_lock = threading.Lock()


def get_loader(
    registry_path: str = DEFAULT_REGISTRY_PATH,
    **kwargs
) -> ModelLoader:
    """
    Get singleton model loader instance.
    
    Args:
        registry_path: Path to registry
        **kwargs: Additional loader arguments
    
    Returns:
        ModelLoader instance
    """
    global _loader_instance
    
    with _loader_lock:
        if _loader_instance is None:
            _loader_instance = ModelLoader(registry_path, **kwargs)
        return _loader_instance


def reset_loader() -> None:
    """Reset singleton loader instance."""
    global _loader_instance
    
    with _loader_lock:
        _loader_instance = None


# ============================================================================
# Convenience Functions
# ============================================================================

def load_model_from_registry(
    model_id: Optional[str] = None,
    registry_path: str = DEFAULT_REGISTRY_PATH
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load model from registry (convenience function).
    
    Args:
        model_id: Model ID (None = current best)
        registry_path: Path to registry
    
    Returns:
        Tuple of (U, V, metadata)
    """
    loader = get_loader(registry_path, auto_load=False)
    
    if model_id:
        return loader.load_model(model_id)
    else:
        return loader.load_current_best()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Model Loader")
    print("=" * 60)
    
    # Test without actual registry
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, 'registry.json')
        
        # Create empty registry first
        registry = ModelRegistry(registry_path)
        
        # Create loader
        loader = ModelLoader(registry_path, auto_load=False)
        
        print(f"Loader created")
        print(f"Current model: {loader.get_current_model()}")
        print(f"Stats: {loader.get_stats()}")
        
        # Test without models - should handle gracefully
        try:
            loader.load_current_best()
        except ValueError as e:
            print(f"Expected error: {e}")
    
    print("\n" + "=" * 60)
    print("Model loader tests passed!")
