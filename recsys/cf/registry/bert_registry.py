"""
BERT Embeddings Registry Module.

This module provides registry functionality for BERT/PhoBERT embeddings:
- Track embedding versions
- Manage embedding files
- Link embeddings to CF models

Example:
    >>> from recsys.cf.registry import BERTEmbeddingsRegistry
    >>> bert_registry = BERTEmbeddingsRegistry()
    >>> bert_registry.register_embeddings('data/embeddings/v1', metadata)
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import json
import os
import hashlib
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_BERT_REGISTRY_PATH = "artifacts/cf/bert_registry.json"

EMBEDDING_TYPES = {
    'product': 'product_embeddings',
    'user': 'user_embeddings',
    'merged': 'merged_embeddings'
}


# ============================================================================
# BERT Embeddings Registry
# ============================================================================

class BERTEmbeddingsRegistry:
    """
    Registry for managing BERT/PhoBERT embeddings.
    
    Features:
    - Track embedding versions
    - Store generation metadata
    - Link to CF models
    - Version comparison
    
    Example:
        >>> registry = BERTEmbeddingsRegistry()
        >>> registry.register_embeddings(
        ...     embedding_path='data/embeddings/v1',
        ...     model_name='vinai/phobert-base',
        ...     num_items=1423,
        ...     embedding_dim=768
        ... )
    """
    
    def __init__(
        self,
        registry_path: str = DEFAULT_BERT_REGISTRY_PATH,
        auto_create: bool = True
    ):
        """
        Initialize BERT embeddings registry.
        
        Args:
            registry_path: Path to bert_registry.json
            auto_create: Create registry if doesn't exist
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.registry_path.exists():
            self._registry = self._load_registry()
        elif auto_create:
            self._registry = self._create_empty_registry()
            self._save_registry()
            logger.info(f"Created BERT embeddings registry at {registry_path}")
        else:
            raise FileNotFoundError(f"Registry not found: {registry_path}")
    
    def _create_empty_registry(self) -> Dict:
        """Create empty BERT registry structure."""
        return {
            'current_best': None,
            'embeddings': {},
            'metadata': {
                'registry_version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'num_embeddings': 0
            }
        }
    
    def _load_registry(self) -> Dict:
        """Load registry from file."""
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        self._registry['metadata']['last_updated'] = datetime.now().isoformat()
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self._registry, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def generate_version_id(prefix: str = "bert") -> str:
        """Generate version identifier."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"
    
    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute MD5 hash of embedding file."""
        hasher = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def validate_embedding_files(
        self,
        embedding_path: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate that embedding files exist.
        
        Args:
            embedding_path: Path to embeddings folder
        
        Returns:
            Tuple of (is_valid, missing_files)
        """
        path = Path(embedding_path)
        
        if not path.exists():
            return False, [str(embedding_path)]
        
        # Check for at least one embedding file
        expected_files = [
            'product_embeddings.pt',
            'product_embeddings.npy',
            'item_embeddings.pt',
            'item_embeddings.npy'
        ]
        
        found = False
        for f in expected_files:
            if (path / f).exists():
                found = True
                break
        
        if not found:
            return False, expected_files
        
        return True, []
    
    def register_embeddings(
        self,
        embedding_path: str,
        model_name: str,
        num_items: int,
        embedding_dim: int,
        generation_config: Optional[Dict] = None,
        text_fields_used: Optional[List[str]] = None,
        data_version: Optional[str] = None,
        git_commit: Optional[str] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Register BERT embeddings.
        
        Args:
            embedding_path: Path to embedding files
            model_name: BERT model name (e.g., 'vinai/phobert-base')
            num_items: Number of items embedded
            embedding_dim: Embedding dimension (e.g., 768)
            generation_config: Generation parameters
            text_fields_used: Text fields used (e.g., ['name', 'description'])
            data_version: Data version hash
            git_commit: Git commit hash
            version: Version string (auto-generated if None)
        
        Returns:
            str: Embedding version ID
        """
        # Validate
        is_valid, missing = self.validate_embedding_files(embedding_path)
        if not is_valid:
            raise ValueError(f"Invalid embedding path or missing files: {missing}")
        
        # Generate version
        version = version or self.generate_version_id()
        
        # Create entry
        entry = {
            'version': version,
            'path': str(embedding_path),
            'created_at': datetime.now().isoformat(),
            'model_name': model_name,
            'num_items': num_items,
            'embedding_dim': embedding_dim,
            'data_version': data_version or 'unknown',
            'git_commit': git_commit,
            'generation_config': generation_config or {},
            'text_fields_used': text_fields_used or ['product_name', 'description'],
            'linked_models': []  # CF models using these embeddings
        }
        
        # Register
        self._registry['embeddings'][version] = entry
        self._registry['metadata']['num_embeddings'] = len(self._registry['embeddings'])
        
        # Set as current if first
        if self._registry['current_best'] is None:
            self._registry['current_best'] = version
        
        self._save_registry()
        
        logger.info(f"Registered BERT embeddings: {version}")
        return version
    
    def get_embeddings(self, version: str) -> Optional[Dict]:
        """Get embeddings info by version."""
        return self._registry['embeddings'].get(version)
    
    def get_current_best(self) -> Optional[Dict]:
        """Get current best embeddings."""
        version = self._registry.get('current_best')
        if not version:
            return None
        
        return {
            'version': version,
            'info': self._registry['embeddings'].get(version)
        }
    
    def set_current_best(self, version: str) -> bool:
        """
        Set current best embeddings version.
        
        Args:
            version: Version to set as best
        
        Returns:
            True if successful
        """
        if version not in self._registry['embeddings']:
            raise KeyError(f"Embeddings version not found: {version}")
        
        self._registry['current_best'] = version
        self._save_registry()
        
        logger.info(f"Set current best BERT embeddings: {version}")
        return True
    
    def link_to_model(self, embedding_version: str, model_id: str) -> bool:
        """
        Link embeddings to a CF model.
        
        Args:
            embedding_version: BERT embeddings version
            model_id: CF model ID using these embeddings
        
        Returns:
            True if successful
        """
        if embedding_version not in self._registry['embeddings']:
            raise KeyError(f"Embeddings not found: {embedding_version}")
        
        linked = self._registry['embeddings'][embedding_version]['linked_models']
        if model_id not in linked:
            linked.append(model_id)
            self._save_registry()
        
        return True
    
    def list_embeddings(self, include_linked_models: bool = True) -> List[Dict]:
        """
        List all registered embeddings.
        
        Args:
            include_linked_models: Include linked CF models
        
        Returns:
            List of embedding info dicts
        """
        result = []
        current = self._registry.get('current_best')
        
        for version, info in self._registry['embeddings'].items():
            entry = {
                'version': version,
                'model_name': info['model_name'],
                'num_items': info['num_items'],
                'embedding_dim': info['embedding_dim'],
                'created_at': info['created_at'],
                'is_current': version == current
            }
            
            if include_linked_models:
                entry['linked_models'] = info.get('linked_models', [])
            
            result.append(entry)
        
        # Sort by created_at descending
        result.sort(key=lambda x: x['created_at'], reverse=True)
        
        return result
    
    def delete_embeddings(
        self,
        version: str,
        delete_files: bool = False
    ) -> bool:
        """
        Delete embeddings from registry.
        
        Args:
            version: Version to delete
            delete_files: Also delete embedding files
        
        Returns:
            True if successful
        """
        if version not in self._registry['embeddings']:
            raise KeyError(f"Embeddings not found: {version}")
        
        if version == self._registry.get('current_best'):
            raise ValueError("Cannot delete current best embeddings")
        
        # Check for linked models
        linked = self._registry['embeddings'][version].get('linked_models', [])
        if linked:
            raise ValueError(f"Cannot delete embeddings linked to models: {linked}")
        
        embedding_path = self._registry['embeddings'][version]['path']
        
        # Delete from registry
        del self._registry['embeddings'][version]
        self._registry['metadata']['num_embeddings'] = len(self._registry['embeddings'])
        self._save_registry()
        
        # Delete files
        if delete_files and os.path.exists(embedding_path):
            import shutil
            shutil.rmtree(embedding_path)
            logger.info(f"Deleted embedding files: {embedding_path}")
        
        logger.info(f"Deleted embeddings: {version}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        embeddings = self._registry['embeddings']
        
        stats = {
            'total_embeddings': len(embeddings),
            'current_best': self._registry.get('current_best'),
            'last_updated': self._registry['metadata']['last_updated']
        }
        
        # Models using embeddings
        all_linked = set()
        for emb in embeddings.values():
            all_linked.update(emb.get('linked_models', []))
        stats['total_linked_models'] = len(all_linked)
        
        return stats


# ============================================================================
# Convenience Functions
# ============================================================================

def get_bert_registry(
    registry_path: str = DEFAULT_BERT_REGISTRY_PATH
) -> BERTEmbeddingsRegistry:
    """Get or create BERT embeddings registry."""
    return BERTEmbeddingsRegistry(registry_path)


def load_bert_embeddings(
    version: Optional[str] = None,
    registry_path: str = DEFAULT_BERT_REGISTRY_PATH
) -> Tuple[np.ndarray, Dict]:
    """
    Load BERT embeddings from registry.
    
    Args:
        version: Version to load (None = current best)
        registry_path: Path to registry
    
    Returns:
        Tuple of (embeddings_array, metadata)
    """
    import torch
    
    registry = BERTEmbeddingsRegistry(registry_path, auto_create=False)
    
    if version:
        info = registry.get_embeddings(version)
    else:
        best = registry.get_current_best()
        if not best:
            raise ValueError("No current best embeddings")
        info = best['info']
    
    if not info:
        raise ValueError("Embeddings not found")
    
    path = Path(info['path'])
    
    # Try different file formats
    if (path / 'product_embeddings.pt').exists():
        embeddings = torch.load(path / 'product_embeddings.pt', map_location='cpu', weights_only=False)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
    elif (path / 'product_embeddings.npy').exists():
        embeddings = np.load(path / 'product_embeddings.npy')
    elif (path / 'item_embeddings.pt').exists():
        embeddings = torch.load(path / 'item_embeddings.pt', map_location='cpu', weights_only=False)
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()
    elif (path / 'item_embeddings.npy').exists():
        embeddings = np.load(path / 'item_embeddings.npy')
    else:
        raise FileNotFoundError(f"No embedding files found in {path}")
    
    return embeddings, info


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing BERT Embeddings Registry")
    print("=" * 60)
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, 'bert_registry.json')
        
        # Create registry
        registry = BERTEmbeddingsRegistry(registry_path)
        
        print(f"Registry created")
        print(f"Stats: {registry.get_stats()}")
        print(f"Current best: {registry.get_current_best()}")
        
        # List (empty)
        embeddings = registry.list_embeddings()
        print(f"Embeddings list: {embeddings}")
    
    print("\n" + "=" * 60)
    print("BERT registry tests passed!")
