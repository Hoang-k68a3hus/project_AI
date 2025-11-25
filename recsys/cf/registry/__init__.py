"""
CF Model Registry Module.

This module provides centralized management for CF model versions:
- Model registration and versioning
- Best model selection based on metrics
- Model loading for serving
- BERT embeddings management
- Audit trail and lineage tracking

Modules:
    registry: Main ModelRegistry class for model management
    model_loader: ModelLoader for loading models in serving
    bert_registry: BERTEmbeddingsRegistry for BERT embeddings
    utils: Utility functions (version generation, git, hashing)

Example:
    >>> from recsys.cf.registry import ModelRegistry, ModelLoader
    >>> 
    >>> # Register a model
    >>> registry = ModelRegistry()
    >>> model_id = registry.register_model(
    ...     artifacts_path='artifacts/cf/als/v1_20250115',
    ...     model_type='als',
    ...     hyperparameters={'factors': 64, 'regularization': 0.1},
    ...     metrics={'ndcg@10': 0.25, 'recall@10': 0.30},
    ...     training_info={'training_time_seconds': 120}
    ... )
    >>> 
    >>> # Select best model
    >>> best = registry.select_best_model(metric='ndcg@10')
    >>> 
    >>> # Load for serving
    >>> loader = ModelLoader()
    >>> U, V, metadata = loader.load_current_best()
    >>> loader.reload_model()  # Hot reload
"""

# Core registry classes
from .registry import (
    # Main class
    ModelRegistry,
    
    # Schema helpers
    create_empty_registry,
    validate_registry_schema,
    
    # Convenience functions
    get_registry,
    register_model,
    select_best_model,
    
    # Constants
    DEFAULT_REGISTRY_PATH,
    REQUIRED_MODEL_FILES,
    MODEL_STATUS,
)

# Model loader
from .model_loader import (
    # Main class
    ModelLoader,
    
    # Data classes
    ModelState,
    LoaderStats,
    
    # Singleton pattern
    get_loader,
    reset_loader,
    
    # Convenience functions
    load_model_from_registry,
)

# BERT embeddings registry
from .bert_registry import (
    # Main class
    BERTEmbeddingsRegistry,
    
    # Convenience functions
    get_bert_registry,
    load_bert_embeddings,
    
    # Constants
    DEFAULT_BERT_REGISTRY_PATH,
    EMBEDDING_TYPES,
)

# Utility functions
from .utils import (
    # Version management
    generate_version_id,
    parse_version_id,
    compare_versions,
    
    # Git integration
    get_git_commit,
    get_git_commit_short,
    get_git_branch,
    is_git_clean,
    
    # Hash computation
    compute_file_hash,
    compute_directory_hash,
    compute_data_version,
    
    # Path validation
    validate_model_path,
    ensure_directory,
    
    # Backup & migration
    backup_registry,
    restore_registry,
    list_backups,
    
    # Artifact utilities
    copy_model_artifacts,
    cleanup_old_artifacts,
    
    # Metadata utilities
    create_model_metadata,
    load_model_metadata,
    save_model_metadata,
)


__all__ = [
    # === Registry ===
    'ModelRegistry',
    'create_empty_registry',
    'validate_registry_schema',
    'get_registry',
    'register_model',
    'select_best_model',
    'DEFAULT_REGISTRY_PATH',
    'REQUIRED_MODEL_FILES',
    'MODEL_STATUS',
    
    # === Model Loader ===
    'ModelLoader',
    'ModelState',
    'LoaderStats',
    'get_loader',
    'reset_loader',
    'load_model_from_registry',
    
    # === BERT Registry ===
    'BERTEmbeddingsRegistry',
    'get_bert_registry',
    'load_bert_embeddings',
    'DEFAULT_BERT_REGISTRY_PATH',
    'EMBEDDING_TYPES',
    
    # === Version Utils ===
    'generate_version_id',
    'parse_version_id',
    'compare_versions',
    
    # === Git Utils ===
    'get_git_commit',
    'get_git_commit_short',
    'get_git_branch',
    'is_git_clean',
    
    # === Hash Utils ===
    'compute_file_hash',
    'compute_directory_hash',
    'compute_data_version',
    
    # === Path Utils ===
    'validate_model_path',
    'ensure_directory',
    
    # === Backup Utils ===
    'backup_registry',
    'restore_registry',
    'list_backups',
    
    # === Artifact Utils ===
    'copy_model_artifacts',
    'cleanup_old_artifacts',
    
    # === Metadata Utils ===
    'create_model_metadata',
    'load_model_metadata',
    'save_model_metadata',
]
