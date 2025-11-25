"""
Model Registry Module for CF Models.

This module provides a centralized registry for managing CF model versions:
- Register new models with metadata
- Select best model based on metrics
- Track model lineage and versioning
- Audit trail for all registry operations

Example:
    >>> from recsys.cf.registry import ModelRegistry
    >>> registry = ModelRegistry()
    >>> registry.register_model('artifacts/cf/als/v1_20250115', metadata)
    >>> best = registry.select_best_model(metric='ndcg@10')
"""

from typing import Dict, List, Set, Optional, Union, Any, Tuple
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import json
import os
import shutil
import logging
import subprocess

logger = logging.getLogger(__name__)


# ============================================================================
# Registry Constants
# ============================================================================

DEFAULT_REGISTRY_PATH = "artifacts/cf/registry.json"
REQUIRED_MODEL_FILES = {
    'als': ['als_U.npy', 'als_V.npy', 'als_params.json', 'als_metadata.json'],
    'bpr': ['bpr_U.npy', 'bpr_V.npy', 'bpr_params.json', 'bpr_metadata.json'],
    'bert_als': ['bert_als_U.npy', 'bert_als_V.npy', 'bert_als_params.json', 'bert_als_metadata.json'],
}

MODEL_STATUS = {
    'active': 'active',
    'archived': 'archived',
    'failed': 'failed',
}


# ============================================================================
# Registry Schema
# ============================================================================

def create_empty_registry() -> Dict[str, Any]:
    """Create empty registry structure."""
    return {
        'current_best': None,
        'models': {},
        'bert_embeddings': {},
        'metadata': {
            'registry_version': '1.1',
            'last_updated': datetime.now().isoformat(),
            'num_models': 0,
            'num_embeddings': 0,
            'selection_criteria': 'ndcg@10'
        }
    }


def validate_registry_schema(registry: Dict) -> bool:
    """Validate registry structure."""
    required_keys = ['current_best', 'models', 'metadata']
    for key in required_keys:
        if key not in registry:
            return False
    return True


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """
    Central registry for CF model management.
    
    Features:
    - Register new models with metadata
    - Select best model based on metrics
    - List and compare models
    - Archive and delete models
    - Audit trail logging
    
    Example:
        >>> registry = ModelRegistry('artifacts/cf/registry.json')
        >>> registry.register_model('artifacts/cf/als/v1_20250115', {...})
        >>> best = registry.select_best_model(metric='ndcg@10')
    """
    
    def __init__(
        self,
        registry_path: str = DEFAULT_REGISTRY_PATH,
        auto_create: bool = True,
        audit_log_path: Optional[str] = None
    ):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry.json file
            auto_create: Create registry if doesn't exist
            audit_log_path: Path to audit log file (default: logs/registry_audit.log)
        """
        self.registry_path = Path(registry_path)
        self.audit_log_path = Path(audit_log_path or "logs/registry_audit.log")
        
        # Create parent directories
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load or create registry
        if self.registry_path.exists():
            self._registry = self._load_registry()
        elif auto_create:
            self._registry = create_empty_registry()
            self._save_registry()
            logger.info(f"Created new registry at {registry_path}")
        else:
            raise FileNotFoundError(f"Registry not found: {registry_path}")
    
    def _load_registry(self) -> Dict:
        """Load registry from JSON file."""
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
        
        if not validate_registry_schema(registry):
            raise ValueError("Invalid registry schema")
        
        return registry
    
    def _save_registry(self) -> None:
        """Save registry to JSON file."""
        self._registry['metadata']['last_updated'] = datetime.now().isoformat()
        
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self._registry, f, indent=2, ensure_ascii=False)
    
    def _audit_log(self, action: str, model_id: str, details: str = "") -> None:
        """Write entry to audit log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp} | {action} | {model_id} | {details}\n"
        
        with open(self.audit_log_path, 'a', encoding='utf-8') as f:
            f.write(entry)
    
    @staticmethod
    def generate_version_id(prefix: str = "v") -> str:
        """
        Generate version identifier.
        
        Format: v{N}_{YYYYMMDD}_{HHMMSS}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}"
    
    @staticmethod
    def generate_model_id(model_type: str, version: str) -> str:
        """Generate model identifier: {type}_{version}."""
        return f"{model_type}_{version}"
    
    @staticmethod
    def get_git_commit() -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def validate_artifacts(
        self,
        artifacts_path: str,
        model_type: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate that all required model files exist.
        
        Args:
            artifacts_path: Path to model artifacts folder
            model_type: Type of model (als, bpr, bert_als)
        
        Returns:
            Tuple of (is_valid, missing_files)
        """
        path = Path(artifacts_path)
        
        if model_type not in REQUIRED_MODEL_FILES:
            return False, [f"Unknown model type: {model_type}"]
        
        required = REQUIRED_MODEL_FILES[model_type]
        missing = []
        
        for file in required:
            if not (path / file).exists():
                missing.append(file)
        
        return len(missing) == 0, missing
    
    def register_model(
        self,
        artifacts_path: str,
        model_type: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        training_info: Dict[str, Any],
        data_version: Optional[str] = None,
        git_commit: Optional[str] = None,
        baseline_comparison: Optional[Dict[str, float]] = None,
        bert_integration: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        overwrite: bool = False
    ) -> str:
        """
        Register a new model in the registry.
        
        Args:
            artifacts_path: Path to model artifacts folder
            model_type: Type of model (als, bpr, bert_als)
            hyperparameters: Model hyperparameters dict
            metrics: Evaluation metrics dict
            training_info: Training information dict
            data_version: Data version hash
            git_commit: Git commit hash (auto-detected if None)
            baseline_comparison: Comparison with baseline
            bert_integration: BERT integration info (for hybrid models)
            version: Version string (auto-generated if None)
            overwrite: Overwrite if model_id exists
        
        Returns:
            str: Registered model_id
        
        Raises:
            ValueError: If artifacts invalid or model_id exists without overwrite
        """
        # Validate artifacts
        is_valid, missing = self.validate_artifacts(artifacts_path, model_type)
        if not is_valid:
            raise ValueError(f"Missing required files: {missing}")
        
        # Generate version and model_id
        version = version or self.generate_version_id()
        model_id = self.generate_model_id(model_type, version)
        
        # Check for duplicates
        if model_id in self._registry['models'] and not overwrite:
            logger.warning(f"Model {model_id} already exists, skipping")
            return model_id
        
        # Get git commit
        git_commit = git_commit or self.get_git_commit()
        
        # Create model entry
        model_entry = {
            'model_type': model_type,
            'version': version,
            'path': str(artifacts_path),
            'created_at': datetime.now().isoformat(),
            'data_version': data_version or 'unknown',
            'git_commit': git_commit,
            'hyperparameters': hyperparameters,
            'metrics': metrics,
            'training_info': training_info,
            'status': MODEL_STATUS['active']
        }
        
        # Add optional fields
        if baseline_comparison:
            model_entry['baseline_comparison'] = baseline_comparison
        
        if bert_integration:
            model_entry['bert_integration'] = bert_integration
        
        # Register
        self._registry['models'][model_id] = model_entry
        self._registry['metadata']['num_models'] = len(self._registry['models'])
        
        # Save
        self._save_registry()
        
        # Audit
        metric_str = f"ndcg@10={metrics.get('ndcg@10', 'N/A')}"
        self._audit_log('REGISTER', model_id, metric_str)
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def select_best_model(
        self,
        metric: str = 'ndcg@10',
        min_improvement: float = 0.0,
        model_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Select best model based on metric.
        
        Args:
            metric: Metric to optimize (e.g., 'ndcg@10', 'recall@10')
            min_improvement: Minimum improvement over baseline (0.1 = 10%)
            model_type: Filter by model type (None = all)
        
        Returns:
            Dict with best model info or None if no valid models
        """
        # Filter active models
        candidates = []
        for model_id, model in self._registry['models'].items():
            if model['status'] != MODEL_STATUS['active']:
                continue
            
            if model_type and model['model_type'] != model_type:
                continue
            
            # Check baseline improvement
            if min_improvement > 0 and 'baseline_comparison' in model:
                baseline_key = f"improvement_{metric.replace('@', '')}"
                if model['baseline_comparison'].get(baseline_key, 0) < min_improvement:
                    continue
            
            # Get metric value
            metric_value = model['metrics'].get(metric, 0)
            candidates.append((model_id, metric_value, model))
        
        if not candidates:
            logger.warning("No valid candidates for best model selection")
            return None
        
        # Sort by metric (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_id, best_value, best_model = candidates[0]
        
        # Get previous best for comparison
        prev_best = self._registry.get('current_best')
        prev_value = None
        if prev_best:
            prev_value = self._registry['models'].get(
                prev_best['model_id'], {}
            ).get('metrics', {}).get(metric)
        
        # Update current_best
        self._registry['current_best'] = {
            'model_id': best_id,
            'model_type': best_model['model_type'],
            'version': best_model['version'],
            'path': best_model['path'],
            'selection_metric': metric,
            'selection_value': best_value,
            'selected_at': datetime.now().isoformat(),
            'selected_by': 'auto'
        }
        
        # Update selection criteria
        self._registry['metadata']['selection_criteria'] = metric
        
        # Save
        self._save_registry()
        
        # Audit
        improvement = ""
        if prev_value:
            pct = ((best_value - prev_value) / prev_value) * 100
            improvement = f"improvement=+{pct:.1f}%"
        self._audit_log('SELECT_BEST', best_id, f"{metric}={best_value:.4f} {improvement}")
        
        logger.info(f"Selected best model: {best_id} ({metric}={best_value:.4f})")
        
        return {
            'model_id': best_id,
            'model_info': best_model,
            'metric': metric,
            'value': best_value
        }
    
    def get_current_best(self) -> Optional[Dict[str, Any]]:
        """Get current best model info."""
        best = self._registry.get('current_best')
        if not best:
            return None
        
        model_id = best['model_id']
        return {
            'model_id': model_id,
            'model_info': self._registry['models'].get(model_id),
            **best
        }
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model info by ID."""
        return self._registry['models'].get(model_id)
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        sort_by: str = 'created_at',
        ascending: bool = False
    ) -> 'pd.DataFrame':
        """
        List models with filtering and sorting.
        
        Args:
            model_type: Filter by type (als, bpr, bert_als)
            status: Filter by status (active, archived, failed)
            sort_by: Column to sort by
            ascending: Sort order
        
        Returns:
            DataFrame with model info
        """
        import pandas as pd
        
        rows = []
        for model_id, model in self._registry['models'].items():
            # Apply filters
            if model_type and model['model_type'] != model_type:
                continue
            if status and model['status'] != status:
                continue
            
            row = {
                'model_id': model_id,
                'model_type': model['model_type'],
                'version': model['version'],
                'status': model['status'],
                'created_at': model['created_at'],
            }
            
            # Add metrics
            for metric, value in model.get('metrics', {}).items():
                row[metric] = value
            
            # Add training info
            row['training_time'] = model.get('training_info', {}).get('training_time_seconds', 0)
            
            rows.append(row)
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        
        return df
    
    def compare_models(
        self,
        model_ids: List[str],
        metrics: List[str] = ['recall@10', 'ndcg@10', 'coverage']
    ) -> 'pd.DataFrame':
        """
        Compare multiple models side-by-side.
        
        Args:
            model_ids: List of model IDs to compare
            metrics: Metrics to include
        
        Returns:
            DataFrame with comparison
        """
        import pandas as pd
        
        rows = []
        for model_id in model_ids:
            model = self._registry['models'].get(model_id)
            if not model:
                logger.warning(f"Model {model_id} not found")
                continue
            
            row = {
                'model_id': model_id,
                'model_type': model['model_type'],
            }
            
            # Add hyperparameters
            row['factors'] = model.get('hyperparameters', {}).get('factors', 'N/A')
            
            # Add metrics
            for metric in metrics:
                row[metric] = model.get('metrics', {}).get(metric, None)
            
            # Training time
            row['training_time'] = model.get('training_info', {}).get('training_time_seconds', 0)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def archive_model(self, model_id: str) -> bool:
        """
        Archive a model (mark as archived).
        
        Args:
            model_id: Model to archive
        
        Returns:
            True if successful
        """
        if model_id not in self._registry['models']:
            raise KeyError(f"Model not found: {model_id}")
        
        # Check if it's current best
        current_best = self._registry.get('current_best', {})
        if current_best and current_best.get('model_id') == model_id:
            logger.warning(f"Cannot archive current best model: {model_id}")
            return False
        
        # Archive
        self._registry['models'][model_id]['status'] = MODEL_STATUS['archived']
        self._save_registry()
        
        self._audit_log('ARCHIVE', model_id, 'reason=manual')
        logger.info(f"Archived model: {model_id}")
        
        return True
    
    def delete_model(
        self,
        model_id: str,
        delete_files: bool = False
    ) -> bool:
        """
        Delete a model from registry.
        
        Args:
            model_id: Model to delete
            delete_files: Also delete model files
        
        Returns:
            True if successful
        """
        if model_id not in self._registry['models']:
            raise KeyError(f"Model not found: {model_id}")
        
        # Check if it's current best
        current_best = self._registry.get('current_best', {})
        if current_best and current_best.get('model_id') == model_id:
            raise ValueError(f"Cannot delete current best model: {model_id}")
        
        model = self._registry['models'][model_id]
        model_path = model['path']
        
        # Delete from registry
        del self._registry['models'][model_id]
        self._registry['metadata']['num_models'] = len(self._registry['models'])
        self._save_registry()
        
        # Delete files if requested
        if delete_files and os.path.exists(model_path):
            shutil.rmtree(model_path)
            logger.info(f"Deleted model files: {model_path}")
        
        self._audit_log('DELETE', model_id, f'delete_files={delete_files}')
        logger.info(f"Deleted model from registry: {model_id}")
        
        return True
    
    def update_model_status(
        self,
        model_id: str,
        status: str
    ) -> bool:
        """
        Update model status.
        
        Args:
            model_id: Model to update
            status: New status (active, archived, failed)
        
        Returns:
            True if successful
        """
        if model_id not in self._registry['models']:
            raise KeyError(f"Model not found: {model_id}")
        
        if status not in MODEL_STATUS:
            raise ValueError(f"Invalid status: {status}")
        
        self._registry['models'][model_id]['status'] = status
        self._save_registry()
        
        self._audit_log('UPDATE_STATUS', model_id, f'status={status}')
        return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        models = self._registry['models']
        
        # Handle current_best safely (can be None)
        current_best = self._registry.get('current_best')
        current_best_id = None
        if current_best and isinstance(current_best, dict):
            current_best_id = current_best.get('model_id')
        
        stats = {
            'total_models': len(models),
            'active_models': sum(1 for m in models.values() if m['status'] == 'active'),
            'archived_models': sum(1 for m in models.values() if m['status'] == 'archived'),
            'by_type': {},
            'current_best': current_best_id,
            'last_updated': self._registry['metadata']['last_updated']
        }
        
        # Count by type
        for model in models.values():
            mtype = model['model_type']
            stats['by_type'][mtype] = stats['by_type'].get(mtype, 0) + 1
        
        return stats


# ============================================================================
# Convenience Functions
# ============================================================================

def get_registry(registry_path: str = DEFAULT_REGISTRY_PATH) -> ModelRegistry:
    """Get or create model registry."""
    return ModelRegistry(registry_path)


def register_model(
    artifacts_path: str,
    model_type: str,
    hyperparameters: Dict,
    metrics: Dict,
    training_info: Dict,
    registry_path: str = DEFAULT_REGISTRY_PATH,
    **kwargs
) -> str:
    """
    Register a model (convenience function).
    
    Returns:
        str: Model ID
    """
    registry = ModelRegistry(registry_path)
    return registry.register_model(
        artifacts_path=artifacts_path,
        model_type=model_type,
        hyperparameters=hyperparameters,
        metrics=metrics,
        training_info=training_info,
        **kwargs
    )


def select_best_model(
    metric: str = 'ndcg@10',
    registry_path: str = DEFAULT_REGISTRY_PATH,
    **kwargs
) -> Optional[Dict]:
    """
    Select best model (convenience function).
    
    Returns:
        Best model info or None
    """
    registry = ModelRegistry(registry_path)
    return registry.select_best_model(metric=metric, **kwargs)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Model Registry")
    print("=" * 60)
    
    import tempfile
    
    # Create temp registry
    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, 'registry.json')
        
        # Initialize
        registry = ModelRegistry(registry_path)
        print(f"Created registry at: {registry_path}")
        
        # Check empty registry
        stats = registry.get_registry_stats()
        print(f"\nEmpty registry stats: {stats}")
        
        # Can't register without actual files, but test the schema
        print("\n--- Registry Schema ---")
        print(f"Registry version: {registry._registry['metadata']['registry_version']}")
        print(f"Current best: {registry.get_current_best()}")
        
        # List models (empty)
        models_df = registry.list_models()
        print(f"\nModels DataFrame shape: {models_df.shape}")
        
    print("\n" + "=" * 60)
    print("Registry tests passed!")
