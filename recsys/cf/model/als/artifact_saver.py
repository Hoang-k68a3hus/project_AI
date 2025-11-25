"""
Step 7: Save Artifacts for ALS Model

This module handles persistence of all ALS training artifacts including:
- User and item embeddings (U, V matrices)
- Training parameters and hyperparameters
- Evaluation metrics and baseline comparisons
- Metadata with score ranges for Task 08 hybrid reranking

Key Components:
    1. ScoreRange: Dataclass for CF score statistics
    2. ALSArtifacts: Dataclass container for all artifacts
    3. compute_score_range(): Calculate score statistics on validation set
    4. save_*(): Individual save functions for each artifact type
    5. save_als_complete(): One-line orchestrator for complete saving

Critical Feature - Score Range Computation:
    - Runs U @ V.T on validation users to get real score distribution
    - Computes percentiles (p01, p99) for robust normalization
    - Essential for Task 08 hybrid reranking global normalization
    - Prevents score range mismatch between CF and content-based models

Usage:
    >>> from recsys.cf.model.als.artifact_saver import save_als_complete
    >>> 
    >>> # Save complete artifacts with score range
    >>> artifacts = save_als_complete(
    ...     user_embeddings=U,
    ...     item_embeddings=V,
    ...     params={'factors': 64, 'regularization': 0.01, ...},
    ...     metrics={'recall@10': 0.234, ...},
    ...     validation_user_indices=[10, 25, 42, ...],  # Validation users for score range
    ...     data_version_hash='abc123',
    ...     output_dir='artifacts/cf/als'
    ... )
    >>> 
    >>> print(f"Score range: [{artifacts.metadata['score_range']['p01']:.3f}, "
    ...       f"{artifacts.metadata['score_range']['p99']:.3f}]")

Author: Copilot
Date: 2025-01-15
"""

import numpy as np
import json
import pickle
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime
import platform
import psutil
import subprocess

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ScoreRange:
    """
    CF score range statistics for global normalization in Task 08.
    
    Computed by running U @ V.T on validation set and analyzing score distribution.
    
    Attributes:
        method: How scores were computed (e.g., 'validation_set', 'train_set')
        min: Minimum score observed
        max: Maximum score observed
        mean: Average score
        std: Standard deviation
        p01: 1st percentile (robust minimum)
        p99: 99th percentile (robust maximum)
        num_samples: Number of user-item pairs sampled
    
    Example:
        >>> score_range = ScoreRange(
        ...     method='validation_set',
        ...     min=0.0,
        ...     max=1.48,
        ...     mean=0.32,
        ...     std=0.21,
        ...     p01=0.01,
        ...     p99=1.12,
        ...     num_samples=50000
        ... )
    """
    method: str
    min: float
    max: float
    mean: float
    std: float
    p01: float
    p99: float
    num_samples: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ALSArtifacts:
    """
    Container for all ALS training artifacts.
    
    Attributes:
        model_type: 'als' or 'bpr'
        output_dir: Base directory where artifacts are saved
        embeddings_path: Path to saved embeddings
        params_path: Path to saved parameters
        metrics_path: Path to saved metrics
        metadata_path: Path to saved metadata
        model_path: Optional path to serialized model object
        params: Training parameters dictionary
        metrics: Evaluation metrics dictionary
        metadata: Metadata dictionary including score_range
    
    Example:
        >>> artifacts = ALSArtifacts(
        ...     model_type='als',
        ...     output_dir=Path('artifacts/cf/als'),
        ...     embeddings_path=Path('artifacts/cf/als/als_U.npy'),
        ...     params={'factors': 64},
        ...     metrics={'recall@10': 0.234},
        ...     metadata={'score_range': {...}}
        ... )
    """
    model_type: str
    output_dir: Path
    embeddings_path: Optional[Path] = None
    params_path: Optional[Path] = None
    metrics_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    model_path: Optional[Path] = None
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_type': self.model_type,
            'output_dir': str(self.output_dir),
            'embeddings_path': str(self.embeddings_path) if self.embeddings_path else None,
            'params_path': str(self.params_path) if self.params_path else None,
            'metrics_path': str(self.metrics_path) if self.metrics_path else None,
            'metadata_path': str(self.metadata_path) if self.metadata_path else None,
            'model_path': str(self.model_path) if self.model_path else None,
            'params': self.params,
            'metrics': self.metrics,
            'metadata': self.metadata
        }
    
    def summary(self) -> str:
        """Generate summary string of saved artifacts."""
        lines = [
            f"\n{'='*70}",
            f"ALS Artifacts Summary",
            f"{'='*70}",
            f"Model Type: {self.model_type}",
            f"Output Directory: {self.output_dir}",
            f"\nSaved Files:",
            f"  - Embeddings: {self.embeddings_path.name if self.embeddings_path else 'N/A'}",
            f"  - Parameters: {self.params_path.name if self.params_path else 'N/A'}",
            f"  - Metrics: {self.metrics_path.name if self.metrics_path else 'N/A'}",
            f"  - Metadata: {self.metadata_path.name if self.metadata_path else 'N/A'}",
            f"  - Model Object: {self.model_path.name if self.model_path else 'N/A'}",
        ]
        
        if 'score_range' in self.metadata:
            sr = self.metadata['score_range']
            lines.extend([
                f"\nScore Range (for Task 08 normalization):",
                f"  - Method: {sr.get('method', 'N/A')}",
                f"  - Range: [{sr.get('min', 0):.4f}, {sr.get('max', 0):.4f}]",
                f"  - Mean ± Std: {sr.get('mean', 0):.4f} ± {sr.get('std', 0):.4f}",
                f"  - Robust Range [p01, p99]: [{sr.get('p01', 0):.4f}, {sr.get('p99', 0):.4f}]",
                f"  - Samples: {sr.get('num_samples', 0):,}"
            ])
        
        if self.metrics:
            lines.append(f"\nKey Metrics:")
            for k, v in sorted(self.metrics.items())[:6]:  # Show first 6 metrics
                if isinstance(v, (int, float)):
                    lines.append(f"  - {k}: {v:.4f}")
                else:
                    lines.append(f"  - {k}: {v}")
        
        lines.append(f"{'='*70}\n")
        return '\n'.join(lines)


# ============================================================================
# Score Range Computation (Critical for Task 08)
# ============================================================================

def compute_score_range(
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    validation_user_indices: Optional[List[int]] = None,
    max_samples: int = 100000,
    method: str = 'validation_set'
) -> ScoreRange:
    """
    Compute CF score range statistics for global normalization in Task 08.
    
    This function runs U @ V.T on validation users and computes statistical
    properties of the score distribution. These statistics are essential for
    normalizing CF scores to [0, 1] range in the hybrid reranking system.
    
    Args:
        user_factors: User embedding matrix U (num_users, factors)
        item_factors: Item embedding matrix V (num_items, factors)
        validation_user_indices: List of validation user indices to sample scores from.
                                 If None, samples randomly from all users.
        max_samples: Maximum number of user-item pairs to sample (memory limit)
        method: Description of how validation set was selected
    
    Returns:
        ScoreRange object with min, max, mean, std, p01, p99
    
    Example:
        >>> # Compute on validation users
        >>> val_users = [10, 25, 42, 67, 89]  # Validation user indices
        >>> score_range = compute_score_range(U, V, val_users)
        >>> print(f"Score range: [{score_range.p01:.3f}, {score_range.p99:.3f}]")
        Score range: [0.012, 1.123]
        >>> 
        >>> # Use in Task 08 normalization
        >>> normalized_cf_score = (raw_score - score_range.p01) / (score_range.p99 - score_range.p01)
    
    Notes:
        - Uses percentiles (p01, p99) instead of min/max for robustness to outliers
        - Samples intelligently to avoid OOM: max_samples limits total pairs
        - Essential for fair comparison between CF and content-based scores in Task 08
        - Validation set preferred over training set to avoid overfitting bias
    """
    logger.info(f"Computing CF score range (method: {method})")
    
    num_users, factors = user_factors.shape
    num_items = item_factors.shape[0]
    
    # Determine which users to sample from
    if validation_user_indices is not None and len(validation_user_indices) > 0:
        user_indices = np.array(validation_user_indices)
        logger.info(f"Using {len(user_indices)} validation users")
    else:
        # Sample random users if no validation set provided
        sample_size = min(1000, num_users)
        user_indices = np.random.choice(num_users, sample_size, replace=False)
        logger.info(f"No validation users provided, sampling {sample_size} random users")
    
    # Compute scores batch by batch to avoid OOM
    num_val_users = len(user_indices)
    items_per_user = num_items
    total_pairs = num_val_users * items_per_user
    
    if total_pairs > max_samples:
        # Sample items for each user
        items_per_user = max_samples // num_val_users
        logger.info(f"Limiting to {items_per_user} items per user (total samples: {num_val_users * items_per_user:,})")
    
    scores_list = []
    
    for user_idx in user_indices:
        # Compute scores for this user
        user_scores = user_factors[user_idx] @ item_factors.T  # (num_items,)
        
        # Sample items if needed
        if items_per_user < num_items:
            item_sample = np.random.choice(num_items, items_per_user, replace=False)
            user_scores = user_scores[item_sample]
        
        scores_list.append(user_scores)
    
    # Concatenate all scores
    all_scores = np.concatenate(scores_list)
    
    logger.info(f"Collected {len(all_scores):,} score samples")
    
    # Compute statistics
    score_min = float(np.min(all_scores))
    score_max = float(np.max(all_scores))
    score_mean = float(np.mean(all_scores))
    score_std = float(np.std(all_scores))
    score_p01 = float(np.percentile(all_scores, 1))
    score_p99 = float(np.percentile(all_scores, 99))
    
    logger.info(f"Score statistics:")
    logger.info(f"  Min: {score_min:.4f}")
    logger.info(f"  Max: {score_max:.4f}")
    logger.info(f"  Mean: {score_mean:.4f}")
    logger.info(f"  Std: {score_std:.4f}")
    logger.info(f"  P01: {score_p01:.4f}")
    logger.info(f"  P99: {score_p99:.4f}")
    
    score_range = ScoreRange(
        method=method,
        min=score_min,
        max=score_max,
        mean=score_mean,
        std=score_std,
        p01=score_p01,
        p99=score_p99,
        num_samples=len(all_scores)
    )
    
    return score_range


# ============================================================================
# Individual Save Functions
# ============================================================================

def save_embeddings(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    output_dir: Path,
    model_type: str = 'als',
    prefix: Optional[str] = None
) -> Tuple[Path, Path]:
    """
    Save user and item embeddings as NumPy .npy files.
    
    Args:
        user_embeddings: User factors U (num_users, factors)
        item_embeddings: Item factors V (num_items, factors)
        output_dir: Directory to save embeddings
        model_type: 'als' or 'bpr'
        prefix: Optional prefix for filenames (default: model_type)
    
    Returns:
        Tuple of (user_embeddings_path, item_embeddings_path)
    
    Example:
        >>> U_path, V_path = save_embeddings(U, V, Path('artifacts/cf/als'))
        >>> print(f"Saved to {U_path} and {V_path}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = prefix or model_type
    
    # Save user embeddings
    user_path = output_dir / f"{prefix}_U.npy"
    np.save(user_path, user_embeddings)
    logger.info(f"Saved user embeddings: {user_path} (shape: {user_embeddings.shape})")
    
    # Save item embeddings
    item_path = output_dir / f"{prefix}_V.npy"
    np.save(item_path, item_embeddings)
    logger.info(f"Saved item embeddings: {item_path} (shape: {item_embeddings.shape})")
    
    return user_path, item_path


def save_params(
    params: Dict[str, Any],
    output_dir: Path,
    model_type: str = 'als'
) -> Path:
    """
    Save training parameters as JSON.
    
    Args:
        params: Dictionary with hyperparameters and training config
        output_dir: Directory to save params
        model_type: 'als' or 'bpr'
    
    Returns:
        Path to saved params.json
    
    Example:
        >>> params = {
        ...     'factors': 64,
        ...     'regularization': 0.01,
        ...     'iterations': 15,
        ...     'alpha': 40,
        ...     'random_seed': 42,
        ...     'training_time_seconds': 45.2
        ... }
        >>> path = save_params(params, Path('artifacts/cf/als'))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    params_path = output_dir / f"{model_type}_params.json"
    
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    logger.info(f"Saved parameters: {params_path}")
    
    return params_path


def save_metrics(
    metrics: Dict[str, Any],
    output_dir: Path,
    model_type: str = 'als'
) -> Path:
    """
    Save evaluation metrics as JSON.
    
    Args:
        metrics: Dictionary with evaluation results including baseline comparisons
        output_dir: Directory to save metrics
        model_type: 'als' or 'bpr'
    
    Returns:
        Path to saved metrics.json
    
    Example:
        >>> metrics = {
        ...     'recall@10': 0.234,
        ...     'recall@20': 0.312,
        ...     'ndcg@10': 0.189,
        ...     'ndcg@20': 0.221,
        ...     'baseline_recall@10': 0.145,
        ...     'baseline_ndcg@10': 0.102,
        ...     'improvement_recall@10': '+61.4%'
        ... }
        >>> path = save_metrics(metrics, Path('artifacts/cf/als'))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_dir / f"{model_type}_metrics.json"
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved metrics: {metrics_path}")
    
    return metrics_path


def save_metadata(
    metadata: Dict[str, Any],
    output_dir: Path,
    model_type: str = 'als',
    score_range: Optional[ScoreRange] = None,
    data_version_hash: Optional[str] = None,
    include_system_info: bool = True
) -> Path:
    """
    Save metadata including score range, timestamps, git commit, system info.
    
    Args:
        metadata: Base metadata dictionary (will be extended)
        output_dir: Directory to save metadata
        model_type: 'als' or 'bpr'
        score_range: ScoreRange object for Task 08 normalization (CRITICAL)
        data_version_hash: Data version hash from Task 01
        include_system_info: Whether to include CPU/GPU/memory info
    
    Returns:
        Path to saved metadata.json
    
    Example:
        >>> metadata = {'custom_field': 'value'}
        >>> score_range = compute_score_range(U, V, val_users)
        >>> path = save_metadata(
        ...     metadata,
        ...     Path('artifacts/cf/als'),
        ...     score_range=score_range,
        ...     data_version_hash='abc123'
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start with provided metadata
    full_metadata = metadata.copy()
    
    # Add timestamp
    full_metadata['timestamp'] = datetime.now().isoformat()
    
    # Add data version
    if data_version_hash:
        full_metadata['data_version_hash'] = data_version_hash
    
    # Add git commit (if in git repo)
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        full_metadata['git_commit'] = git_commit
        logger.info(f"Git commit: {git_commit[:8]}")
    except Exception as e:
        logger.warning(f"Could not get git commit: {e}")
        full_metadata['git_commit'] = None
    
    # Add system info
    if include_system_info:
        full_metadata['system_info'] = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'cpu_model': platform.processor() or 'Unknown'
        }
    
    # Add score range (CRITICAL for Task 08)
    if score_range:
        full_metadata['score_range'] = score_range.to_dict()
        logger.info(f"Added score range for Task 08 normalization: "
                   f"[{score_range.p01:.4f}, {score_range.p99:.4f}]")
    else:
        logger.warning("No score_range provided - Task 08 hybrid reranking will need manual normalization!")
    
    # Save
    metadata_path = output_dir / f"{model_type}_metadata.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(full_metadata, f, indent=2)
    
    logger.info(f"Saved metadata: {metadata_path}")
    
    return metadata_path


def save_model_object(
    model,
    output_dir: Path,
    model_type: str = 'als'
) -> Path:
    """
    Save serialized model object (optional, for implicit.als models).
    
    Args:
        model: Trained model object (e.g., AlternatingLeastSquares instance)
        output_dir: Directory to save model
        model_type: 'als' or 'bpr'
    
    Returns:
        Path to saved model.pkl
    
    Example:
        >>> from implicit.als import AlternatingLeastSquares
        >>> model = AlternatingLeastSquares(factors=64)
        >>> # ... train model ...
        >>> path = save_model_object(model, Path('artifacts/cf/als'))
    
    Notes:
        - This is optional; embeddings alone are sufficient for serving
        - Useful for reproducibility and debugging
        - BPR custom models may not be picklable
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / f"{model_type}_model.pkl"
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model object: {model_path}")
        return model_path
    except Exception as e:
        logger.warning(f"Could not serialize model object: {e}")
        return None


# ============================================================================
# Complete Artifact Saving (One-Line Orchestrator)
# ============================================================================

def save_als_complete(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: Union[str, Path],
    validation_user_indices: Optional[List[int]] = None,
    data_version_hash: Optional[str] = None,
    model_object=None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    model_type: str = 'als',
    compute_score_range_flag: bool = True
) -> ALSArtifacts:
    """
    Save all ALS artifacts in one call (embeddings, params, metrics, metadata).
    
    This is the recommended way to save all artifacts. It handles:
    - Embedding persistence (.npy files)
    - Parameter saving (JSON)
    - Metrics saving (JSON)
    - Metadata with score range (JSON) - CRITICAL for Task 08
    - Optional model object serialization
    
    Args:
        user_embeddings: User factors U
        item_embeddings: Item factors V
        params: Training parameters dictionary
        metrics: Evaluation metrics dictionary (from ALSEvaluator)
        output_dir: Directory to save all artifacts
        validation_user_indices: Validation users for score range computation
        data_version_hash: Data version hash from Task 01
        model_object: Optional trained model object to serialize
        additional_metadata: Additional metadata fields
        model_type: 'als' or 'bpr'
        compute_score_range_flag: Whether to compute score range (disable for testing)
    
    Returns:
        ALSArtifacts object with paths to all saved files
    
    Example:
        >>> # After training and evaluation
        >>> artifacts = save_als_complete(
        ...     user_embeddings=U,
        ...     item_embeddings=V,
        ...     params={'factors': 64, 'regularization': 0.01, 'iterations': 15},
        ...     metrics={'recall@10': 0.234, 'ndcg@10': 0.189},
        ...     output_dir='artifacts/cf/als',
        ...     validation_user_indices=[10, 25, 42, 67],
        ...     data_version_hash='abc123def456',
        ...     model_object=trained_als_model
        ... )
        >>> 
        >>> print(artifacts.summary())
        >>> # Artifacts saved to artifacts/cf/als/
        >>> # - als_U.npy, als_V.npy
        >>> # - als_params.json, als_metrics.json, als_metadata.json
        >>> # - Score range for Task 08: [0.012, 1.123]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Saving {model_type.upper()} artifacts to {output_dir}")
    logger.info(f"{'='*70}")
    
    # 1. Save embeddings
    user_path, item_path = save_embeddings(
        user_embeddings, item_embeddings, output_dir, model_type
    )
    
    # 2. Save params
    params_path = save_params(params, output_dir, model_type)
    
    # 3. Save metrics
    metrics_path = save_metrics(metrics, output_dir, model_type)
    
    # 4. Compute score range (CRITICAL for Task 08)
    score_range = None
    if compute_score_range_flag:
        logger.info("\nComputing CF score range for Task 08 hybrid normalization...")
        score_range = compute_score_range(
            user_embeddings,
            item_embeddings,
            validation_user_indices=validation_user_indices,
            method='validation_set' if validation_user_indices else 'random_sample'
        )
    else:
        logger.warning("Score range computation disabled - Task 08 will need manual normalization!")
    
    # 5. Save metadata with score range
    metadata = additional_metadata.copy() if additional_metadata else {}
    metadata['model_type'] = model_type
    metadata['num_users'] = user_embeddings.shape[0]
    metadata['num_items'] = item_embeddings.shape[0]
    metadata['factors'] = user_embeddings.shape[1]
    
    metadata_path = save_metadata(
        metadata,
        output_dir,
        model_type,
        score_range=score_range,
        data_version_hash=data_version_hash,
        include_system_info=True
    )
    
    # 6. Save model object (optional)
    model_path = None
    if model_object is not None:
        model_path = save_model_object(model_object, output_dir, model_type)
    
    # Create artifacts container
    artifacts = ALSArtifacts(
        model_type=model_type,
        output_dir=output_dir,
        embeddings_path=user_path,
        params_path=params_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        model_path=model_path,
        params=params,
        metrics=metrics,
        metadata=metadata if score_range else {}
    )
    
    # Add score_range to metadata dict in artifacts
    if score_range:
        artifacts.metadata['score_range'] = score_range.to_dict()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"All artifacts saved successfully!")
    logger.info(f"{'='*70}\n")
    
    return artifacts


# ============================================================================
# Load Functions (Bonus)
# ============================================================================

def load_als_artifacts(
    artifacts_dir: Union[str, Path],
    model_type: str = 'als'
) -> ALSArtifacts:
    """
    Load all ALS artifacts from a directory.
    
    Args:
        artifacts_dir: Directory containing saved artifacts
        model_type: 'als' or 'bpr'
    
    Returns:
        ALSArtifacts object with loaded data
    
    Example:
        >>> artifacts = load_als_artifacts('artifacts/cf/als')
        >>> U = np.load(artifacts.embeddings_path)  # User embeddings
        >>> score_range = artifacts.metadata['score_range']
        >>> print(f"Score normalization range: [{score_range['p01']}, {score_range['p99']}]")
    """
    artifacts_dir = Path(artifacts_dir)
    
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
    
    # Load params
    params_path = artifacts_dir / f"{model_type}_params.json"
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Load metrics
    metrics_path = artifacts_dir / f"{model_type}_metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load metadata
    metadata_path = artifacts_dir / f"{model_type}_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Embedding paths
    user_emb_path = artifacts_dir / f"{model_type}_U.npy"
    item_emb_path = artifacts_dir / f"{model_type}_V.npy"
    
    # Model object path (optional)
    model_obj_path = artifacts_dir / f"{model_type}_model.pkl"
    model_path = model_obj_path if model_obj_path.exists() else None
    
    artifacts = ALSArtifacts(
        model_type=model_type,
        output_dir=artifacts_dir,
        embeddings_path=user_emb_path,
        params_path=params_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        model_path=model_path,
        params=params,
        metrics=metrics,
        metadata=metadata
    )
    
    logger.info(f"Loaded artifacts from {artifacts_dir}")
    
    return artifacts


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ALS Artifact Saver Module - Demo")
    print("="*70)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Synthetic data
    np.random.seed(42)
    
    num_users = 100
    num_items = 50
    factors = 16
    
    U = np.random.randn(num_users, factors) * 0.1
    V = np.random.randn(num_items, factors) * 0.1
    
    # Validation users (10% of users)
    val_user_indices = list(range(0, num_users, 10))  # [0, 10, 20, ...]
    
    print("\n" + "-"*70)
    print("Example 1: Compute Score Range")
    print("-"*70)
    
    score_range = compute_score_range(
        U, V,
        validation_user_indices=val_user_indices,
        max_samples=10000,
        method='validation_set'
    )
    
    print(f"\nScore Range for Task 08 Normalization:")
    print(f"  Min: {score_range.min:.4f}")
    print(f"  Max: {score_range.max:.4f}")
    print(f"  Mean ± Std: {score_range.mean:.4f} ± {score_range.std:.4f}")
    print(f"  Robust Range [p01, p99]: [{score_range.p01:.4f}, {score_range.p99:.4f}]")
    print(f"  Samples: {score_range.num_samples:,}")
    
    print("\n" + "-"*70)
    print("Example 2: Save Complete Artifacts")
    print("-"*70)
    
    output_dir = Path("temp_als_artifacts")
    
    params = {
        'factors': factors,
        'regularization': 0.01,
        'iterations': 15,
        'alpha': 10,
        'random_seed': 42,
        'training_time_seconds': 45.2
    }
    
    metrics = {
        'recall@10': 0.234,
        'recall@20': 0.312,
        'ndcg@10': 0.189,
        'ndcg@20': 0.221,
        'baseline_recall@10': 0.145,
        'baseline_ndcg@10': 0.102,
        'improvement_recall@10': '+61.4%'
    }
    
    artifacts = save_als_complete(
        user_embeddings=U,
        item_embeddings=V,
        params=params,
        metrics=metrics,
        output_dir=output_dir,
        validation_user_indices=val_user_indices,
        data_version_hash='demo_abc123',
        model_type='als',
        compute_score_range_flag=True
    )
    
    print(artifacts.summary())
    
    print("\n" + "-"*70)
    print("Example 3: Load Artifacts")
    print("-"*70)
    
    loaded_artifacts = load_als_artifacts(output_dir, model_type='als')
    
    print(f"\nLoaded artifacts from {output_dir}")
    print(f"Score range available: {'score_range' in loaded_artifacts.metadata}")
    
    if 'score_range' in loaded_artifacts.metadata:
        sr = loaded_artifacts.metadata['score_range']
        print(f"Loaded score range: [{sr['p01']:.4f}, {sr['p99']:.4f}]")
        
        # Demo: Normalize a score using loaded range
        raw_score = 0.5
        normalized = (raw_score - sr['p01']) / (sr['p99'] - sr['p01'])
        print(f"\nTask 08 Normalization Demo:")
        print(f"  Raw CF score: {raw_score:.4f}")
        print(f"  Normalized to [0, 1]: {normalized:.4f}")
    
    print("\n" + "-"*70)
    print("Example 4: Individual Save Functions")
    print("-"*70)
    
    # Save embeddings only
    user_path, item_path = save_embeddings(U, V, output_dir, model_type='als')
    print(f"Saved embeddings: {user_path.name}, {item_path.name}")
    
    # Save params only
    params_path = save_params(params, output_dir, model_type='als')
    print(f"Saved params: {params_path.name}")
    
    # Save metrics only
    metrics_path = save_metrics(metrics, output_dir, model_type='als')
    print(f"Saved metrics: {metrics_path.name}")
    
    # Cleanup demo artifacts
    import shutil
    shutil.rmtree(output_dir)
    print(f"\nCleaned up demo artifacts: {output_dir}")
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Use save_als_complete() for one-line artifact saving")
    print("2. Always provide validation_user_indices for accurate score range")
    print("3. Score range is CRITICAL for Task 08 hybrid reranking normalization")
    print("4. Robust percentiles (p01, p99) preferred over min/max")
    print("="*70)
