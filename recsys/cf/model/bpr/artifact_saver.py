"""
BPR Artifact Saving Module (Task 02 - Step 7)

This module handles saving BPR training artifacts:
- User embeddings (U.npy)
- Item embeddings (V.npy)
- Training parameters (params.json)
- Metrics (metrics.json)
- Metadata with score range for Task 08 normalization

Score Range:
The score range is critical for hybrid reranking (Task 08).
It provides min/max/mean/std/percentiles of U @ V.T scores
for global normalization across different models.
"""

import logging
import json
from typing import Dict, Set, Tuple, Optional, Any, List
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime
import subprocess

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScoreRange:
    """
    Score distribution statistics for normalization.
    
    Used in Task 08 hybrid reranking to normalize CF scores
    before combining with content-based and popularity scores.
    """
    method: str  # How scores were computed (e.g., "validation_set", "random_sample")
    min: float
    max: float
    mean: float
    std: float
    p01: float  # 1st percentile
    p99: float  # 99th percentile
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BPRArtifacts:
    """
    Container for all BPR artifacts.
    
    Attributes:
        user_embeddings_path: Path to U.npy
        item_embeddings_path: Path to V.npy
        params_path: Path to params.json
        metrics_path: Path to metrics.json
        metadata_path: Path to metadata.json
        score_range: Score distribution for normalization
    """
    user_embeddings_path: Path
    item_embeddings_path: Path
    params_path: Path
    metrics_path: Path
    metadata_path: Path
    score_range: Optional[ScoreRange] = None
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "BPR Artifacts:",
            f"  User embeddings: {self.user_embeddings_path}",
            f"  Item embeddings: {self.item_embeddings_path}",
            f"  Parameters: {self.params_path}",
            f"  Metrics: {self.metrics_path}",
            f"  Metadata: {self.metadata_path}",
        ]
        if self.score_range:
            lines.append(f"  Score range: [{self.score_range.min:.3f}, {self.score_range.max:.3f}]")
        return "\n".join(lines)


def compute_bpr_score_range(
    U: np.ndarray,
    V: np.ndarray,
    user_indices: Optional[List[int]] = None,
    sample_size: int = 10000,
    random_seed: int = 42
) -> ScoreRange:
    """
    Compute score range statistics for normalization.
    
    This is critical for Task 08 hybrid reranking.
    
    Args:
        U: User embeddings (num_users, factors)
        V: Item embeddings (num_items, factors)
        user_indices: Optional specific users for validation set
        sample_size: Number of random (user, item) pairs to sample
        random_seed: Random seed
    
    Returns:
        ScoreRange with min/max/mean/std/percentiles
    
    Example:
        >>> score_range = compute_bpr_score_range(U, V, sample_size=10000)
        >>> print(f"Score range: [{score_range.min:.3f}, {score_range.max:.3f}]")
    """
    logger.info("Computing BPR score range for normalization...")
    
    rng = np.random.default_rng(random_seed)
    num_users, factors = U.shape
    num_items = V.shape[0]
    
    if user_indices is not None:
        # Use specific validation users
        method = "validation_set"
        scores = []
        for u_idx in user_indices:
            user_scores = U[u_idx] @ V.T
            scores.extend(user_scores.tolist())
        scores = np.array(scores)
    else:
        # Random sampling
        method = "random_sample"
        user_samples = rng.integers(0, num_users, sample_size)
        item_samples = rng.integers(0, num_items, sample_size)
        
        scores = np.array([
            U[u] @ V[i] for u, i in zip(user_samples, item_samples)
        ])
    
    score_range = ScoreRange(
        method=method,
        min=float(scores.min()),
        max=float(scores.max()),
        mean=float(scores.mean()),
        std=float(scores.std()),
        p01=float(np.percentile(scores, 1)),
        p99=float(np.percentile(scores, 99))
    )
    
    logger.info(f"Score range computed: [{score_range.min:.3f}, {score_range.max:.3f}]")
    logger.info(f"  Method: {method}")
    logger.info(f"  Mean: {score_range.mean:.3f}, Std: {score_range.std:.3f}")
    logger.info(f"  P01-P99: [{score_range.p01:.3f}, {score_range.p99:.3f}]")
    
    return score_range


def save_embeddings(
    U: np.ndarray,
    V: np.ndarray,
    output_dir: Path,
    prefix: str = "bpr"
) -> Tuple[Path, Path]:
    """
    Save user and item embeddings.
    
    Args:
        U: User embeddings
        V: Item embeddings
        output_dir: Output directory
        prefix: Filename prefix
    
    Returns:
        Tuple of (U_path, V_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    u_path = output_dir / f"{prefix}_U.npy"
    v_path = output_dir / f"{prefix}_V.npy"
    
    np.save(u_path, U)
    np.save(v_path, V)
    
    logger.info(f"Embeddings saved: U={u_path}, V={v_path}")
    
    return u_path, v_path


def save_params(
    params: Dict[str, Any],
    output_dir: Path,
    filename: str = "bpr_params.json"
) -> Path:
    """
    Save training parameters.
    
    Args:
        params: Parameter dictionary
        output_dir: Output directory
        filename: Output filename
    
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path = output_dir / filename
    
    # Convert numpy types to Python types
    params_clean = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            params_clean[key] = value.tolist()
        elif isinstance(value, (np.int32, np.int64)):
            params_clean[key] = int(value)
        elif isinstance(value, (np.float32, np.float64)):
            params_clean[key] = float(value)
        else:
            params_clean[key] = value
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params_clean, f, indent=2)
    
    logger.info(f"Parameters saved: {path}")
    
    return path


def save_metrics(
    metrics: Dict[str, Any],
    output_dir: Path,
    filename: str = "bpr_metrics.json"
) -> Path:
    """
    Save evaluation metrics.
    
    Args:
        metrics: Metrics dictionary
        output_dir: Output directory
        filename: Output filename
    
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path = output_dir / filename
    
    # Convert numpy types
    metrics_clean = {}
    for key, value in metrics.items():
        if isinstance(value, (np.float32, np.float64)):
            metrics_clean[key] = float(value)
        elif isinstance(value, dict):
            metrics_clean[key] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in value.items()
            }
        else:
            metrics_clean[key] = value
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics_clean, f, indent=2)
    
    logger.info(f"Metrics saved: {path}")
    
    return path


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()[:8]
    except Exception:
        return None


def save_metadata(
    output_dir: Path,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    score_range: Optional[ScoreRange] = None,
    data_version_hash: Optional[str] = None,
    training_time_seconds: Optional[float] = None,
    filename: str = "bpr_metadata.json"
) -> Path:
    """
    Save comprehensive metadata including score range.
    
    Args:
        output_dir: Output directory
        params: Training parameters
        metrics: Evaluation metrics
        score_range: Score distribution for Task 08
        data_version_hash: Data version hash from Task 01
        training_time_seconds: Total training time
        filename: Output filename
    
    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "model_type": "bpr",
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "data_version_hash": data_version_hash,
        "training_time_seconds": training_time_seconds,
        "parameters": params,
        "metrics": metrics
    }
    
    # Add score range for Task 08 normalization
    if score_range:
        metadata["score_range"] = score_range.to_dict()
    
    path = output_dir / filename
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Metadata saved: {path}")
    
    return path


def save_bpr_complete(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: str,
    validation_user_indices: Optional[List[int]] = None,
    data_version_hash: Optional[str] = None,
    training_time_seconds: Optional[float] = None,
    prefix: str = "bpr"
) -> BPRArtifacts:
    """
    Save complete BPR training artifacts.
    
    This function saves all artifacts in the proper structure:
    - {prefix}_U.npy: User embeddings
    - {prefix}_V.npy: Item embeddings
    - {prefix}_params.json: Training parameters
    - {prefix}_metrics.json: Evaluation metrics
    - {prefix}_metadata.json: Full metadata with score range
    
    Args:
        user_embeddings: User embedding matrix U
        item_embeddings: Item embedding matrix V
        params: Training parameters
        metrics: Evaluation metrics
        output_dir: Output directory path
        validation_user_indices: User indices for score range computation
        data_version_hash: Data version hash from Task 01
        training_time_seconds: Total training time
        prefix: Filename prefix (default: "bpr")
    
    Returns:
        BPRArtifacts with paths to all saved files
    
    Example:
        >>> artifacts = save_bpr_complete(
        ...     user_embeddings=U,
        ...     item_embeddings=V,
        ...     params={'factors': 64, 'learning_rate': 0.05},
        ...     metrics={'recall@10': 0.22, 'ndcg@10': 0.18},
        ...     output_dir='artifacts/cf/bpr',
        ...     validation_user_indices=[10, 25, 42, 67]
        ... )
        >>> print(artifacts.summary())
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("Saving BPR Artifacts")
    logger.info("="*60)
    
    # Save embeddings
    u_path, v_path = save_embeddings(
        user_embeddings, item_embeddings, output_dir, prefix
    )
    
    # Save parameters
    params_path = save_params(params, output_dir, f"{prefix}_params.json")
    
    # Save metrics
    metrics_path = save_metrics(metrics, output_dir, f"{prefix}_metrics.json")
    
    # Compute score range for Task 08
    score_range = compute_bpr_score_range(
        user_embeddings,
        item_embeddings,
        user_indices=validation_user_indices
    )
    
    # Save metadata with score range
    metadata_path = save_metadata(
        output_dir=output_dir,
        params=params,
        metrics=metrics,
        score_range=score_range,
        data_version_hash=data_version_hash,
        training_time_seconds=training_time_seconds,
        filename=f"{prefix}_metadata.json"
    )
    
    artifacts = BPRArtifacts(
        user_embeddings_path=u_path,
        item_embeddings_path=v_path,
        params_path=params_path,
        metrics_path=metrics_path,
        metadata_path=metadata_path,
        score_range=score_range
    )
    
    logger.info("="*60)
    logger.info("BPR Artifacts saved successfully")
    logger.info("="*60)
    logger.info(artifacts.summary())
    
    return artifacts


def load_bpr_artifacts(
    artifact_dir: str,
    prefix: str = "bpr"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load BPR artifacts from directory.
    
    Args:
        artifact_dir: Directory containing artifacts
        prefix: Filename prefix
    
    Returns:
        Tuple of (U, V, params, metrics, metadata)
    
    Example:
        >>> U, V, params, metrics, metadata = load_bpr_artifacts('artifacts/cf/bpr')
        >>> print(f"Loaded: U={U.shape}, V={V.shape}")
        >>> print(f"Score range: {metadata['score_range']}")
    """
    artifact_dir = Path(artifact_dir)
    
    logger.info(f"Loading BPR artifacts from {artifact_dir}")
    
    # Load embeddings
    u_path = artifact_dir / f"{prefix}_U.npy"
    v_path = artifact_dir / f"{prefix}_V.npy"
    
    U = np.load(u_path)
    V = np.load(v_path)
    
    logger.info(f"Loaded embeddings: U={U.shape}, V={V.shape}")
    
    # Load params
    params_path = artifact_dir / f"{prefix}_params.json"
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Load metrics
    metrics_path = artifact_dir / f"{prefix}_metrics.json"
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Load metadata
    metadata_path = artifact_dir / f"{prefix}_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info("BPR artifacts loaded successfully")
    
    return U, V, params, metrics, metadata


# Main execution example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("BPR Artifact Saver Demo")
    print("="*60)
    
    # Create synthetic data
    np.random.seed(42)
    num_users, num_items, factors = 1000, 500, 64
    
    U = np.random.randn(num_users, factors).astype(np.float32) * 0.1
    V = np.random.randn(num_items, factors).astype(np.float32) * 0.1
    
    params = {
        'factors': factors,
        'learning_rate': 0.05,
        'regularization': 0.0001,
        'epochs': 50,
        'hard_ratio': 0.3
    }
    
    metrics = {
        'recall@10': 0.22,
        'recall@20': 0.31,
        'ndcg@10': 0.18,
        'ndcg@20': 0.22,
        'improvement_vs_baseline': '45.2%'
    }
    
    # Test score range computation
    print("\n" + "-"*60)
    print("Computing score range")
    print("-"*60)
    
    score_range = compute_bpr_score_range(U, V, sample_size=5000)
    print(f"Score range: {score_range}")
    
    # Test complete save
    print("\n" + "-"*60)
    print("Saving complete artifacts")
    print("-"*60)
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = save_bpr_complete(
            user_embeddings=U,
            item_embeddings=V,
            params=params,
            metrics=metrics,
            output_dir=tmpdir,
            validation_user_indices=[0, 100, 200, 300, 400],
            training_time_seconds=120.5
        )
        
        print(f"\n{artifacts.summary()}")
        
        # Test loading
        print("\n" + "-"*60)
        print("Loading artifacts")
        print("-"*60)
        
        U2, V2, params2, metrics2, metadata2 = load_bpr_artifacts(tmpdir)
        print(f"Loaded: U={U2.shape}, V={V2.shape}")
        print(f"Score range from metadata: {metadata2.get('score_range', {})}")
    
    print("\n" + "="*60)
    print("Demo complete!")
