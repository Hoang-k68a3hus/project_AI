"""
Complete ALS Training Pipeline - Convenience Script

This script provides a simple interface to run the full ALS training pipeline
from data loading to artifact saving, similar to the DataProcessor pattern.

Key Features:
- Single function call to run complete pipeline
- Automatic data loading from Task 01 outputs
- Model initialization, training, evaluation, and artifact saving
- Score range computation for Task 08 hybrid reranking
- Comprehensive logging and error handling

Usage:
    >>> from scripts.run_als_complete import run_als_pipeline
    >>> 
    >>> # Run with default settings
    >>> artifacts = run_als_pipeline()
    >>> 
    >>> # Custom configuration
    >>> artifacts = run_als_pipeline(
    ...     data_dir='data/processed',
    ...     output_dir='artifacts/cf/als',
    ...     factors=64,
    ...     regularization=0.01,
    ...     iterations=15,
    ...     alpha=10
    ... )

Author: Copilot
Date: 2025-01-15
"""

import os
import sys
import logging
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from scipy.sparse import load_npz, csr_matrix
import json
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import ALS modules
from recsys.cf.model.als import (
    ALSModelInitializer,
    ALSTrainer,
    EmbeddingExtractor,
    ALSEvaluator,
    save_als_complete
)


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(log_file: str = "logs/cf/als_complete.log") -> logging.Logger:
    """
    Configure logging for ALS pipeline.
    
    Args:
        log_file: Path to log file
    
    Returns:
        Logger instance
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger("als_complete")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_training_matrix(data_dir: str = "data/processed") -> csr_matrix:
    """
    Load confidence matrix for ALS training.
    
    Args:
        data_dir: Directory containing Task 01 outputs
    
    Returns:
        CSR matrix (num_users, num_items)
    
    Example:
        >>> X_train = load_training_matrix()
        >>> print(f"Matrix shape: {X_train.shape}")
    """
    data_dir = Path(data_dir)
    matrix_path = data_dir / "X_train_confidence.npz"
    
    if not matrix_path.exists():
        raise FileNotFoundError(f"Training matrix not found: {matrix_path}")
    
    X_train = load_npz(matrix_path)
    logger.info(f"Loaded training matrix: {X_train.shape}, nnz: {X_train.nnz:,}")
    
    return X_train


def load_id_mappings(data_dir: str = "data/processed") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load user and item ID mappings with trainable user re-mapping.
    
    CRITICAL: Handles LOCAL vs GLOBAL index mapping
    - Global mappings: All 300K users (0-299,999)
    - Training matrix: LOCAL indices for trainable users only (0-25,999)
    - This function loads trainable_user_mapping.json to resolve the mismatch
    
    Args:
        data_dir: Directory containing Task 01 outputs
    
    Returns:
        Tuple of (mappings_dict, metadata_dict)
    
    Example:
        >>> mappings, metadata = load_id_mappings()
        >>> print(f"Trainable users: {len(mappings['user_to_idx'])}")
    """
    data_dir = Path(data_dir)
    mappings_path = data_dir / "user_item_mappings.json"
    
    if not mappings_path.exists():
        raise FileNotFoundError(f"ID mappings not found: {mappings_path}")
    
    with open(mappings_path, 'r') as f:
        data = json.load(f)
    
    # Load global mappings (all users)
    global_user_to_idx = data['user_to_idx']
    global_idx_to_user = {int(k): v for k, v in data['idx_to_user'].items()}
    
    # Item mappings (no re-mapping needed)
    item_to_idx = data['item_to_idx']
    idx_to_item = {int(k): v for k, v in data['idx_to_item'].items()}
    
    logger.info(f"Loaded GLOBAL mappings: {len(global_user_to_idx)} users, {len(item_to_idx)} items")
    
    # === CRITICAL: Load Trainable User Re-mapping ===
    trainable_map_path = data_dir / "trainable_user_mapping.json"
    if trainable_map_path.exists():
        logger.info("Loading trainable user re-mapping...")
        
        with open(trainable_map_path, 'r') as f:
            trainable_map = json.load(f)
        
        # Create LOCAL -> GLOBAL -> USER_ID mapping chain
        u_idx_cf_to_u_idx = {int(k): int(v) for k, v in trainable_map['u_idx_cf_to_u_idx'].items()}
        
        # Build model-compatible mappings: Local Index -> User ID
        user_to_idx = {}  # user_id -> local_idx (for model input)
        idx_to_user = {}  # local_idx -> user_id (for model output/logging)
        
        for local_idx, global_idx in u_idx_cf_to_u_idx.items():
            if global_idx in global_idx_to_user:
                user_id = global_idx_to_user[global_idx]
                idx_to_user[local_idx] = user_id
                user_to_idx[user_id] = local_idx
        
        logger.info(f"✓ Re-mapped to {len(idx_to_user)} TRAINABLE users (local indices)")
        
        mappings = {
            'user_to_idx': user_to_idx,
            'idx_to_user': idx_to_user,
            'item_to_idx': item_to_idx,
            'idx_to_item': idx_to_item,
            '_global_user_to_idx': global_user_to_idx,  # Keep for reference
            '_global_idx_to_user': global_idx_to_user   # Keep for reference
        }
    else:
        # Fallback: Use global mappings (for backward compatibility)
        logger.warning(
            "⚠️  trainable_user_mapping.json not found! Using global mappings. "
            "This may cause index mismatch if X_train uses local indices."
        )
        mappings = {
            'user_to_idx': global_user_to_idx,
            'idx_to_user': global_idx_to_user,
            'item_to_idx': item_to_idx,
            'idx_to_item': idx_to_item
        }
    
    logger.info(f"Final mappings: {len(mappings['user_to_idx'])} users, {len(mappings['item_to_idx'])} items")
    
    metadata = {
        'positive_threshold': data.get('positive_threshold', 4.0),
        'hard_negative_threshold': data.get('hard_negative_threshold', 3.0),
        'data_hash': data.get('data_hash'),
        'timestamp': data.get('timestamp')
    }
    
    return mappings, metadata


def load_user_sets(data_dir: str = "data/processed") -> Tuple[Dict[int, set], Dict[int, set]]:
    """
    Load training and test positive sets.
    
    Args:
        data_dir: Directory containing Task 01 outputs
    
    Returns:
        Tuple of (user_pos_train, user_pos_test)
    
    Example:
        >>> train_sets, test_sets = load_user_sets()
        >>> print(f"Train users: {len(train_sets)}, Test users: {len(test_sets)}")
    """
    data_dir = Path(data_dir)
    
    train_path = data_dir / "user_pos_train.pkl"
    test_path = data_dir / "user_pos_test.pkl"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training sets not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test sets not found: {test_path}")
    
    with open(train_path, 'rb') as f:
        user_pos_train = pickle.load(f)
    
    with open(test_path, 'rb') as f:
        user_pos_test = pickle.load(f)
    
    logger.info(f"Loaded user sets: {len(user_pos_train)} train, {len(user_pos_test)} test")
    
    return user_pos_train, user_pos_test


def load_all_data(data_dir: str = "data/processed") -> Dict[str, Any]:
    """
    Load all required data for ALS training.
    
    Args:
        data_dir: Directory containing Task 01 outputs
    
    Returns:
        Dictionary with keys: X_train, mappings, metadata, user_pos_train, user_pos_test
    
    Example:
        >>> data = load_all_data()
        >>> X_train = data['X_train']
        >>> mappings = data['mappings']
    """
    logger.info("Loading all data from Task 01 outputs...")
    
    X_train = load_training_matrix(data_dir)
    mappings, metadata = load_id_mappings(data_dir)
    user_pos_train, user_pos_test = load_user_sets(data_dir)
    
    data = {
        'X_train': X_train,
        'mappings': mappings,
        'metadata': metadata,
        'user_pos_train': user_pos_train,
        'user_pos_test': user_pos_test
    }
    
    logger.info("Data loading complete!")
    
    return data


# ============================================================================
# Model Training Functions
# ============================================================================

def initialize_als_model(
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 15,
    alpha: float = 10,
    use_gpu: bool = False,
    random_seed: int = 42
):
    """
    Initialize ALS model with given hyperparameters.
    
    Args:
        factors: Embedding dimension
        regularization: L2 regularization penalty
        iterations: Number of ALS iterations
        alpha: Confidence scaling parameter
        use_gpu: Use GPU acceleration (requires cupy)
        random_seed: Random seed for reproducibility
    
    Returns:
        Initialized ALS model (implicit.als.AlternatingLeastSquares)
    
    Example:
        >>> model = initialize_als_model(factors=64, alpha=10)
        >>> print(f"Model: {model}")
    """
    initializer = ALSModelInitializer(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        use_gpu=use_gpu,
        random_seed=random_seed
    )
    
    model = initializer.initialize_model()
    
    logger.info(f"Model initialized: factors={factors}, reg={regularization}, "
                f"iter={iterations}, alpha={alpha}, gpu={use_gpu}")
    
    return model


def train_als_model(
    model,
    X_train: csr_matrix,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = 5,
    track_memory: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train ALS model with progress tracking.
    
    Args:
        model: Initialized ALS model
        X_train: Training confidence matrix
        checkpoint_dir: Directory for checkpoints (optional)
        checkpoint_interval: Save checkpoint every N iterations
        track_memory: Enable memory profiling
    
    Returns:
        Tuple of (trained_model, training_summary)
    
    Example:
        >>> model = initialize_als_model()
        >>> trained_model, summary = train_als_model(model, X_train)
        >>> print(f"Training time: {summary['training_time']:.2f}s")
    """
    trainer = ALSTrainer(
        model=model,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        track_memory=track_memory
    )
    
    training_summary = trainer.fit(X_train, show_progress=True)
    
    logger.info(f"Training completed: {training_summary['training_time']:.2f}s, "
                f"{training_summary['iterations']} iterations")
    
    return trainer.model, training_summary


def extract_embeddings(
    model,
    normalize: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract user and item embeddings from trained model.
    
    Args:
        model: Trained ALS model
        normalize: Apply L2 normalization
    
    Returns:
        Tuple of (U, V) where U is user embeddings, V is item embeddings
    
    Example:
        >>> U, V = extract_embeddings(model, normalize=True)
        >>> print(f"U: {U.shape}, V: {V.shape}")
    """
    extractor = EmbeddingExtractor(model=model, normalize=normalize)
    U, V = extractor.get_embeddings()
    
    logger.info(f"Embeddings extracted: U={U.shape}, V={V.shape}, normalized={normalize}")
    
    return U, V


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_als_model(
    U: np.ndarray,
    V: np.ndarray,
    mappings: Dict[str, Any],
    user_pos_train: Dict[int, set],
    user_pos_test: Dict[int, set],
    k_values: list = [10, 20],
    compare_baseline: bool = True
):
    """
    Evaluate ALS model with baseline comparison.
    
    Args:
        U: User embeddings
        V: Item embeddings
        mappings: ID mappings dictionary
        user_pos_train: Training positive sets
        user_pos_test: Test positive sets
        k_values: K values for Recall@K and NDCG@K
        compare_baseline: Compare with popularity baseline
    
    Returns:
        EvaluationResult object
    
    Example:
        >>> results = evaluate_als_model(U, V, mappings, train_sets, test_sets)
        >>> results.print_summary()
    """
    evaluator = ALSEvaluator(
        user_factors=U,
        item_factors=V,
        user_to_idx=mappings['user_to_idx'],
        idx_to_user=mappings['idx_to_user'],
        item_to_idx=mappings['item_to_idx'],
        idx_to_item=mappings['idx_to_item'],
        user_pos_train=user_pos_train,
        user_pos_test=user_pos_test
    )
    
    results = evaluator.evaluate(
        k_values=k_values,
        filter_seen=True,
        compare_baseline=compare_baseline,
        baseline_source='train',
        model_type='als'
    )
    
    logger.info(f"Evaluation completed: Recall@10={results.metrics.get('recall@10', 0):.4f}, "
                f"NDCG@10={results.metrics.get('ndcg@10', 0):.4f}")
    
    return results


# ============================================================================
# Artifact Saving Functions
# ============================================================================

def extract_validation_users(
    user_pos_test: Dict[int, set],
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> list:
    """
    Extract validation users from test set for score range computation.
    
    Args:
        user_pos_test: Test positive sets
        val_ratio: Ratio of test users for validation
        random_seed: Random seed
    
    Returns:
        List of validation user indices
    
    Example:
        >>> val_users = extract_validation_users(test_sets, val_ratio=0.1)
        >>> print(f"Validation users: {len(val_users)}")
    """
    np.random.seed(random_seed)
    
    test_users = list(user_pos_test.keys())
    num_val = max(1, int(len(test_users) * val_ratio))
    val_users = np.random.choice(test_users, num_val, replace=False).tolist()
    
    logger.info(f"Extracted {len(val_users)} validation users ({val_ratio*100:.1f}% of test set)")
    
    return val_users


def save_artifacts(
    U: np.ndarray,
    V: np.ndarray,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: str = "artifacts/cf/als",
    validation_user_indices: Optional[list] = None,
    data_version_hash: Optional[str] = None
):
    """
    Save all ALS artifacts with score range for Task 08.
    
    Args:
        U: User embeddings
        V: Item embeddings
        params: Training parameters
        metrics: Evaluation metrics
        output_dir: Output directory
        validation_user_indices: Validation users for score range
        data_version_hash: Data version hash from Task 01
    
    Returns:
        ALSArtifacts object
    
    Example:
        >>> artifacts = save_artifacts(U, V, params, metrics, validation_user_indices=val_users)
        >>> print(artifacts.summary())
    """
    artifacts = save_als_complete(
        user_embeddings=U,
        item_embeddings=V,
        params=params,
        metrics=metrics,
        output_dir=output_dir,
        validation_user_indices=validation_user_indices,
        data_version_hash=data_version_hash,
        model_type='als',
        compute_score_range_flag=True
    )
    
    logger.info(f"Artifacts saved to {output_dir}")
    
    return artifacts


# ============================================================================
# Complete Pipeline Function
# ============================================================================

def run_als_pipeline(
    data_dir: str = "data/processed",
    output_dir: str = "artifacts/cf/als",
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 15,
    alpha: float = 10,
    use_gpu: bool = False,
    random_seed: int = 42,
    normalize_embeddings: bool = False,
    k_values: list = [10, 20],
    compare_baseline: bool = True,
    val_ratio: float = 0.1,
    checkpoint_dir: Optional[str] = None,
    checkpoint_interval: int = 5,
    track_memory: bool = False,
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete ALS training pipeline from data loading to artifact saving.
    
    This is the main function that orchestrates all 7 steps:
        Step 1: Load data from Task 01
        Step 2: Initialize ALS model
        Step 3: Train model
        Step 4: Extract embeddings
        Step 5: Generate recommendations (during evaluation)
        Step 6: Evaluate with baseline comparison
        Step 7: Save artifacts with score range for Task 08
    
    Args:
        data_dir: Directory with Task 01 preprocessed data
        output_dir: Directory to save artifacts
        factors: Embedding dimension (default: 64)
        regularization: L2 regularization (default: 0.01)
        iterations: Number of ALS iterations (default: 15)
        alpha: Confidence scaling (default: 10 for sentiment-enhanced)
        use_gpu: Use GPU acceleration (default: False)
        random_seed: Random seed for reproducibility (default: 42)
        normalize_embeddings: L2 normalize embeddings (default: False)
        k_values: K values for Recall@K and NDCG@K (default: [10, 20])
        compare_baseline: Compare with popularity baseline (default: True)
        val_ratio: Ratio of test users for validation (default: 0.1)
        checkpoint_dir: Directory for training checkpoints (optional)
        checkpoint_interval: Save checkpoint every N iterations (default: 5)
        track_memory: Enable memory profiling (default: False)
        log_file: Path to log file (optional)
    
    Returns:
        Dictionary with keys:
            - model: Trained ALS model
            - U: User embeddings
            - V: Item embeddings
            - training_summary: Training metrics
            - evaluation_results: Evaluation metrics and comparisons
            - artifacts: Saved artifact paths and metadata
            - data: Loaded data (for debugging)
    
    Example:
        >>> # Run with default settings
        >>> results = run_als_pipeline()
        >>> 
        >>> # Access results
        >>> print(f"Recall@10: {results['evaluation_results'].metrics['recall@10']:.3f}")
        >>> print(f"Artifacts: {results['artifacts'].output_dir}")
        >>> 
        >>> # Custom configuration
        >>> results = run_als_pipeline(
        ...     factors=128,
        ...     regularization=0.05,
        ...     iterations=20,
        ...     alpha=15,
        ...     val_ratio=0.2
        ... )
    """
    # Setup logging
    if log_file:
        global logger
        logger = setup_logging(log_file)
    
    logger.info("="*80)
    logger.info("COMPLETE ALS TRAINING PIPELINE")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        # Step 1: Load data
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Load Data from Task 01")
        logger.info("="*80)
        data = load_all_data(data_dir)
        
        # Step 2: Initialize model
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Initialize ALS Model")
        logger.info("="*80)
        model = initialize_als_model(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            alpha=alpha,
            use_gpu=use_gpu,
            random_seed=random_seed
        )
        
        # Step 3: Train model
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Train ALS Model")
        logger.info("="*80)
        trained_model, training_summary = train_als_model(
            model=model,
            X_train=data['X_train'],
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            track_memory=track_memory
        )
        
        # Step 4: Extract embeddings
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Extract Embeddings")
        logger.info("="*80)
        U, V = extract_embeddings(trained_model, normalize=normalize_embeddings)
        
        # Extract validation users
        val_users = extract_validation_users(
            data['user_pos_test'],
            val_ratio=val_ratio,
            random_seed=random_seed
        )
        
        # Remaining test users for evaluation
        test_users_set = set(data['user_pos_test'].keys())
        val_users_set = set(val_users)
        test_users = list(test_users_set - val_users_set)
        
        # Filter test sets for evaluation
        user_pos_test_eval = {u: data['user_pos_test'][u] for u in test_users}
        
        # Steps 5 & 6: Generate recommendations and evaluate
        logger.info("\n" + "="*80)
        logger.info("STEPS 5 & 6: Generate Recommendations and Evaluate")
        logger.info("="*80)
        evaluation_results = evaluate_als_model(
            U=U,
            V=V,
            mappings=data['mappings'],
            user_pos_train=data['user_pos_train'],
            user_pos_test=user_pos_test_eval,
            k_values=k_values,
            compare_baseline=compare_baseline
        )
        
        # Print evaluation summary
        evaluation_results.print_summary()
        
        # Step 7: Save artifacts
        logger.info("\n" + "="*80)
        logger.info("STEP 7: Save Artifacts with Score Range")
        logger.info("="*80)
        
        # Prepare parameters
        params = {
            'factors': factors,
            'regularization': regularization,
            'iterations': iterations,
            'alpha': alpha,
            'use_gpu': use_gpu,
            'random_seed': random_seed,
            'normalize_embeddings': normalize_embeddings,
            'training_time_seconds': training_summary.get('training_time', 0)
        }
        
        # Prepare metrics (flatten EvaluationResult)
        metrics = evaluation_results.metrics.copy()
        metrics.update({
            f'baseline_{k}': v 
            for k, v in evaluation_results.baseline_metrics.items()
        })
        metrics.update({
            f'improvement_{k}': v 
            for k, v in evaluation_results.improvement.items()
        })
        
        # Save artifacts with score range
        artifacts = save_artifacts(
            U=U,
            V=V,
            params=params,
            metrics=metrics,
            output_dir=output_dir,
            validation_user_indices=val_users,
            data_version_hash=data['metadata'].get('data_hash')
        )
        
        # Print artifacts summary
        print(artifacts.summary())
        
        # Complete
        total_time = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Artifacts saved to: {output_dir}")
        logger.info(f"Recall@10: {evaluation_results.metrics.get('recall@10', 0):.4f}")
        logger.info(f"NDCG@10: {evaluation_results.metrics.get('ndcg@10', 0):.4f}")
        logger.info(f"Improvement vs baseline: {evaluation_results.improvement.get('improvement_recall@10', 'N/A')}")
        logger.info("="*80)
        
        return {
            'model': trained_model,
            'U': U,
            'V': V,
            'training_summary': training_summary,
            'evaluation_results': evaluation_results,
            'artifacts': artifacts,
            'data': data,
            'val_users': val_users,
            'test_users': test_users
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


# ============================================================================
# Quick Test Function
# ============================================================================

def test_als_pipeline():
    """
    Quick test of ALS pipeline with default settings.
    
    Example:
        >>> from scripts.run_als_complete import test_als_pipeline
        >>> test_als_pipeline()
    """
    print("Testing ALS Complete Pipeline...")
    print("="*80)
    
    results = run_als_pipeline(
        data_dir="data/processed",
        output_dir="artifacts/cf/als",
        factors=64,
        regularization=0.01,
        iterations=15,
        alpha=10,
        k_values=[10, 20],
        val_ratio=0.1
    )
    
    print("\nTest completed!")
    print(f"Recall@10: {results['evaluation_results'].metrics['recall@10']:.4f}")
    print(f"NDCG@10: {results['evaluation_results'].metrics['ndcg@10']:.4f}")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Test the complete ALS pipeline.
    """
    print("="*80)
    print("ALS Complete Training Pipeline - Test Run")
    print("="*80)
    
    try:
        results = test_als_pipeline()
        print("\n" + "="*80)
        print("SUCCESS: Pipeline completed successfully!")
        print("="*80)
    except Exception as e:
        print("\n" + "="*80)
        print(f"ERROR: Pipeline failed - {e}")
        print("="*80)
        raise
