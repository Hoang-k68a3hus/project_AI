"""
Model Training Pipeline Script.
Trains ALS and BPR models and registers the best one.

Usage:
    python scripts/train_both_models.py [--model als|bpr|both] [--auto-select]
    
Options:
    --model als|bpr|both   Which model(s) to train (default: both)
    --auto-select          Automatically register best model
    --skip-eval            Skip evaluation after training
    --force                Force training even if recent model exists
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    retry, PipelineTracker, PipelineLock,
    setup_logging, send_pipeline_alert, get_git_commit
)


# =============================================================================
# Configuration
# =============================================================================

TRAINING_CONFIG = {
    'processed_dir': PROJECT_ROOT / 'data' / 'processed',
    'artifacts_dir': PROJECT_ROOT / 'artifacts' / 'cf',
    'checkpoints_dir': PROJECT_ROOT / 'checkpoints',
    'registry_path': PROJECT_ROOT / 'artifacts' / 'cf' / 'registry.json',
    
    # ALS hyperparameters
    'als': {
        'factors': 64,
        'regularization': 0.1,
        'iterations': 15,
        'alpha': 10,
        'use_gpu': False,
        'calculate_training_loss': True
    },
    
    # BPR hyperparameters
    'bpr': {
        'factors': 64,
        'learning_rate': 0.05,
        'regularization': 0.0001,
        'epochs': 50,
        'neg_sample_ratio': 0.3  # 30% hard negatives
    },
    
    # Evaluation settings
    'eval_k_values': [5, 10, 20],
    'primary_metric': 'recall',  # For model selection
    'primary_k': 10  # Recall@10
}


# =============================================================================
# Training Functions
# =============================================================================

def load_training_data(logger: logging.Logger) -> Dict[str, Any]:
    """
    Load all required training data.
    
    Returns:
        Dict with loaded data arrays and mappings
    """
    import numpy as np
    import scipy.sparse as sp
    import pickle
    
    processed_dir = TRAINING_CONFIG['processed_dir']
    
    # Load sparse matrices
    X_confidence = sp.load_npz(processed_dir / 'X_train_confidence.npz')
    X_binary = sp.load_npz(processed_dir / 'X_train_binary.npz')
    
    # Load mappings
    with open(processed_dir / 'user_item_mappings.json', 'r') as f:
        mappings = json.load(f)
    
    # Load user sets
    with open(processed_dir / 'user_pos_train.pkl', 'rb') as f:
        user_pos_train = pickle.load(f)
    
    with open(processed_dir / 'user_hard_neg_train.pkl', 'rb') as f:
        user_hard_neg_train = pickle.load(f)
    
    # Load stats
    with open(processed_dir / 'data_stats.json', 'r') as f:
        data_stats = json.load(f)
    
    logger.info(f"Loaded training data: {X_confidence.shape[0]} users x {X_confidence.shape[1]} items")
    
    return {
        'X_confidence': X_confidence,
        'X_binary': X_binary,
        'mappings': mappings,
        'user_pos_train': user_pos_train,
        'user_hard_neg_train': user_hard_neg_train,
        'data_stats': data_stats,
        'data_hash': mappings.get('data_hash')
    }


@retry(max_attempts=2, backoff_factor=2.0)
def train_als_model(
    data: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Train ALS model using implicit library.
    
    Returns:
        Dict with model, embeddings, and training info
    """
    import numpy as np
    from implicit.als import AlternatingLeastSquares
    
    config = TRAINING_CONFIG['als']
    
    logger.info("Training ALS model...")
    logger.info(f"  factors={config['factors']}, reg={config['regularization']}, "
                f"iter={config['iterations']}, alpha={config['alpha']}")
    
    # Initialize model
    model = AlternatingLeastSquares(
        factors=config['factors'],
        regularization=config['regularization'],
        iterations=config['iterations'],
        alpha=config['alpha'],
        use_gpu=config['use_gpu'],
        calculate_training_loss=config['calculate_training_loss'],
        random_state=42
    )
    
    # Train on confidence matrix (item x user format for implicit)
    X_train = data['X_confidence'].T.tocsr()  # Transpose to item x user
    
    start_time = datetime.now()
    model.fit(X_train, show_progress=True)
    training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"ALS training completed in {training_time:.1f}s")
    
    # Extract embeddings
    U = np.array(model.user_factors)  # (num_users, factors)
    V = np.array(model.item_factors)  # (num_items, factors)
    
    return {
        'model': model,
        'U': U,
        'V': V,
        'model_type': 'als',
        'training_time': training_time,
        'config': config,
        'loss': model.loss if hasattr(model, 'loss') else None
    }


@retry(max_attempts=2, backoff_factor=2.0)
def train_bpr_model(
    data: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Train BPR model using implicit library.
    
    Returns:
        Dict with model, embeddings, and training info
    """
    import numpy as np
    from implicit.bpr import BayesianPersonalizedRanking
    
    config = TRAINING_CONFIG['bpr']
    
    logger.info("Training BPR model...")
    logger.info(f"  factors={config['factors']}, lr={config['learning_rate']}, "
                f"reg={config['regularization']}, epochs={config['epochs']}")
    
    # Initialize model
    model = BayesianPersonalizedRanking(
        factors=config['factors'],
        learning_rate=config['learning_rate'],
        regularization=config['regularization'],
        iterations=config['epochs'],
        random_state=42,
        verify_negative_samples=True
    )
    
    # Train on binary matrix (item x user format)
    X_train = data['X_binary'].T.tocsr()
    
    start_time = datetime.now()
    model.fit(X_train, show_progress=True)
    training_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"BPR training completed in {training_time:.1f}s")
    
    # Extract embeddings
    U = np.array(model.user_factors)
    V = np.array(model.item_factors)
    
    return {
        'model': model,
        'U': U,
        'V': V,
        'model_type': 'bpr',
        'training_time': training_time,
        'config': config
    }


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(
    model_result: Dict[str, Any],
    data: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Evaluate trained model using Recall@K and NDCG@K.
    
    Returns:
        Dict with evaluation metrics
    """
    import numpy as np
    
    model_type = model_result['model_type']
    U = model_result['U']
    V = model_result['V']
    user_pos_train = data['user_pos_train']
    
    logger.info(f"Evaluating {model_type.upper()} model...")
    
    # Load test data
    processed_dir = TRAINING_CONFIG['processed_dir']
    
    try:
        import pickle
        with open(processed_dir / 'user_pos_test.pkl', 'rb') as f:
            user_pos_test = pickle.load(f)
    except FileNotFoundError:
        logger.warning("Test data not found, using 20% holdout from train")
        # Fallback: use last 20% of each user's positives as test
        user_pos_test = {}
        for u_idx, items in user_pos_train.items():
            if len(items) >= 2:
                items_list = list(items)
                split_idx = max(1, int(len(items_list) * 0.8))
                user_pos_test[u_idx] = set(items_list[split_idx:])
    
    k_values = TRAINING_CONFIG['eval_k_values']
    
    # Compute predictions for test users
    metrics = {f'recall@{k}': [] for k in k_values}
    metrics.update({f'ndcg@{k}': [] for k in k_values})
    
    test_users = list(user_pos_test.keys())
    
    for u_idx in test_users:
        if u_idx >= U.shape[0]:
            continue
            
        # Compute scores
        scores = U[u_idx] @ V.T
        
        # Filter out training items
        train_items = user_pos_train.get(u_idx, set())
        scores[list(train_items)] = -np.inf
        
        # Get top-K predictions
        max_k = max(k_values)
        top_k_items = np.argsort(scores)[-max_k:][::-1]
        
        # Ground truth
        test_items = user_pos_test[u_idx]
        
        for k in k_values:
            preds_k = set(top_k_items[:k])
            
            # Recall@K
            hits = len(preds_k & test_items)
            recall = hits / min(len(test_items), k) if test_items else 0
            metrics[f'recall@{k}'].append(recall)
            
            # NDCG@K
            dcg = 0.0
            for i, item in enumerate(top_k_items[:k]):
                if item in test_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[f'ndcg@{k}'].append(ndcg)
    
    # Average metrics
    avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}
    avg_metrics['num_test_users'] = len(test_users)
    
    logger.info(f"  Recall@10: {avg_metrics['recall@10']:.4f}")
    logger.info(f"  NDCG@10: {avg_metrics['ndcg@10']:.4f}")
    
    return avg_metrics


# =============================================================================
# Model Saving & Registration
# =============================================================================

def save_model(
    model_result: Dict[str, Any],
    metrics: Dict[str, Any],
    data: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Save model artifacts to disk.
    
    Returns:
        Dict with save paths and model_id
    """
    import numpy as np
    
    model_type = model_result['model_type']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f"{model_type}_{timestamp}"
    
    # Create output directory
    output_dir = TRAINING_CONFIG['artifacts_dir'] / model_type / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {model_type.upper()} model to {output_dir}")
    
    # Save embeddings
    np.save(output_dir / f'{model_type}_U.npy', model_result['U'])
    np.save(output_dir / f'{model_type}_V.npy', model_result['V'])
    
    # Save parameters
    params = {
        'model_type': model_type,
        'config': model_result['config'],
        'training_time': model_result['training_time'],
        'num_users': model_result['U'].shape[0],
        'num_items': model_result['V'].shape[0],
        'factors': model_result['U'].shape[1]
    }
    
    with open(output_dir / f'{model_type}_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    # Save metrics
    with open(output_dir / f'{model_type}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save metadata
    metadata = {
        'model_id': model_id,
        'model_type': model_type,
        'created_at': datetime.now().isoformat(),
        'data_hash': data['data_hash'],
        'git_commit': get_git_commit(),
        'score_range': {
            'min': 0.0,
            'max': float(model_result['U'].shape[1])  # Approximate max score
        }
    }
    
    with open(output_dir / f'{model_type}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'model_id': model_id,
        'output_dir': str(output_dir),
        'files': [f'{model_type}_U.npy', f'{model_type}_V.npy', 
                  f'{model_type}_params.json', f'{model_type}_metrics.json',
                  f'{model_type}_metadata.json']
    }


def register_model(
    model_id: str,
    model_type: str,
    metrics: Dict[str, Any],
    output_dir: Path,
    logger: logging.Logger
) -> bool:
    """
    Register model in the registry.
    
    Returns:
        True if registered as best model
    """
    registry_path = TRAINING_CONFIG['registry_path']
    
    # Load or create registry
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {'models': [], 'current_best': None}
    
    # Create model entry
    entry = {
        'model_id': model_id,
        'model_type': model_type,
        'registered_at': datetime.now().isoformat(),
        'metrics': metrics,
        'path': str(output_dir),
        'is_active': False
    }
    
    # Check if this is the best model
    is_best = False
    primary_metric = f"{TRAINING_CONFIG['primary_metric']}@{TRAINING_CONFIG['primary_k']}"
    current_score = metrics.get(primary_metric, 0)
    
    if registry['current_best']:
        # Find current best's score
        best_model = next(
            (m for m in registry['models'] if m['model_id'] == registry['current_best']),
            None
        )
        if best_model:
            best_score = best_model['metrics'].get(primary_metric, 0)
            if current_score > best_score:
                is_best = True
                logger.info(f"New best model! {primary_metric}: {current_score:.4f} > {best_score:.4f}")
        else:
            is_best = True
    else:
        is_best = True
    
    if is_best:
        # Deactivate previous best
        for m in registry['models']:
            m['is_active'] = False
        
        entry['is_active'] = True
        registry['current_best'] = model_id
    
    registry['models'].append(entry)
    
    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Registered model: {model_id} (is_best={is_best})")
    
    return is_best


# =============================================================================
# Main Pipeline
# =============================================================================

def train_models(
    model_types: List[str] = ['als', 'bpr'],
    auto_select: bool = True,
    skip_eval: bool = False,
    force: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Main training pipeline.
    
    Args:
        model_types: List of models to train ('als', 'bpr')
        auto_select: Automatically register best model
        skip_eval: Skip evaluation
        force: Force training even if recent model exists
        logger: Logger instance
        
    Returns:
        Dict with training results
    """
    if logger is None:
        logger = setup_logging("model_training")
    
    tracker = PipelineTracker()
    result = {
        'pipeline': 'model_training',
        'started_at': datetime.now().isoformat(),
        'models': {}
    }
    
    with PipelineLock("model_training") as lock:
        if not lock.acquired:
            msg = "Model training already running"
            logger.warning(msg)
            result['status'] = 'skipped'
            result['message'] = msg
            return result
        
        run_id = tracker.start_run("model_training", {'models': model_types})
        
        try:
            # Load data
            logger.info("Loading training data...")
            data = load_training_data(logger)
            result['data_hash'] = data['data_hash']
            
            trained_models = []
            
            # Train each model type
            for model_type in model_types:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_type.upper()} model")
                logger.info(f"{'='*60}")
                
                try:
                    # Train
                    if model_type == 'als':
                        model_result = train_als_model(data, logger)
                    elif model_type == 'bpr':
                        model_result = train_bpr_model(data, logger)
                    else:
                        logger.error(f"Unknown model type: {model_type}")
                        continue
                    
                    # Evaluate
                    if not skip_eval:
                        metrics = evaluate_model(model_result, data, logger)
                    else:
                        metrics = {}
                    
                    # Save
                    save_result = save_model(model_result, metrics, data, logger)
                    
                    trained_models.append({
                        'model_id': save_result['model_id'],
                        'model_type': model_type,
                        'metrics': metrics,
                        'output_dir': save_result['output_dir'],
                        'training_time': model_result['training_time']
                    })
                    
                    result['models'][model_type] = {
                        'model_id': save_result['model_id'],
                        'metrics': metrics,
                        'training_time': model_result['training_time']
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_type}: {e}")
                    result['models'][model_type] = {'status': 'failed', 'error': str(e)}
            
            # Auto-select best model
            if auto_select and trained_models:
                logger.info("\n" + "="*60)
                logger.info("Selecting best model...")
                
                primary_metric = f"{TRAINING_CONFIG['primary_metric']}@{TRAINING_CONFIG['primary_k']}"
                best_model = max(
                    trained_models,
                    key=lambda m: m['metrics'].get(primary_metric, 0)
                )
                
                is_best = register_model(
                    best_model['model_id'],
                    best_model['model_type'],
                    best_model['metrics'],
                    Path(best_model['output_dir']),
                    logger
                )
                
                result['selected_model'] = best_model['model_id']
                result['is_new_best'] = is_best
            
            # Success
            result['status'] = 'success'
            result['finished_at'] = datetime.now().isoformat()
            
            tracker.complete_run(run_id, {
                'status': 'success',
                'models_trained': len(trained_models),
                'selected_model': result.get('selected_model')
            })
            
            logger.info("\nModel training completed successfully!")
            
            # Send alert
            send_pipeline_alert(
                'model_training',
                'success',
                f"Trained {len(trained_models)} models. Best: {result.get('selected_model')}",
                severity='info'
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Training pipeline failed: {error_msg}")
            
            result['status'] = 'failed'
            result['error'] = error_msg
            
            tracker.fail_run(run_id, error_msg)
            
            send_pipeline_alert(
                'model_training',
                'failed',
                f"Training failed: {error_msg}",
                severity='error'
            )
            
            raise
    
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train recommendation models"
    )
    parser.add_argument(
        '--model', '-m',
        choices=['als', 'bpr', 'both'],
        default='both',
        help='Which model(s) to train'
    )
    parser.add_argument(
        '--auto-select',
        action='store_true',
        default=True,
        help='Automatically register best model'
    )
    parser.add_argument(
        '--no-auto-select',
        action='store_false',
        dest='auto_select',
        help='Do not auto-register best model'
    )
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip evaluation after training'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force training even if recent model exists'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Determine models to train
    if args.model == 'both':
        model_types = ['als', 'bpr']
    else:
        model_types = [args.model]
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging("model_training", level=level)
    
    try:
        result = train_models(
            model_types=model_types,
            auto_select=args.auto_select,
            skip_eval=args.skip_eval,
            force=args.force,
            logger=logger
        )
        
        print(f"\n{'='*60}")
        print(f"Training Result: {result['status'].upper()}")
        print(f"{'='*60}")
        
        for model_type, model_info in result.get('models', {}).items():
            print(f"\n{model_type.upper()}:")
            if 'model_id' in model_info:
                print(f"  Model ID: {model_info['model_id']}")
                print(f"  Training Time: {model_info['training_time']:.1f}s")
                if model_info.get('metrics'):
                    print(f"  Recall@10: {model_info['metrics'].get('recall@10', 0):.4f}")
                    print(f"  NDCG@10: {model_info['metrics'].get('ndcg@10', 0):.4f}")
            else:
                print(f"  Status: {model_info.get('status', 'unknown')}")
                if 'error' in model_info:
                    print(f"  Error: {model_info['error']}")
        
        if 'selected_model' in result:
            print(f"\nSelected Best Model: {result['selected_model']}")
        
        sys.exit(0 if result['status'] == 'success' else 1)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
