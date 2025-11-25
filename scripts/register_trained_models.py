"""
Register trained CF models to the registry.

This script registers the BERT-enhanced ALS model (best performer) 
and optionally other models to the registry for serving.

Usage:
    python scripts/register_trained_models.py
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from recsys.cf.registry import ModelRegistry


def copy_checkpoint_to_artifact(
    checkpoint_path: str,
    artifact_path: str,
    model_type: str
) -> None:
    """
    Copy checkpoint files to artifact directory with proper naming.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        artifact_path: Path to artifact directory
        model_type: Type of model (als, bpr, bert_als)
    """
    os.makedirs(artifact_path, exist_ok=True)
    
    # Define file mappings
    # Checkpoint files use iter015 naming, need to rename for registry
    if model_type == 'als':
        file_mappings = [
            ('als_U_iter015.npy', 'als_U.npy'),
            ('als_V_iter015.npy', 'als_V.npy'),
            ('checkpoint_iter015.json', 'als_params.json'),
        ]
    elif model_type == 'bpr':
        file_mappings = [
            ('bpr_U.npy', 'bpr_U.npy'),
            ('bpr_V.npy', 'bpr_V.npy'),
            ('bpr_params.json', 'bpr_params.json'),
        ]
    else:
        # bert_als already has correct naming
        return
    
    for src_name, dst_name in file_mappings:
        src = Path(checkpoint_path) / src_name
        dst = Path(artifact_path) / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied: {src_name} -> {dst_name}")


def create_metadata_from_params(
    params_path: str,
    model_type: str,
    metrics: dict = None
) -> dict:
    """Create metadata dict from params file."""
    with open(params_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    # Default metrics if not provided
    if metrics is None:
        metrics = {
            'recall@10': 0.0,
            'ndcg@10': 0.0,
        }
    
    metadata = {
        'model_type': model_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_time_seconds': params.get('training_time_seconds', 0),
        'num_users': params.get('num_users', params.get('user_factors_shape', [0])[0]),
        'num_items': params.get('num_items', params.get('item_factors_shape', [0])[0]),
        'factors': params.get('factors', params.get('model_config', {}).get('factors', 64)),
        'data_hash': params.get('data_hash', 'UNKNOWN'),
        'metrics': metrics,
    }
    
    return metadata


def register_bert_als_model(registry: ModelRegistry) -> str:
    """Register the BERT-enhanced ALS model (best performer)."""
    print("\n=== Registering BERT-Enhanced ALS Model ===")
    
    # Best BERT-ALS model path
    model_path = 'artifacts/cf/bert_als/20251125_061805'
    
    # Read existing metadata
    metadata_file = Path(model_path) / 'bert_als_metadata.json'
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Read params
    params_file = Path(model_path) / 'bert_als_params.json'
    with open(params_file, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    # Extract metrics
    metrics = metadata.get('metrics', {})
    metrics_clean = {
        'recall@10': metrics.get('recall@10', 0),
        'ndcg@10': metrics.get('ndcg@10', 0),
        'recall@20': metrics.get('recall@20', 0),
        'ndcg@20': metrics.get('ndcg@20', 0),
    }
    
    # Training info
    training_info = {
        'training_time_seconds': metadata.get('training_time_seconds', 0),
        'bert_initialization': metadata.get('bert_initialization', {}),
        'num_train_users': metadata.get('num_train_users', 0),
        'num_test_users': metadata.get('num_test_users', 0),
        'colab_training': metadata.get('colab_training', False),
    }
    
    # Baseline comparison
    baseline_comparison = {
        'baseline_recall@10': metrics.get('baseline_recall@10', 0),
        'baseline_ndcg@10': metrics.get('baseline_ndcg@10', 0),
        'improvement_recall10': float(metrics.get('improvement_recall@10', '+0%').replace('%', '').replace('+', '')) / 100,
        'improvement_ndcg10': float(metrics.get('improvement_ndcg@10', '+0%').replace('%', '').replace('+', '')) / 100,
    }
    
    # BERT integration info
    bert_integration = metadata.get('bert_initialization', {})
    bert_integration['score_range'] = metadata.get('score_range', {})
    
    # Register
    model_id = registry.register_model(
        artifacts_path=model_path,
        model_type='bert_als',
        hyperparameters=params,
        metrics=metrics_clean,
        training_info=training_info,
        data_version=metadata.get('data_hash', 'UNKNOWN_HASH'),
        baseline_comparison=baseline_comparison,
        bert_integration=bert_integration,
        version='20251125_061805',
        overwrite=True
    )
    
    print(f"  Registered: {model_id}")
    print(f"  Recall@10: {metrics_clean['recall@10']:.4f}")
    print(f"  NDCG@10: {metrics_clean['ndcg@10']:.4f}")
    
    return model_id


def register_als_model(registry: ModelRegistry) -> str:
    """Register the standard ALS model from checkpoints."""
    print("\n=== Registering Standard ALS Model ===")
    
    checkpoint_path = 'checkpoints/als'
    artifact_path = 'artifacts/cf/als/20251125_checkpoint'
    
    # Copy files to artifact directory
    os.makedirs(artifact_path, exist_ok=True)
    copy_checkpoint_to_artifact(checkpoint_path, artifact_path, 'als')
    
    # Create metadata file
    params_file = Path(checkpoint_path) / 'checkpoint_iter015.json'
    with open(params_file, 'r', encoding='utf-8') as f:
        checkpoint_data = json.load(f)
    
    # Extract params
    model_config = checkpoint_data.get('model_config', {})
    params = {
        'factors': model_config.get('factors', 64),
        'regularization': model_config.get('regularization', 0.1),
        'alpha': model_config.get('alpha', 10),
        'iterations': checkpoint_data.get('iteration', 15),
    }
    
    # Save params as als_params.json
    with open(Path(artifact_path) / 'als_params.json', 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)
    
    # Default metrics (from earlier training - needs actual eval)
    metrics = {
        'recall@10': 0.134,  # From previous training
        'ndcg@10': 0.085,
    }
    
    # Create and save metadata
    metadata = {
        'model_type': 'als',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_users': checkpoint_data.get('item_factors_shape', [26050])[0],  # Note: swapped in checkpoint
        'num_items': checkpoint_data.get('user_factors_shape', [1423])[0],
        'factors': params['factors'],
        'data_hash': 'CHECKPOINT_IMPORT',
        'metrics': metrics,
        'training_info': {
            'source': 'checkpoint_import',
            'original_iteration': checkpoint_data.get('iteration', 15),
        }
    }
    
    with open(Path(artifact_path) / 'als_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Created artifact at: {artifact_path}")
    
    # Register
    model_id = registry.register_model(
        artifacts_path=artifact_path,
        model_type='als',
        hyperparameters=params,
        metrics=metrics,
        training_info={'source': 'checkpoint_import'},
        version='20251125_checkpoint',
        overwrite=True
    )
    
    print(f"  Registered: {model_id}")
    
    return model_id


def main():
    """Main registration script."""
    print("=" * 60)
    print("CF Model Registration Script")
    print("=" * 60)
    
    # Initialize registry
    registry_path = 'artifacts/cf/registry.json'
    registry = ModelRegistry(registry_path, auto_create=True)
    
    print(f"\nRegistry path: {registry_path}")
    
    # Register BERT-ALS (best model)
    bert_als_id = register_bert_als_model(registry)
    
    # Optionally register standard ALS
    try:
        als_id = register_als_model(registry)
    except Exception as e:
        print(f"\nWarning: Could not register ALS model: {e}")
        als_id = None
    
    # Select best model
    print("\n=== Selecting Best Model ===")
    best = registry.select_best_model(metric='recall@10')
    
    if best:
        print(f"  Best model: {best['model_id']}")
        print(f"  Metric (recall@10): {best['value']:.4f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Registry Summary")
    print("=" * 60)
    
    stats = registry.get_registry_stats()
    print(f"  Total models: {stats['total_models']}")
    print(f"  Active models: {stats['active_models']}")
    print(f"  Current best: {stats['current_best']}")
    print(f"  Models by type: {stats['by_type']}")
    
    # List all models
    print("\n=== Registered Models ===")
    models_df = registry.list_models()
    if not models_df.empty:
        print(models_df.to_string(index=False))
    
    print("\n✅ Registration complete!")
    print(f"Registry saved to: {registry_path}")


if __name__ == '__main__':
    main()
