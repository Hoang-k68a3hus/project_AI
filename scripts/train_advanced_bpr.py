"""
Training Script for Advanced BPR with Smart Sampling

This script trains the enhanced BPR model with:
1. BERT-initialized embeddings (optional)
2. Advanced negative sampling (contextual, sentiment-contrast, dynamic)
3. Modern optimizers (AdamW, AdaGrad) with differential regularization
4. Embedding dropout and gradient clipping
5. Learning rate warmup and decay

Features:
- Contextual negatives: BERT-similar items user hasn't bought
- Sentiment-contrast negatives: Items with opposite sentiment
- Dynamic difficulty: Harder negatives as training progresses
- AdamW optimizer with separate weight decay for user/item
- Early stopping and checkpointing

Usage:
    python scripts/train_advanced_bpr.py --config config/advanced_bpr_config.yaml
    python scripts/train_advanced_bpr.py --factors 64 --optimizer adamw --epochs 50
    python scripts/train_advanced_bpr.py --use-bert --contextual-ratio 0.2 --dynamic-sampling

Author: VieComRec Team
Date: 2025-11-26
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Set

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from recsys.cf.model.bpr import (
    BPRDataLoader,
    AdvancedTripletSampler,
    SamplingStrategy,
    DynamicSamplingConfig,
    AdvancedBPRTrainer,
    OptimizerType,
    OptimizerConfig,
    TrainingConfig,
    SchedulerType,
    save_bpr_complete
)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Advanced BPR with Smart Sampling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--bert-embeddings', type=str,
                       default='data/processed/content_based_embeddings/product_embeddings.pt',
                       help='Path to BERT embeddings')
    parser.add_argument('--output-dir', type=str, default='artifacts/cf/bpr',
                       help='Output directory for artifacts')
    
    # Model architecture
    parser.add_argument('--factors', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--use-bert', action='store_true',
                       help='Initialize item embeddings from BERT')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Mini-batch size')
    parser.add_argument('--samples-per-positive', type=int, default=5,
                       help='Negative samples per positive per epoch')
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['sgd', 'adagrad', 'adam', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--user-weight-decay', type=float, default=None,
                       help='Separate weight decay for user embeddings')
    parser.add_argument('--item-weight-decay', type=float, default=None,
                       help='Separate weight decay for item embeddings')
    parser.add_argument('--scheduler', type=str, default='warmup_cosine',
                       choices=['constant', 'linear', 'cosine', 'exponential', 'warmup_cosine'],
                       help='Learning rate schedule')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs for LR schedule')
    
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Embedding dropout rate')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping norm (0 to disable)')
    
    # Sampling strategy
    parser.add_argument('--hard-ratio', type=float, default=0.25,
                       help='Fraction from hard negatives')
    parser.add_argument('--contextual-ratio', type=float, default=0.20,
                       help='Fraction from contextual negatives (BERT-similar)')
    parser.add_argument('--sentiment-ratio', type=float, default=0.15,
                       help='Fraction from sentiment-contrast negatives')
    parser.add_argument('--popular-ratio', type=float, default=0.10,
                       help='Fraction from popular negatives')
    parser.add_argument('--dynamic-sampling', action='store_true',
                       help='Enable dynamic difficulty adjustment')
    parser.add_argument('--difficulty-schedule', type=str, default='cosine',
                       choices=['linear', 'exponential', 'cosine'],
                       help='How difficulty increases over training')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--min-delta', type=float, default=0.001,
                       help='Minimum improvement for early stopping')
    parser.add_argument('--eval-every', type=int, default=5,
                       help='Evaluate every N epochs')
    
    # Other
    parser.add_argument('--checkpoint-every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def _load_pickle(path: Path, fallback) -> Any:
    """Utility to load pickle files with graceful fallback."""
    if path.exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    logger.warning(f"  Missing file: {path}")
    return fallback() if callable(fallback) else fallback


def _get_git_commit_hash() -> str:
    """Return current git commit hash (cross-platform)."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return 'unknown'


def load_data(args) -> Dict[str, Any]:
    """Load training data from processed Task 01 artifacts."""
    logger.info("Loading training data from processed artifacts...")
    
    data_dir = Path(args.data_dir)
    loader = BPRDataLoader(base_path=data_dir)
    loader_data = loader.load_all()
    
    # Optional test positives for evaluation
    user_pos_test_path = data_dir / 'user_pos_test.pkl'
    user_pos_test = _load_pickle(user_pos_test_path, dict)
    logger.info(f"  Loaded user_pos_test for {len(user_pos_test):,} users")
    
    data = {
        'positive_pairs': loader_data['positive_pairs'],
        'user_pos_train': loader_data['user_pos_sets'],
        'user_hard_neg_train': loader_data['hard_neg_sets'],
        'user_pos_test': user_pos_test,
        'num_users': loader_data['num_users'],
        'num_items': loader_data['num_items'],
        'mappings': loader_data.get('mappings', {})
    }
    
    logger.info(f"  Users: {data['num_users']:,}")
    logger.info(f"  Items: {data['num_items']:,}")
    logger.info(f"  Positive pairs: {len(data['positive_pairs']):,}")
    logger.info(f"  Users with hard negatives: {len(data['user_hard_neg_train']):,}")
    
    return data


def load_bert_embeddings(bert_path: str, num_items: int) -> Optional[np.ndarray]:
    """Load BERT embeddings for items."""
    bert_path = Path(bert_path)
    
    if not bert_path.exists():
        logger.warning(f"BERT embeddings not found at {bert_path}")
        return None
    
    logger.info(f"Loading BERT embeddings from {bert_path}...")
    
    try:
        if bert_path.suffix == '.pt':
            import torch
            embeddings = torch.load(bert_path)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.numpy()
            elif isinstance(embeddings, dict):
                # Handle dict format (item_idx -> embedding)
                embed_dim = len(next(iter(embeddings.values())))
                arr = np.zeros((num_items, embed_dim))
                for raw_idx, emb in embeddings.items():
                    try:
                        idx = int(raw_idx)
                    except (TypeError, ValueError):
                        logger.debug(f"  Skipping BERT embedding key {raw_idx!r} (non-integer)")
                        continue
                    if 0 <= idx < num_items:
                        arr[idx] = emb if isinstance(emb, np.ndarray) else emb.numpy()
                embeddings = arr
        else:
            embeddings = np.load(bert_path)
        
        logger.info(f"  Loaded embeddings shape: {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        logger.error(f"  Error loading BERT embeddings: {e}")
        return None


def load_item_metadata(data_dir: str) -> Dict[str, Any]:
    """Load item metadata for advanced sampling."""
    import pandas as pd
    
    metadata = {
        'item_sentiment_scores': {},
        'item_popularity': {},
        'item_categories': {},
        'item_brands': {}
    }
    
    product_path = Path(data_dir).parent / 'published_data' / 'data_product.csv'
    
    if product_path.exists():
        try:
            df = pd.read_csv(product_path, encoding='utf-8')
            logger.info(f"Loaded product metadata: {len(df)} items")
            
            # Load mappings
            mappings_path = Path(data_dir) / 'user_item_mappings.json'
            if mappings_path.exists():
                with open(mappings_path, 'r') as f:
                    mappings = json.load(f)
                item_to_idx = mappings.get('item_to_idx', {})
            else:
                item_to_idx = {}
            
            for _, row in df.iterrows():
                product_id = str(row.get('product_id', ''))
                if product_id in item_to_idx:
                    idx = item_to_idx[product_id]
                    
                    # Popularity
                    if 'num_sold_time' in row:
                        metadata['item_popularity'][idx] = float(row['num_sold_time'] or 0)
                    
                    # Brand
                    if 'brand' in row and pd.notna(row['brand']):
                        metadata['item_brands'][idx] = str(row['brand'])
                    
                    # Category (if available)
                    if 'category_level_2' in row and pd.notna(row['category_level_2']):
                        metadata['item_categories'][idx] = str(row['category_level_2'])
                    
                    # Sentiment (use avg_star as proxy)
                    if 'avg_star' in row:
                        # Normalize to [0, 1]
                        metadata['item_sentiment_scores'][idx] = float(row['avg_star'] or 3.0) / 5.0
            
            logger.info(f"  Popularity scores: {len(metadata['item_popularity'])}")
            logger.info(f"  Sentiment scores: {len(metadata['item_sentiment_scores'])}")
            logger.info(f"  Categories: {len(metadata['item_categories'])}")
            logger.info(f"  Brands: {len(metadata['item_brands'])}")
            
        except Exception as e:
            logger.warning(f"Error loading product metadata: {e}")
    
    return metadata


def project_bert_to_factors(
    bert_embeddings: np.ndarray,
    target_dim: int,
    random_seed: int = 42
) -> np.ndarray:
    """Project BERT embeddings to target dimension using SVD."""
    from scipy.linalg import svd
    
    logger.info(f"Projecting BERT embeddings {bert_embeddings.shape} -> {target_dim} dims...")
    
    # Center embeddings
    centered = bert_embeddings - bert_embeddings.mean(axis=0)
    
    # SVD projection
    U, S, Vt = svd(centered, full_matrices=False)
    
    # Project to target dimension
    projected = U[:, :target_dim] * S[:target_dim]
    
    # Normalize
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    projected = projected / norms * 0.01  # Scale similar to random init
    
    logger.info(f"  Projected shape: {projected.shape}")
    
    return projected


def create_sampler(
    args,
    data: Dict[str, Any],
    bert_embeddings: Optional[np.ndarray],
    metadata: Dict[str, Any]
) -> AdvancedTripletSampler:
    """Create advanced triplet sampler."""
    logger.info("Creating advanced triplet sampler...")
    
    strategy = SamplingStrategy(
        hard_ratio=args.hard_ratio,
        contextual_ratio=args.contextual_ratio,
        sentiment_contrast_ratio=args.sentiment_ratio,
        popular_ratio=args.popular_ratio
    )
    logger.info(f"  Strategy: {strategy.to_dict()}")
    
    dynamic_config = DynamicSamplingConfig(
        enable_dynamic=args.dynamic_sampling,
        warmup_epochs=args.warmup_epochs,
        difficulty_schedule=args.difficulty_schedule,
        initial_difficulty=0.3,
        final_difficulty=0.8,
        resample_hard_negatives=True
    )
    logger.info(f"  Dynamic sampling: {args.dynamic_sampling}")
    
    sampler = AdvancedTripletSampler(
        positive_pairs=data['positive_pairs'],
        user_pos_sets=data['user_pos_train'],
        num_items=data['num_items'],
        hard_neg_sets=data.get('user_hard_neg_train'),
        item_embeddings=bert_embeddings,
        item_categories=metadata.get('item_categories'),
        item_sentiment_scores=metadata.get('item_sentiment_scores'),
        item_popularity=metadata.get('item_popularity'),
        strategy=strategy,
        dynamic_config=dynamic_config,
        samples_per_positive=args.samples_per_positive,
        random_seed=args.seed
    )
    
    return sampler


def create_trainer(
    args,
    data: Dict[str, Any],
    bert_embeddings: Optional[np.ndarray]
) -> AdvancedBPRTrainer:
    """Create advanced BPR trainer."""
    logger.info("Creating advanced BPR trainer...")
    
    # Optimizer config
    optimizer_map = {
        'sgd': OptimizerType.SGD,
        'adagrad': OptimizerType.ADAGRAD,
        'adam': OptimizerType.ADAM,
        'adamw': OptimizerType.ADAMW
    }
    
    scheduler_map = {
        'constant': SchedulerType.CONSTANT,
        'linear': SchedulerType.LINEAR,
        'cosine': SchedulerType.COSINE,
        'exponential': SchedulerType.EXPONENTIAL,
        'warmup_cosine': SchedulerType.WARMUP_COSINE
    }
    
    opt_config = OptimizerConfig(
        optimizer_type=optimizer_map[args.optimizer],
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        user_weight_decay=args.user_weight_decay,
        item_weight_decay=args.item_weight_decay,
        scheduler_type=scheduler_map[args.scheduler],
        warmup_epochs=args.warmup_epochs
    )
    logger.info(f"  Optimizer: {args.optimizer}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    
    training_config = TrainingConfig(
        factors=args.factors,
        epochs=args.epochs,
        samples_per_positive=args.samples_per_positive,
        batch_size=args.batch_size,
        dropout_rate=args.dropout,
        gradient_clip=args.gradient_clip,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        random_seed=args.seed
    )
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  Gradient clip: {args.gradient_clip}")
    
    # Prepare item embeddings
    init_item_embeddings = None
    if args.use_bert and bert_embeddings is not None:
        init_item_embeddings = project_bert_to_factors(
            bert_embeddings, args.factors, args.seed
        )
    
    # Checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = Path(args.output_dir) / f'checkpoints_{timestamp}'
    
    trainer = AdvancedBPRTrainer(
        num_users=data['num_users'],
        num_items=data['num_items'],
        training_config=training_config,
        optimizer_config=opt_config,
        init_item_embeddings=init_item_embeddings,
        checkpoint_dir=checkpoint_dir
    )
    
    return trainer


def save_results(
    args,
    trainer: AdvancedBPRTrainer,
    training_result: Dict[str, Any],
    sampler: AdvancedTripletSampler,
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """Save training results and artifacts."""
    logger.info("Saving training artifacts...")
    
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f'advanced_bpr_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Get embeddings
    U, V = trainer.get_embeddings()
    
    # Compute score range
    sample_users = np.random.choice(len(U), min(1000, len(U)), replace=False)
    all_scores = []
    for u in sample_users:
        scores = U[u] @ V.T
        all_scores.extend(scores[:100])
    all_scores = np.array(all_scores)
    score_range = {
        'min': float(all_scores.min()),
        'max': float(all_scores.max()),
        'mean': float(all_scores.mean()),
        'std': float(all_scores.std()),
        'p01': float(np.percentile(all_scores, 1)),
        'p99': float(np.percentile(all_scores, 99))
    }
    
    # Save embeddings
    np.save(run_dir / 'BPR_U.npy', U)
    np.save(run_dir / 'BPR_V.npy', V)
    logger.info(f"  Saved embeddings: U={U.shape}, V={V.shape}")
    
    # Save parameters
    params = {
        'model_type': 'advanced_bpr',
        'factors': args.factors,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'user_weight_decay': args.user_weight_decay,
        'item_weight_decay': args.item_weight_decay,
        'scheduler': args.scheduler,
        'warmup_epochs': args.warmup_epochs,
        'dropout': args.dropout,
        'gradient_clip': args.gradient_clip,
        'batch_size': args.batch_size,
        'samples_per_positive': args.samples_per_positive,
        'use_bert_init': args.use_bert,
        'sampling_strategy': {
            'hard_ratio': args.hard_ratio,
            'contextual_ratio': args.contextual_ratio,
            'sentiment_ratio': args.sentiment_ratio,
            'popular_ratio': args.popular_ratio,
            'dynamic_sampling': args.dynamic_sampling
        }
    }
    with open(run_dir / 'params.json', 'w') as f:
        json.dump(params, f, indent=2)
    
    # Save metrics
    metrics = {
        'best_metric': training_result['best_metric'],
        'best_epoch': training_result['best_epoch'],
        'total_time_seconds': training_result['total_time'],
        'final_train_loss': training_result['history']['train_loss'][-1] if training_result['history']['train_loss'] else None
    }
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save metadata
    metadata = {
        'model_type': 'advanced_bpr',
        'timestamp': timestamp,
        'num_users': data['num_users'],
        'num_items': data['num_items'],
        'num_positive_pairs': len(data['positive_pairs']),
        'score_range': score_range,
        'training_history': training_result['history'],
        'sampling_stats': sampler.get_sampling_stats(),
        'git_commit': _get_git_commit_hash()
    }
    with open(run_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"  Artifacts saved to: {run_dir}")
    
    return {
        'output_dir': str(run_dir),
        'params': params,
        'metrics': metrics,
        'metadata': metadata
    }


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("Advanced BPR Training")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # 1. Load data
    data = load_data(args)
    
    # 2. Load BERT embeddings (optional)
    bert_embeddings = None
    if args.use_bert or args.contextual_ratio > 0:
        bert_embeddings = load_bert_embeddings(args.bert_embeddings, data['num_items'])
    
    # 3. Load item metadata
    metadata = load_item_metadata(args.data_dir)
    
    # 4. Create sampler
    sampler = create_sampler(args, data, bert_embeddings, metadata)
    
    # 5. Create trainer
    trainer = create_trainer(args, data, bert_embeddings)
    
    # 6. Train model
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)
    
    training_result = trainer.fit(
        sampler=sampler,
        user_pos_test=data.get('user_pos_test'),
        user_pos_train=data.get('user_pos_train'),
        eval_metric='Recall@10',
        verbose=True
    )
    
    # 7. Save results
    results = save_results(args, trainer, training_result, sampler, data)
    
    # 8. Summary
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Best Recall@10: {training_result['best_metric']:.4f}")
    logger.info(f"Best epoch: {training_result['best_epoch'] + 1}")
    logger.info(f"Artifacts: {results['output_dir']}")
    
    # Print sampling stats
    sampling_stats = sampler.get_sampling_stats()
    logger.info("\nSampling Statistics:")
    logger.info(f"  Total samples: {sampling_stats['total_samples']:,}")
    logger.info(f"  Hard ratio: {sampling_stats['hard_ratio']:.1%}")
    logger.info(f"  Contextual ratio: {sampling_stats['contextual_ratio']:.1%}")
    logger.info(f"  Sentiment ratio: {sampling_stats['sentiment_ratio']:.1%}")
    logger.info(f"  Popular ratio: {sampling_stats['popular_ratio']:.1%}")
    logger.info(f"  Random ratio: {sampling_stats['random_ratio']:.1%}")
    
    return results


if __name__ == '__main__':
    main()
