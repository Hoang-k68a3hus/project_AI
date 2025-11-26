"""
Train BERT-Enhanced BPR Model Script

This script trains the BERT-Enhanced BPR model with:
1. PhoBERT-initialized item factors
2. Sentiment-aware confidence weighting

Usage:
    python scripts/train_bert_bpr.py --config config/bert_bpr_config.yaml
    python scripts/train_bert_bpr.py --factors 64 --epochs 50 --sentiment-weighting

Author: VieComRec Team
Date: 2025-11-26
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from recsys.cf.model.bert_enhanced_bpr import (
    BERTEnhancedBPR,
    SentimentAwareConfig,
    compute_score_range_bpr
)
from recsys.cf.data.processing.bpr_data import BPRDataPreparer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/cf/bert_bpr_training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def load_processed_data(data_dir: Path) -> dict:
    """
    Load processed data from Task 01 pipeline.
    
    Args:
        data_dir: Path to data/processed directory
    
    Returns:
        Dictionary with interactions DataFrame and mappings
    """
    logger.info(f"Loading processed data from {data_dir}")
    
    # Load interactions
    interactions_path = data_dir / "interactions.parquet"
    if interactions_path.exists():
        interactions_df = pd.read_parquet(interactions_path)
        logger.info(f"Loaded {len(interactions_df):,} interactions")
    else:
        raise FileNotFoundError(f"Interactions file not found: {interactions_path}")
    
    # Load mappings
    mappings_path = data_dir / "user_item_mappings.json"
    if mappings_path.exists():
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        logger.info(f"Loaded mappings: {mappings['metadata']['num_users']} users, "
                   f"{mappings['metadata']['num_items']} items")
    else:
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")
    
    return {
        'interactions_df': interactions_df,
        'mappings': mappings,
        'item_to_idx': mappings['item_to_idx'],
        'num_users': mappings['metadata']['num_users'],
        'num_items': mappings['metadata']['num_items']
    }


def prepare_bpr_data_with_confidence(
    interactions_df: pd.DataFrame,
    positive_threshold: float = 4.0,
    hard_negative_threshold: float = 3.0
) -> dict:
    """
    Prepare BPR training data with confidence scores.
    
    Args:
        interactions_df: DataFrame with interactions
        positive_threshold: Rating threshold for positive
        hard_negative_threshold: Rating threshold for hard negatives
    
    Returns:
        Dictionary with training data and confidence scores
    """
    logger.info("Preparing BPR data with sentiment-aware confidence...")
    
    # Filter to trainable users (from Task 01)
    if 'is_trainable_user' in interactions_df.columns:
        trainable_df = interactions_df[interactions_df['is_trainable_user'] == True].copy()
        logger.info(f"Filtered to trainable users: {len(trainable_df):,} interactions")
    else:
        trainable_df = interactions_df.copy()
        logger.warning("is_trainable_user column not found, using all data")
    
    # Use BPRDataPreparer for data preparation
    preparer = BPRDataPreparer(
        positive_threshold=positive_threshold,
        hard_negative_threshold=hard_negative_threshold,
        top_k_popular=50,
        hard_negative_ratio=0.3
    )
    
    # Check if confidence_score exists
    if 'confidence_score' in trainable_df.columns:
        logger.info("Using sentiment-aware confidence scores")
        data = preparer.get_bpr_training_data_with_confidence(
            trainable_df,
            user_col='u_idx',
            item_col='i_idx',
            rating_col='rating',
            confidence_col='confidence_score'
        )
    else:
        logger.warning("confidence_score not found, using standard BPR data")
        data = preparer.get_bpr_training_data(
            trainable_df,
            user_col='u_idx',
            item_col='i_idx',
            rating_col='rating'
        )
        # Create dummy confidence scores
        data['confidence_scores'] = trainable_df[
            trainable_df['rating'] >= positive_threshold
        ]['rating'].values.astype(np.float32)
    
    return data


def train_bert_bpr(args):
    """Main training function."""
    logger.info("="*80)
    logger.info("BERT-ENHANCED BPR TRAINING")
    logger.info("="*80)
    
    start_time = time.time()
    
    # Paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bert_embeddings_path = Path(args.bert_embeddings) if args.bert_embeddings else None
    
    # Load data
    logger.info("\n" + "-"*40)
    logger.info("Step 1: Load Data")
    logger.info("-"*40)
    
    processed_data = load_processed_data(data_dir)
    
    # Prepare BPR data with confidence
    logger.info("\n" + "-"*40)
    logger.info("Step 2: Prepare BPR Data")
    logger.info("-"*40)
    
    bpr_data = prepare_bpr_data_with_confidence(
        processed_data['interactions_df'],
        positive_threshold=args.positive_threshold,
        hard_negative_threshold=args.hard_negative_threshold
    )
    
    # Configure sentiment weighting
    logger.info("\n" + "-"*40)
    logger.info("Step 3: Configure Model")
    logger.info("-"*40)
    
    sentiment_config = SentimentAwareConfig(
        use_sentiment_weighting=args.sentiment_weighting,
        confidence_col='confidence_score',
        confidence_min=1.0,
        confidence_max=6.0,
        positive_confidence_threshold=args.confidence_threshold,
        suspicious_penalty=args.suspicious_penalty,
        high_confidence_bonus=args.confidence_bonus
    )
    
    logger.info(f"Sentiment weighting: {sentiment_config.use_sentiment_weighting}")
    logger.info(f"Confidence threshold: {sentiment_config.positive_confidence_threshold}")
    logger.info(f"Suspicious penalty: {sentiment_config.suspicious_penalty}")
    logger.info(f"High confidence bonus: {sentiment_config.high_confidence_bonus}")
    
    # Initialize model
    model = BERTEnhancedBPR(
        bert_embeddings_path=str(bert_embeddings_path) if bert_embeddings_path and bert_embeddings_path.exists() else None,
        factors=args.factors,
        projection_method=args.projection_method,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        lr_decay=args.lr_decay,
        lr_decay_every=args.lr_decay_every,
        sentiment_config=sentiment_config,
        random_seed=args.seed
    )
    
    logger.info(f"Model initialized:")
    logger.info(f"  Factors: {args.factors}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Regularization: {args.regularization}")
    logger.info(f"  BERT embeddings: {bert_embeddings_path}")
    
    # Train model
    logger.info("\n" + "-"*40)
    logger.info("Step 4: Train Model")
    logger.info("-"*40)
    
    # Get item_to_idx for BERT alignment
    item_to_idx = processed_data.get('item_to_idx', None)
    
    summary = model.fit(
        positive_pairs=bpr_data['positive_pairs'],
        user_pos_sets=bpr_data['user_pos_sets'],
        num_users=processed_data['num_users'],
        num_items=processed_data['num_items'],
        item_to_idx=item_to_idx,
        hard_neg_sets=bpr_data.get('hard_neg_sets'),
        confidence_scores=bpr_data.get('confidence_scores'),
        epochs=args.epochs,
        samples_per_positive=args.samples_per_positive,
        hard_ratio=args.hard_ratio,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        early_stopping_metric='recall@10',
        show_progress=True,
        checkpoint_dir=output_dir / "checkpoints" if args.save_checkpoints else None,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Compute score range for normalization
    logger.info("\n" + "-"*40)
    logger.info("Step 5: Compute Score Range")
    logger.info("-"*40)
    
    U, V = model.get_embeddings()
    score_range = compute_score_range_bpr(U, V, sample_size=1000)
    
    # Save artifacts
    logger.info("\n" + "-"*40)
    logger.info("Step 6: Save Artifacts")
    logger.info("-"*40)
    
    # Add metadata
    metadata = {
        'data_hash': processed_data['mappings'].get('metadata', {}).get('data_hash'),
        'training_config': {
            'factors': args.factors,
            'learning_rate': args.learning_rate,
            'regularization': args.regularization,
            'epochs': args.epochs,
            'positive_threshold': args.positive_threshold,
            'hard_negative_threshold': args.hard_negative_threshold,
            'sentiment_weighting': args.sentiment_weighting,
            'confidence_threshold': args.confidence_threshold,
        },
        'score_range': score_range,
        'training_summary': {
            'total_duration_seconds': summary['total_duration_seconds'],
            'epochs_completed': summary['epochs_completed'],
            'best_epoch': summary['best_epoch'],
            'final_loss': summary['final_loss'],
            'bert_init_used': summary['bert_init_used'],
        }
    }
    
    model.save_artifacts(output_dir, metadata=metadata)
    
    # Save training summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, 'w') as f:
        # Convert numpy types for JSON serialization
        summary_json = json.loads(
            json.dumps(summary, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x)
        )
        json.dump(summary_json, f, indent=2)
    
    total_time = time.time() - start_time
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"Epochs completed: {summary['epochs_completed']}")
    logger.info(f"Best epoch: {summary['best_epoch']}")
    logger.info(f"Final loss: {summary['final_loss']:.4f}")
    logger.info(f"BERT init used: {summary['bert_init_used']}")
    logger.info(f"Sentiment weighting: {summary['sentiment_weighting_enabled']}")
    logger.info(f"Artifacts saved to: {output_dir}")
    logger.info("="*80)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train BERT-Enhanced BPR Model with Sentiment-Aware Confidence"
    )
    
    # Data paths
    parser.add_argument(
        '--data-dir', type=str, default='data/processed',
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--output-dir', type=str, default='artifacts/cf/bert_bpr',
        help='Path to output directory for artifacts'
    )
    parser.add_argument(
        '--bert-embeddings', type=str, 
        default='data/processed/content_based_embeddings/product_embeddings.pt',
        help='Path to BERT embeddings file (.pt)'
    )
    
    # Model hyperparameters
    parser.add_argument('--factors', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--learning-rate', type=float, default=0.05, help='Initial learning rate')
    parser.add_argument('--regularization', type=float, default=0.0001, help='L2 regularization')
    parser.add_argument('--lr-decay', type=float, default=0.9, help='Learning rate decay factor')
    parser.add_argument('--lr-decay-every', type=int, default=10, help='Decay LR every N epochs')
    parser.add_argument('--projection-method', type=str, default='svd', 
                       choices=['svd', 'pca'], help='BERT projection method')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='Mini-batch size')
    parser.add_argument('--samples-per-positive', type=int, default=5, 
                       help='Samples per positive pair per epoch')
    parser.add_argument('--hard-ratio', type=float, default=0.3, 
                       help='Fraction of hard negatives (rest random)')
    parser.add_argument('--patience', type=int, default=5, 
                       help='Early stopping patience (epochs)')
    
    # Thresholds
    parser.add_argument('--positive-threshold', type=float, default=4.0, 
                       help='Rating threshold for positive')
    parser.add_argument('--hard-negative-threshold', type=float, default=3.0,
                       help='Rating threshold for hard negatives')
    
    # Sentiment-aware weighting
    parser.add_argument('--sentiment-weighting', action='store_true', default=True,
                       help='Enable sentiment-aware confidence weighting')
    parser.add_argument('--no-sentiment-weighting', action='store_false', dest='sentiment_weighting',
                       help='Disable sentiment-aware confidence weighting')
    parser.add_argument('--confidence-threshold', type=float, default=4.5,
                       help='Confidence threshold for trusted positives')
    parser.add_argument('--suspicious-penalty', type=float, default=0.5,
                       help='Weight penalty for suspicious reviews')
    parser.add_argument('--confidence-bonus', type=float, default=1.5,
                       help='Weight bonus for high-confidence reviews')
    
    # Checkpointing
    parser.add_argument('--save-checkpoints', action='store_true', default=False,
                       help='Save checkpoints during training')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Other
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create logs directory
    Path('logs/cf').mkdir(parents=True, exist_ok=True)
    
    # Run training
    train_bert_bpr(args)


if __name__ == "__main__":
    main()
