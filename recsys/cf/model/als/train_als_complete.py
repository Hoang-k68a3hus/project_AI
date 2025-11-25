"""
Complete ALS Training Pipeline Orchestrator

This script runs the full ALS training pipeline from data loading to artifact saving:
- Step 1: Load preprocessed data from Task 01
- Step 2: Initialize ALS model with configuration
- Step 3: Train model with progress tracking
- Step 4: Extract and normalize embeddings
- Step 5: Generate recommendations for evaluation
- Step 6: Evaluate with baseline comparison
- Step 7: Save all artifacts with score range for Task 08

Usage:
    # Default configuration
    python scripts/train_als_complete.py
    
    # Custom configuration
    python scripts/train_als_complete.py --config config/als_config.yaml --output artifacts/cf/als
    
    # With validation set for score range
    python scripts/train_als_complete.py --val-ratio 0.1
    
    # GPU acceleration
    python scripts/train_als_complete.py --use-gpu

Author: Copilot
Date: 2025-01-15
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy.sparse import csr_matrix
import json
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import ALS modules
from recsys.cf.model.als import (
    ALSModelInitializer,
    ALSTrainer,
    EmbeddingExtractor,
    ALSRecommender,
    ALSEvaluator,
    save_als_complete
)


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(log_file: Optional[str] = None, verbose: bool = False) -> logging.Logger:
    """
    Configure logging for the training pipeline.
    
    Args:
        log_file: Path to log file (default: logs/cf/als_training.log)
        verbose: If True, set log level to DEBUG
    
    Returns:
        Logger instance
    """
    if log_file is None:
        log_file = "logs/cf/als_training.log"
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("als_pipeline")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
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
# Data Loading from Task 01 Outputs
# ============================================================================

class ALSDataLoader:
    """
    Load preprocessed data from Task 01 pipeline outputs.
    
    Expected files in data_dir:
        - X_train_confidence.npz: CSR matrix for ALS training
        - user_item_mappings.json: ID mappings (user_to_idx, item_to_idx, etc.)
        - user_pos_train.pkl: Training positive sets
        - user_pos_test.pkl: Test positive sets
        - data_stats.json: Data statistics (optional)
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        logger.info(f"Initializing ALSDataLoader from {self.data_dir}")
    
    def load_training_matrix(self) -> csr_matrix:
        """Load confidence matrix for ALS training."""
        matrix_path = self.data_dir / "X_train_confidence.npz"
        
        if not matrix_path.exists():
            raise FileNotFoundError(
                f"Training matrix not found: {matrix_path}\n"
                f"Please run Task 01 data pipeline first (scripts/run_task01_complete.py)"
            )
        
        from scipy.sparse import load_npz
        X_train = load_npz(matrix_path)
        
        logger.info(f"Loaded training matrix: {X_train.shape}, density: {X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.6f}")
        
        return X_train
    
    def load_id_mappings(self) -> Dict[str, Any]:
        """
        Load user and item ID mappings with trainable user re-mapping.
        
        CRITICAL: Handles LOCAL vs GLOBAL index mapping
        - Global mappings: All 300K users (0-299,999)
        - Training matrix: LOCAL indices for trainable users only (0-25,999)
        - This method loads trainable_user_mapping.json to resolve the mismatch
        """
        mappings_path = self.data_dir / "user_item_mappings.json"
        
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
        trainable_map_path = self.data_dir / "trainable_user_mapping.json"
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
        
        # Extract metadata
        metadata = {
            'positive_threshold': data.get('positive_threshold', 4.0),
            'hard_negative_threshold': data.get('hard_negative_threshold', 3.0),
            'data_hash': data.get('data_hash'),
            'timestamp': data.get('timestamp')
        }
        
        return mappings, metadata
    
    def load_user_sets(self) -> Tuple[Dict[int, set], Dict[int, set]]:
        """Load training and test positive sets."""
        import pickle
        
        train_path = self.data_dir / "user_pos_train.pkl"
        test_path = self.data_dir / "user_pos_test.pkl"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training sets not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test sets not found: {test_path}")
        
        with open(train_path, 'rb') as f:
            user_pos_train = pickle.load(f)
        
        with open(test_path, 'rb') as f:
            user_pos_test = pickle.load(f)
        
        logger.info(f"Loaded user sets: {len(user_pos_train)} train users, {len(user_pos_test)} test users")
        
        return user_pos_train, user_pos_test
    
    def load_data_stats(self) -> Optional[Dict[str, Any]]:
        """Load data statistics (optional)."""
        stats_path = self.data_dir / "data_stats.json"
        
        if not stats_path.exists():
            logger.warning(f"Data stats not found: {stats_path}")
            return None
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        logger.info(f"Loaded data statistics")
        
        return stats
    
    def load_all(self) -> Dict[str, Any]:
        """Load all required data for ALS training."""
        logger.info("Loading all data from Task 01 outputs...")
        
        data = {
            'X_train': self.load_training_matrix(),
            'user_pos_train': None,
            'user_pos_test': None,
            'mappings': None,
            'metadata': None,
            'stats': None
        }
        
        # Load mappings
        mappings, metadata = self.load_id_mappings()
        data['mappings'] = mappings
        data['metadata'] = metadata
        
        # Load user sets
        user_pos_train, user_pos_test = self.load_user_sets()
        data['user_pos_train'] = user_pos_train
        data['user_pos_test'] = user_pos_test
        
        # Load stats (optional)
        data['stats'] = self.load_data_stats()
        
        logger.info("Data loading complete!")
        
        return data


# ============================================================================
# Configuration Management
# ============================================================================

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to YAML config file (optional)
    
    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        'factors': 64,
        'regularization': 0.01,
        'iterations': 15,
        'alpha': 10,
        'random_seed': 42,
        'normalize_embeddings': False,
        'k_values': [10, 20],
        'compare_baseline': True,
        'baseline_source': 'train'
    }
    
    if config_path is None:
        logger.info("Using default configuration")
        return default_config
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return default_config
    
    with open(config_path, 'r') as f:
        file_config = yaml.safe_load(f)
    
    # Merge with defaults (file config takes precedence)
    config = default_config.copy()
    
    # Extract ALS section if present
    if 'als' in file_config:
        als_config = file_config['als']
        config.update({
            'factors': als_config.get('factors', config['factors']),
            'regularization': als_config.get('regularization', config['regularization']),
            'iterations': als_config.get('iterations', config['iterations']),
            'alpha': als_config.get('alpha', config['alpha']),
            'random_seed': als_config.get('random_seed', config['random_seed'])
        })
    
    # Extract evaluation section if present
    if 'evaluation' in file_config:
        eval_config = file_config['evaluation']
        config.update({
            'k_values': eval_config.get('k_values', config['k_values'])
        })
    
    logger.info(f"Loaded configuration from {config_path}")
    
    return config


# ============================================================================
# Validation Set Extraction
# ============================================================================

def extract_validation_users(
    user_pos_train: Dict[int, set],
    user_pos_test: Dict[int, set],
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[list, list]:
    """
    Extract validation users from test set for score range computation.
    
    Args:
        user_pos_train: Training positive sets
        user_pos_test: Test positive sets
        val_ratio: Ratio of test users to use for validation (default: 0.1)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (validation_user_indices, remaining_test_user_indices)
    """
    np.random.seed(random_seed)
    
    # Get all test users
    test_users = list(user_pos_test.keys())
    
    # Sample validation users
    num_val = max(1, int(len(test_users) * val_ratio))
    val_users = np.random.choice(test_users, num_val, replace=False).tolist()
    
    # Remaining test users
    test_users_set = set(test_users)
    val_users_set = set(val_users)
    remaining_test_users = list(test_users_set - val_users_set)
    
    logger.info(f"Extracted {len(val_users)} validation users ({val_ratio*100:.1f}% of test set)")
    logger.info(f"Remaining test users: {len(remaining_test_users)}")
    
    return val_users, remaining_test_users


# ============================================================================
# Main Training Pipeline
# ============================================================================

class ALSTrainingPipeline:
    """
    Complete ALS training pipeline orchestrator.
    
    Executes all 7 steps:
        1. Load data (handled by ALSDataLoader)
        2. Initialize model
        3. Train model
        4. Extract embeddings
        5. Generate recommendations
        6. Evaluate with baseline
        7. Save artifacts with score range
    """
    
    def __init__(
        self,
        data_dir: str = "data/processed",
        output_dir: str = "artifacts/cf/als",
        config: Optional[Dict[str, Any]] = None,
        val_ratio: float = 0.1,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 5,
        track_memory: bool = False
    ):
        """
        Initialize training pipeline.
        
        Args:
            data_dir: Directory with Task 01 preprocessed data
            output_dir: Directory to save artifacts
            config: Training configuration dictionary
            val_ratio: Ratio of test users for validation (score range)
            checkpoint_dir: Directory for training checkpoints
            checkpoint_interval: Save checkpoint every N iterations
            track_memory: Enable memory profiling
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        self.val_ratio = val_ratio
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.track_memory = track_memory
        
        # Pipeline state
        self.data = None
        self.model = None
        self.trainer = None
        self.U = None
        self.V = None
        self.evaluation_results = None
        self.artifacts = None
        self.val_users = None
        self.test_users = None
        
        logger.info("="*80)
        logger.info("ALS Training Pipeline Initialized")
        logger.info("="*80)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Configuration: {self.config}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Returns:
            Dictionary with all results and artifacts
        """
        start_time = time.time()
        
        try:
            # Step 1: Load data
            logger.info("\n" + "="*80)
            logger.info("STEP 1: Loading preprocessed data from Task 01")
            logger.info("="*80)
            self.data = self._load_data()
            
            # Extract validation users
            self.val_users, self.test_users = extract_validation_users(
                self.data['user_pos_train'],
                self.data['user_pos_test'],
                val_ratio=self.val_ratio,
                random_seed=self.config.get('random_seed', 42)
            )
            
            # Step 2: Initialize model
            logger.info("\n" + "="*80)
            logger.info("STEP 2: Initializing ALS model")
            logger.info("="*80)
            self.model = self._initialize_model()
            
            # Step 3: Train model
            logger.info("\n" + "="*80)
            logger.info("STEP 3: Training ALS model")
            logger.info("="*80)
            training_summary = self._train_model()
            
            # Step 4: Extract embeddings
            logger.info("\n" + "="*80)
            logger.info("STEP 4: Extracting embeddings")
            logger.info("="*80)
            self.U, self.V = self._extract_embeddings()
            
            # Step 5 & 6: Evaluate (includes recommendation generation)
            logger.info("\n" + "="*80)
            logger.info("STEP 5 & 6: Evaluating model")
            logger.info("="*80)
            self.evaluation_results = self._evaluate_model()
            
            # Step 7: Save artifacts
            logger.info("\n" + "="*80)
            logger.info("STEP 7: Saving artifacts with score range")
            logger.info("="*80)
            self.artifacts = self._save_artifacts(training_summary)
            
            # Pipeline summary
            total_time = time.time() - start_time
            logger.info("\n" + "="*80)
            logger.info("ALS TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f}m)")
            logger.info(f"Artifacts saved to: {self.output_dir}")
            
            # Print results summary
            self.evaluation_results.print_summary()
            print(self.artifacts.summary())
            
            return {
                'data': self.data,
                'model': self.model,
                'embeddings': {'U': self.U, 'V': self.V},
                'evaluation': self.evaluation_results,
                'artifacts': self.artifacts,
                'total_time': total_time
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _load_data(self) -> Dict[str, Any]:
        """Step 1: Load data from Task 01."""
        loader = ALSDataLoader(data_dir=str(self.data_dir))
        data = loader.load_all()
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Training matrix: {data['X_train'].shape}")
        logger.info(f"  Train users: {len(data['user_pos_train'])}")
        logger.info(f"  Test users: {len(data['user_pos_test'])}")
        
        return data
    
    def _initialize_model(self):
        """Step 2: Initialize ALS model."""
        initializer = ALSModelInitializer(
            factors=self.config.get('factors', 64),
            regularization=self.config.get('regularization', 0.01),
            iterations=self.config.get('iterations', 15),
            alpha=self.config.get('alpha', 10),
            random_seed=self.config.get('random_seed', 42)
        )
        
        model = initializer.initialize_model()
        
        logger.info("Model initialized:")
        logger.info(f"  Factors: {model.factors}")
        logger.info(f"  Regularization: {model.regularization}")
        logger.info(f"  Iterations: {model.iterations}")
        logger.info(f"  Alpha: {model.alpha}")
        
        return model
    
    def _train_model(self) -> Dict[str, Any]:
        """Step 3: Train ALS model."""
        self.trainer = ALSTrainer(
            model=self.model,
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_interval=self.checkpoint_interval,
            track_memory=self.track_memory
        )
        
        training_summary = self.trainer.fit(
            X_train=self.data['X_train'],
            show_progress=True
        )
        
        logger.info(f"Training completed:")
        logger.info(f"  Training time: {training_summary['training_time']:.2f}s")
        logger.info(f"  Iterations: {training_summary['iterations']}")
        
        if 'peak_memory_mb' in training_summary:
            logger.info(f"  Peak memory: {training_summary['peak_memory_mb']:.2f} MB")
        
        return training_summary
    
    def _extract_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Step 4: Extract embeddings."""
        normalize = self.config.get('normalize_embeddings', False)
        
        extractor = EmbeddingExtractor(
            model=self.trainer.model,
            normalize=normalize
        )
        
        U, V = extractor.get_embeddings()
        
        logger.info(f"Embeddings extracted:")
        logger.info(f"  User embeddings (U): {U.shape}")
        logger.info(f"  Item embeddings (V): {V.shape}")
        logger.info(f"  Normalized: {normalize}")
        
        # Get embedding statistics
        stats = extractor.get_embedding_statistics()
        logger.info(f"  User embedding norms: mean={stats['user_norms']['mean']:.4f}, std={stats['user_norms']['std']:.4f}")
        logger.info(f"  Item embedding norms: mean={stats['item_norms']['mean']:.4f}, std={stats['item_norms']['std']:.4f}")
        
        return U, V
    
    def _evaluate_model(self):
        """Steps 5 & 6: Generate recommendations and evaluate."""
        evaluator = ALSEvaluator(
            user_factors=self.U,
            item_factors=self.V,
            user_to_idx=self.data['mappings']['user_to_idx'],
            idx_to_user=self.data['mappings']['idx_to_user'],
            item_to_idx=self.data['mappings']['item_to_idx'],
            idx_to_item=self.data['mappings']['idx_to_item'],
            user_pos_train=self.data['user_pos_train'],
            user_pos_test={u: self.data['user_pos_test'][u] for u in self.test_users}  # Use remaining test users
        )
        
        results = evaluator.evaluate(
            k_values=self.config.get('k_values', [10, 20]),
            filter_seen=True,
            compare_baseline=self.config.get('compare_baseline', True),
            baseline_source=self.config.get('baseline_source', 'train'),
            model_type='als'
        )
        
        return results
    
    def _save_artifacts(self, training_summary: Dict[str, Any]):
        """Step 7: Save all artifacts with score range."""
        # Prepare parameters
        params = {
            'factors': self.config.get('factors', 64),
            'regularization': self.config.get('regularization', 0.01),
            'iterations': self.config.get('iterations', 15),
            'alpha': self.config.get('alpha', 10),
            'random_seed': self.config.get('random_seed', 42),
            'normalize_embeddings': self.config.get('normalize_embeddings', False),
            'training_time_seconds': training_summary.get('training_time', 0)
        }
        
        # Prepare metrics (flatten EvaluationResult)
        metrics = self.evaluation_results.metrics.copy()
        metrics.update({
            f'baseline_{k}': v 
            for k, v in self.evaluation_results.baseline_metrics.items()
        })
        metrics.update({
            f'improvement_{k}': v 
            for k, v in self.evaluation_results.improvement.items()
        })
        
        # Additional metadata
        additional_metadata = {
            'num_train_users': len(self.data['user_pos_train']),
            'num_test_users': len(self.test_users),
            'num_val_users': len(self.val_users),
            'val_ratio': self.val_ratio,
            'data_stats': self.data.get('stats')
        }
        
        # Save complete artifacts
        artifacts = save_als_complete(
            user_embeddings=self.U,
            item_embeddings=self.V,
            params=params,
            metrics=metrics,
            output_dir=str(self.output_dir),
            validation_user_indices=self.val_users,
            data_version_hash=self.data['metadata'].get('data_hash'),
            model_object=self.trainer.model,
            additional_metadata=additional_metadata,
            model_type='als',
            compute_score_range_flag=True
        )
        
        return artifacts


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Complete ALS Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory with Task 01 preprocessed data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/cf/als',
        help='Directory to save artifacts'
    )
    
    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file'
    )
    
    # Model hyperparameters (override config)
    parser.add_argument(
        '--factors',
        type=int,
        default=None,
        help='Embedding dimension'
    )
    
    parser.add_argument(
        '--regularization',
        type=float,
        default=None,
        help='L2 regularization'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Number of ALS iterations'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='Confidence scaling parameter'
    )
    
    parser.add_argument(
        '--normalize-embeddings',
        action='store_true',
        help='L2 normalize embeddings'
    )
    
    # Evaluation
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[10, 20],
        help='K values for Recall@K and NDCG@K'
    )
    
    parser.add_argument(
        '--no-baseline',
        action='store_true',
        help='Skip baseline comparison'
    )
    
    # Validation
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Ratio of test users for validation (score range)'
    )
    
    # Training options
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory for training checkpoints'
    )
    
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=5,
        help='Save checkpoint every N iterations'
    )
    
    parser.add_argument(
        '--track-memory',
        action='store_true',
        help='Enable memory profiling'
    )
    
    # Logging
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(log_file=args.log_file, verbose=args.verbose)
    
    logger.info("="*80)
    logger.info("ALS COMPLETE TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI arguments
    if args.factors is not None:
        config['factors'] = args.factors
    if args.regularization is not None:
        config['regularization'] = args.regularization
    if args.iterations is not None:
        config['iterations'] = args.iterations
    if args.alpha is not None:
        config['alpha'] = args.alpha
    if args.normalize_embeddings:
        config['normalize_embeddings'] = True
    if args.k_values:
        config['k_values'] = args.k_values
    if args.no_baseline:
        config['compare_baseline'] = False
    
    logger.info(f"Final configuration: {config}")
    
    # Create pipeline
    pipeline = ALSTrainingPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        val_ratio=args.val_ratio,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        track_memory=args.track_memory
    )
    
    # Run pipeline
    try:
        results = pipeline.run()
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Status: SUCCESS")
        logger.info(f"Total time: {results['total_time']:.2f}s")
        logger.info(f"Artifacts: {results['artifacts'].output_dir}")
        logger.info(f"Key metrics:")
        for k, v in sorted(results['evaluation'].metrics.items())[:4]:
            logger.info(f"  {k}: {v:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error("\n" + "="*80)
        logger.error("PIPELINE EXECUTION FAILED")
        logger.error("="*80)
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
