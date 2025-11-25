"""
ALS Training Pipeline - Step 1: Matrix Preparation Module

This module implements Step 1 of the ALS training pipeline with sentiment-enhanced
confidence scores. It provides comprehensive functionality for:

1. Loading and preparing confidence-weighted matrices
2. Confidence scaling strategies (raw vs normalized)
3. Preference matrix derivation (binary and continuous)
4. Alpha scaling recommendations
5. Matrix validation and quality analysis
6. Integration with DataProcessor and ALSDataPreparer

Key Features:
- Sentiment-enhanced confidence: rating (1-5) + comment_quality (0-1) → [1.0, 6.0]
- Optional normalization to [0, 1] range
- Binary preference derivation (threshold-based)
- Continuous preference derivation (normalized)
- Alpha scaling recommendations based on confidence range
- Comprehensive validation and statistics
- Ready for implicit library (transpose to item-user format)

Usage:
    >>> from recsys.cf.model.als.pre_data import ALSMatrixPreparer
    >>> preparer = ALSMatrixPreparer(base_path='data/processed')
    >>> matrices = preparer.prepare_complete_als_data(normalize_confidence=False)
    >>> X_train = matrices['X_train_confidence']
    >>> summary = matrices['training_summary']

Author: AI Assistant
Date: 2025-11-23
"""

import os
import json
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Import from data layer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from recsys.cf.data import DataProcessor
from recsys.cf.data.processing.als_data import ALSDataPreparer


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(log_file: str = "logs/cf/als_matrix_preparation.log") -> logging.Logger:
    """
    Configure logging for ALS matrix preparation.
    
    Args:
        log_file: Path to log file
    
    Returns:
        Logger instance
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger("als_matrix_prep")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
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
# Main ALS Matrix Preparation Class
# ============================================================================

class ALSMatrixPreparer:
    """
    Complete Step 1 implementation for ALS training pipeline.
    
    This class orchestrates all Step 1 sub-tasks:
    - 1.1: Load confidence matrix from processed data
    - 1.2: Apply confidence scaling strategies
    - 1.3: Derive preference matrices (binary and continuous)
    - Validate matrices and provide training summary
    - Prepare data for implicit library
    
    Attributes:
        base_path: Base directory for processed data
        processor: DataProcessor instance for data loading
        als_preparer: ALSDataPreparer instance for matrix operations
        normalize_confidence: Whether to normalize confidence to [0, 1]
        alpha: Alpha scaling factor for ALS
    
    Example:
        >>> preparer = ALSMatrixPreparer(
        ...     base_path='data/processed',
        ...     normalize_confidence=False,
        ...     alpha=10.0
        ... )
        >>> matrices = preparer.prepare_complete_als_data()
        >>> X_train = matrices['X_train_confidence']
        >>> summary = matrices['training_summary']
        >>> print(f"Matrix shape: {X_train.shape}")
        >>> print(f"Recommended alpha: {summary['recommended_alpha']}")
    """
    
    def __init__(
        self,
        base_path: str = 'data/processed',
        normalize_confidence: bool = False,
        alpha: float = 10.0,
        min_confidence: float = 1.0,
        max_confidence: float = 6.0
    ):
        """
        Initialize ALSMatrixPreparer.
        
        Args:
            base_path: Base directory for processed data files
            normalize_confidence: If True, normalize confidence to [0, 1]
            alpha: Alpha scaling factor for ALS training
            min_confidence: Expected minimum confidence value (default: 1.0)
            max_confidence: Expected maximum confidence value (default: 6.0)
        """
        self.base_path = Path(base_path)
        self.normalize_confidence = normalize_confidence
        self.alpha = alpha
        
        # Initialize DataProcessor
        self.processor = DataProcessor(
            base_path='data/published_data',
            positive_threshold=4.0,
            hard_negative_threshold=3.0
        )
        
        # Initialize ALSDataPreparer
        self.als_preparer = ALSDataPreparer(
            normalize_confidence=normalize_confidence,
            min_confidence=min_confidence,
            max_confidence=max_confidence
        )
        
        # Cache for loaded data
        self._interactions_df = None
        self._mappings = None
        self._num_users = None
        self._num_items = None
        
        logger.info("="*80)
        logger.info("ALS MATRIX PREPARER INITIALIZED")
        logger.info("="*80)
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Normalize confidence: {self.normalize_confidence}")
        logger.info(f"Alpha scaling: {self.alpha}")
        logger.info(f"Confidence range: [{min_confidence}, {max_confidence}]")
    
    # ========================================================================
    # Step 1.1: Load Confidence Matrix
    # ========================================================================
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Load preprocessed interactions and ID mappings (Step 1.1).
        
        Loads from Task 01 outputs:
        - interactions.parquet: Full interactions with confidence_score
        - user_item_mappings.json: ID mappings and metadata
        
        Returns:
            Tuple of (interactions_df, mappings_dict)
        
        Raises:
            FileNotFoundError: If required files don't exist
        
        Example:
            >>> df, mappings = preparer.load_processed_data()
            >>> print(f"Loaded {len(df)} interactions")
            >>> print(f"Users: {mappings['metadata']['num_users']}")
            >>> print(f"Items: {mappings['metadata']['num_items']}")
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1.1: LOAD PROCESSED DATA")
        logger.info("="*80)
        
        # Check if already loaded
        if self._interactions_df is not None and self._mappings is not None:
            logger.info("Using cached data")
            return self._interactions_df, self._mappings
        
        # Load interactions
        interactions_path = self.base_path / 'interactions.parquet'
        if not interactions_path.exists():
            raise FileNotFoundError(
                f"Interactions file not found: {interactions_path}\n"
                f"Please run Task 01 data pipeline first."
            )
        
        logger.info(f"Loading interactions from: {interactions_path}")
        self._interactions_df = pd.read_parquet(interactions_path)
        logger.info(f"Loaded {len(self._interactions_df):,} interactions")
        
        # Validate required columns
        required_cols = ['u_idx', 'i_idx', 'confidence_score', 'rating', 
                        'is_trainable_user', 'is_positive']
        missing_cols = [col for col in required_cols if col not in self._interactions_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in interactions: {missing_cols}\n"
                f"Available columns: {list(self._interactions_df.columns)}"
            )
        
        # Load mappings
        mappings_path = self.base_path / 'user_item_mappings.json'
        if not mappings_path.exists():
            raise FileNotFoundError(
                f"Mappings file not found: {mappings_path}\n"
                f"Please run Task 01 data pipeline first."
            )
        
        logger.info(f"Loading ID mappings from: {mappings_path}")
        with open(mappings_path, 'r', encoding='utf-8') as f:
            self._mappings = json.load(f)
        
        # Extract dimensions
        self._num_users = self._mappings['metadata']['num_users']
        self._num_items = self._mappings['metadata']['num_items']
        
        logger.info(f"Matrix dimensions: ({self._num_users:,}, {self._num_items:,})")
        logger.info("✓ Data loaded successfully")
        
        return self._interactions_df, self._mappings
    
    def prepare_confidence_matrix(
        self,
        trainable_only: bool = True,
        validate_matrix: bool = True
    ) -> Tuple[csr_matrix, Dict[str, float]]:
        """
        Build confidence-weighted CSR matrix for ALS (Step 1.1).
        
        Args:
            trainable_only: If True, only include trainable users (≥2 interactions)
            validate_matrix: If True, run validation checks
        
        Returns:
            Tuple of (csr_matrix, stats_dict)
            - csr_matrix: Sparse confidence matrix (num_users, num_items)
            - stats_dict: Statistics (sparsity, mean, std, etc.)
        
        Example:
            >>> X_conf, stats = preparer.prepare_confidence_matrix(trainable_only=True)
            >>> print(f"Sparsity: {stats['sparsity']:.4%}")
            >>> print(f"Mean confidence: {stats['mean_confidence']:.3f}")
        """
        # Load data if not already loaded
        if self._interactions_df is None:
            self.load_processed_data()
        
        # Filter to trainable users if requested
        df_train = self._interactions_df.copy()
        if trainable_only:
            logger.info("Filtering to trainable users only (is_trainable_user=True)...")
            initial_count = len(df_train)
            df_train = df_train[df_train['is_trainable_user'] == True]
            logger.info(
                f"Filtered: {initial_count:,} → {len(df_train):,} interactions "
                f"({len(df_train)/initial_count*100:.2f}% retained)"
            )
        
        # Build confidence matrix using ALSDataPreparer
        X_confidence, stats = self.als_preparer.prepare_confidence_matrix(
            interactions_df=df_train,
            num_users=self._num_users,
            num_items=self._num_items,
            user_col='u_idx',
            item_col='i_idx',
            confidence_col='confidence_score'
        )
        
        # Validate if requested
        if validate_matrix:
            logger.info("\nValidating confidence matrix...")
            is_valid = self.als_preparer.validate_matrix(
                matrix=X_confidence,
                interactions_df=df_train,
                user_col='u_idx',
                item_col='i_idx'
            )
            if not is_valid:
                logger.warning("⚠ Matrix validation failed!")
        
        return X_confidence, stats
    
    # ========================================================================
    # Step 1.2: Confidence Scaling Strategy
    # ========================================================================
    
    def get_alpha_recommendations(
        self,
        X_confidence: csr_matrix
    ) -> Dict[str, any]:
        """
        Get alpha scaling recommendations based on confidence distribution (Step 1.2).
        
        Args:
            X_confidence: Confidence matrix
        
        Returns:
            Dict with recommended alpha values and rationale
        
        Example:
            >>> X_conf, _ = preparer.prepare_confidence_matrix()
            >>> recommendations = preparer.get_alpha_recommendations(X_conf)
            >>> print(f"Recommended: {recommendations['recommended']}")
            >>> print(f"Rationale: {recommendations['rationale']}")
        """
        return self.als_preparer.get_recommended_alpha(
            X_confidence=X_confidence,
            is_normalized=self.normalize_confidence
        )
    
    def apply_alpha_scaling(
        self,
        X_confidence: csr_matrix,
        alpha: Optional[float] = None
    ) -> csr_matrix:
        """
        Apply alpha scaling to confidence matrix (Step 1.2).
        
        Args:
            X_confidence: Confidence matrix
            alpha: Alpha scaling factor (uses self.alpha if None)
        
        Returns:
            Scaled confidence matrix
        
        Formula:
            C_scaled[u,i] = 1 + alpha * confidence[u,i]
        
        Example:
            >>> X_conf, _ = preparer.prepare_confidence_matrix()
            >>> X_scaled = preparer.apply_alpha_scaling(X_conf, alpha=10.0)
        """
        if alpha is None:
            alpha = self.alpha
        
        return self.als_preparer.apply_alpha_scaling(X_confidence, alpha)
    
    def analyze_confidence_distribution(
        self,
        interactions_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze confidence score distribution by rating level (Step 1.2).
        
        Args:
            interactions_df: DataFrame with interactions (uses cached if None)
        
        Returns:
            Dict with distribution statistics per rating level
        
        Purpose:
            Verify that confidence weighting differentiates within rating levels
        
        Example:
            >>> dist = preparer.analyze_confidence_distribution()
            >>> print(dist['rating_5'])  # 5-star distribution
        """
        if interactions_df is None:
            if self._interactions_df is None:
                self.load_processed_data()
            interactions_df = self._interactions_df
        
        return self.als_preparer.get_confidence_distribution(
            interactions_df=interactions_df,
            confidence_col='confidence_score',
            rating_col='rating'
        )
    
    # ========================================================================
    # Step 1.3: Preference Matrix Derivation
    # ========================================================================
    
    def derive_binary_preference(
        self,
        X_confidence: csr_matrix,
        threshold: float = 4.5
    ) -> csr_matrix:
        """
        Derive binary preference matrix from confidence (Step 1.3).
        
        Args:
            X_confidence: Confidence matrix
            threshold: Confidence threshold (≥ threshold → preference = 1)
        
        Returns:
            Binary preference matrix (values 0 or 1)
        
        Formula:
            P[u,i] = 1 if confidence[u,i] >= threshold, else 0
        
        Example:
            >>> X_conf, _ = preparer.prepare_confidence_matrix()
            >>> P_binary = preparer.derive_binary_preference(X_conf, threshold=4.5)
            >>> print(f"Positive preferences: {P_binary.data.sum()}")
        """
        return self.als_preparer.derive_binary_preference_matrix(
            X_confidence=X_confidence,
            threshold=threshold
        )
    
    def derive_continuous_preference(
        self,
        X_confidence: csr_matrix,
        normalize_to_01: bool = True
    ) -> csr_matrix:
        """
        Derive continuous preference matrix from confidence (Step 1.3).
        
        Args:
            X_confidence: Confidence matrix
            normalize_to_01: If True, normalize to [0, 1] range
        
        Returns:
            Continuous preference matrix
        
        Formula:
            If normalize_to_01=True:
                P[u,i] = (confidence[u,i] - 1) / 5
            Else:
                P[u,i] = confidence[u,i]
        
        Example:
            >>> X_conf, _ = preparer.prepare_confidence_matrix()
            >>> P_cont = preparer.derive_continuous_preference(X_conf)
        """
        return self.als_preparer.derive_continuous_preference_matrix(
            X_confidence=X_confidence,
            normalize_to_01=normalize_to_01
        )
    
    # ========================================================================
    # Integration with Implicit Library
    # ========================================================================
    
    def prepare_for_implicit(
        self,
        X_confidence: csr_matrix,
        transpose: bool = True
    ) -> csr_matrix:
        """
        Prepare matrix for implicit library (item-user format).
        
        Args:
            X_confidence: Confidence matrix (user-item format)
            transpose: If True, transpose to item-user format
        
        Returns:
            Matrix in implicit library format (items × users)
        
        Note:
            implicit library expects shape (num_items, num_users)
        
        Example:
            >>> X_conf, _ = preparer.prepare_confidence_matrix()
            >>> X_train = preparer.prepare_for_implicit(X_conf)
            >>> print(X_train.shape)  # (num_items, num_users)
        """
        return self.als_preparer.prepare_for_implicit_library(
            X_confidence=X_confidence,
            transpose=transpose
        )
    
    # ========================================================================
    # Complete Pipeline & Training Summary
    # ========================================================================
    
    def prepare_complete_als_data(
        self,
        trainable_only: bool = True,
        include_binary_preference: bool = False,
        include_continuous_preference: bool = False,
        binary_threshold: float = 4.5,
        prepare_for_training: bool = True
    ) -> Dict[str, any]:
        """
        Complete Step 1 pipeline: Prepare all ALS matrices and summary.
        
        This is the main entry point for Step 1, executing all sub-tasks:
        1.1: Load confidence matrix
        1.2: Analyze confidence distribution and get alpha recommendations
        1.3: Derive preference matrices (optional)
        - Validate matrices
        - Prepare for implicit library
        - Generate training summary
        
        Args:
            trainable_only: Filter to trainable users only
            include_binary_preference: Generate binary preference matrix
            include_continuous_preference: Generate continuous preference matrix
            binary_threshold: Threshold for binary preference derivation
            prepare_for_training: Transpose matrix for implicit library
        
        Returns:
            Dict with all prepared matrices and metadata:
            {
                'X_train_confidence': csr_matrix (user × item),
                'X_train_implicit': csr_matrix (item × user, if prepare_for_training=True),
                'P_binary': csr_matrix (optional),
                'P_continuous': csr_matrix (optional),
                'num_users': int,
                'num_items': int,
                'confidence_stats': Dict,
                'alpha_recommendations': Dict,
                'training_summary': Dict,
                'ready_for_training': bool
            }
        
        Example:
            >>> preparer = ALSMatrixPreparer(base_path='data/processed')
            >>> data = preparer.prepare_complete_als_data(
            ...     trainable_only=True,
            ...     include_binary_preference=True
            ... )
            >>> X_train = data['X_train_implicit']  # For implicit library
            >>> summary = data['training_summary']
            >>> print(f"Recommended alpha: {summary['recommended_alpha']}")
        """
        logger.info("\n" + "="*80)
        logger.info("COMPLETE ALS MATRIX PREPARATION PIPELINE")
        logger.info("="*80)
        
        results = {}
        
        # Step 1.1: Load and build confidence matrix
        logger.info("\n[1/5] Building confidence matrix...")
        X_confidence, conf_stats = self.prepare_confidence_matrix(
            trainable_only=trainable_only,
            validate_matrix=True
        )
        results['X_train_confidence'] = X_confidence
        results['confidence_stats'] = conf_stats
        results['num_users'] = self._num_users
        results['num_items'] = self._num_items
        
        # Step 1.2: Analyze confidence distribution
        logger.info("\n[2/5] Analyzing confidence distribution...")
        conf_distribution = self.analyze_confidence_distribution()
        results['confidence_distribution'] = conf_distribution
        
        # Step 1.2: Get alpha recommendations
        logger.info("\n[3/5] Computing alpha recommendations...")
        alpha_rec = self.get_alpha_recommendations(X_confidence)
        results['alpha_recommendations'] = alpha_rec
        
        # Step 1.3: Derive preference matrices (optional)
        if include_binary_preference:
            logger.info("\n[4/5] Deriving binary preference matrix...")
            P_binary = self.derive_binary_preference(X_confidence, binary_threshold)
            results['P_binary'] = P_binary
        
        if include_continuous_preference:
            logger.info("\n[4/5] Deriving continuous preference matrix...")
            P_continuous = self.derive_continuous_preference(X_confidence, normalize_to_01=True)
            results['P_continuous'] = P_continuous
        
        # Prepare for implicit library
        if prepare_for_training:
            logger.info("\n[5/5] Preparing matrix for implicit library (transpose)...")
            X_train_implicit = self.prepare_for_implicit(X_confidence, transpose=True)
            results['X_train_implicit'] = X_train_implicit
        
        # Generate comprehensive training summary
        logger.info("\nGenerating training summary...")
        training_summary = self.als_preparer.get_als_training_summary(
            X_confidence=X_confidence,
            alpha=self.alpha,
            normalize=self.normalize_confidence
        )
        results['training_summary'] = training_summary
        results['ready_for_training'] = True
        
        # Log final summary
        logger.info("\n" + "="*80)
        logger.info("ALS STEP 1 COMPLETE")
        logger.info("="*80)
        logger.info(f"Matrix shape (user × item):    {X_confidence.shape}")
        if prepare_for_training:
            logger.info(f"Matrix shape (item × user):    {results['X_train_implicit'].shape}")
        logger.info(f"Interactions:                  {X_confidence.nnz:,}")
        logger.info(f"Sparsity:                      {conf_stats['sparsity']:.4%}")
        logger.info(f"Mean confidence:               {conf_stats['mean_confidence']:.3f}")
        logger.info(f"Confidence range:              [{conf_stats['min_confidence']:.3f}, {conf_stats['max_confidence']:.3f}]")
        logger.info(f"Recommended alpha:             {alpha_rec['recommended']}")
        logger.info(f"Current alpha:                 {self.alpha}")
        logger.info("="*80)
        logger.info("✓ Ready for ALS training (Step 2)")
        
        return results
    
    # ========================================================================
    # Save/Load Artifacts
    # ========================================================================
    
    def save_matrices(
        self,
        matrices: Dict[str, any],
        output_dir: str = 'artifacts/cf/als/step1'
    ) -> Dict[str, str]:
        """
        Save prepared matrices and metadata to disk.
        
        Args:
            matrices: Dict from prepare_complete_als_data()
            output_dir: Output directory for artifacts
        
        Returns:
            Dict mapping artifact name to saved file path
        
        Example:
            >>> data = preparer.prepare_complete_als_data()
            >>> saved_paths = preparer.save_matrices(data, output_dir='artifacts/cf/als/step1')
            >>> print(f"Saved {len(saved_paths)} artifacts")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save confidence matrix
        if 'X_train_confidence' in matrices:
            from scipy.sparse import save_npz
            conf_path = output_path / 'X_train_confidence.npz'
            save_npz(conf_path, matrices['X_train_confidence'])
            saved_paths['X_train_confidence'] = str(conf_path)
            logger.info(f"Saved confidence matrix: {conf_path}")
        
        # Save implicit library matrix
        if 'X_train_implicit' in matrices:
            from scipy.sparse import save_npz
            impl_path = output_path / 'X_train_implicit.npz'
            save_npz(impl_path, matrices['X_train_implicit'])
            saved_paths['X_train_implicit'] = str(impl_path)
            logger.info(f"Saved implicit matrix: {impl_path}")
        
        # Save binary preference
        if 'P_binary' in matrices:
            from scipy.sparse import save_npz
            binary_path = output_path / 'P_binary.npz'
            save_npz(binary_path, matrices['P_binary'])
            saved_paths['P_binary'] = str(binary_path)
            logger.info(f"Saved binary preference: {binary_path}")
        
        # Save continuous preference
        if 'P_continuous' in matrices:
            from scipy.sparse import save_npz
            cont_path = output_path / 'P_continuous.npz'
            save_npz(cont_path, matrices['P_continuous'])
            saved_paths['P_continuous'] = str(cont_path)
            logger.info(f"Saved continuous preference: {cont_path}")
        
        # Save metadata and summary
        metadata = {
            'num_users': matrices.get('num_users'),
            'num_items': matrices.get('num_items'),
            'confidence_stats': matrices.get('confidence_stats'),
            'alpha_recommendations': matrices.get('alpha_recommendations'),
            'training_summary': matrices.get('training_summary'),
            'normalize_confidence': self.normalize_confidence,
            'alpha': self.alpha
        }
        
        metadata_path = output_path / 'step1_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        saved_paths['metadata'] = str(metadata_path)
        logger.info(f"Saved metadata: {metadata_path}")
        
        logger.info(f"\n✓ Saved {len(saved_paths)} artifacts to {output_dir}")
        
        return saved_paths


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_prepare_als_matrix(
    base_path: str = 'data/processed',
    normalize: bool = False,
    alpha: float = 10.0,
    trainable_only: bool = True
) -> Tuple[csr_matrix, Dict[str, any]]:
    """
    Quick function to prepare ALS confidence matrix with defaults.
    
    Args:
        base_path: Path to processed data
        normalize: Normalize confidence to [0, 1]
        alpha: Alpha scaling factor
        trainable_only: Filter to trainable users
    
    Returns:
        Tuple of (X_train_implicit, training_summary)
    
    Example:
        >>> X_train, summary = quick_prepare_als_matrix()
        >>> print(f"Ready for training: {summary['ready_for_training']}")
    """
    preparer = ALSMatrixPreparer(
        base_path=base_path,
        normalize_confidence=normalize,
        alpha=alpha
    )
    
    results = preparer.prepare_complete_als_data(
        trainable_only=trainable_only,
        prepare_for_training=True
    )
    
    return results['X_train_implicit'], results['training_summary']


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    """
    Test ALS Matrix Preparation - Step 1 Complete
    """
    print("="*80)
    print("TESTING ALS MATRIX PREPARATION - STEP 1")
    print("="*80)
    
    try:
        # Initialize preparer
        preparer = ALSMatrixPreparer(
            base_path='data/processed',
            normalize_confidence=False,
            alpha=10.0
        )
        
        # Execute complete Step 1 pipeline
        print("\n[TEST 1] Complete Step 1 Pipeline")
        results = preparer.prepare_complete_als_data(
            trainable_only=True,
            include_binary_preference=True,
            include_continuous_preference=True,
            prepare_for_training=True
        )
        
        # Verify outputs
        print("\n[TEST 2] Verify Outputs")
        assert 'X_train_confidence' in results, "Missing confidence matrix"
        assert 'X_train_implicit' in results, "Missing implicit matrix"
        assert 'P_binary' in results, "Missing binary preference"
        assert 'P_continuous' in results, "Missing continuous preference"
        assert 'training_summary' in results, "Missing training summary"
        
        X_conf = results['X_train_confidence']
        X_impl = results['X_train_implicit']
        summary = results['training_summary']
        
        print(f"✓ Confidence matrix shape: {X_conf.shape}")
        print(f"✓ Implicit matrix shape: {X_impl.shape}")
        print(f"✓ Interactions: {X_conf.nnz:,}")
        print(f"✓ Sparsity: {summary['sparsity']:.4%}")
        print(f"✓ Recommended alpha: {summary['recommended_alpha']}")
        
        # Test alpha recommendations
        print("\n[TEST 3] Alpha Recommendations")
        alpha_rec = preparer.get_alpha_recommendations(X_conf)
        print(f"✓ Recommended: {alpha_rec['recommended']}")
        print(f"✓ Alternatives: {alpha_rec['alternatives']}")
        print(f"✓ Rationale: {alpha_rec['rationale']}")
        
        # Test save artifacts
        print("\n[TEST 4] Save Artifacts")
        saved_paths = preparer.save_matrices(
            results,
            output_dir='artifacts/cf/als/step1_test'
        )
        print(f"✓ Saved {len(saved_paths)} artifacts")
        
        # Quick prepare test
        print("\n[TEST 5] Quick Prepare Function")
        X_quick, summary_quick = quick_prepare_als_matrix(
            base_path='data/processed',
            normalize=False,
            alpha=10.0
        )
        print(f"✓ Quick matrix shape: {X_quick.shape}")
        print(f"✓ Ready for training: {summary_quick['ready_for_training']}")
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nStep 1 is complete and ready for Step 2 (Model Training)")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        print("\n" + "="*80)
        print("TEST FAILED ✗")
        print("="*80)
        raise
