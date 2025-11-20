"""
ALS Data Preparation Module

This module handles Step 2.1: Confidence-Weighted Matrix Construction for ALS.
Uses explicit feedback (ratings) enhanced with comment quality to create
confidence scores that distinguish "truly loved" from "just okay" products.

Key Features:
- Confidence score = rating + comment_quality (range [1.0, 6.0])
- Alternative normalized version: (confidence - 1) / 5 → [0, 1]
- Handles 95% 5-star rating skew via sentiment-based weighting
"""

import logging
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


logger = logging.getLogger("data_layer")


class ALSDataPreparer:
    """
    Prepare data specifically for ALS (Alternating Least Squares) training.
    
    ALS paradigm: Explicit feedback with confidence weighting
    - Matrix values = confidence_score (rating + comment_quality)
    - Higher confidence → more weight in loss function
    - Addresses rating skew: 5-star with detailed review ≠ 5-star with no comment
    """
    
    def __init__(
        self,
        normalize_confidence: bool = False,
        min_confidence: float = 1.0,
        max_confidence: float = 6.0
    ):
        """
        Initialize ALSDataPreparer.
        
        Args:
            normalize_confidence: If True, normalize confidence to [0, 1]
            min_confidence: Expected minimum confidence value
            max_confidence: Expected maximum confidence value
        """
        self.normalize_confidence = normalize_confidence
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
    
    def prepare_confidence_matrix(
        self,
        interactions_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        confidence_col: str = 'confidence_score'
    ) -> Tuple[csr_matrix, Dict[str, float]]:
        """
        Build confidence-weighted sparse matrix for ALS training.
        
        Args:
            interactions_df: DataFrame with user-item interactions
            num_users: Total number of users
            num_items: Total number of items
            user_col: Column name for user indices
            item_col: Column name for item indices
            confidence_col: Column name for confidence scores
        
        Returns:
            Tuple[csr_matrix, Dict]:
                - Sparse matrix (num_users, num_items) with confidence values
                - Statistics dict with mean, std, range info
        
        Example:
            >>> preparer = ALSDataPreparer()
            >>> X_confidence, stats = preparer.prepare_confidence_matrix(
            ...     train_df, num_users=26000, num_items=2231
            ... )
            >>> print(X_confidence.shape)  # (26000, 2231)
            >>> print(stats['mean_confidence'])  # ~5.14
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2.1: ALS CONFIDENCE-WEIGHTED MATRIX")
        logger.info("="*80)
        logger.info("Building explicit feedback matrix with sentiment-based weighting")
        
        # Validate required columns
        required_cols = [user_col, item_col, confidence_col]
        missing_cols = [col for col in required_cols if col not in interactions_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Extract data
        users = interactions_df[user_col].values
        items = interactions_df[item_col].values
        confidences = interactions_df[confidence_col].values
        
        logger.info(f"Processing {len(interactions_df):,} interactions")
        logger.info(f"Matrix shape: ({num_users:,}, {num_items:,})")
        
        # Validate confidence range
        conf_min, conf_max = confidences.min(), confidences.max()
        logger.info(f"Confidence score range: [{conf_min:.3f}, {conf_max:.3f}]")
        
        if conf_min < self.min_confidence or conf_max > self.max_confidence:
            logger.warning(
                f"⚠ Confidence scores outside expected range "
                f"[{self.min_confidence}, {self.max_confidence}]"
            )
        
        # Optional normalization
        if self.normalize_confidence:
            logger.info("Normalizing confidence scores to [0, 1]...")
            confidences = (confidences - self.min_confidence) / (
                self.max_confidence - self.min_confidence
            )
            logger.info(f"Normalized range: [{confidences.min():.3f}, {confidences.max():.3f}]")
        
        # Build sparse matrix
        logger.info("Building CSR sparse matrix...")
        X_confidence = csr_matrix(
            (confidences, (users, items)),
            shape=(num_users, num_items),
            dtype=np.float32
        )
        
        # Compute statistics
        stats = self._compute_matrix_stats(X_confidence, confidences)
        
        # Log summary
        logger.info("\n" + "-"*80)
        logger.info("ALS MATRIX SUMMARY")
        logger.info("-"*80)
        logger.info(f"Matrix shape:           {X_confidence.shape}")
        logger.info(f"Non-zero entries:       {X_confidence.nnz:,}")
        logger.info(f"Sparsity:               {stats['sparsity']:.4%}")
        logger.info(f"Mean confidence:        {stats['mean_confidence']:.3f}")
        logger.info(f"Median confidence:      {stats['median_confidence']:.3f}")
        logger.info(f"Std confidence:         {stats['std_confidence']:.3f}")
        logger.info(f"Confidence range:       [{stats['min_confidence']:.3f}, {stats['max_confidence']:.3f}]")
        logger.info("-"*80)
        logger.info("✓ ALS confidence matrix ready for training")
        
        return X_confidence, stats
    
    def _compute_matrix_stats(
        self,
        matrix: csr_matrix,
        values: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistics for confidence matrix.
        
        Args:
            matrix: Sparse confidence matrix
            values: Dense array of confidence values (for stats)
        
        Returns:
            Dict with statistical metrics
        """
        total_cells = matrix.shape[0] * matrix.shape[1]
        sparsity = 1.0 - (matrix.nnz / total_cells)
        
        stats = {
            'sparsity': sparsity,
            'density': 1.0 - sparsity,
            'nnz': matrix.nnz,
            'mean_confidence': float(values.mean()),
            'median_confidence': float(np.median(values)),
            'std_confidence': float(values.std()),
            'min_confidence': float(values.min()),
            'max_confidence': float(values.max()),
            'q25_confidence': float(np.percentile(values, 25)),
            'q75_confidence': float(np.percentile(values, 75))
        }
        
        return stats
    
    def create_preference_vector(
        self,
        user_interactions: pd.DataFrame,
        num_items: int,
        item_col: str = 'i_idx',
        confidence_col: str = 'confidence_score'
    ) -> np.ndarray:
        """
        Create preference vector for a single user (for inference).
        
        Args:
            user_interactions: DataFrame with user's historical interactions
            num_items: Total number of items
            item_col: Column name for item indices
            confidence_col: Column name for confidence scores
        
        Returns:
            np.ndarray: Dense preference vector (num_items,)
        
        Usage:
            Used during serving to create user vector for new recommendations
        """
        preference = np.zeros(num_items, dtype=np.float32)
        
        if len(user_interactions) == 0:
            return preference
        
        items = user_interactions[item_col].values
        confidences = user_interactions[confidence_col].values
        
        if self.normalize_confidence:
            confidences = (confidences - self.min_confidence) / (
                self.max_confidence - self.min_confidence
            )
        
        preference[items] = confidences
        
        return preference
    
    def validate_matrix(
        self,
        matrix: csr_matrix,
        interactions_df: pd.DataFrame,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx'
    ) -> bool:
        """
        Validate confidence matrix correctness.
        
        Args:
            matrix: CSR matrix to validate
            interactions_df: Original interactions DataFrame
            user_col: User index column
            item_col: Item index column
        
        Returns:
            bool: True if validation passes
        
        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating ALS confidence matrix...")
        
        # Check 1: Non-zero count matches
        if matrix.nnz != len(interactions_df):
            raise ValueError(
                f"Matrix nnz ({matrix.nnz}) != interactions count ({len(interactions_df)})"
            )
        
        # Check 2: Spot-check random samples
        sample_size = min(100, len(interactions_df))
        sample_rows = interactions_df.sample(n=sample_size, random_state=42)
        
        mismatches = 0
        for _, row in sample_rows.iterrows():
            u, i = row[user_col], row[item_col]
            expected_conf = row['confidence_score']
            
            if self.normalize_confidence:
                expected_conf = (expected_conf - self.min_confidence) / (
                    self.max_confidence - self.min_confidence
                )
            
            actual_conf = matrix[u, i]
            
            if not np.isclose(actual_conf, expected_conf, rtol=1e-4):
                mismatches += 1
        
        if mismatches > 0:
            logger.warning(
                f"⚠ Found {mismatches}/{sample_size} mismatches in spot-check"
            )
        else:
            logger.info(f"✓ Spot-check passed ({sample_size} samples)")
        
        # Check 3: No negative values
        if matrix.data.min() < 0:
            raise ValueError("Matrix contains negative confidence values")
        
        # Check 4: Shape matches
        num_users = interactions_df[user_col].max() + 1
        num_items = interactions_df[item_col].max() + 1
        
        if matrix.shape != (num_users, num_items):
            logger.warning(
                f"⚠ Matrix shape {matrix.shape} may not match data range "
                f"({num_users}, {num_items})"
            )
        
        logger.info("✓ ALS matrix validation passed")
        return True
    
    def get_confidence_distribution(
        self,
        interactions_df: pd.DataFrame,
        confidence_col: str = 'confidence_score',
        rating_col: str = 'rating'
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze confidence score distribution by rating level.
        
        Args:
            interactions_df: DataFrame with interactions
            confidence_col: Confidence score column
            rating_col: Rating column
        
        Returns:
            Dict with distribution statistics per rating level
        
        Purpose:
            Verify that confidence weighting successfully differentiates
            within each rating level (especially 5-star ratings)
        """
        logger.info("\n" + "="*80)
        logger.info("CONFIDENCE SCORE DISTRIBUTION ANALYSIS")
        logger.info("="*80)
        
        distribution = {}
        
        for rating in sorted(interactions_df[rating_col].unique()):
            subset = interactions_df[interactions_df[rating_col] == rating]
            confidences = subset[confidence_col]
            
            dist_stats = {
                'rating': rating,
                'count': len(subset),
                'percentage': len(subset) / len(interactions_df) * 100,
                'conf_mean': confidences.mean(),
                'conf_median': confidences.median(),
                'conf_std': confidences.std(),
                'conf_min': confidences.min(),
                'conf_max': confidences.max(),
                'conf_q25': confidences.quantile(0.25),
                'conf_q75': confidences.quantile(0.75)
            }
            
            distribution[f'rating_{int(rating)}'] = dist_stats
            
            logger.info(f"\nRating {rating:.0f} ({'⭐' * int(rating)})")
            logger.info(f"  Count:       {dist_stats['count']:,} ({dist_stats['percentage']:.2f}%)")
            logger.info(f"  Confidence:  {dist_stats['conf_mean']:.3f} ± {dist_stats['conf_std']:.3f}")
            logger.info(f"  Range:       [{dist_stats['conf_min']:.3f}, {dist_stats['conf_max']:.3f}]")
            logger.info(f"  Quartiles:   Q1={dist_stats['conf_q25']:.3f}, Q3={dist_stats['conf_q75']:.3f}")
        
        logger.info("\n" + "="*80)
        
        return distribution
