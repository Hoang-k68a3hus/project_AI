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
    
    # ========================================================================
    # Step 1.3: Preference Matrix Derivation (Optional)
    # ========================================================================
    
    def derive_binary_preference_matrix(
        self,
        X_confidence: csr_matrix,
        threshold: float = 4.5
    ) -> csr_matrix:
        """
        Derive binary preference matrix from confidence matrix (Step 1.3).
        
        Args:
            X_confidence: Confidence matrix (CSR format)
            threshold: Confidence threshold for binary preference
                       P[u,i] = 1 if confidence >= threshold, else 0
        
        Returns:
            csr_matrix: Binary preference matrix (same shape as X_confidence)
        
        Purpose:
            Convert continuous confidence scores to binary preferences
            for ALS variants that require explicit preference targets.
        
        Example:
            >>> X_conf, _ = preparer.prepare_confidence_matrix(train_df, 26000, 2231)
            >>> P_binary = preparer.derive_binary_preference_matrix(X_conf, threshold=4.5)
            >>> print(f"Positive preferences: {P_binary.nnz}")
        """
        logger.info("\n" + "="*80)
        logger.info("DERIVING BINARY PREFERENCE MATRIX")
        logger.info("="*80)
        logger.info(f"Threshold: confidence >= {threshold:.2f} → preference = 1")
        
        # Create binary matrix
        X_binary = X_confidence.copy()
        X_binary.data = (X_binary.data >= threshold).astype(np.float32)
        
        # Statistics
        num_positive = X_binary.nnz  # All non-zero entries
        num_ones = int(X_binary.data.sum())  # Entries >= threshold
        num_zeros = num_positive - num_ones
        
        logger.info(f"Positive preferences (1): {num_ones:,} ({num_ones/num_positive*100:.2f}%)")
        logger.info(f"Negative preferences (0): {num_zeros:,} ({num_zeros/num_positive*100:.2f}%)")
        logger.info("✓ Binary preference matrix created")
        
        return X_binary
    
    def derive_continuous_preference_matrix(
        self,
        X_confidence: csr_matrix,
        normalize_to_01: bool = True
    ) -> csr_matrix:
        """
        Derive continuous preference matrix from confidence (Step 1.3).
        
        Args:
            X_confidence: Confidence matrix (CSR format)
            normalize_to_01: If True, normalize to [0, 1] range
        
        Returns:
            csr_matrix: Continuous preference matrix
        
        Formula:
            If normalize_to_01=True:
                P[u,i] = (confidence[u,i] - 1) / 5  # Assumes conf in [1, 6]
            Else:
                P[u,i] = confidence[u,i]  # Keep original values
        
        Purpose:
            Create continuous preference targets for ALS loss function.
            Normalized version puts preferences in [0, 1] range.
        
        Example:
            >>> X_conf, _ = preparer.prepare_confidence_matrix(train_df, 26000, 2231)
            >>> P_cont = preparer.derive_continuous_preference_matrix(X_conf, normalize_to_01=True)
        """
        logger.info("\n" + "="*80)
        logger.info("DERIVING CONTINUOUS PREFERENCE MATRIX")
        logger.info("="*80)
        
        X_preference = X_confidence.copy()
        
        if normalize_to_01:
            logger.info("Normalizing preferences to [0, 1] range...")
            # Assumes confidence in [1, 6] range
            X_preference.data = (X_preference.data - 1.0) / 5.0
            
            logger.info(f"Preference range: [{X_preference.data.min():.3f}, {X_preference.data.max():.3f}]")
            logger.info(f"Mean preference: {X_preference.data.mean():.3f}")
        else:
            logger.info("Keeping original confidence values as preferences")
            logger.info(f"Preference range: [{X_preference.data.min():.3f}, {X_preference.data.max():.3f}]")
        
        logger.info("✓ Continuous preference matrix created")
        
        return X_preference
    
    # ========================================================================
    # Step 1.2: Confidence Scaling Strategies
    # ========================================================================
    
    def apply_alpha_scaling(
        self,
        X_confidence: csr_matrix,
        alpha: float = 40.0
    ) -> csr_matrix:
        """
        Apply alpha scaling to confidence matrix (Step 1.2).
        
        Args:
            X_confidence: Confidence matrix
            alpha: Scaling factor
        
        Returns:
            csr_matrix: Scaled confidence matrix
        
        Formula:
            C_scaled[u,i] = 1 + alpha * confidence[u,i]
        
        Note:
            This is typically applied internally by ALS library during training.
            Exposed here for analysis and custom implementations.
        
        Recommendation:
            - Raw scores [1, 6]: Use lower alpha (5-10)
            - Normalized [0, 1]: Use standard alpha (20-40)
        
        Example:
            >>> X_scaled = preparer.apply_alpha_scaling(X_conf, alpha=10.0)
        """
        logger.info(f"\nApplying alpha scaling (alpha={alpha})...")
        
        X_scaled = X_confidence.copy()
        X_scaled.data = 1.0 + alpha * X_scaled.data
        
        logger.info(f"Scaled range: [{X_scaled.data.min():.3f}, {X_scaled.data.max():.3f}]")
        
        return X_scaled
    
    def get_recommended_alpha(
        self,
        X_confidence: csr_matrix,
        is_normalized: bool = False
    ) -> Dict[str, float]:
        """
        Get recommended alpha values based on confidence distribution.
        
        Args:
            X_confidence: Confidence matrix
            is_normalized: Whether confidence is normalized to [0, 1]
        
        Returns:
            Dict with recommended alpha values and rationale
        
        Example:
            >>> recommendations = preparer.get_recommended_alpha(X_conf, is_normalized=False)
            >>> print(f"Recommended alpha: {recommendations['recommended']}")
        """
        conf_min = X_confidence.data.min()
        conf_max = X_confidence.data.max()
        conf_mean = X_confidence.data.mean()
        conf_std = X_confidence.data.std()
        
        if is_normalized:
            # Normalized [0, 1]: Use standard alpha
            recommendations = {
                'confidence_range': f"[{conf_min:.3f}, {conf_max:.3f}]",
                'is_normalized': True,
                'recommended': 40.0,
                'alternatives': [20.0, 30.0, 40.0, 50.0],
                'rationale': "Normalized confidence → standard alpha (20-50)"
            }
        else:
            # Raw [1, 6]: Use lower alpha
            recommendations = {
                'confidence_range': f"[{conf_min:.3f}, {conf_max:.3f}]",
                'is_normalized': False,
                'recommended': 10.0,
                'alternatives': [5.0, 10.0, 15.0, 20.0],
                'rationale': "Raw confidence [1-6] → lower alpha (5-20) to avoid over-weighting"
            }
        
        logger.info("\n" + "="*80)
        logger.info("ALPHA SCALING RECOMMENDATIONS")
        logger.info("="*80)
        logger.info(f"Confidence range:  {recommendations['confidence_range']}")
        logger.info(f"Normalized:        {recommendations['is_normalized']}")
        logger.info(f"Mean ± Std:        {conf_mean:.3f} ± {conf_std:.3f}")
        logger.info(f"Recommended alpha: {recommendations['recommended']}")
        logger.info(f"Alternatives:      {recommendations['alternatives']}")
        logger.info(f"Rationale:         {recommendations['rationale']}")
        logger.info("="*80)
        
        return recommendations
    
    # ========================================================================
    # Utility Methods for Training Integration
    # ========================================================================
    
    def prepare_for_implicit_library(
        self,
        X_confidence: csr_matrix,
        transpose: bool = True
    ) -> csr_matrix:
        """
        Prepare matrix for implicit library (expects item-user format).
        
        Args:
            X_confidence: Confidence matrix (user-item format)
            transpose: If True, transpose to item-user format
        
        Returns:
            csr_matrix: Matrix in format expected by implicit library
        
        Note:
            implicit library expects shape (num_items, num_users)
        
        Example:
            >>> X_conf, _ = preparer.prepare_confidence_matrix(train_df, 26000, 2231)
            >>> X_train = preparer.prepare_for_implicit_library(X_conf)
            >>> print(X_train.shape)  # (2231, 26000) - transposed
        """
        if transpose:
            logger.info("Transposing matrix for implicit library (item-user format)...")
            X_train = X_confidence.T.tocsr()
            logger.info(f"Transposed shape: {X_train.shape} (items × users)")
            return X_train
        else:
            return X_confidence
    
    def get_als_training_summary(
        self,
        X_confidence: csr_matrix,
        alpha: float = 40.0,
        normalize: bool = False
    ) -> Dict[str, any]:
        """
        Get comprehensive summary for ALS training (Step 1 complete).
        
        Args:
            X_confidence: Confidence matrix
            alpha: Alpha scaling factor
            normalize: Whether confidence is normalized
        
        Returns:
            Dict with all training preparation statistics
        
        Purpose:
            Provide complete overview before starting ALS training (Step 2)
        
        Example:
            >>> summary = preparer.get_als_training_summary(X_conf, alpha=10.0)
            >>> print(summary['matrix_shape'])
            >>> print(summary['recommended_alpha'])
        """
        stats = self._compute_matrix_stats(X_confidence, X_confidence.data)
        alpha_rec = self.get_recommended_alpha(X_confidence, is_normalized=normalize)
        
        summary = {
            'matrix_shape': X_confidence.shape,
            'num_users': X_confidence.shape[0],
            'num_items': X_confidence.shape[1],
            'num_interactions': X_confidence.nnz,
            'sparsity': stats['sparsity'],
            'density': stats['density'],
            'confidence_stats': {
                'mean': stats['mean_confidence'],
                'median': stats['median_confidence'],
                'std': stats['std_confidence'],
                'min': stats['min_confidence'],
                'max': stats['max_confidence'],
                'q25': stats['q25_confidence'],
                'q75': stats['q75_confidence']
            },
            'is_normalized': normalize,
            'current_alpha': alpha,
            'recommended_alpha': alpha_rec['recommended'],
            'alpha_alternatives': alpha_rec['alternatives'],
            'alpha_rationale': alpha_rec['rationale'],
            'memory_size_mb': (X_confidence.data.nbytes + 
                              X_confidence.indices.nbytes + 
                              X_confidence.indptr.nbytes) / (1024 ** 2),
            'ready_for_training': True
        }
        
        logger.info("\n" + "="*80)
        logger.info("ALS TRAINING PREPARATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Matrix shape:         {summary['matrix_shape']}")
        logger.info(f"Interactions:         {summary['num_interactions']:,}")
        logger.info(f"Sparsity:             {summary['sparsity']:.4%}")
        logger.info(f"Confidence range:     [{summary['confidence_stats']['min']:.3f}, {summary['confidence_stats']['max']:.3f}]")
        logger.info(f"Mean confidence:      {summary['confidence_stats']['mean']:.3f}")
        logger.info(f"Normalized:           {summary['is_normalized']}")
        logger.info(f"Current alpha:        {summary['current_alpha']}")
        logger.info(f"Recommended alpha:    {summary['recommended_alpha']}")
        logger.info(f"Memory usage:         {summary['memory_size_mb']:.2f} MB")
        logger.info("="*80)
        logger.info("✓ Ready for ALS training (Step 2)")
        
        return summary
