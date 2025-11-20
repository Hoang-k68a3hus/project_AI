"""
Temporal Split Module for Collaborative Filtering

This module implements leave-one-out temporal splitting with positive-only test set.
Supports train/test/val splits with proper chronological ordering.

Key Features:
- Leave-one-out split: Latest positive interaction per user → test
- Positive-only test: Only interactions with rating ≥4 go to test set
- Edge case handling: Users with insufficient data, all-negative users
- Temporal validation: No data leakage (test timestamps > train timestamps)
- Optional validation set: 2nd latest positive → val

Author: Data Team
Created: 2025-01-15
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalSplitter:
    """
    Temporal splitter for leave-one-out evaluation with positive-only test set.
    
    This class handles:
    1. Sorting interactions per user by timestamp
    2. Selecting latest positive interaction for test
    3. Handling edge cases (insufficient positives, all-negative users)
    4. Optional validation set creation
    5. Temporal validation (no data leakage)
    
    Usage:
        splitter = TemporalSplitter(positive_threshold=4)
        train_df, test_df, val_df = splitter.split(
            interactions_df, 
            method='leave_one_out',
            use_validation=False
        )
    """
    
    def __init__(self, positive_threshold: float = 4.0):
        """
        Initialize temporal splitter.
        
        Args:
            positive_threshold: Minimum rating to consider as positive (default: 4.0)
        """
        self.positive_threshold = positive_threshold
        self.split_metadata = {}
        
    def split(
        self,
        interactions_df: pd.DataFrame,
        method: str = 'leave_one_out',
        use_validation: bool = False,
        timestamp_col: str = 'cmt_date',
        user_col: str = 'u_idx',
        rating_col: str = 'rating'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split interactions into train/test(/val) sets with temporal ordering.
        
        Args:
            interactions_df: DataFrame with interactions (must have u_idx, rating, timestamp)
            method: Split method ('leave_one_out' or 'leave_k_out')
            use_validation: Whether to create validation set
            timestamp_col: Name of timestamp column
            user_col: Name of user column
            rating_col: Name of rating column
            
        Returns:
            Tuple of (train_df, test_df, val_df)
            - train_df: All interactions except test/val
            - test_df: Latest positive interaction per user (rating ≥ positive_threshold)
            - val_df: Optional 2nd latest positive interaction per user (or None)
        """
        logger.info(f"Starting temporal split with method: {method}")
        logger.info(f"Input: {len(interactions_df)} interactions from {interactions_df[user_col].nunique()} users")
        
        # Validate inputs
        self._validate_inputs(interactions_df, timestamp_col, user_col, rating_col)
        
        # Add is_positive flag if not exists
        if 'is_positive' not in interactions_df.columns:
            interactions_df['is_positive'] = (interactions_df[rating_col] >= self.positive_threshold).astype(int)
        
        # Sort and split
        if method == 'leave_one_out':
            train_df, test_df, val_df = self._leave_one_out_split(
                interactions_df, 
                use_validation, 
                timestamp_col, 
                user_col, 
                rating_col
            )
        else:
            raise ValueError(f"Unsupported split method: {method}")
        
        # Validate temporal ordering
        self._validate_temporal_ordering(train_df, test_df, val_df, timestamp_col, user_col)
        
        # Store metadata
        self._compute_split_metadata(train_df, test_df, val_df, user_col)
        
        logger.info(f"Split complete: Train={len(train_df)}, Test={len(test_df)}, Val={len(val_df) if val_df is not None else 0}")
        
        return train_df, test_df, val_df
    
    def _validate_inputs(
        self, 
        df: pd.DataFrame, 
        timestamp_col: str, 
        user_col: str, 
        rating_col: str
    ):
        """Validate input DataFrame has required columns and no missing values."""
        required_cols = [user_col, rating_col, timestamp_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for NaT/missing timestamps
        if df[timestamp_col].isna().any():
            num_missing = df[timestamp_col].isna().sum()
            logger.error(f"Found {num_missing} rows with missing timestamps - CRITICAL ERROR")
            raise ValueError(
                f"DataFrame contains {num_missing} rows with missing timestamps. "
                "These must be removed in preprocessing (Step 1.1) to avoid data leakage."
            )
        
        # Check rating range
        if (df[rating_col] < 1.0).any() or (df[rating_col] > 5.0).any():
            invalid_count = ((df[rating_col] < 1.0) | (df[rating_col] > 5.0)).sum()
            logger.warning(f"Found {invalid_count} ratings outside [1.0, 5.0] range")
    
    def _leave_one_out_split(
        self,
        df: pd.DataFrame,
        use_validation: bool,
        timestamp_col: str,
        user_col: str,
        rating_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Perform leave-one-out split with positive-only test set.
        
        Strategy:
        1. Sort interactions per user by timestamp (ascending)
        2. Find latest POSITIVE interaction (rating ≥ threshold) → test
        3. If use_validation: Find 2nd latest positive → val
        4. Remaining interactions → train
        
        Edge Cases:
        - User with 0 positives: All → train, no test/val (should be rare after Step 2.3)
        - User with 1 positive: positive → test if no val, else → train
        - User with 2 positives: 1 → test, 1 → train (or val if enabled)
        - Latest interaction is negative: Take previous positive for test
        """
        train_rows = []
        test_rows = []
        val_rows = []
        
        # Statistics tracking
        stats = {
            'users_with_test': 0,
            'users_with_val': 0,
            'users_no_test': 0,
            'users_train_only': 0
        }
        
        # Process each user
        for u_idx, user_group in df.groupby(user_col):
            # Sort by timestamp (ascending), break ties by rating (descending)
            user_sorted = user_group.sort_values(
                by=[timestamp_col, rating_col],
                ascending=[True, False]
            ).reset_index(drop=True)
            
            # Identify positive interactions
            positive_mask = user_sorted['is_positive'] == 1
            positives = user_sorted[positive_mask]
            
            num_positives = len(positives)
            
            # Edge Case: No positives (should be rare after Step 2.3 filtering)
            if num_positives == 0:
                # All interactions → train (no test/val possible)
                train_rows.append(user_sorted)
                stats['users_no_test'] += 1
                continue
            
            # Edge Case: Only 1 positive
            if num_positives == 1:
                if use_validation:
                    # Can't create both test and val from 1 positive
                    # Use positive for train, no test/val
                    train_rows.append(user_sorted)
                    stats['users_train_only'] += 1
                else:
                    # Use the 1 positive for test
                    test_idx = positives.index[-1]  # Latest (and only) positive
                    test_interaction = user_sorted.iloc[test_idx]
                    test_rows.append(user_sorted.iloc[[test_idx]])
                    
                    # Only interactions BEFORE test timestamp → train
                    test_timestamp = test_interaction[timestamp_col]
                    train_mask = (user_sorted[timestamp_col] < test_timestamp)
                    train_interactions = user_sorted[train_mask]
                    if not train_interactions.empty:
                        train_rows.append(train_interactions)
                    
                    stats['users_with_test'] += 1
                continue
            
            # Edge Case: 2+ positives
            if use_validation and num_positives >= 2:
                # Latest positive → test
                test_idx = positives.index[-1]
                test_interaction = user_sorted.iloc[test_idx]
                test_rows.append(user_sorted.iloc[[test_idx]])
                
                # 2nd latest positive → val
                val_idx = positives.index[-2]
                val_interaction = user_sorted.iloc[val_idx]
                val_rows.append(user_sorted.iloc[[val_idx]])
                
                # Rest → train (ONLY interactions BEFORE val timestamp)
                # This prevents data leakage: train cannot contain future interactions
                val_timestamp = val_interaction[timestamp_col]
                train_mask = (user_sorted[timestamp_col] < val_timestamp)
                train_interactions = user_sorted[train_mask]
                if not train_interactions.empty:
                    train_rows.append(train_interactions)
                
                stats['users_with_test'] += 1
                stats['users_with_val'] += 1
            else:
                # Latest positive → test
                test_idx = positives.index[-1]
                test_interaction = user_sorted.iloc[test_idx]
                test_rows.append(user_sorted.iloc[[test_idx]])
                
                # Rest → train (ONLY interactions BEFORE test timestamp)
                # This prevents data leakage: train cannot contain future interactions
                test_timestamp = test_interaction[timestamp_col]
                train_mask = (user_sorted[timestamp_col] < test_timestamp)
                train_interactions = user_sorted[train_mask]
                if not train_interactions.empty:
                    train_rows.append(train_interactions)
                
                stats['users_with_test'] += 1
        
        # Concatenate results
        train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
        test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
        val_df = pd.concat(val_rows, ignore_index=True) if val_rows else None
        
        # Log statistics
        logger.info(f"Split statistics:")
        logger.info(f"  - Users with test: {stats['users_with_test']}")
        logger.info(f"  - Users with val: {stats['users_with_val']}")
        logger.info(f"  - Users with no test (0 positives): {stats['users_no_test']}")
        logger.info(f"  - Users train-only (insufficient positives): {stats['users_train_only']}")
        
        return train_df, test_df, val_df
    
    def _validate_temporal_ordering(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        timestamp_col: str,
        user_col: str
    ):
        """
        Validate no data leakage: test timestamps > train/val timestamps per user.
        
        Raises:
            ValueError: If temporal ordering is violated
        """
        if test_df.empty:
            logger.warning("Test set is empty - skipping temporal validation")
            return
        
        logger.info("Validating temporal ordering (no data leakage)...")
        
        violations = []
        
        # Check train vs test
        for u_idx in test_df[user_col].unique():
            test_times = test_df[test_df[user_col] == u_idx][timestamp_col]
            train_times = train_df[train_df[user_col] == u_idx][timestamp_col]
            
            if not train_times.empty:
                test_min = test_times.min()
                train_max = train_times.max()
                
                if test_min < train_max:
                    violations.append(f"User {u_idx}: test_min ({test_min}) < train_max ({train_max})")
        
        # Check val vs train (if val exists)
        if val_df is not None and not val_df.empty:
            for u_idx in val_df[user_col].unique():
                val_times = val_df[val_df[user_col] == u_idx][timestamp_col]
                train_times = train_df[train_df[user_col] == u_idx][timestamp_col]
                
                if not train_times.empty:
                    val_min = val_times.min()
                    train_max = train_times.max()
                    
                    if val_min < train_max:
                        violations.append(f"User {u_idx}: val_min ({val_min}) < train_max ({train_max})")
        
        if violations:
            logger.error(f"Found {len(violations)} temporal ordering violations:")
            for v in violations[:10]:  # Log first 10
                logger.error(f"  - {v}")
            raise ValueError(f"Temporal ordering violated for {len(violations)} users")
        
        logger.info("Temporal ordering validated - no data leakage detected ✓")
    
    def _compute_split_metadata(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        user_col: str
    ):
        """Compute and store split metadata."""
        self.split_metadata = {
            'created_at': datetime.now().isoformat(),
            'positive_threshold': self.positive_threshold,
            'train': {
                'num_interactions': len(train_df),
                'num_users': train_df[user_col].nunique() if not train_df.empty else 0,
                'num_positives': int((train_df['is_positive'] == 1).sum()) if not train_df.empty else 0,
            },
            'test': {
                'num_interactions': len(test_df),
                'num_users': test_df[user_col].nunique() if not test_df.empty else 0,
                'num_positives': int((test_df['is_positive'] == 1).sum()) if not test_df.empty else 0,
                'all_positive': int((test_df['is_positive'] == 1).all()) if not test_df.empty else True,
            },
            'val': {
                'num_interactions': len(val_df) if val_df is not None else 0,
                'num_users': val_df[user_col].nunique() if val_df is not None and not val_df.empty else 0,
                'num_positives': int((val_df['is_positive'] == 1).sum()) if val_df is not None and not val_df.empty else 0,
            } if val_df is not None else None
        }
    
    def get_split_metadata(self) -> Dict:
        """
        Get metadata about the split.
        
        Returns:
            Dict with split statistics:
            - created_at: Timestamp
            - positive_threshold: Threshold used
            - train/test/val: num_interactions, num_users, num_positives
        """
        return self.split_metadata
    
    def save_split_metadata(self, output_path: str):
        """
        Save split metadata to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.split_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Split metadata saved to {output_path}")
    
    def validate_positive_only_test(self, test_df: pd.DataFrame, rating_col: str = 'rating') -> bool:
        """
        Validate that test set only contains positive interactions.
        
        Args:
            test_df: Test DataFrame
            rating_col: Name of rating column
            
        Returns:
            True if all test interactions are positive, False otherwise
        """
        if test_df.empty:
            logger.warning("Test set is empty")
            return True
        
        all_positive = (test_df[rating_col] >= self.positive_threshold).all()
        
        if not all_positive:
            num_negative = (test_df[rating_col] < self.positive_threshold).sum()
            logger.error(f"Test set contains {num_negative} negative interactions (rating < {self.positive_threshold})")
            return False
        
        logger.info(f"Test set validation passed: All {len(test_df)} interactions are positive ✓")
        return True
    
    def get_user_split_summary(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        user_col: str = 'u_idx'
    ) -> pd.DataFrame:
        """
        Get per-user split summary.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            val_df: Validation DataFrame (optional)
            user_col: Name of user column
            
        Returns:
            DataFrame with columns: u_idx, train_count, test_count, val_count, has_test, has_val
        """
        all_users = pd.concat([
            train_df[[user_col]] if not train_df.empty else pd.DataFrame(),
            test_df[[user_col]] if not test_df.empty else pd.DataFrame(),
            val_df[[user_col]] if val_df is not None and not val_df.empty else pd.DataFrame()
        ])[user_col].unique()
        
        summary_rows = []
        
        for u_idx in all_users:
            train_count = len(train_df[train_df[user_col] == u_idx]) if not train_df.empty else 0
            test_count = len(test_df[test_df[user_col] == u_idx]) if not test_df.empty else 0
            val_count = len(val_df[val_df[user_col] == u_idx]) if val_df is not None and not val_df.empty else 0
            
            summary_rows.append({
                user_col: u_idx,
                'train_count': train_count,
                'test_count': test_count,
                'val_count': val_count,
                'has_test': test_count > 0,
                'has_val': val_count > 0
            })
        
        return pd.DataFrame(summary_rows)
