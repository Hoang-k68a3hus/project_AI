"""
Temporal Split Module for Collaborative Filtering

This module implements leave-one-out temporal splitting with optional negative holdouts
and implicit negative sampling for unbiased offline evaluation. It supports train/test/val splits with proper
chronological ordering.

Key Features:
- Leave-one-out split: Latest positive interaction per user → test
- Optional negative holdouts sourced from explicit dislikes
- Edge case handling: Users with insufficient data, all-negative users
- Temporal validation: No data leakage (test timestamps > train timestamps)
- Optional validation set: 2nd latest positive → val

Author: Data Team
Created: 2025-01-15
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalSplitter:
    """
    Temporal splitter for leave-one-out evaluation with optional negative holdouts
    and implicit negatives for ranking metrics.
    
    This class handles:
    1. Sorting interactions per user by timestamp
    2. Selecting latest positive interaction for test
    3. Reserving representative negative interactions when available
    4. Handling edge cases (insufficient positives, all-negative users)
    5. Optional validation set creation
    6. Temporal validation (no data leakage)
    
    Usage:
        splitter = TemporalSplitter(positive_threshold=4)
        train_df, test_df, val_df = splitter.split(
            interactions_df, 
            method='leave_one_out',
            use_validation=False
        )
    """
    
    def __init__(
        self,
        positive_threshold: float = 4.0,
        include_negative_holdout: bool = True,
        hard_negative_threshold: Optional[float] = None,
        implicit_negative_per_user: int = 50,
        implicit_negative_strategy: str = 'popular',
        implicit_negative_max_candidates: Optional[int] = 500,
        random_state: Optional[int] = 42
    ):
        """
        Initialize temporal splitter.
        
        Args:
            positive_threshold: Minimum rating to consider as positive (default: 4.0)
            include_negative_holdout: Whether to reserve explicit negatives for testing
            hard_negative_threshold: Optional rating threshold defining explicit negatives
            implicit_negative_per_user: Number of implicit negatives to sample per user
            implicit_negative_strategy: Strategy for sampling negatives ('popular' or 'random')
            implicit_negative_max_candidates: Cap on candidate pool for implicit negatives
            random_state: Seed for implicit negative sampling reproducibility
        """
        self.positive_threshold = positive_threshold
        self.include_negative_holdout = include_negative_holdout
        self.hard_negative_threshold = (
            hard_negative_threshold
            if hard_negative_threshold is not None
            else max(1.0, positive_threshold - 1.0)
        )
        self.implicit_negative_per_user = max(0, implicit_negative_per_user)
        self.implicit_negative_strategy = implicit_negative_strategy
        self.implicit_negative_max_candidates = implicit_negative_max_candidates
        self._rng = np.random.default_rng(random_state)
        self.split_metadata = {}
        
    def split(
        self,
        interactions_df: pd.DataFrame,
        method: str = 'leave_one_out',
        use_validation: bool = False,
        timestamp_col: str = 'cmt_date',
        user_col: str = 'u_idx',
        rating_col: str = 'rating',
        item_col: str = 'i_idx'
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
        candidate_pool = None
        if self.implicit_negative_per_user > 0:
            candidate_pool = self._prepare_candidate_pool(
                interactions_df,
                item_col=item_col
            )
        
        if method == 'leave_one_out':
            train_df, test_df, val_df = self._leave_one_out_split(
                interactions_df, 
                use_validation, 
                timestamp_col, 
                user_col, 
                rating_col,
                item_col,
                candidate_pool
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
        rating_col: str,
        item_col: str,
        candidate_pool: Optional[List]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Perform leave-one-out split with unbiased evaluation holdouts.
        
        Strategy:
        1. Sort interactions per user by timestamp (ascending)
        2. Find latest POSITIVE interaction (rating ≥ threshold) → test
        3. If use_validation: Find 2nd latest positive → val
        4. Remaining interactions → train
        5. Optionally attach explicit and implicit negatives for evaluation
        
        Edge Cases:
        - User with 0 positives: All → train, no test/val (should be rare after Step 2.3)
        - User with 1 positive: positive → test if no val, else → train
        - User with 2 positives: 1 → test, 1 → train (or val if enabled)
        - Latest interaction is negative: Take previous positive for test
        """
        train_rows: List[pd.DataFrame] = []
        test_rows: List[pd.DataFrame] = []
        val_rows: List[pd.DataFrame] = []
        
        # Statistics tracking
        stats = {
            'users_with_test': 0,
            'users_with_val': 0,
            'users_no_test': 0,
            'users_train_only': 0,
            'users_with_negative_holdout': 0,
            'users_with_implicit_negatives': 0,
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
            
            # Edge Case: Only 1 positive when validation is required
            if use_validation and num_positives == 1:
                train_rows.append(user_sorted)
                stats['users_train_only'] += 1
                continue
            
            holdout_indices: List[int] = []
            holdout_timestamps: List[pd.Timestamp] = []
            
            # Optional validation interaction (second latest positive)
            if use_validation and num_positives >= 2:
                val_interaction = positives.iloc[-2]
                val_idx_in_sorted = val_interaction.name
                val_row = user_sorted.loc[[val_idx_in_sorted]].copy()
                val_row['holdout_type'] = 'validation'
                val_rows.append(val_row)
                
                holdout_indices.append(val_idx_in_sorted)
                holdout_timestamps.append(val_interaction[timestamp_col])
                stats['users_with_val'] += 1
            
            # Latest positive → test
            test_interaction = positives.iloc[-1]
            test_idx_in_sorted = test_interaction.name
            test_row = user_sorted.loc[[test_idx_in_sorted]].copy()
            test_row['holdout_type'] = 'positive'
            test_rows.append(test_row)
            
            holdout_indices.append(test_idx_in_sorted)
            holdout_timestamps.append(test_interaction[timestamp_col])
            stats['users_with_test'] += 1
            
            # Optional explicit negative holdout
            negative_interaction = self._select_negative_holdout(
                user_sorted=user_sorted,
                exclude_indices=holdout_indices,
                rating_col=rating_col
            )
            if negative_interaction is not None:
                negative_idx = negative_interaction.name
                negative_row = user_sorted.loc[[negative_idx]].copy()
                negative_row['holdout_type'] = 'negative'
                test_rows.append(negative_row)
                
                holdout_indices.append(negative_idx)
                holdout_timestamps.append(negative_interaction[timestamp_col])
                stats['users_with_negative_holdout'] += 1
            
            # Keep only interactions BEFORE the earliest holdout timestamp
            cutoff_timestamp = min(holdout_timestamps) if holdout_timestamps else None
            if cutoff_timestamp is None:
                continue
            
            train_mask = user_sorted[timestamp_col] < cutoff_timestamp
            train_interactions = user_sorted[train_mask]
            if holdout_indices:
                train_interactions = train_interactions.drop(index=holdout_indices, errors='ignore')
            
            if not train_interactions.empty:
                train_rows.append(train_interactions)
            
            # Optional implicit negatives (popular/random unseen items)
            implicit_neg_df = self._generate_implicit_negatives(
                user_sorted=user_sorted,
                user_col=user_col,
                item_col=item_col,
                rating_col=rating_col,
                timestamp_col=timestamp_col,
                reference_timestamp=test_interaction[timestamp_col],
                candidate_pool=candidate_pool
            )
            if implicit_neg_df is not None:
                test_rows.append(implicit_neg_df)
                stats['users_with_implicit_negatives'] += 1
        
        # Concatenate results
        train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
        test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
        val_df = pd.concat(val_rows, ignore_index=True) if val_rows else None
        
        # Log statistics
        logger.info("Split statistics:")
        logger.info(f"  - Users with test: {stats['users_with_test']}")
        logger.info(f"  - Users with val: {stats['users_with_val']}")
        logger.info(f"  - Users with negative holdout: {stats['users_with_negative_holdout']}")
        logger.info(f"  - Users with implicit negatives: {stats['users_with_implicit_negatives']}")
        logger.info(f"  - Users with no test (0 positives): {stats['users_no_test']}")
        logger.info(f"  - Users train-only (insufficient positives): {stats['users_train_only']}")
        
        return train_df, test_df, val_df
    
    def _select_negative_holdout(
        self,
        user_sorted: pd.DataFrame,
        exclude_indices: List[int],
        rating_col: str
    ) -> Optional[pd.Series]:
        """
        Select the most recent explicit negative interaction for evaluation.
        
        Args:
            user_sorted: Sorted DataFrame for a single user
            exclude_indices: Indices already reserved for other holdouts
            rating_col: Column holding rating values
        
        Returns:
            pd.Series representing the chosen negative interaction, or None
        """
        if not self.include_negative_holdout:
            return None
        
        negative_mask = user_sorted['is_positive'] == 0
        if self.hard_negative_threshold is not None:
            negative_mask &= user_sorted[rating_col] <= self.hard_negative_threshold
        
        negative_candidates = user_sorted[negative_mask]
        if negative_candidates.empty:
            return None
        
        if exclude_indices:
            negative_candidates = negative_candidates[
                ~negative_candidates.index.isin(exclude_indices)
            ]
        
        if negative_candidates.empty:
            return None
        
        return negative_candidates.iloc[-1]
    
    def _prepare_candidate_pool(
        self,
        df: pd.DataFrame,
        item_col: str
    ) -> Optional[List]:
        """
        Build candidate pool for implicit negative sampling.
        
        Returns:
            Ordered list of candidate item IDs or None if disabled.
        """
        if self.implicit_negative_per_user <= 0:
            return None
        
        item_counts = df[item_col].value_counts()
        if self.implicit_negative_max_candidates is not None:
            item_counts = item_counts.head(self.implicit_negative_max_candidates)
        
        if self.implicit_negative_strategy == 'random':
            return item_counts.index.tolist()
        
        # Default: popularity ordered list
        return item_counts.index.tolist()
    
    def _generate_implicit_negatives(
        self,
        user_sorted: pd.DataFrame,
        user_col: str,
        item_col: str,
        rating_col: str,
        timestamp_col: str,
        reference_timestamp: pd.Timestamp,
        candidate_pool: Optional[List]
    ) -> Optional[pd.DataFrame]:
        """
        Generate synthetic implicit negatives for unbiased evaluation.
        """
        if (
            self.implicit_negative_per_user <= 0
            or candidate_pool is None
            or user_sorted.empty
        ):
            return None
        
        user_id = user_sorted[user_col].iloc[0]
        seen_items = set(user_sorted[item_col].tolist())
        available_candidates = [item for item in candidate_pool if item not in seen_items]
        if not available_candidates:
            return None
        
        sample_size = min(self.implicit_negative_per_user, len(available_candidates))
        
        if self.implicit_negative_strategy == 'random':
            sampled_items = self._rng.choice(
                available_candidates,
                size=sample_size,
                replace=False
            ).tolist()
        else:  # popularity ordered
            sampled_items = available_candidates[:sample_size]
        
        rows = []
        base_columns = list(user_sorted.columns)
        if 'holdout_type' not in base_columns:
            base_columns.append('holdout_type')
        if 'is_positive' not in base_columns:
            base_columns.append('is_positive')
        
        for item_id in sampled_items:
            row_data = {col: np.nan for col in base_columns}
            row_data[user_col] = user_id
            row_data[item_col] = item_id
            row_data[rating_col] = 0.0
            row_data['is_positive'] = 0
            row_data[timestamp_col] = reference_timestamp
            if 'confidence_score' in user_sorted.columns:
                row_data['confidence_score'] = 0.0
            row_data['holdout_type'] = 'implicit_negative'
            rows.append(row_data)
        
        return pd.DataFrame(rows, columns=base_columns)
    
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
                'num_negatives': int((test_df['is_positive'] == 0).sum()) if not test_df.empty else 0,
                'all_positive': int((test_df['is_positive'] == 1).all()) if not test_df.empty else True,
                'negative_holdout_enabled': self.include_negative_holdout,
                'implicit_negative_per_user': self.implicit_negative_per_user,
                'implicit_negative_strategy': self.implicit_negative_strategy,
                'num_implicit_negatives': (
                    int((test_df['holdout_type'] == 'implicit_negative').sum())
                    if not test_df.empty and 'holdout_type' in test_df.columns
                    else 0
                ),
                'holdout_type_counts': (
                    test_df['holdout_type'].value_counts().to_dict()
                    if not test_df.empty and 'holdout_type' in test_df.columns else {}
                ),
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
