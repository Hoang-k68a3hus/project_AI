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
        implicit_negative_per_user: int = 0,
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
        
        OPTIMIZED: Uses vectorized operations instead of per-user loops for 10-100x speedup.
        """
        import time
        start_time = time.time()
        
        logger.info("Starting optimized leave-one-out split (vectorized)...")
        logger.info(f"Processing {len(df):,} interactions from {df[user_col].nunique():,} users")
        
        # Determine total steps (8 base + 1 optional for implicit negatives)
        total_steps = 9 if (self.implicit_negative_per_user > 0 and candidate_pool is not None) else 8
        
        # OPTIMIZATION 1: Sort entire dataframe once (much faster than per-group sorting)
        logger.info(f"Step 1/{total_steps}: Sorting dataframe by user, timestamp, rating...")
        sort_start = time.time()
        df_sorted = df.sort_values(
            by=[user_col, timestamp_col, rating_col],
            ascending=[True, True, False]
        ).reset_index(drop=True)
        logger.info(f"  ✓ Sorting completed in {time.time() - sort_start:.2f}s")
        
        # OPTIMIZATION 2: Use vectorized groupby operations to find latest positives
        logger.info(f"Step 2/{total_steps}: Extracting positive interactions...")
        pos_start = time.time()
        
        # Get latest positive per user (for test)
        positives_df = df_sorted[df_sorted['is_positive'] == 1].copy()
        
        if positives_df.empty:
            logger.warning("No positive interactions found - all data goes to train")
            return df_sorted, pd.DataFrame(), None
        
        logger.info(f"  Found {len(positives_df):,} positive interactions from {positives_df[user_col].nunique():,} users")
        logger.info(f"  ✓ Positive extraction completed in {time.time() - pos_start:.2f}s")
        
        # Get latest positive per user (using groupby().tail(1) is vectorized)
        logger.info(f"Step 3/{total_steps}: Finding latest positive per user (test candidates)...")
        test_start = time.time()
        test_candidates = positives_df.groupby(user_col, sort=False).tail(1).copy()
        test_candidates['holdout_type'] = 'positive'
        logger.info(f"  Found {len(test_candidates):,} test candidates")
        logger.info(f"  ✓ Test candidate selection completed in {time.time() - test_start:.2f}s")
        
        # Get 2nd latest positive per user (for validation if needed)
        val_candidates = None
        if use_validation:
            logger.info(f"Step 4/{total_steps}: Finding 2nd latest positive per user (validation candidates)...")
            val_start = time.time()
            # Get last 2 positives per user, then take the first one (2nd latest)
            last_two_positives = positives_df.groupby(user_col, sort=False).tail(2)
            # Filter to only users with >= 2 positives
            user_counts = last_two_positives.groupby(user_col).size()
            users_with_multiple = user_counts[user_counts >= 2].index
            if len(users_with_multiple) > 0:
                val_candidates = last_two_positives[
                    last_two_positives[user_col].isin(users_with_multiple)
                ].groupby(user_col, sort=False).head(1).copy()
                val_candidates['holdout_type'] = 'validation'
                logger.info(f"  Found {len(val_candidates):,} validation candidates")
            logger.info(f"  ✓ Validation candidate selection completed in {time.time() - val_start:.2f}s")
        
        # OPTIMIZATION 3: Filter out users with insufficient positives
        if use_validation:
            logger.info(f"Step 5/{total_steps}: Filtering users with insufficient positives...")
            filter_start = time.time()
            # Remove users with only 1 positive from test (they go to train)
            positive_counts = positives_df.groupby(user_col).size()
            users_with_single_positive = positive_counts[positive_counts == 1].index
            before_count = len(test_candidates)
            test_candidates = test_candidates[
                ~test_candidates[user_col].isin(users_with_single_positive)
            ]
            logger.info(f"  Filtered out {before_count - len(test_candidates):,} users with only 1 positive")
            logger.info(f"  ✓ Filtering completed in {time.time() - filter_start:.2f}s")
        
        # Get test indices (must be after filtering test_candidates)
        logger.info(f"Step 6/{total_steps}: Building holdout indices...")
        idx_start = time.time()
        test_indices = set(test_candidates.index) if not test_candidates.empty else set()
        val_indices = set(val_candidates.index) if val_candidates is not None and not val_candidates.empty else set()
        logger.info(f"  Test indices: {len(test_indices):,}, Val indices: {len(val_indices):,}")
        logger.info(f"  ✓ Index building completed in {time.time() - idx_start:.2f}s")
        
        # OPTIMIZATION 4: Vectorized negative holdout selection
        negative_holdouts = None
        if self.include_negative_holdout:
            logger.info(f"Step 7/{total_steps}: Selecting negative holdouts...")
            neg_start = time.time()
            negatives_df = df_sorted[
                (df_sorted['is_positive'] == 0) & 
                (df_sorted[rating_col] <= self.hard_negative_threshold)
            ].copy()
            
            if not negatives_df.empty:
                logger.info(f"  Found {len(negatives_df):,} negative interactions")
                # Get latest negative per user (excluding test/val indices)
                latest_negatives = negatives_df[
                    ~negatives_df.index.isin(test_indices | val_indices)
                ].groupby(user_col, sort=False).tail(1)
                
                # Only keep negatives from users who have test data (after filtering)
                if not test_candidates.empty:
                    latest_negatives = latest_negatives[
                        latest_negatives[user_col].isin(test_candidates[user_col])
                    ]
                else:
                    latest_negatives = pd.DataFrame()
                
                if not latest_negatives.empty:
                    negative_holdouts = latest_negatives.copy()
                    negative_holdouts['holdout_type'] = 'negative'
                    test_indices.update(negative_holdouts.index)
                    logger.info(f"  Selected {len(negative_holdouts):,} negative holdouts")
            logger.info(f"  ✓ Negative holdout selection completed in {time.time() - neg_start:.2f}s")
        
        # OPTIMIZATION 5: Build test_df efficiently
        logger.info(f"Step 8/{total_steps}: Building final splits...")
        split_start = time.time()
        
        test_rows_list = []
        if not test_candidates.empty:
            test_rows_list.append(test_candidates)
        if negative_holdouts is not None and not negative_holdouts.empty:
            test_rows_list.append(negative_holdouts)
        
        test_df = pd.concat(test_rows_list, ignore_index=True) if test_rows_list else pd.DataFrame()
        logger.info(f"  Built test_df: {len(test_df):,} interactions")
        
        # OPTIMIZATION 6: Calculate cutoff timestamps per user (vectorized)
        holdout_indices_all = test_indices | val_indices
        
        if holdout_indices_all:
            logger.info(f"  Computing cutoff timestamps for {len(holdout_indices_all):,} holdout interactions...")
            holdout_df = df_sorted.loc[list(holdout_indices_all)].copy()
            
            if not holdout_df.empty:
                # Get earliest holdout timestamp per user
                cutoff_timestamps = holdout_df.groupby(user_col)[timestamp_col].min()
                logger.info(f"  Computed cutoffs for {len(cutoff_timestamps):,} users")
                
                # OPTIMIZATION 7: Vectorized train split - keep interactions before cutoff
                logger.info("  Building train split (filtering by cutoff timestamps)...")
                # Create a mapping from user to cutoff timestamp
                df_sorted_with_cutoff = df_sorted.copy()
                df_sorted_with_cutoff['cutoff_time'] = df_sorted_with_cutoff[user_col].map(cutoff_timestamps)
                
                # Train = interactions before cutoff AND not in holdout indices
                # Note: cutoff_time.isna() handles users without holdouts (they all go to train)
                train_mask = (
                    (df_sorted_with_cutoff[timestamp_col] < df_sorted_with_cutoff['cutoff_time']) |
                    (df_sorted_with_cutoff['cutoff_time'].isna())
                ) & (~df_sorted_with_cutoff.index.isin(holdout_indices_all))
                
                train_df = df_sorted_with_cutoff[train_mask].drop(columns=['cutoff_time'], errors='ignore').copy()
                logger.info(f"  Built train_df: {len(train_df):,} interactions")
            else:
                train_df = df_sorted[~df_sorted.index.isin(holdout_indices_all)].copy()
                logger.info(f"  Built train_df: {len(train_df):,} interactions (no holdouts)")
        else:
            # No holdouts - all data goes to train
            train_df = df_sorted.copy()
            logger.info(f"  Built train_df: {len(train_df):,} interactions (all data)")
        
        # Build val_df
        val_df = val_candidates.copy() if val_candidates is not None and not val_candidates.empty else None
        if val_df is not None:
            logger.info(f"  Built val_df: {len(val_df):,} interactions")
        
        logger.info(f"  ✓ Final split building completed in {time.time() - split_start:.2f}s")
        
        # OPTIMIZATION 8: Implicit negatives (batch process if needed)
        if self.implicit_negative_per_user > 0 and candidate_pool is not None and not test_df.empty:
            logger.info(f"Step 9/{total_steps}: Generating implicit negatives...")
            implicit_start = time.time()
            implicit_neg_list = []
            # Only process users with test data
            test_users = test_df[test_df['holdout_type'] == 'positive'][user_col].unique()
            logger.info(f"  Processing {len(test_users):,} users for implicit negatives...")
            
            processed = 0
            for user_id in test_users:
                processed += 1
                if processed % 1000 == 0:
                    logger.info(f"    Progress: {processed:,}/{len(test_users):,} users ({processed/len(test_users)*100:.1f}%)")
                # Get positive test interaction for this user (should be exactly 1)
                user_positive_test = test_df[
                    (test_df[user_col] == user_id) & 
                    (test_df['holdout_type'] == 'positive')
                ]
                if user_positive_test.empty:
                    continue
                
                # Use timestamp from positive test interaction
                test_timestamp = user_positive_test[timestamp_col].iloc[0]
                user_all_interactions = df_sorted[df_sorted[user_col] == user_id]
                seen_items = set(user_all_interactions[item_col].unique())
                
                available_candidates = [item for item in candidate_pool if item not in seen_items]
                if not available_candidates:
                    continue
                
                sample_size = min(self.implicit_negative_per_user, len(available_candidates))
                
                if self.implicit_negative_strategy == 'random':
                    sampled_items = self._rng.choice(
                        available_candidates,
                        size=sample_size,
                        replace=False
                    ).tolist()
                else:
                    sampled_items = available_candidates[:sample_size]
                
                # Create implicit negative rows
                for item_id in sampled_items:
                    row_data = {
                        user_col: user_id,
                        item_col: item_id,
                        rating_col: 0.0,
                        'is_positive': 0,
                        timestamp_col: test_timestamp,
                        'holdout_type': 'implicit_negative'
                    }
                    # Copy other columns from test row if they exist
                    for col in df_sorted.columns:
                        if col not in row_data:
                            row_data[col] = np.nan
                    implicit_neg_list.append(row_data)
            
            if implicit_neg_list:
                implicit_neg_df = pd.DataFrame(implicit_neg_list)
                test_df = pd.concat([test_df, implicit_neg_df], ignore_index=True)
                logger.info(f"  Generated {len(implicit_neg_list):,} implicit negatives")
            logger.info(f"  ✓ Implicit negative generation completed in {time.time() - implicit_start:.2f}s")
        
        # Compute statistics
        stats = {
            'users_with_test': test_df[test_df['holdout_type'] == 'positive'][user_col].nunique() if not test_df.empty else 0,
            'users_with_val': val_df[user_col].nunique() if val_df is not None and not val_df.empty else 0,
            'users_no_test': df_sorted[user_col].nunique() - (test_df[test_df['holdout_type'] == 'positive'][user_col].nunique() if not test_df.empty else 0),
            'users_train_only': 0,  # Simplified for optimization
            'users_with_negative_holdout': negative_holdouts[user_col].nunique() if negative_holdouts is not None and not negative_holdouts.empty else 0,
            'users_with_implicit_negatives': test_df[test_df['holdout_type'] == 'implicit_negative'][user_col].nunique() if not test_df.empty and 'holdout_type' in test_df.columns else 0,
        }
        
        # Log statistics
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("Split statistics:")
        logger.info(f"  - Users with test: {stats['users_with_test']:,}")
        logger.info(f"  - Users with val: {stats['users_with_val']:,}")
        logger.info(f"  - Users with negative holdout: {stats['users_with_negative_holdout']:,}")
        logger.info(f"  - Users with implicit negatives: {stats['users_with_implicit_negatives']:,}")
        logger.info(f"  - Users with no test (0 positives): {stats['users_no_test']:,}")
        logger.info("=" * 80)
        logger.info(f"✓ Temporal split completed in {total_time:.2f}s")
        logger.info(f"  Final sizes - Train: {len(train_df):,}, Test: {len(test_df):,}, Val: {len(val_df) if val_df is not None else 0:,}")
        
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
        
        OPTIMIZED: Uses vectorized groupby operations instead of per-user loops.
        
        Raises:
            ValueError: If temporal ordering is violated
        """
        if test_df.empty:
            logger.warning("Test set is empty - skipping temporal validation")
            return
        
        logger.info("Validating temporal ordering (no data leakage)...")
        
        violations = []
        
        # OPTIMIZATION: Vectorized validation using groupby().agg()
        # Check train vs test
        if not train_df.empty:
            # Get max train timestamp per user
            train_max_per_user = train_df.groupby(user_col)[timestamp_col].max()
            
            # Get min test timestamp per user (only for users in both sets)
            test_users_in_train = test_df[test_df[user_col].isin(train_max_per_user.index)]
            if not test_users_in_train.empty:
                test_min_per_user = test_users_in_train.groupby(user_col)[timestamp_col].min()
                
                # Find violations: test_min < train_max
                common_users = test_min_per_user.index.intersection(train_max_per_user.index)
                if len(common_users) > 0:
                    violations_mask = test_min_per_user[common_users] < train_max_per_user[common_users]
                    violating_users = violations_mask[violations_mask].index
                    
                    if len(violating_users) > 0:
                        for u_idx in violating_users:
                            violations.append(
                                f"User {u_idx}: test_min ({test_min_per_user[u_idx]}) < "
                                f"train_max ({train_max_per_user[u_idx]})"
                            )
        
        # Check val vs train (if val exists)
        if val_df is not None and not val_df.empty and not train_df.empty:
            # Get max train timestamp per user
            train_max_per_user = train_df.groupby(user_col)[timestamp_col].max()
            
            # Get min val timestamp per user
            val_users_in_train = val_df[val_df[user_col].isin(train_max_per_user.index)]
            if not val_users_in_train.empty:
                val_min_per_user = val_users_in_train.groupby(user_col)[timestamp_col].min()
                
                # Find violations: val_min < train_max
                common_users = val_min_per_user.index.intersection(train_max_per_user.index)
                if len(common_users) > 0:
                    violations_mask = val_min_per_user[common_users] < train_max_per_user[common_users]
                    violating_users = violations_mask[violations_mask].index
                    
                    if len(violating_users) > 0:
                        for u_idx in violating_users:
                            violations.append(
                                f"User {u_idx}: val_min ({val_min_per_user[u_idx]}) < "
                                f"train_max ({train_max_per_user[u_idx]})"
                            )
        
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
