"""
User Filtering & Segmentation Module

This module handles Step 2.3: User Filtering and Segmentation for CF Training.
Segments users into trainable (≥2 interactions, ≥1 positive) vs cold-start
to address the high sparsity problem (90% of users have only 1 interaction).

Key Features:
- Trainable users: ≥2 interactions AND ≥1 positive → CF training
- Cold-start users: 1 interaction or no positives → content-based serving
- Special handling: 2 interactions with both negative → force cold-start
- Iterative filtering: Apply min item interactions after user filtering
"""

import logging
from typing import Dict, Tuple, Set

import pandas as pd
import numpy as np


logger = logging.getLogger("data_layer")


class UserFilter:
    """
    Filter and segment users for CF training based on interaction count and quality.
    
    Strategy:
    - Trainable users: Have enough data for collaborative patterns (≥2 interactions, ≥1 positive)
    - Cold-start users: Insufficient CF data, serve via content-based + popularity
    - Rationale: 90% users have ≤1 interaction → can't learn CF patterns
    """
    
    def __init__(
        self,
        min_user_interactions: int = 2,
        min_user_positives: int = 1,
        min_item_positives: int = 5,
        positive_threshold: float = 4.0
    ):
        """
        Initialize UserFilter.
        
        Args:
            min_user_interactions: Minimum total interactions for trainable user (default: 2)
            min_user_positives: Minimum positive interactions required (default: 1)
            min_item_positives: Minimum positive interactions per item (default: 5)
            positive_threshold: Rating threshold for positive interactions (default: 4.0)
        """
        self.min_user_interactions = min_user_interactions
        self.min_user_positives = min_user_positives
        self.min_item_positives = min_item_positives
        self.positive_threshold = positive_threshold
    
    def segment_users(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        rating_col: str = 'rating'
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Segment users into trainable vs cold-start based on interaction count.
        
        Args:
            interactions_df: DataFrame with user interactions
            user_col: User ID column name
            rating_col: Rating column name
        
        Returns:
            Tuple[DataFrame, Dict]:
                - Updated DataFrame with 'is_trainable_user' column
                - Statistics dict with segmentation metrics
        
        Example:
            >>> filter = UserFilter(min_user_interactions=2, min_user_positives=1)
            >>> df_segmented, stats = filter.segment_users(interactions_df)
            >>> print(stats['trainable_users'])  # ~26,000 (~8.6% of total)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2.3: USER FILTERING & SEGMENTATION")
        logger.info("="*80)
        logger.info("Segmenting users: Trainable (≥2 interactions, ≥1 positive) vs Cold-start")
        logger.info(f"Configuration:")
        logger.info(f"  - Min user interactions: {self.min_user_interactions}")
        logger.info(f"  - Min user positives: {self.min_user_positives}")
        logger.info(f"  - Positive threshold: rating ≥ {self.positive_threshold}")
        
        # Check if is_positive column exists (from Step 2.2)
        if 'is_positive' not in interactions_df.columns:
            raise ValueError("Missing 'is_positive' column. Run Step 2.2 (BPR labels) first.")
        
        # Drop existing segmentation column if present
        if 'is_trainable_user' in interactions_df.columns:
            interactions_df = interactions_df.drop(columns=['is_trainable_user'])
        
        # Count interactions per user
        user_stats = interactions_df.groupby(user_col).agg(
            total_interactions=('rating', 'count'),
            positive_interactions=('is_positive', 'sum'),
            negative_interactions=('is_positive', lambda x: (~x).sum()),
            avg_rating=(rating_col, 'mean')
        ).reset_index()
        
        logger.info(f"\nUser interaction statistics:")
        logger.info(f"  Total unique users: {len(user_stats):,}")
        logger.info(f"  Mean interactions per user: {user_stats['total_interactions'].mean():.2f}")
        logger.info(f"  Median interactions per user: {user_stats['total_interactions'].median():.1f}")
        logger.info(f"  Max interactions per user: {user_stats['total_interactions'].max()}")
        
        # Define trainable users
        logger.info("\n" + "-"*80)
        logger.info("TRAINABLE USER CRITERIA")
        logger.info("-"*80)
        
        trainable_mask = (
            (user_stats['total_interactions'] >= self.min_user_interactions) &
            (user_stats['positive_interactions'] >= self.min_user_positives)
        )
        
        # Special case: 2 interactions with both negative → force cold-start
        special_case_mask = (
            (user_stats['total_interactions'] == 2) &
            (user_stats['positive_interactions'] == 0)
        )
        trainable_mask = trainable_mask & ~special_case_mask
        
        user_stats['is_trainable_user'] = trainable_mask
        
        # Compute statistics
        num_trainable = trainable_mask.sum()
        num_cold_start = len(user_stats) - num_trainable
        pct_trainable = num_trainable / len(user_stats) * 100
        
        logger.info(f"Trainable users: {num_trainable:,} ({pct_trainable:.2f}%)")
        logger.info(f"  Criteria: ≥{self.min_user_interactions} interactions AND ≥{self.min_user_positives} positive")
        logger.info(f"Cold-start users: {num_cold_start:,} ({100-pct_trainable:.2f}%)")
        logger.info(f"  Criteria: <{self.min_user_interactions} interactions OR no positives")
        
        # Log special cases
        num_special_cases = special_case_mask.sum()
        if num_special_cases > 0:
            logger.info(f"\nSpecial cases (forced to cold-start): {num_special_cases:,}")
            logger.info(f"  Users with exactly 2 interactions, both negative (rating < {self.positive_threshold})")
        
        # Merge back to interactions
        interactions_df = interactions_df.merge(
            user_stats[[user_col, 'is_trainable_user']],
            on=user_col,
            how='left'
        )
        
        # Compute trainable user interaction stats
        trainable_interactions = interactions_df[interactions_df['is_trainable_user']]
        trainable_user_ids = user_stats[user_stats['is_trainable_user']][user_col].unique()
        
        avg_interactions_trainable = user_stats[trainable_mask]['total_interactions'].mean()
        
        logger.info(f"\nTrainable user details:")
        logger.info(f"  Interactions from trainable users: {len(trainable_interactions):,}")
        logger.info(f"  Average interactions per trainable user: {avg_interactions_trainable:.2f}")
        logger.info(f"  Positive rate among trainable users: {trainable_interactions[rating_col].ge(self.positive_threshold).mean()*100:.2f}%")
        
        # Estimate matrix density (will be more accurate after item filtering)
        num_items = interactions_df['product_id'].nunique() if 'product_id' in interactions_df.columns else interactions_df['i_idx'].nunique()
        estimated_density = len(trainable_interactions) / (num_trainable * num_items) * 100
        
        logger.info(f"\nEstimated CF matrix density:")
        logger.info(f"  Trainable users: {num_trainable:,}")
        logger.info(f"  Total items: {num_items:,}")
        logger.info(f"  Interactions: {len(trainable_interactions):,}")
        logger.info(f"  Density: {estimated_density:.4f}%")
        
        # Build statistics dict
        stats = {
            'total_users': len(user_stats),
            'trainable_users': num_trainable,
            'cold_start_users': num_cold_start,
            'trainable_percentage': pct_trainable,
            'cold_start_percentage': 100 - pct_trainable,
            'trainable_interactions': len(trainable_interactions),
            'avg_interactions_per_trainable_user': avg_interactions_trainable,
            'estimated_matrix_density': estimated_density,
            'special_cases': num_special_cases
        }
        
        logger.info("\n" + "="*80)
        logger.info("✓ User segmentation completed")
        
        return interactions_df, stats
    
    def filter_items_iteratively(
        self,
        interactions_df: pd.DataFrame,
        item_col: str = 'product_id',
        rating_col: str = 'rating'
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Iteratively filter items with insufficient positive interactions.
        
        This is done AFTER user filtering to remove items that don't have
        enough positive interactions from trainable users.
        
        Args:
            interactions_df: DataFrame with user interactions and 'is_trainable_user' column
            item_col: Item ID column name
            rating_col: Rating column name
        
        Returns:
            Tuple[DataFrame, Dict]:
                - Filtered DataFrame (items with sufficient positives only)
                - Statistics dict
        
        Algorithm:
            1. Filter to trainable users only
            2. Count positive interactions per item
            3. Remove items with < min_item_positives
            4. Repeat until stable (no more items removed)
        """
        logger.info("\n" + "="*80)
        logger.info("ITERATIVE ITEM FILTERING")
        logger.info("="*80)
        logger.info(f"Removing items with < {self.min_item_positives} positive interactions from trainable users")
        
        if 'is_trainable_user' not in interactions_df.columns:
            raise ValueError("Must run segment_users() before filter_items_iteratively()")
        
        initial_size = len(interactions_df)
        initial_items = interactions_df[item_col].nunique()
        
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            iteration += 1
            size_before = len(interactions_df)
            items_before = interactions_df[item_col].nunique()
            
            # Filter to trainable users only for counting
            trainable_df = interactions_df[interactions_df['is_trainable_user']]
            
            # Count positive interactions per item
            item_pos_counts = trainable_df[
                trainable_df[rating_col] >= self.positive_threshold
            ].groupby(item_col).size()
            
            # Identify cold items (< min positives)
            cold_items = item_pos_counts[item_pos_counts < self.min_item_positives].index
            num_cold_items = len(cold_items)
            
            if num_cold_items == 0:
                logger.info(f"Iteration {iteration}: Converged (no more items to remove)")
                break
            
            # Remove cold items from ALL interactions (trainable + cold-start)
            interactions_df = interactions_df[~interactions_df[item_col].isin(cold_items)]
            
            size_after = len(interactions_df)
            items_after = interactions_df[item_col].nunique()
            
            logger.info(f"Iteration {iteration}:")
            logger.info(f"  Removed {num_cold_items:,} cold items (< {self.min_item_positives} positives)")
            logger.info(f"  Items: {items_before:,} → {items_after:,}")
            logger.info(f"  Interactions: {size_before:,} → {size_after:,}")
        
        if iteration == max_iterations:
            logger.warning(f"⚠ Reached max iterations ({max_iterations}) without convergence")
        
        # Final statistics
        final_size = len(interactions_df)
        final_items = interactions_df[item_col].nunique()
        retention_rate = final_size / initial_size * 100
        
        logger.info("\n" + "-"*80)
        logger.info("ITEM FILTERING SUMMARY")
        logger.info("-"*80)
        logger.info(f"Initial items:       {initial_items:,}")
        logger.info(f"Final items:         {final_items:,}")
        logger.info(f"Items removed:       {initial_items - final_items:,}")
        logger.info(f"Initial interactions: {initial_size:,}")
        logger.info(f"Final interactions:   {final_size:,}")
        logger.info(f"Retention rate:       {retention_rate:.2f}%")
        logger.info(f"Iterations:           {iteration}")
        logger.info("-"*80)
        logger.info("✓ Item filtering completed")
        
        stats = {
            'initial_items': initial_items,
            'final_items': final_items,
            'items_removed': initial_items - final_items,
            'initial_interactions': initial_size,
            'final_interactions': final_size,
            'retention_rate': retention_rate,
            'iterations': iteration
        }
        
        return interactions_df, stats
    
    def apply_complete_filtering(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'product_id',
        rating_col: str = 'rating'
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Apply complete filtering pipeline: user segmentation + iterative item filtering.
        
        Args:
            interactions_df: Raw interactions DataFrame
            user_col: User ID column
            item_col: Item ID column
            rating_col: Rating column
        
        Returns:
            Tuple[DataFrame, Dict]:
                - Fully filtered DataFrame
                - Combined statistics from both steps
        
        Workflow:
            1. Segment users into trainable vs cold-start
            2. Iteratively filter items (only count trainable users)
            3. Return filtered data with both trainable and cold-start users
        """
        logger.info("\n" + "#"*80)
        logger.info("# STEP 2.3: COMPLETE USER & ITEM FILTERING PIPELINE")
        logger.info("#"*80)
        
        # Step 1: User segmentation
        interactions_df, user_stats = self.segment_users(
            interactions_df, user_col, rating_col
        )
        
        # Step 2: Iterative item filtering
        interactions_df, item_stats = self.filter_items_iteratively(
            interactions_df, item_col, rating_col
        )
        
        # Recompute user stats after item filtering
        trainable_df = interactions_df[interactions_df['is_trainable_user']]
        num_trainable_users = trainable_df[user_col].nunique()
        num_items = interactions_df[item_col].nunique()
        matrix_density = len(trainable_df) / (num_trainable_users * num_items) * 100
        
        logger.info("\n" + "="*80)
        logger.info("FINAL FILTERING SUMMARY")
        logger.info("="*80)
        logger.info(f"Trainable users:     {num_trainable_users:,}")
        logger.info(f"Cold-start users:    {interactions_df[~interactions_df['is_trainable_user']][user_col].nunique():,}")
        logger.info(f"Final items:         {num_items:,}")
        logger.info(f"Trainable interactions: {len(trainable_df):,}")
        logger.info(f"Total interactions:  {len(interactions_df):,}")
        logger.info(f"Matrix density:      {matrix_density:.4f}%")
        logger.info("="*80)
        logger.info("✓ Complete filtering pipeline finished")
        
        # Combine stats
        combined_stats = {
            **user_stats,
            **item_stats,
            'final_trainable_users': num_trainable_users,
            'final_matrix_density': matrix_density
        }
        
        return interactions_df, combined_stats
    
    def get_trainable_user_set(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id'
    ) -> Set[int]:
        """
        Get set of trainable user IDs.
        
        Args:
            interactions_df: DataFrame with 'is_trainable_user' column
            user_col: User ID column
        
        Returns:
            Set of trainable user IDs
        
        Usage:
            Used to filter data for CF training (exclude cold-start users)
        """
        if 'is_trainable_user' not in interactions_df.columns:
            raise ValueError("Must run segment_users() first")
        
        trainable_users = set(
            interactions_df[interactions_df['is_trainable_user']][user_col].unique()
        )
        
        return trainable_users
    
    def get_cold_start_user_set(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id'
    ) -> Set[int]:
        """
        Get set of cold-start user IDs.
        
        Args:
            interactions_df: DataFrame with 'is_trainable_user' column
            user_col: User ID column
        
        Returns:
            Set of cold-start user IDs
        
        Usage:
            Used to route users to content-based serving
        """
        if 'is_trainable_user' not in interactions_df.columns:
            raise ValueError("Must run segment_users() first")
        
        cold_start_users = set(
            interactions_df[~interactions_df['is_trainable_user']][user_col].unique()
        )
        
        return cold_start_users
    
    def filter_to_trainable_only(
        self,
        interactions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter DataFrame to trainable users only.
        
        Args:
            interactions_df: DataFrame with 'is_trainable_user' column
        
        Returns:
            Filtered DataFrame (trainable users only)
        
        Usage:
            Used before CF training to exclude cold-start users
        """
        if 'is_trainable_user' not in interactions_df.columns:
            raise ValueError("Must run segment_users() first")
        
        trainable_df = interactions_df[interactions_df['is_trainable_user']].copy()
        
        logger.info(f"\nFiltered to trainable users only:")
        logger.info(f"  Original size: {len(interactions_df):,}")
        logger.info(f"  Trainable size: {len(trainable_df):,}")
        logger.info(f"  Retention: {len(trainable_df)/len(interactions_df)*100:.2f}%")
        
        return trainable_df
    
    def validate_segmentation(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        rating_col: str = 'rating'
    ) -> bool:
        """
        Validate user segmentation correctness.
        
        Args:
            interactions_df: DataFrame with 'is_trainable_user' column
            user_col: User ID column
            rating_col: Rating column
        
        Returns:
            bool: True if validation passes
        
        Raises:
            ValueError: If validation fails
        """
        logger.info("\nValidating user segmentation...")
        
        if 'is_trainable_user' not in interactions_df.columns:
            raise ValueError("No 'is_trainable_user' column found")
        
        # Check trainable users meet criteria
        trainable_users = interactions_df[interactions_df['is_trainable_user']].groupby(user_col).agg(
            total=(rating_col, 'count'),
            positives=(rating_col, lambda x: (x >= self.positive_threshold).sum())
        )
        
        # All trainable users should have ≥ min_user_interactions
        invalid_interactions = trainable_users[trainable_users['total'] < self.min_user_interactions]
        if len(invalid_interactions) > 0:
            raise ValueError(
                f"Found {len(invalid_interactions)} trainable users with "
                f"< {self.min_user_interactions} interactions"
            )
        
        # All trainable users should have ≥ min_user_positives
        invalid_positives = trainable_users[trainable_users['positives'] < self.min_user_positives]
        if len(invalid_positives) > 0:
            raise ValueError(
                f"Found {len(invalid_positives)} trainable users with "
                f"< {self.min_user_positives} positive interactions"
            )
        
        # Check cold-start users don't meet criteria
        cold_start_users = interactions_df[~interactions_df['is_trainable_user']].groupby(user_col).agg(
            total=(rating_col, 'count'),
            positives=(rating_col, lambda x: (x >= self.positive_threshold).sum())
        )
        
        # Cold-start users should fail at least one criterion
        meets_both_criteria = (
            (cold_start_users['total'] >= self.min_user_interactions) &
            (cold_start_users['positives'] >= self.min_user_positives)
        )
        
        if meets_both_criteria.any():
            raise ValueError(
                f"Found {meets_both_criteria.sum()} cold-start users that meet trainable criteria"
            )
        
        logger.info("✓ User segmentation validation passed")
        return True
