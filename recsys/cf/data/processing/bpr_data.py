"""
BPR Data Preparation Module

This module handles Step 2.2: Positive/Negative Labels with Hard Negative Mining for BPR.
Implements dual-strategy hard negative mining to combat data sparsity.

Key Features:
- Positive signal: rating >= 4
- Hard Negative Strategy 1: Explicit negatives (rating <= 3)
- Hard Negative Strategy 2: Implicit negatives from popularity (Top-50 items NOT bought)
- Sampling: 30% hard negatives + 70% random negatives
"""

import logging
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


logger = logging.getLogger("data_layer")


class BPRDataPreparer:
    """
    Prepare data specifically for BPR (Bayesian Personalized Ranking) training.
    
    BPR paradigm: Pairwise ranking with positive/negative sampling
    - Positive: Items user explicitly liked (rating >= 4)
    - Hard Negatives: Items user disliked OR popular items user ignored
    - Rationale: Informative negatives improve ranking quality vs random sampling
    """
    
    def __init__(
        self,
        positive_threshold: float = 4.0,
        hard_negative_threshold: float = 3.0,
        top_k_popular: int = 50,
        hard_negative_ratio: float = 0.3
    ):
        """
        Initialize BPRDataPreparer.
        
        Args:
            positive_threshold: Rating threshold for positive interactions
            hard_negative_threshold: Rating threshold for explicit hard negatives
            top_k_popular: Number of top popular items for implicit negatives
            hard_negative_ratio: Fraction of negatives from hard negatives (rest random)
        """
        self.positive_threshold = positive_threshold
        self.hard_negative_threshold = hard_negative_threshold
        self.top_k_popular = top_k_popular
        self.hard_negative_ratio = hard_negative_ratio
    
    def create_positive_labels(
        self,
        interactions_df: pd.DataFrame,
        rating_col: str = 'rating'
    ) -> pd.DataFrame:
        """
        Create binary positive labels based on rating threshold.
        
        Args:
            interactions_df: DataFrame with ratings
            rating_col: Column name for ratings
        
        Returns:
            DataFrame with added 'is_positive' column (0/1)
        
        Example:
            >>> preparer = BPRDataPreparer(positive_threshold=4.0)
            >>> df = preparer.create_positive_labels(interactions_df)
            >>> df['is_positive'].value_counts()
            1    331440  # rating >= 4
            0      6604  # rating < 4
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2.2: BPR POSITIVE LABELS")
        logger.info("="*80)
        logger.info(f"Positive threshold: rating >= {self.positive_threshold}")
        
        # Create binary labels
        interactions_df['is_positive'] = (
            interactions_df[rating_col] >= self.positive_threshold
        ).astype(int)
        
        # Log statistics
        num_positive = interactions_df['is_positive'].sum()
        num_negative = len(interactions_df) - num_positive
        pct_positive = num_positive / len(interactions_df) * 100
        
        logger.info(f"\nPositive interactions:  {num_positive:,} ({pct_positive:.2f}%)")
        logger.info(f"Negative interactions:  {num_negative:,} ({100-pct_positive:.2f}%)")
        
        # Distribution by rating
        logger.info("\nPositive label distribution by rating:")
        for rating, group in interactions_df.groupby(rating_col):
            pos_count = group['is_positive'].sum()
            total = len(group)
            logger.info(
                f"  Rating {rating:.0f}: {pos_count:,}/{total:,} positive "
                f"({pos_count/total*100:.1f}%)"
            )
        
        logger.info("✓ Positive labels created")
        
        return interactions_df
    
    def mine_hard_negatives(
        self,
        interactions_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
        rating_col: str = 'rating',
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        popularity_col: str = 'num_sold_time'
    ) -> Tuple[pd.DataFrame, Dict[str, Set[int]]]:
        """
        Mine hard negatives using dual strategy.
        
        Strategy 1 - Explicit Hard Negatives:
            Items with rating <= hard_negative_threshold (user bought but disliked)
        
        Strategy 2 - Implicit Hard Negatives:
            Top-K popular items user DIDN'T interact with
            Logic: "Hot product but you didn't buy → implicit negative preference"
        
        Args:
            interactions_df: DataFrame with user-item interactions
            products_df: Optional DataFrame with product metadata (for popularity)
            rating_col: Rating column name
            user_col: User index column
            item_col: Item index column
            popularity_col: Popularity metric column (in products_df)
        
        Returns:
            Tuple[DataFrame, Dict]:
                - Updated interactions_df with 'is_hard_negative' and 'hard_neg_source'
                - Dict mapping u_idx to Set of hard negative item indices
        
        Example:
            >>> df, hard_neg_sets = preparer.mine_hard_negatives(
            ...     train_df, products_df
            ... )
            >>> df['is_hard_negative'].value_counts()
            0    335000  # Not hard negative
            1      3044  # Hard negative (explicit or implicit)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2.2: HARD NEGATIVE MINING")
        logger.info("="*80)
        logger.info("Dual strategy: Explicit (low ratings) + Implicit (popular items not bought)")
        
        # Strategy 1: Explicit hard negatives
        logger.info(f"\nStrategy 1: Explicit hard negatives (rating <= {self.hard_negative_threshold})")
        explicit_hard_neg = interactions_df[rating_col] <= self.hard_negative_threshold
        num_explicit = explicit_hard_neg.sum()
        logger.info(f"  Found {num_explicit:,} explicit hard negatives")
        
        # Initialize columns
        interactions_df['is_hard_negative'] = explicit_hard_neg.astype(int)
        interactions_df['hard_neg_source'] = 'none'
        interactions_df.loc[explicit_hard_neg, 'hard_neg_source'] = 'explicit'
        
        # Strategy 2: Implicit hard negatives from popularity
        logger.info(f"\nStrategy 2: Implicit hard negatives (Top-{self.top_k_popular} popular items)")
        
        implicit_hard_neg_sets = {}
        
        if products_df is not None and popularity_col in products_df.columns:
            # Identify top-K popular items
            top_popular_items = self._identify_top_popular_items(
                products_df, 
                item_col='product_id',
                popularity_col=popularity_col
            )
            
            logger.info(f"  Top-{self.top_k_popular} popular items identified")
            
            # For each user, find popular items they DIDN'T buy
            implicit_hard_neg_sets = self._find_implicit_negatives(
                interactions_df,
                top_popular_items,
                user_col=user_col,
                item_col=item_col
            )
            
            total_implicit = sum(len(items) for items in implicit_hard_neg_sets.values())
            avg_implicit = total_implicit / len(implicit_hard_neg_sets) if implicit_hard_neg_sets else 0
            
            logger.info(f"  Generated {total_implicit:,} implicit hard negatives")
            logger.info(f"  Average {avg_implicit:.1f} per user")
        else:
            logger.warning("⚠ Products DataFrame not provided - skipping implicit negatives")
        
        # Combine explicit and implicit hard negatives
        combined_hard_neg_sets = self._combine_hard_negatives(
            interactions_df,
            implicit_hard_neg_sets,
            user_col=user_col,
            item_col=item_col
        )
        
        # Log summary
        logger.info("\n" + "-"*80)
        logger.info("HARD NEGATIVE MINING SUMMARY")
        logger.info("-"*80)
        
        total_hard_neg = interactions_df['is_hard_negative'].sum()
        pct_hard_neg = total_hard_neg / len(interactions_df) * 100
        
        logger.info(f"Total hard negatives:      {total_hard_neg:,} ({pct_hard_neg:.2f}%)")
        logger.info(f"  - Explicit (low rating): {num_explicit:,}")
        
        if implicit_hard_neg_sets:
            num_implicit = total_hard_neg - num_explicit
            logger.info(f"  - Implicit (popularity): {num_implicit:,}")
        
        logger.info(f"Users with hard negatives: {len(combined_hard_neg_sets):,}")
        logger.info("-"*80)
        logger.info("✓ Hard negative mining completed")
        
        return interactions_df, combined_hard_neg_sets
    
    def _identify_top_popular_items(
        self,
        products_df: pd.DataFrame,
        item_col: str = 'product_id',
        popularity_col: str = 'num_sold_time'
    ) -> Set[int]:
        """
        Identify top-K most popular items by sales count.
        
        Args:
            products_df: DataFrame with product metadata
            item_col: Item ID column
            popularity_col: Popularity metric column
        
        Returns:
            Set of top-K popular item IDs
        """
        # Handle missing popularity values
        products_sorted = products_df.copy()
        products_sorted[popularity_col] = products_sorted[popularity_col].fillna(0)
        
        # Sort and take top-K
        products_sorted = products_sorted.sort_values(
            popularity_col, 
            ascending=False
        ).head(self.top_k_popular)
        
        top_items = set(products_sorted[item_col].values)
        
        logger.info(f"    Popular items range: {products_sorted[popularity_col].min():.0f} - {products_sorted[popularity_col].max():.0f} sales")
        
        return top_items
    
    def _find_implicit_negatives(
        self,
        interactions_df: pd.DataFrame,
        top_popular_items: Set[int],
        user_col: str = 'u_idx',
        item_col: str = 'i_idx'
    ) -> Dict[int, Set[int]]:
        """
        For each user, find popular items they DIDN'T interact with.
        
        Args:
            interactions_df: User-item interactions
            top_popular_items: Set of top popular item IDs
            user_col: User index column
            item_col: Item index column
        
        Returns:
            Dict mapping user_idx -> Set of implicit negative item indices
        """
        # Build user interaction sets
        user_items = interactions_df.groupby(user_col)[item_col].apply(set).to_dict()
        
        implicit_negatives = {}
        
        for user_idx, interacted_items in user_items.items():
            # Items that are popular but user didn't buy
            implicit_neg_items = top_popular_items - interacted_items
            
            if implicit_neg_items:
                implicit_negatives[user_idx] = implicit_neg_items
        
        return implicit_negatives
    
    def _combine_hard_negatives(
        self,
        interactions_df: pd.DataFrame,
        implicit_hard_neg_sets: Dict[int, Set[int]],
        user_col: str = 'u_idx',
        item_col: str = 'i_idx'
    ) -> Dict[int, Set[int]]:
        """
        Combine explicit and implicit hard negatives per user.
        
        Args:
            interactions_df: DataFrame with explicit hard negatives marked
            implicit_hard_neg_sets: Dict of implicit hard negatives per user
            user_col: User index column
            item_col: Item index column
        
        Returns:
            Dict mapping user_idx -> Set of all hard negative item indices
        """
        combined = {}
        
        # Get explicit hard negatives from interactions
        explicit_df = interactions_df[
            interactions_df['hard_neg_source'] == 'explicit'
        ]
        
        for user_idx, group in explicit_df.groupby(user_col):
            combined[user_idx] = set(group[item_col].values)
        
        # Add implicit hard negatives
        for user_idx, implicit_items in implicit_hard_neg_sets.items():
            if user_idx in combined:
                combined[user_idx].update(implicit_items)
            else:
                combined[user_idx] = implicit_items
        
        return combined
    
    def build_positive_sets(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx'
    ) -> Dict[int, Set[int]]:
        """
        Build positive item sets per user (for negative sampling).
        
        Args:
            interactions_df: DataFrame with 'is_positive' column
            user_col: User index column
            item_col: Item index column
        
        Returns:
            Dict mapping user_idx -> Set of positive item indices
        
        Usage:
            Used during BPR training to exclude positive items when sampling negatives
        """
        logger.info("\nBuilding user positive item sets...")
        
        positive_df = interactions_df[interactions_df['is_positive'] == 1]
        
        user_pos_sets = positive_df.groupby(user_col)[item_col].apply(set).to_dict()
        
        num_users = len(user_pos_sets)
        avg_pos = sum(len(items) for items in user_pos_sets.values()) / num_users
        
        logger.info(f"  Users with positives: {num_users:,}")
        logger.info(f"  Average positives per user: {avg_pos:.2f}")
        logger.info("✓ Positive sets built")
        
        return user_pos_sets
    
    def create_binary_matrix(
        self,
        interactions_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx'
    ) -> Tuple[csr_matrix, Dict[str, float]]:
        """
        Build binary sparse matrix for BPR (optional).
        
        Args:
            interactions_df: DataFrame with positive interactions only
            num_users: Total number of users
            num_items: Total number of items
            user_col: User index column
            item_col: Item index column
        
        Returns:
            Tuple[csr_matrix, Dict]:
                - Binary matrix (1 for positive interactions, 0 elsewhere)
                - Statistics dict
        
        Note:
            This matrix is OPTIONAL for BPR. Most BPR implementations
            sample on-the-fly during training rather than using pre-built matrix.
        """
        logger.info("\nBuilding binary matrix (optional for BPR)...")
        
        # Filter to positive interactions only
        positive_df = interactions_df[interactions_df['is_positive'] == 1]
        
        users = positive_df[user_col].values
        items = positive_df[item_col].values
        values = np.ones(len(positive_df), dtype=np.float32)
        
        # Build sparse matrix
        X_binary = csr_matrix(
            (values, (users, items)),
            shape=(num_users, num_items),
            dtype=np.float32
        )
        
        # Compute stats
        total_cells = num_users * num_items
        sparsity = 1.0 - (X_binary.nnz / total_cells)
        
        stats = {
            'sparsity': sparsity,
            'density': 1.0 - sparsity,
            'nnz': X_binary.nnz,
            'shape': X_binary.shape
        }
        
        logger.info(f"  Matrix shape: {X_binary.shape}")
        logger.info(f"  Non-zero entries: {X_binary.nnz:,}")
        logger.info(f"  Sparsity: {sparsity:.4%}")
        logger.info("✓ Binary matrix built")
        
        return X_binary, stats
    
    def get_sampling_strategy_info(self) -> Dict[str, float]:
        """
        Get information about negative sampling strategy for BPR training.
        
        Returns:
            Dict with sampling ratios and recommendations
        
        Usage:
            Pass this to BPR training module to guide negative sampling
        """
        return {
            'hard_negative_ratio': self.hard_negative_ratio,
            'random_negative_ratio': 1.0 - self.hard_negative_ratio,
            'positive_threshold': self.positive_threshold,
            'hard_negative_threshold': self.hard_negative_threshold,
            'recommendation': (
                f"Sample {self.hard_negative_ratio*100:.0f}% from hard negatives, "
                f"{(1-self.hard_negative_ratio)*100:.0f}% from random unseen items"
            )
        }
    
    def validate_labels(
        self,
        interactions_df: pd.DataFrame,
        rating_col: str = 'rating'
    ) -> bool:
        """
        Validate positive/negative labels consistency.
        
        Args:
            interactions_df: DataFrame with labels
            rating_col: Rating column
        
        Returns:
            bool: True if validation passes
        
        Raises:
            ValueError: If validation fails
        """
        logger.info("\nValidating BPR labels...")
        
        # Check 1: is_positive matches threshold
        expected_positive = (
            interactions_df[rating_col] >= self.positive_threshold
        ).astype(int)
        
        if not (interactions_df['is_positive'] == expected_positive).all():
            raise ValueError("is_positive labels don't match rating threshold")
        
        # Check 2: Hard negatives should have low ratings OR be implicit
        explicit_hard_neg = interactions_df[
            (interactions_df['is_hard_negative'] == 1) &
            (interactions_df['hard_neg_source'] == 'explicit')
        ]
        
        if len(explicit_hard_neg) > 0:
            invalid = explicit_hard_neg[
                explicit_hard_neg[rating_col] > self.hard_negative_threshold
            ]
            
            if len(invalid) > 0:
                raise ValueError(
                    f"Found {len(invalid)} explicit hard negatives with "
                    f"rating > {self.hard_negative_threshold}"
                )
        
        # Check 3: No overlap between positive and hard negative (for explicit)
        overlap = interactions_df[
            (interactions_df['is_positive'] == 1) &
            (interactions_df['is_hard_negative'] == 1) &
            (interactions_df['hard_neg_source'] == 'explicit')
        ]
        
        if len(overlap) > 0:
            raise ValueError(
                f"Found {len(overlap)} interactions marked as both positive and "
                f"explicit hard negative"
            )
        
        logger.info("✓ BPR labels validation passed")
        return True
