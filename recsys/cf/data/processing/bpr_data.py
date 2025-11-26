"""
BPR Data Preparation Module

This module handles Step 2.2: Positive/Negative Labels with Hard Negative Mining for BPR.
Implements dual-strategy hard negative mining to combat data sparsity.

Key Features:
- Positive signal: rating >= 4
- Hard Negative Strategy 1: Explicit negatives (rating <= 3)
- Hard Negative Strategy 2: Implicit negatives from popularity (Top-50 items NOT bought)
- Sampling: 30% hard negatives + 70% random negatives
- Sentiment-aware confidence scoring for weighted BPR training

Sentiment-Aware Enhancement:
- Uses confidence_score (rating + comment_quality) to weight triplets
- Higher weight for genuine positive reviews (high sentiment score)
- Lower weight for suspicious reviews (rating/sentiment mismatch)
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any

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
    
    # ========================================================================
    # Step 1: BPR Positive Pairs Methods (for Training)
    # ========================================================================
    
    def build_positive_pairs(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx'
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Build positive pairs list from DataFrame for BPR training (Step 1.2).
        
        Args:
            interactions_df: DataFrame with 'is_positive' column (must have 0/1)
            user_col: User index column
            item_col: Item index column
        
        Returns:
            Tuple[np.ndarray, Dict]:
                - Array of shape (N, 2) with columns [u_idx, i_idx]
                - Statistics dictionary
        
        Usage:
            Positive pairs are used in BPR training:
            1. For each (u, i_pos) pair, sample a negative i_neg
            2. Compute BPR loss: -log(sigmoid(score(u, i_pos) - score(u, i_neg)))
        
        Example:
            >>> preparer = BPRDataPreparer(positive_threshold=4.0)
            >>> pairs, stats = preparer.build_positive_pairs(train_df)
            >>> print(f"Total positive pairs: {len(pairs):,}")
            >>> print(f"First 5 pairs: {pairs[:5]}")
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1.2: BUILD BPR POSITIVE PAIRS")
        logger.info("="*80)
        
        # Check for is_positive column
        if 'is_positive' not in interactions_df.columns:
            logger.info("Creating is_positive column...")
            if 'rating' in interactions_df.columns:
                interactions_df = self.create_positive_labels(interactions_df)
            else:
                raise ValueError(
                    "DataFrame must have 'is_positive' column or 'rating' column "
                    "to compute positive labels"
                )
        
        # Filter to positive interactions
        positive_df = interactions_df[interactions_df['is_positive'] == 1]
        
        # Extract pairs
        users = positive_df[user_col].values
        items = positive_df[item_col].values
        
        pairs = np.column_stack([users, items])
        
        # Compute statistics
        num_pairs = len(pairs)
        num_unique_users = len(np.unique(users))
        num_unique_items = len(np.unique(items))
        avg_pairs_per_user = num_pairs / num_unique_users if num_unique_users > 0 else 0
        
        stats = {
            'num_pairs': num_pairs,
            'num_unique_users': num_unique_users,
            'num_unique_items': num_unique_items,
            'avg_pairs_per_user': avg_pairs_per_user,
            'positive_threshold': self.positive_threshold,
            'pair_shape': pairs.shape
        }
        
        logger.info(f"Positive pairs:       {num_pairs:,}")
        logger.info(f"Unique users:         {num_unique_users:,}")
        logger.info(f"Unique items:         {num_unique_items:,}")
        logger.info(f"Avg pairs per user:   {avg_pairs_per_user:.2f}")
        logger.info("✓ Positive pairs built")
        
        return pairs, stats
    
    def build_positive_pairs_from_sets(
        self,
        user_pos_sets: Dict[int, Set[int]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Build positive pairs list from user positive sets (Step 1.2 alternative).
        
        This method creates the same output as build_positive_pairs() but uses
        precomputed user_pos_sets instead of scanning DataFrame.
        
        Args:
            user_pos_sets: Dict mapping u_idx -> Set of positive i_idx
                          (from build_positive_sets())
        
        Returns:
            Tuple[np.ndarray, Dict]:
                - Array of shape (N, 2) with columns [u_idx, i_idx]
                - Statistics dictionary
        
        Example:
            >>> user_pos_sets = preparer.build_positive_sets(train_df)
            >>> pairs, stats = preparer.build_positive_pairs_from_sets(user_pos_sets)
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1.2: BUILD BPR POSITIVE PAIRS (FROM SETS)")
        logger.info("="*80)
        
        # Expand user-item pairs
        all_pairs = []
        for user_idx, item_set in user_pos_sets.items():
            for item_idx in item_set:
                all_pairs.append([user_idx, item_idx])
        
        pairs = np.array(all_pairs, dtype=np.int64)
        
        # Compute statistics
        num_pairs = len(pairs)
        num_unique_users = len(user_pos_sets)
        all_items = set()
        for items in user_pos_sets.values():
            all_items.update(items)
        num_unique_items = len(all_items)
        avg_pairs_per_user = num_pairs / num_unique_users if num_unique_users > 0 else 0
        
        stats = {
            'num_pairs': num_pairs,
            'num_unique_users': num_unique_users,
            'num_unique_items': num_unique_items,
            'avg_pairs_per_user': avg_pairs_per_user,
            'pair_shape': pairs.shape
        }
        
        logger.info(f"Positive pairs:       {num_pairs:,}")
        logger.info(f"Unique users:         {num_unique_users:,}")
        logger.info(f"Unique items:         {num_unique_items:,}")
        logger.info(f"Avg pairs per user:   {avg_pairs_per_user:.2f}")
        logger.info("✓ Positive pairs built from sets")
        
        return pairs, stats
    
    def get_bpr_training_data(
        self,
        interactions_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        rating_col: str = 'rating'
    ) -> Dict[str, Any]:
        """
        Get complete BPR training data (Step 1 complete).
        
        This method orchestrates the full data preparation pipeline:
        1. Create positive labels (is_positive column)
        2. Build positive sets (user -> set of positive items)
        3. Build positive pairs (array of [u, i] for training)
        4. Mine hard negatives (explicit + implicit)
        5. Return comprehensive training data
        
        Args:
            interactions_df: DataFrame with user-item interactions
            products_df: Optional product metadata for implicit negative mining
            user_col: User index column
            item_col: Item index column
            rating_col: Rating column
        
        Returns:
            Dict containing:
                - 'positive_pairs': np.ndarray of shape (N, 2)
                - 'user_pos_sets': Dict[u_idx, Set[i_idx]]
                - 'hard_neg_sets': Dict[u_idx, Set[i_idx]]
                - 'num_users': Total number of users
                - 'num_items': Total number of items
                - 'stats': Comprehensive statistics
        
        Example:
            >>> data = preparer.get_bpr_training_data(train_df, products_df)
            >>> print(f"Positive pairs: {len(data['positive_pairs']):,}")
            >>> print(f"Users with hard negatives: {len(data['hard_neg_sets']):,}")
        """
        logger.info("\n" + "="*80)
        logger.info("BPR DATA PREPARATION - COMPLETE PIPELINE")
        logger.info("="*80)
        
        # Step 1: Create positive labels
        df_labeled = self.create_positive_labels(interactions_df.copy(), rating_col)
        
        # Step 2: Build positive sets
        user_pos_sets = self.build_positive_sets(
            df_labeled, user_col, item_col
        )
        
        # Step 3: Build positive pairs
        positive_pairs, pairs_stats = self.build_positive_pairs(
            df_labeled, user_col, item_col
        )
        
        # Step 4: Mine hard negatives
        _, hard_neg_sets = self.mine_hard_negatives(
            df_labeled, products_df, rating_col, user_col, item_col
        )
        
        # Compute comprehensive statistics
        num_users = df_labeled[user_col].nunique()
        num_items = df_labeled[item_col].nunique()
        
        stats = {
            'num_users': num_users,
            'num_items': num_items,
            'num_positive_pairs': len(positive_pairs),
            'num_users_with_positives': len(user_pos_sets),
            'num_users_with_hard_negatives': len(hard_neg_sets),
            'avg_positives_per_user': pairs_stats['avg_pairs_per_user'],
            'avg_hard_negatives_per_user': (
                sum(len(s) for s in hard_neg_sets.values()) / len(hard_neg_sets)
                if hard_neg_sets else 0
            ),
            'positive_threshold': self.positive_threshold,
            'hard_negative_threshold': self.hard_negative_threshold,
            'hard_negative_ratio': self.hard_negative_ratio,
            'top_k_popular': self.top_k_popular
        }
        
        logger.info("\n" + "-"*80)
        logger.info("BPR DATA PREPARATION COMPLETE")
        logger.info("-"*80)
        logger.info(f"Users: {num_users:,}, Items: {num_items:,}")
        logger.info(f"Positive pairs: {len(positive_pairs):,}")
        logger.info(f"Users with hard negatives: {len(hard_neg_sets):,}")
        logger.info("-"*80)
        
        return {
            'positive_pairs': positive_pairs,
            'user_pos_sets': user_pos_sets,
            'hard_neg_sets': hard_neg_sets,
            'num_users': num_users,
            'num_items': num_items,
            'stats': stats
        }
    
    # ========================================================================
    # Sentiment-Aware BPR Methods (Enhanced for fake review detection)
    # ========================================================================
    
    def build_positive_pairs_with_confidence(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        confidence_col: str = 'confidence_score'
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Build positive pairs with associated confidence scores for sentiment-aware BPR.
        
        This method extracts positive pairs along with their confidence scores
        (rating + comment_quality) for weighted BPR training. Higher confidence
        means the review is more trustworthy (genuine positive vs spam/fake).
        
        Confidence Score Range:
        - Min: 1.0 (rating 1 + quality 0)
        - Max: 6.0 (rating 5 + quality 1)
        - Threshold: >= 4.5 for "trusted" positives (rating 4 + quality 0.5)
        
        Args:
            interactions_df: DataFrame with 'is_positive' and 'confidence_score' columns
            user_col: User index column
            item_col: Item index column
            confidence_col: Confidence score column (rating + comment_quality)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]:
                - pairs: Array of shape (N, 2) with [u_idx, i_idx]
                - confidence_scores: Array of shape (N,) with confidence values
                - stats: Statistics dictionary including confidence distribution
        
        Example:
            >>> preparer = BPRDataPreparer()
            >>> pairs, scores, stats = preparer.build_positive_pairs_with_confidence(df)
            >>> print(f"Positive pairs: {len(pairs):,}")
            >>> print(f"Mean confidence: {stats['confidence_mean']:.3f}")
            >>> print(f"High confidence ratio: {stats['high_confidence_ratio']:.2%}")
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1.2+: BUILD SENTIMENT-AWARE BPR POSITIVE PAIRS")
        logger.info("="*80)
        
        # Check for required columns
        if 'is_positive' not in interactions_df.columns:
            logger.info("Creating is_positive column...")
            if 'rating' in interactions_df.columns:
                interactions_df = self.create_positive_labels(interactions_df)
            else:
                raise ValueError("DataFrame must have 'is_positive' or 'rating' column")
        
        if confidence_col not in interactions_df.columns:
            logger.warning(
                f"Column '{confidence_col}' not found. "
                f"Using rating as confidence score (no sentiment enhancement)."
            )
            confidence_col = 'rating'
        
        # Filter to positive interactions
        positive_df = interactions_df[interactions_df['is_positive'] == 1].copy()
        
        # Extract pairs and confidence scores
        users = positive_df[user_col].values
        items = positive_df[item_col].values
        confidence_scores = positive_df[confidence_col].values.astype(np.float32)
        
        pairs = np.column_stack([users, items])
        
        # Compute statistics
        num_pairs = len(pairs)
        num_unique_users = len(np.unique(users))
        num_unique_items = len(np.unique(items))
        avg_pairs_per_user = num_pairs / num_unique_users if num_unique_users > 0 else 0
        
        # Confidence distribution stats
        confidence_mean = float(confidence_scores.mean())
        confidence_std = float(confidence_scores.std())
        confidence_min = float(confidence_scores.min())
        confidence_max = float(confidence_scores.max())
        
        # High confidence threshold (rating 4 + quality 0.5 = 4.5)
        high_confidence_threshold = 4.5
        high_confidence_count = (confidence_scores >= high_confidence_threshold).sum()
        high_confidence_ratio = high_confidence_count / num_pairs if num_pairs > 0 else 0
        
        # Suspicious reviews (high rating but low confidence)
        # e.g., rating 5 but confidence < 5.3 means comment_quality < 0.3
        suspicious_threshold = 5.3
        suspicious_count = (
            (positive_df['rating'] == 5) & 
            (confidence_scores < suspicious_threshold)
        ).sum() if 'rating' in positive_df.columns else 0
        suspicious_ratio = suspicious_count / num_pairs if num_pairs > 0 else 0
        
        stats = {
            'num_pairs': num_pairs,
            'num_unique_users': num_unique_users,
            'num_unique_items': num_unique_items,
            'avg_pairs_per_user': avg_pairs_per_user,
            'positive_threshold': self.positive_threshold,
            'pair_shape': pairs.shape,
            # Confidence stats
            'confidence_col': confidence_col,
            'confidence_mean': confidence_mean,
            'confidence_std': confidence_std,
            'confidence_min': confidence_min,
            'confidence_max': confidence_max,
            'confidence_p25': float(np.percentile(confidence_scores, 25)),
            'confidence_p50': float(np.percentile(confidence_scores, 50)),
            'confidence_p75': float(np.percentile(confidence_scores, 75)),
            'high_confidence_threshold': high_confidence_threshold,
            'high_confidence_count': int(high_confidence_count),
            'high_confidence_ratio': high_confidence_ratio,
            'suspicious_count': int(suspicious_count),
            'suspicious_ratio': suspicious_ratio
        }
        
        logger.info(f"Positive pairs:           {num_pairs:,}")
        logger.info(f"Unique users:             {num_unique_users:,}")
        logger.info(f"Unique items:             {num_unique_items:,}")
        logger.info(f"Avg pairs per user:       {avg_pairs_per_user:.2f}")
        logger.info("\nConfidence Score Distribution:")
        logger.info(f"  Min:    {confidence_min:.3f}")
        logger.info(f"  Mean:   {confidence_mean:.3f}")
        logger.info(f"  Max:    {confidence_max:.3f}")
        logger.info(f"  Std:    {confidence_std:.3f}")
        logger.info(f"\nSentiment Quality Analysis:")
        logger.info(f"  High confidence (>={high_confidence_threshold}): {high_confidence_count:,} ({high_confidence_ratio:.2%})")
        logger.info(f"  Suspicious reviews:      {suspicious_count:,} ({suspicious_ratio:.2%})")
        logger.info("✓ Sentiment-aware positive pairs built")
        
        return pairs, confidence_scores, stats
    
    def get_bpr_training_data_with_confidence(
        self,
        interactions_df: pd.DataFrame,
        products_df: Optional[pd.DataFrame] = None,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        rating_col: str = 'rating',
        confidence_col: str = 'confidence_score'
    ) -> Dict[str, Any]:
        """
        Get complete BPR training data with sentiment-aware confidence scores.
        
        This is the enhanced version of get_bpr_training_data() that includes
        confidence scores for weighted BPR training. Use this for BERT-Enhanced
        BPR with sentiment-aware training.
        
        Pipeline:
        1. Create positive labels
        2. Build positive sets
        3. Build positive pairs WITH confidence scores
        4. Mine hard negatives (explicit + implicit)
        5. Return data with confidence for weighted training
        
        Args:
            interactions_df: DataFrame with user-item interactions
            products_df: Optional product metadata for implicit negative mining
            user_col: User index column
            item_col: Item index column
            rating_col: Rating column
            confidence_col: Confidence score column (rating + comment_quality)
        
        Returns:
            Dict containing:
                - 'positive_pairs': np.ndarray of shape (N, 2)
                - 'confidence_scores': np.ndarray of shape (N,) - for weighted training
                - 'user_pos_sets': Dict[u_idx, Set[i_idx]]
                - 'hard_neg_sets': Dict[u_idx, Set[i_idx]]
                - 'num_users': Total number of users
                - 'num_items': Total number of items
                - 'stats': Comprehensive statistics including confidence distribution
        
        Example:
            >>> data = preparer.get_bpr_training_data_with_confidence(train_df, products_df)
            >>> print(f"Positive pairs: {len(data['positive_pairs']):,}")
            >>> print(f"Mean confidence: {data['stats']['confidence_mean']:.3f}")
            >>> 
            >>> # Use with BERT-Enhanced BPR
            >>> model = BERTEnhancedBPR(bert_embeddings_path=...)
            >>> model.fit(
            ...     positive_pairs=data['positive_pairs'],
            ...     confidence_scores=data['confidence_scores'],
            ...     ...
            ... )
        """
        logger.info("\n" + "="*80)
        logger.info("BPR DATA PREPARATION - SENTIMENT-AWARE PIPELINE")
        logger.info("="*80)
        
        # Step 1: Create positive labels
        df_labeled = self.create_positive_labels(interactions_df.copy(), rating_col)
        
        # Step 2: Build positive sets
        user_pos_sets = self.build_positive_sets(
            df_labeled, user_col, item_col
        )
        
        # Step 3: Build positive pairs WITH confidence scores
        positive_pairs, confidence_scores, pairs_stats = self.build_positive_pairs_with_confidence(
            df_labeled, user_col, item_col, confidence_col
        )
        
        # Step 4: Mine hard negatives
        _, hard_neg_sets = self.mine_hard_negatives(
            df_labeled, products_df, rating_col, user_col, item_col
        )
        
        # Compute comprehensive statistics
        num_users = df_labeled[user_col].nunique()
        num_items = df_labeled[item_col].nunique()
        
        stats = {
            'num_users': num_users,
            'num_items': num_items,
            'num_positive_pairs': len(positive_pairs),
            'num_users_with_positives': len(user_pos_sets),
            'num_users_with_hard_negatives': len(hard_neg_sets),
            'avg_positives_per_user': pairs_stats['avg_pairs_per_user'],
            'avg_hard_negatives_per_user': (
                sum(len(s) for s in hard_neg_sets.values()) / len(hard_neg_sets)
                if hard_neg_sets else 0
            ),
            'positive_threshold': self.positive_threshold,
            'hard_negative_threshold': self.hard_negative_threshold,
            'hard_negative_ratio': self.hard_negative_ratio,
            'top_k_popular': self.top_k_popular,
            # Confidence stats from pairs_stats
            'confidence_col': pairs_stats.get('confidence_col', confidence_col),
            'confidence_mean': pairs_stats.get('confidence_mean'),
            'confidence_std': pairs_stats.get('confidence_std'),
            'confidence_min': pairs_stats.get('confidence_min'),
            'confidence_max': pairs_stats.get('confidence_max'),
            'high_confidence_ratio': pairs_stats.get('high_confidence_ratio'),
            'suspicious_ratio': pairs_stats.get('suspicious_ratio'),
        }
        
        logger.info("\n" + "-"*80)
        logger.info("SENTIMENT-AWARE BPR DATA PREPARATION COMPLETE")
        logger.info("-"*80)
        logger.info(f"Users: {num_users:,}, Items: {num_items:,}")
        logger.info(f"Positive pairs: {len(positive_pairs):,}")
        logger.info(f"Mean confidence: {stats['confidence_mean']:.3f}")
        logger.info(f"High confidence ratio: {stats['high_confidence_ratio']:.2%}")
        logger.info(f"Suspicious review ratio: {stats['suspicious_ratio']:.2%}")
        logger.info("-"*80)
        
        return {
            'positive_pairs': positive_pairs,
            'confidence_scores': confidence_scores,
            'user_pos_sets': user_pos_sets,
            'hard_neg_sets': hard_neg_sets,
            'num_users': num_users,
            'num_items': num_items,
            'stats': stats
        }
    
    def compute_confidence_weighted_stats(
        self,
        confidence_scores: np.ndarray,
        threshold_low: float = 4.0,
        threshold_high: float = 5.0
    ) -> Dict[str, Any]:
        """
        Compute detailed statistics about confidence score distribution.
        
        Useful for understanding the quality of reviews in the dataset
        and tuning the sentiment-aware weighting parameters.
        
        Args:
            confidence_scores: Array of confidence scores
            threshold_low: Lower threshold for "medium confidence"
            threshold_high: Upper threshold for "high confidence"
        
        Returns:
            Dictionary with detailed confidence distribution statistics
        """
        total = len(confidence_scores)
        
        low_conf = (confidence_scores < threshold_low).sum()
        medium_conf = ((confidence_scores >= threshold_low) & 
                       (confidence_scores < threshold_high)).sum()
        high_conf = (confidence_scores >= threshold_high).sum()
        
        return {
            'total_samples': total,
            'threshold_low': threshold_low,
            'threshold_high': threshold_high,
            'low_confidence_count': int(low_conf),
            'low_confidence_pct': low_conf / total if total > 0 else 0,
            'medium_confidence_count': int(medium_conf),
            'medium_confidence_pct': medium_conf / total if total > 0 else 0,
            'high_confidence_count': int(high_conf),
            'high_confidence_pct': high_conf / total if total > 0 else 0,
            'mean': float(confidence_scores.mean()),
            'std': float(confidence_scores.std()),
            'min': float(confidence_scores.min()),
            'max': float(confidence_scores.max()),
            'percentiles': {
                'p10': float(np.percentile(confidence_scores, 10)),
                'p25': float(np.percentile(confidence_scores, 25)),
                'p50': float(np.percentile(confidence_scores, 50)),
                'p75': float(np.percentile(confidence_scores, 75)),
                'p90': float(np.percentile(confidence_scores, 90)),
            }
        }

