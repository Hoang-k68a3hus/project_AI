"""
Matrix Construction Module for Collaborative Filtering

This module builds sparse CSR matrices and auxiliary data structures for CF training:
1. X_train_confidence: CSR matrix with confidence scores for ALS
2. X_train_binary: CSR matrix with binary values for BPR (optional)
3. User positive sets: Fast lookup for seen items
4. Hard negative sets: Explicit and implicit negatives
5. Item popularity: Log-transformed popularity scores
6. User metadata: Trainable vs cold-start segmentation

Key Features:
- Efficient scipy.sparse.csr_matrix construction
- Only trainable users included in matrices
- Top-K popular items tracking for implicit negatives
- Comprehensive validation and statistics

Author: Data Team
Created: 2025-01-15
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict, Tuple, Set, Optional, List
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatrixBuilder:
    """
    Builder for sparse CSR matrices and auxiliary data structures.
    
    This class constructs:
    1. Sparse matrices for ALS (confidence scores) and BPR (binary)
    2. User positive/negative sets for negative sampling
    3. Item popularity with top-K tracking
    4. User segmentation metadata
    
    Usage:
        builder = MatrixBuilder(positive_threshold=4.0, top_k_popular=50)
        
        # Build confidence matrix for ALS
        X_conf = builder.build_confidence_matrix(
            train_df, num_users, num_items, value_col='confidence_score'
        )
        
        # Build auxiliary structures
        user_pos_sets = builder.build_user_positive_sets(train_df)
        user_hard_neg_sets = builder.build_user_hard_negative_sets(train_df)
        item_popularity = builder.build_item_popularity(train_df, num_items)
    """
    
    def __init__(
        self, 
        positive_threshold: float = 4.0,
        hard_negative_threshold: float = 3.0,
        top_k_popular: int = 50
    ):
        """
        Initialize matrix builder.
        
        Args:
            positive_threshold: Rating threshold for positive interactions
            hard_negative_threshold: Rating threshold for hard negatives
            top_k_popular: Number of top popular items to track
        """
        self.positive_threshold = positive_threshold
        self.hard_negative_threshold = hard_negative_threshold
        self.top_k_popular = top_k_popular
        self.build_metadata = {}
    
    def build_confidence_matrix(
        self,
        interactions_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        value_col: str = 'confidence_score'
    ) -> csr_matrix:
        """
        Build sparse CSR matrix with confidence scores for ALS.
        
        Args:
            interactions_df: DataFrame with interactions
            num_users: Total number of users (matrix rows)
            num_items: Total number of items (matrix columns)
            user_col: User index column
            item_col: Item index column
            value_col: Value column (confidence_score or rating)
            
        Returns:
            scipy.sparse.csr_matrix of shape (num_users, num_items)
        """
        logger.info(f"Building confidence matrix for ALS...")
        logger.info(f"  Matrix shape: ({num_users}, {num_items})")
        logger.info(f"  Value column: {value_col}")
        logger.info(f"  Input interactions: {len(interactions_df)}")
        
        # Extract arrays
        rows = interactions_df[user_col].values
        cols = interactions_df[item_col].values
        data = interactions_df[value_col].values
        
        # Validate indices
        if rows.max() >= num_users:
            raise ValueError(f"User index {rows.max()} exceeds num_users {num_users}")
        if cols.max() >= num_items:
            raise ValueError(f"Item index {cols.max()} exceeds num_items {num_items}")
        
        # Build CSR matrix
        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(num_users, num_items),
            dtype=np.float32
        )
        
        # Log statistics
        nnz = matrix.nnz
        sparsity = 1.0 - (nnz / (num_users * num_items))
        
        logger.info(f"  Non-zero entries: {nnz:,}")
        logger.info(f"  Sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)")
        logger.info(f"  Memory: {matrix.data.nbytes / 1024 / 1024:.2f} MB")
        
        # Store metadata
        self.build_metadata['confidence_matrix'] = {
            'shape': (num_users, num_items),
            'nnz': int(nnz),
            'sparsity': float(sparsity),
            'value_range': (float(data.min()), float(data.max())),
            'value_mean': float(data.mean()),
            'value_std': float(data.std())
        }
        
        return matrix
    
    def build_binary_matrix(
        self,
        interactions_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        positive_only: bool = True
    ) -> csr_matrix:
        """
        Build sparse binary CSR matrix for BPR (optional).
        
        Args:
            interactions_df: DataFrame with interactions
            num_users: Total number of users
            num_items: Total number of items
            user_col: User index column
            item_col: Item index column
            positive_only: If True, only include positive interactions
            
        Returns:
            scipy.sparse.csr_matrix with binary values (0 or 1)
        """
        logger.info(f"Building binary matrix for BPR...")
        
        # Filter to positive interactions if requested
        if positive_only:
            if 'is_positive' not in interactions_df.columns:
                raise ValueError("DataFrame must have 'is_positive' column for positive_only=True")
            df_filtered = interactions_df[interactions_df['is_positive'] == 1].copy()
            logger.info(f"  Filtered to {len(df_filtered)} positive interactions")
        else:
            df_filtered = interactions_df.copy()
        
        # Extract arrays (all values = 1)
        rows = df_filtered[user_col].values
        cols = df_filtered[item_col].values
        data = np.ones(len(df_filtered), dtype=np.float32)
        
        # Build CSR matrix
        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(num_users, num_items),
            dtype=np.float32
        )
        
        nnz = matrix.nnz
        sparsity = 1.0 - (nnz / (num_users * num_items))
        
        logger.info(f"  Non-zero entries: {nnz:,}")
        logger.info(f"  Sparsity: {sparsity:.6f}")
        
        # Store metadata
        self.build_metadata['binary_matrix'] = {
            'shape': (num_users, num_items),
            'nnz': int(nnz),
            'sparsity': float(sparsity),
            'positive_only': positive_only
        }
        
        return matrix
    
    def build_user_positive_sets(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx'
    ) -> Dict[int, Set[int]]:
        """
        Build user positive item sets for fast lookup.
        
        Args:
            interactions_df: DataFrame with positive interactions
            user_col: User index column
            item_col: Item index column
            
        Returns:
            Dict mapping u_idx -> Set[i_idx] of positive items
        """
        logger.info("Building user positive sets...")
        
        # Filter to positive interactions
        if 'is_positive' in interactions_df.columns:
            df_positive = interactions_df[interactions_df['is_positive'] == 1].copy()
        else:
            logger.warning("No 'is_positive' column, using all interactions")
            df_positive = interactions_df.copy()
        
        # Group by user and collect item sets
        user_pos_sets = {}
        for u_idx, group in df_positive.groupby(user_col):
            user_pos_sets[u_idx] = set(group[item_col].values)
        
        # Statistics
        total_users = len(user_pos_sets)
        total_positives = sum(len(items) for items in user_pos_sets.values())
        avg_positives = total_positives / total_users if total_users > 0 else 0
        
        logger.info(f"  Users with positives: {total_users:,}")
        logger.info(f"  Total positive interactions: {total_positives:,}")
        logger.info(f"  Avg positives per user: {avg_positives:.2f}")
        
        # Store metadata
        self.build_metadata['user_positive_sets'] = {
            'num_users': total_users,
            'total_positives': total_positives,
            'avg_positives_per_user': float(avg_positives)
        }
        
        return user_pos_sets
    
    def build_user_hard_negative_sets(
        self,
        interactions_df: pd.DataFrame,
        top_k_popular_items: Optional[List[int]] = None,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        rating_col: str = 'rating'
    ) -> Dict[int, Dict[str, Set[int]]]:
        """
        Build user hard negative sets (explicit + implicit).
        
        Args:
            interactions_df: DataFrame with interactions
            top_k_popular_items: List of popular item indices for implicit negatives
            user_col: User index column
            item_col: Item index column
            rating_col: Rating column
            
        Returns:
            Dict mapping u_idx -> {"explicit": set(...), "implicit": set(...)}
        """
        logger.info("Building user hard negative sets...")
        
        user_hard_neg_sets = {}
        
        # 1. Explicit hard negatives (rating <= threshold)
        if 'is_hard_negative' in interactions_df.columns:
            df_explicit = interactions_df[interactions_df['is_hard_negative'] == 1].copy()
        else:
            df_explicit = interactions_df[
                interactions_df[rating_col] <= self.hard_negative_threshold
            ].copy()
        
        for u_idx, group in df_explicit.groupby(user_col):
            if u_idx not in user_hard_neg_sets:
                user_hard_neg_sets[u_idx] = {"explicit": set(), "implicit": set()}
            user_hard_neg_sets[u_idx]["explicit"] = set(group[item_col].values)
        
        num_users_explicit = sum(1 for sets in user_hard_neg_sets.values() if sets["explicit"])
        total_explicit = sum(len(sets["explicit"]) for sets in user_hard_neg_sets.values())
        
        logger.info(f"  Explicit hard negatives:")
        logger.info(f"    Users: {num_users_explicit:,}")
        logger.info(f"    Total: {total_explicit:,}")
        
        # 2. Implicit hard negatives (popular items NOT interacted with)
        if top_k_popular_items is not None and len(top_k_popular_items) > 0:
            popular_set = set(top_k_popular_items)
            
            # Get all users
            all_users = interactions_df[user_col].unique()
            
            for u_idx in all_users:
                # Get user's interacted items
                user_items = set(interactions_df[interactions_df[user_col] == u_idx][item_col].values)
                
                # Implicit negatives = popular items NOT in user's history
                implicit_negs = popular_set - user_items
                
                if implicit_negs:
                    if u_idx not in user_hard_neg_sets:
                        user_hard_neg_sets[u_idx] = {"explicit": set(), "implicit": set()}
                    user_hard_neg_sets[u_idx]["implicit"] = implicit_negs
            
            num_users_implicit = sum(1 for sets in user_hard_neg_sets.values() if sets["implicit"])
            total_implicit = sum(len(sets["implicit"]) for sets in user_hard_neg_sets.values())
            
            logger.info(f"  Implicit hard negatives:")
            logger.info(f"    Users: {num_users_implicit:,}")
            logger.info(f"    Total: {total_implicit:,}")
        
        # Store metadata
        self.build_metadata['user_hard_negative_sets'] = {
            'num_users': len(user_hard_neg_sets),
            'num_users_with_explicit': num_users_explicit,
            'total_explicit': total_explicit,
            'num_users_with_implicit': num_users_implicit if top_k_popular_items else 0,
            'total_implicit': total_implicit if top_k_popular_items else 0
        }
        
        return user_hard_neg_sets
    
    def build_item_popularity(
        self,
        interactions_df: pd.DataFrame,
        num_items: int,
        item_col: str = 'i_idx',
        log_transform: bool = True
    ) -> np.ndarray:
        """
        Build item popularity scores with log-transform.
        
        Args:
            interactions_df: DataFrame with interactions
            num_items: Total number of items
            item_col: Item index column
            log_transform: If True, apply log(1 + count)
            
        Returns:
            np.ndarray of shape (num_items,) with popularity scores
        """
        logger.info("Building item popularity scores...")
        
        # Count interactions per item
        item_counts = interactions_df[item_col].value_counts()
        
        # Initialize popularity array
        popularity = np.zeros(num_items, dtype=np.float32)
        
        # Fill in counts
        for i_idx, count in item_counts.items():
            popularity[i_idx] = count
        
        # Apply log-transform if requested
        if log_transform:
            popularity = np.log1p(popularity)
            logger.info("  Applied log(1 + count) transform")
        
        # Statistics
        num_items_with_interactions = (popularity > 0).sum()
        
        logger.info(f"  Items with interactions: {num_items_with_interactions:,} / {num_items:,}")
        logger.info(f"  Popularity range: [{popularity.min():.2f}, {popularity.max():.2f}]")
        logger.info(f"  Mean popularity: {popularity.mean():.2f}")
        logger.info(f"  Std popularity: {popularity.std():.2f}")
        
        # Store metadata
        self.build_metadata['item_popularity'] = {
            'num_items': num_items,
            'num_items_with_interactions': int(num_items_with_interactions),
            'log_transformed': log_transform,
            'min': float(popularity.min()),
            'max': float(popularity.max()),
            'mean': float(popularity.mean()),
            'std': float(popularity.std())
        }
        
        return popularity
    
    def get_top_k_popular_items(
        self,
        interactions_df: pd.DataFrame,
        k: Optional[int] = None,
        item_col: str = 'i_idx'
    ) -> List[int]:
        """
        Get top-K most popular items by interaction count.
        
        Args:
            interactions_df: DataFrame with interactions
            k: Number of top items (default: self.top_k_popular)
            item_col: Item index column
            
        Returns:
            List of i_idx for top-K popular items
        """
        if k is None:
            k = self.top_k_popular
        
        logger.info(f"Finding top-{k} popular items...")
        
        # Count interactions per item
        item_counts = interactions_df[item_col].value_counts()
        
        # Get top-K
        top_k = item_counts.head(k).index.tolist()
        
        logger.info(f"  Top-{k} items: {top_k[:10]}..." if len(top_k) > 10 else f"  Top-{k} items: {top_k}")
        logger.info(f"  Top item count: {item_counts.iloc[0]:,}")
        logger.info(f"  #{k} item count: {item_counts.iloc[k-1]:,}" if k <= len(item_counts) else "")
        
        # Store metadata
        self.build_metadata['top_k_popular_items'] = {
            'k': k,
            'top_k_items': [int(i) for i in top_k],
            'top_item_count': int(item_counts.iloc[0]) if len(item_counts) > 0 else 0,
            'kth_item_count': int(item_counts.iloc[k-1]) if k <= len(item_counts) else 0
        }
        
        return top_k
    
    def build_user_metadata(
        self,
        interactions_df: pd.DataFrame,
        min_interactions_trainable: int = 2,
        user_col: str = 'u_idx'
    ) -> Dict:
        """
        Build user segmentation metadata (trainable vs cold-start).
        
        Args:
            interactions_df: DataFrame with interactions
            min_interactions_trainable: Minimum interactions for trainable users
            user_col: User index column
            
        Returns:
            Dict with user statistics and segmentation
        """
        logger.info("Building user segmentation metadata...")
        
        # Count interactions per user
        user_counts = interactions_df[user_col].value_counts()
        
        # Segment users
        trainable_users = set(user_counts[user_counts >= min_interactions_trainable].index)
        cold_start_users = set(user_counts[user_counts < min_interactions_trainable].index)
        
        # Statistics
        total_users = len(user_counts)
        num_trainable = len(trainable_users)
        num_cold_start = len(cold_start_users)
        pct_trainable = 100 * num_trainable / total_users if total_users > 0 else 0
        
        logger.info(f"  Total users: {total_users:,}")
        logger.info(f"  Trainable users (>={min_interactions_trainable} interactions): {num_trainable:,} ({pct_trainable:.2f}%)")
        logger.info(f"  Cold-start users (<{min_interactions_trainable} interactions): {num_cold_start:,}")
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "min_interactions_trainable": min_interactions_trainable,
            "total_users": total_users,
            "trainable_users": trainable_users,
            "cold_start_users": cold_start_users,
            "user_interaction_counts": user_counts.to_dict(),
            "stats": {
                "num_trainable": num_trainable,
                "num_cold_start": num_cold_start,
                "pct_trainable": float(pct_trainable),
                "avg_interactions_trainable": float(user_counts[user_counts >= min_interactions_trainable].mean()) if num_trainable > 0 else 0,
                "avg_interactions_cold_start": float(user_counts[user_counts < min_interactions_trainable].mean()) if num_cold_start > 0 else 0
            }
        }
        
        # Store metadata
        self.build_metadata['user_metadata'] = {
            'total_users': total_users,
            'trainable_users': num_trainable,
            'cold_start_users': num_cold_start,
            'pct_trainable': float(pct_trainable)
        }
        
        return metadata
    
    def get_build_metadata(self) -> Dict:
        """
        Get metadata about all built structures.
        
        Returns:
            Dict with comprehensive build statistics
        """
        return self.build_metadata
    
    def save_build_metadata(self, output_path: str):
        """
        Save build metadata to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        # Convert sets to lists for JSON serialization
        metadata_serializable = {}
        for key, value in self.build_metadata.items():
            if isinstance(value, dict):
                metadata_serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, set):
                        metadata_serializable[key][k] = list(v)
                    else:
                        metadata_serializable[key][k] = v
            else:
                metadata_serializable[key] = value
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Build metadata saved to {output_path}")
