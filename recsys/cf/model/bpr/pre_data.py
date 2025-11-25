"""
BPR Data Preparation Module (Task 02 - Step 1)

This module handles loading and preparing data for BPR training:
- Load positive pairs from processed data
- Load user positive sets for negative sampling
- Load hard negative sets (explicit + implicit)
- Validate data integrity

Key Data Structures:
- positive_pairs: np.ndarray of shape (N, 2) with [u_idx, i_idx]
- user_pos_sets: Dict[int, Set[int]] mapping u_idx -> positive item indices
- hard_neg_sets: Dict[int, Set[int]] mapping u_idx -> hard negative item indices
"""

import logging
import json
import pickle
from typing import Dict, Set, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BPRDataLoader:
    """
    Load and prepare data for BPR training.
    
    This class handles loading pre-processed data from Task 01:
    - Positive pairs from interactions
    - User positive sets for negative sampling
    - Hard negative sets (explicit + implicit negatives)
    - Mappings (user_to_idx, item_to_idx)
    
    Attributes:
        base_path: Path to processed data directory
        positive_pairs: Loaded positive pairs array
        user_pos_sets: Loaded user positive item sets
        hard_neg_sets: Loaded hard negative item sets
        mappings: Loaded ID mappings
    """
    
    def __init__(self, base_path: str = 'data/processed'):
        """
        Initialize BPR data loader.
        
        Args:
            base_path: Path to processed data directory
        
        Example:
            >>> loader = BPRDataLoader(base_path='data/processed')
            >>> data = loader.load_all()
        """
        self.base_path = Path(base_path)
        self.positive_pairs = None
        self.user_pos_sets = None
        self.hard_neg_sets = None
        self.mappings = None
        self.num_users = None
        self.num_items = None
        
        logger.info(f"BPRDataLoader initialized: base_path={self.base_path}")
    
    def load_mappings(self, filename: str = 'user_item_mappings.json') -> Dict[str, Any]:
        """
        Load ID mappings from JSON file.
        
        Args:
            filename: Mappings JSON filename
        
        Returns:
            Dictionary with user_to_idx, idx_to_user, item_to_idx, idx_to_item
        """
        filepath = self.base_path / filename
        logger.info(f"Loading mappings from {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.mappings = {
            'user_to_idx': {k: int(v) for k, v in data['user_to_idx'].items()},
            'idx_to_user': {int(k): v for k, v in data['idx_to_user'].items()},
            'item_to_idx': {k: int(v) for k, v in data['item_to_idx'].items()},
            'idx_to_item': {int(k): v for k, v in data['idx_to_item'].items()}
        }
        
        # Extract metadata
        metadata = data.get('metadata', {})
        self.num_users = metadata.get('num_users', len(self.mappings['user_to_idx']))
        self.num_items = metadata.get('num_items', len(self.mappings['item_to_idx']))
        
        logger.info(f"Loaded mappings: {self.num_users:,} users, {self.num_items:,} items")
        
        return self.mappings
    
    def load_user_pos_sets(self, filename: str = 'user_pos_train.pkl') -> Dict[int, Set[int]]:
        """
        Load user positive item sets from pickle file.
        
        Args:
            filename: Pickle filename for user positive sets
        
        Returns:
            Dictionary mapping u_idx -> Set of positive i_idx
        """
        filepath = self.base_path / filename
        logger.info(f"Loading user positive sets from {filepath}")
        
        with open(filepath, 'rb') as f:
            self.user_pos_sets = pickle.load(f)
        
        num_users = len(self.user_pos_sets)
        total_positives = sum(len(items) for items in self.user_pos_sets.values())
        avg_positives = total_positives / num_users if num_users > 0 else 0
        
        logger.info(f"Loaded user positive sets: {num_users:,} users, {total_positives:,} pairs")
        logger.info(f"Average positives per user: {avg_positives:.2f}")
        
        return self.user_pos_sets
    
    def load_hard_neg_sets(self, filename: str = 'user_hard_neg_train.pkl') -> Dict[int, Set[int]]:
        """
        Load user hard negative sets from pickle file.
        
        Args:
            filename: Pickle filename for hard negative sets
        
        Returns:
            Dictionary mapping u_idx -> Set of hard negative i_idx
        """
        filepath = self.base_path / filename
        
        if not filepath.exists():
            logger.warning(f"Hard negative file not found: {filepath}")
            logger.warning("BPR will use random negatives only")
            self.hard_neg_sets = {}
            return self.hard_neg_sets
        
        logger.info(f"Loading hard negative sets from {filepath}")
        
        with open(filepath, 'rb') as f:
            raw_data = pickle.load(f)
        
        # Handle nested structure from MatrixBuilder
        # Expected format: {u_idx: {"explicit": set(...), "implicit": set(...)}}
        self.hard_neg_sets = {}
        
        for user_idx, neg_data in raw_data.items():
            if isinstance(neg_data, dict):
                # Merge explicit and implicit negatives
                combined = set()
                if 'explicit' in neg_data:
                    combined.update(neg_data['explicit'])
                if 'implicit' in neg_data:
                    combined.update(neg_data['implicit'])
                self.hard_neg_sets[user_idx] = combined
            else:
                # Already a flat set
                self.hard_neg_sets[user_idx] = neg_data
        
        num_users = len(self.hard_neg_sets)
        total_negatives = sum(len(items) for items in self.hard_neg_sets.values())
        avg_negatives = total_negatives / num_users if num_users > 0 else 0
        
        logger.info(f"Loaded hard negative sets: {num_users:,} users, {total_negatives:,} items")
        logger.info(f"Average hard negatives per user: {avg_negatives:.2f}")
        
        return self.hard_neg_sets
    
    def build_positive_pairs_from_sets(self) -> np.ndarray:
        """
        Build positive pairs array from user positive sets.
        
        Returns:
            np.ndarray of shape (N, 2) with columns [u_idx, i_idx]
        """
        if self.user_pos_sets is None:
            raise ValueError("User positive sets not loaded. Call load_user_pos_sets() first.")
        
        logger.info("Building positive pairs from user positive sets...")
        
        all_pairs = []
        for user_idx, item_set in self.user_pos_sets.items():
            for item_idx in item_set:
                all_pairs.append([user_idx, item_idx])
        
        self.positive_pairs = np.array(all_pairs, dtype=np.int64)
        
        logger.info(f"Built {len(self.positive_pairs):,} positive pairs")
        
        return self.positive_pairs
    
    def load_positive_pairs_from_interactions(
        self, 
        filename: str = 'interactions.parquet',
        positive_threshold: float = 4.0
    ) -> np.ndarray:
        """
        Load positive pairs from interactions parquet file.
        
        Args:
            filename: Parquet filename for interactions
            positive_threshold: Rating threshold for positive interactions
        
        Returns:
            np.ndarray of shape (N, 2) with columns [u_idx, i_idx]
        """
        filepath = self.base_path / filename
        logger.info(f"Loading positive pairs from {filepath}")
        
        df = pd.read_parquet(filepath)
        
        # Check for required columns
        required_cols = ['u_idx', 'i_idx']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in interactions: {missing}")
        
        # Filter positive interactions
        if 'is_positive' in df.columns:
            positive_df = df[df['is_positive'] == 1]
        elif 'rating' in df.columns:
            positive_df = df[df['rating'] >= positive_threshold]
        else:
            raise ValueError("Need 'is_positive' or 'rating' column to identify positives")
        
        # Extract pairs
        self.positive_pairs = positive_df[['u_idx', 'i_idx']].values.astype(np.int64)
        
        logger.info(f"Loaded {len(self.positive_pairs):,} positive pairs")
        
        return self.positive_pairs
    
    def load_all(self) -> Dict[str, Any]:
        """
        Load all BPR training data.
        
        Returns:
            Dictionary containing:
                - 'positive_pairs': np.ndarray of shape (N, 2)
                - 'user_pos_sets': Dict[int, Set[int]]
                - 'hard_neg_sets': Dict[int, Set[int]]
                - 'mappings': ID mappings
                - 'num_users': Total number of users
                - 'num_items': Total number of items
        """
        logger.info("="*60)
        logger.info("Loading all BPR training data")
        logger.info("="*60)
        
        # Load mappings
        self.load_mappings()
        
        # Load user positive sets
        self.load_user_pos_sets()
        
        # Load hard negative sets
        self.load_hard_neg_sets()
        
        # Build positive pairs from sets
        self.build_positive_pairs_from_sets()
        
        data = {
            'positive_pairs': self.positive_pairs,
            'user_pos_sets': self.user_pos_sets,
            'hard_neg_sets': self.hard_neg_sets,
            'mappings': self.mappings,
            'num_users': self.num_users,
            'num_items': self.num_items
        }
        
        logger.info("="*60)
        logger.info("BPR data loading complete")
        logger.info(f"  Positive pairs: {len(self.positive_pairs):,}")
        logger.info(f"  Users with positives: {len(self.user_pos_sets):,}")
        logger.info(f"  Users with hard negatives: {len(self.hard_neg_sets):,}")
        logger.info(f"  Total users: {self.num_users:,}")
        logger.info(f"  Total items: {self.num_items:,}")
        logger.info("="*60)
        
        return data
    
    def validate_data(self) -> bool:
        """
        Validate loaded data integrity.
        
        Returns:
            True if validation passes
        
        Raises:
            ValueError: If validation fails
        """
        logger.info("Validating BPR data...")
        
        # Check positive pairs shape
        if self.positive_pairs is None or len(self.positive_pairs) == 0:
            raise ValueError("No positive pairs loaded")
        
        if self.positive_pairs.ndim != 2 or self.positive_pairs.shape[1] != 2:
            raise ValueError(f"Invalid positive pairs shape: {self.positive_pairs.shape}")
        
        # Check user indices range
        max_user = self.positive_pairs[:, 0].max()
        if max_user >= self.num_users:
            raise ValueError(f"User index {max_user} exceeds num_users {self.num_users}")
        
        # Check item indices range
        max_item = self.positive_pairs[:, 1].max()
        if max_item >= self.num_items:
            raise ValueError(f"Item index {max_item} exceeds num_items {self.num_items}")
        
        # Check user_pos_sets consistency
        total_pairs_from_sets = sum(len(items) for items in self.user_pos_sets.values())
        if total_pairs_from_sets != len(self.positive_pairs):
            logger.warning(
                f"Mismatch: {total_pairs_from_sets} pairs from sets vs "
                f"{len(self.positive_pairs)} in positive_pairs array"
            )
        
        logger.info("✓ BPR data validation passed")
        return True


def load_bpr_data(base_path: str = 'data/processed') -> Dict[str, Any]:
    """
    Convenience function to load all BPR data.
    
    Args:
        base_path: Path to processed data directory
    
    Returns:
        Dictionary with all BPR training data
    
    Example:
        >>> data = load_bpr_data()
        >>> pairs = data['positive_pairs']
        >>> print(f"Loaded {len(pairs):,} positive pairs")
    """
    loader = BPRDataLoader(base_path=base_path)
    return loader.load_all()


def prepare_bpr_data(
    interactions_df: pd.DataFrame,
    products_df: Optional[pd.DataFrame] = None,
    positive_threshold: float = 4.0,
    hard_negative_threshold: float = 3.0,
    top_k_popular: int = 50
) -> Dict[str, Any]:
    """
    Prepare BPR data from raw DataFrames.
    
    This function uses the BPRDataPreparer from the data layer.
    
    Args:
        interactions_df: DataFrame with user-item interactions
        products_df: Optional product metadata for implicit negatives
        positive_threshold: Rating threshold for positives
        hard_negative_threshold: Rating threshold for hard negatives
        top_k_popular: Number of popular items for implicit negatives
    
    Returns:
        Dictionary with BPR training data
    
    Example:
        >>> data = prepare_bpr_data(train_df, products_df)
        >>> print(f"Positive pairs: {len(data['positive_pairs']):,}")
    """
    from recsys.cf.data.processing.bpr_data import BPRDataPreparer
    
    preparer = BPRDataPreparer(
        positive_threshold=positive_threshold,
        hard_negative_threshold=hard_negative_threshold,
        top_k_popular=top_k_popular
    )
    
    return preparer.get_bpr_training_data(
        interactions_df,
        products_df
    )
