"""
ID Mapping Module for Collaborative Filtering

This module handles Step 3: ID Mapping (Contiguous Indexing).
Maps sparse original IDs (user_id, product_id) to contiguous indices (u_idx, i_idx)
for efficient matrix operations in ALS and BPR.

Key Features:
- Bidirectional mappings (original ↔ contiguous)
- Validation and reversibility checks
- JSON serialization with metadata
- Data hash tracking for versioning
"""

import logging
from typing import Dict, Tuple, Set
import hashlib
from datetime import datetime
import json

import pandas as pd
import numpy as np


logger = logging.getLogger("data_layer")


class IDMapper:
    """
    Class for mapping sparse IDs to contiguous indices.
    
    This class handles:
    - User ID mapping (user_id → u_idx)
    - Item ID mapping (product_id → i_idx)
    - Bidirectional mappings
    - Validation and reversibility
    """
    
    def __init__(self):
        """Initialize IDMapper."""
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
        self.num_users = 0
        self.num_items = 0
        self.metadata = {}
    
    def create_mappings(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'product_id'
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Create bidirectional mappings for users and items.
        
        Args:
            interactions_df: DataFrame with interactions
            user_col: User ID column
            item_col: Item ID column
        
        Returns:
            Tuple of (user_to_idx, idx_to_user, item_to_idx, idx_to_item)
        
        Example:
            >>> mapper = IDMapper()
            >>> u2i, i2u, i2i, i2i = mapper.create_mappings(df)
            >>> print(f"Mapped {len(u2i)} users, {len(i2i)} items")
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 3: ID MAPPING (CONTIGUOUS INDEXING)")
        logger.info("="*80)
        logger.info("Mapping sparse IDs to contiguous indices for efficient matrix operations")
        
        # Extract unique IDs
        logger.info("\n" + "-"*80)
        logger.info("EXTRACTING UNIQUE IDs")
        logger.info("-"*80)
        
        unique_users = sorted(interactions_df[user_col].unique())
        unique_items = sorted(interactions_df[item_col].unique())
        
        logger.info(f"Unique users: {len(unique_users):,}")
        logger.info(f"  Original user_id range: [{min(unique_users)}, {max(unique_users)}]")
        logger.info(f"  Sparse user_id range span: {max(unique_users) - min(unique_users):,}")
        
        logger.info(f"\nUnique items: {len(unique_items):,}")
        logger.info(f"  Original product_id range: [{min(unique_items)}, {max(unique_items)}]")
        logger.info(f"  Sparse product_id range span: {max(unique_items) - min(unique_items):,}")
        
        # Create user mappings
        logger.info("\n" + "-"*80)
        logger.info("USER MAPPING")
        logger.info("-"*80)
        
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.idx_to_user = {idx: user_id for user_id, idx in self.user_to_idx.items()}
        self.num_users = len(unique_users)
        
        logger.info(f"Created user mappings:")
        logger.info(f"  user_id (sparse) → u_idx (contiguous 0-{self.num_users-1})")
        logger.info(f"  Example mappings:")
        
        # Show first 5 mappings
        for i, (user_id, u_idx) in enumerate(list(self.user_to_idx.items())[:5]):
            logger.info(f"    user_id {user_id} → u_idx {u_idx}")
        
        # Create item mappings
        logger.info("\n" + "-"*80)
        logger.info("ITEM MAPPING")
        logger.info("-"*80)
        
        self.item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}
        self.num_items = len(unique_items)
        
        logger.info(f"Created item mappings:")
        logger.info(f"  product_id (sparse) → i_idx (contiguous 0-{self.num_items-1})")
        logger.info(f"  Example mappings:")
        
        # Show first 5 mappings
        for i, (item_id, i_idx) in enumerate(list(self.item_to_idx.items())[:5]):
            logger.info(f"    product_id {item_id} → i_idx {i_idx}")
        
        # Validate mappings
        self._validate_mappings()
        
        logger.info("\n" + "="*80)
        logger.info("✓ ID mappings created successfully")
        
        return self.user_to_idx, self.idx_to_user, self.item_to_idx, self.idx_to_item
    
    def apply_mappings(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'product_id'
    ) -> pd.DataFrame:
        """
        Apply mappings to interactions DataFrame.
        
        Args:
            interactions_df: DataFrame with user_id and product_id columns
            user_col: User ID column
            item_col: Item ID column
        
        Returns:
            DataFrame with added u_idx and i_idx columns
        
        Example:
            >>> df_mapped = mapper.apply_mappings(df)
            >>> print(df_mapped[['user_id', 'u_idx', 'product_id', 'i_idx']].head())
        """
        logger.info("\n" + "-"*80)
        logger.info("APPLYING MAPPINGS TO INTERACTIONS")
        logger.info("-"*80)
        
        if not self.user_to_idx or not self.item_to_idx:
            raise ValueError("Mappings not created. Call create_mappings() first.")
        
        # Map user IDs
        interactions_df['u_idx'] = interactions_df[user_col].map(self.user_to_idx)
        
        # Map item IDs
        interactions_df['i_idx'] = interactions_df[item_col].map(self.item_to_idx)
        
        # Validate no missing mappings
        missing_users = interactions_df['u_idx'].isna().sum()
        missing_items = interactions_df['i_idx'].isna().sum()
        
        if missing_users > 0:
            raise ValueError(f"Found {missing_users} unmapped user_ids")
        if missing_items > 0:
            raise ValueError(f"Found {missing_items} unmapped product_ids")
        
        logger.info(f"Applied mappings to {len(interactions_df):,} interactions")
        logger.info(f"  Added u_idx column: user_id → contiguous [0, {self.num_users-1}]")
        logger.info(f"  Added i_idx column: product_id → contiguous [0, {self.num_items-1}]")
        logger.info(f"✓ No missing mappings detected")
        
        return interactions_df
    
    def _validate_mappings(self):
        """
        Validate bidirectional mappings.
        
        Checks:
        - Contiguous indices (0 to N-1)
        - Bidirectional consistency
        - No duplicate mappings
        """
        logger.info("\n" + "-"*80)
        logger.info("VALIDATING MAPPINGS")
        logger.info("-"*80)
        
        # Check user mappings
        user_indices = set(self.user_to_idx.values())
        expected_user_indices = set(range(self.num_users))
        
        if user_indices != expected_user_indices:
            raise ValueError(f"User indices not contiguous: {user_indices} != {expected_user_indices}")
        
        # Check item mappings
        item_indices = set(self.item_to_idx.values())
        expected_item_indices = set(range(self.num_items))
        
        if item_indices != expected_item_indices:
            raise ValueError(f"Item indices not contiguous: {item_indices} != {expected_item_indices}")
        
        # Check bidirectional consistency
        for user_id, u_idx in self.user_to_idx.items():
            if self.idx_to_user[u_idx] != user_id:
                raise ValueError(f"User mapping inconsistency: {user_id} → {u_idx} → {self.idx_to_user[u_idx]}")
        
        for item_id, i_idx in self.item_to_idx.items():
            if self.idx_to_item[i_idx] != item_id:
                raise ValueError(f"Item mapping inconsistency: {item_id} → {i_idx} → {self.idx_to_item[i_idx]}")
        
        logger.info("✓ User mappings validated:")
        logger.info(f"  - Contiguous indices: [0, {self.num_users-1}]")
        logger.info(f"  - Bidirectional consistency: OK")
        
        logger.info("✓ Item mappings validated:")
        logger.info(f"  - Contiguous indices: [0, {self.num_items-1}]")
        logger.info(f"  - Bidirectional consistency: OK")
    
    def compute_data_hash(self, interactions_df: pd.DataFrame) -> str:
        """
        Compute MD5 hash of interactions data for versioning.
        
        Args:
            interactions_df: DataFrame with interactions
        
        Returns:
            MD5 hash string
        """
        # Sort DataFrame for consistent hashing
        df_sorted = interactions_df.sort_values(['user_id', 'product_id', 'rating']).reset_index(drop=True)
        
        # Compute hash of key columns
        hash_input = df_sorted[['user_id', 'product_id', 'rating']].to_string()
        data_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        
        return data_hash
    
    def save_mappings(
        self,
        output_path: str,
        interactions_df: pd.DataFrame,
        include_metadata: bool = True
    ):
        """
        Save mappings to JSON file with metadata.
        
        Args:
            output_path: Path to save JSON file
            interactions_df: Original interactions DataFrame for hash computation
            include_metadata: Include timestamp, hash, and stats
        
        Example:
            >>> mapper.save_mappings('data/processed/user_item_mappings.json', df)
        """
        logger.info("\n" + "-"*80)
        logger.info("SAVING MAPPINGS TO JSON")
        logger.info("-"*80)
        
        # Convert all mappings to JSON-serializable format (handle numpy types)
        mappings_data = {
            "user_to_idx": {str(k): int(v) for k, v in self.user_to_idx.items()},
            "idx_to_user": {str(k): int(v) for k, v in self.idx_to_user.items()},
            "item_to_idx": {str(k): int(v) for k, v in self.item_to_idx.items()},
            "idx_to_item": {str(k): int(v) for k, v in self.idx_to_item.items()}
        }
        
        if include_metadata:
            data_hash = self.compute_data_hash(interactions_df)
            
            mappings_data["metadata"] = {
                "created_at": datetime.now().isoformat(),
                "num_users": self.num_users,
                "num_items": self.num_items,
                "num_interactions": len(interactions_df),
                "data_hash": data_hash,
                "user_id_range": {
                    "min": int(min(self.user_to_idx.keys())),
                    "max": int(max(self.user_to_idx.keys()))
                },
                "product_id_range": {
                    "min": int(min(self.item_to_idx.keys())),
                    "max": int(max(self.item_to_idx.keys()))
                },
                "u_idx_range": {
                    "min": 0,
                    "max": self.num_users - 1
                },
                "i_idx_range": {
                    "min": 0,
                    "max": self.num_items - 1
                }
            }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mappings_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved mappings to: {output_path}")
        logger.info(f"  User mappings: {self.num_users:,} entries")
        logger.info(f"  Item mappings: {self.num_items:,} entries")
        
        if include_metadata:
            logger.info(f"  Metadata included:")
            logger.info(f"    - Timestamp: {mappings_data['metadata']['created_at']}")
            logger.info(f"    - Data hash: {data_hash[:8]}...")
            logger.info(f"    - User ID range: [{mappings_data['metadata']['user_id_range']['min']}, {mappings_data['metadata']['user_id_range']['max']}]")
            logger.info(f"    - Product ID range: [{mappings_data['metadata']['product_id_range']['min']}, {mappings_data['metadata']['product_id_range']['max']}]")
        
        logger.info("✓ Mappings saved successfully")
    
    def load_mappings(self, input_path: str):
        """
        Load mappings from JSON file.
        
        Args:
            input_path: Path to JSON file
        
        Example:
            >>> mapper = IDMapper()
            >>> mapper.load_mappings('data/processed/user_item_mappings.json')
        """
        logger.info(f"\nLoading mappings from: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            mappings_data = json.load(f)
        
        # Convert string keys back to integers
        self.user_to_idx = {int(k): v for k, v in mappings_data['user_to_idx'].items()}
        self.idx_to_user = {int(k): v for k, v in mappings_data['idx_to_user'].items()}
        self.item_to_idx = {int(k): v for k, v in mappings_data['item_to_idx'].items()}
        self.idx_to_item = {int(k): v for k, v in mappings_data['idx_to_item'].items()}
        
        self.num_users = len(self.user_to_idx)
        self.num_items = len(self.item_to_idx)
        
        if 'metadata' in mappings_data:
            self.metadata = mappings_data['metadata']
            logger.info(f"  Loaded metadata:")
            logger.info(f"    - Created at: {self.metadata['created_at']}")
            logger.info(f"    - Num users: {self.metadata['num_users']:,}")
            logger.info(f"    - Num items: {self.metadata['num_items']:,}")
            logger.info(f"    - Data hash: {self.metadata['data_hash'][:8]}...")
        
        logger.info(f"✓ Loaded {self.num_users:,} user mappings, {self.num_items:,} item mappings")
    
    def get_mapping_stats(self) -> Dict:
        """
        Get statistics about current mappings.
        
        Returns:
            Dict with mapping statistics
        """
        if not self.user_to_idx or not self.item_to_idx:
            return {
                "num_users": 0,
                "num_items": 0,
                "mappings_created": False
            }
        
        original_user_ids = list(self.user_to_idx.keys())
        original_item_ids = list(self.item_to_idx.keys())
        
        user_id_min = min(original_user_ids)
        user_id_max = max(original_user_ids)
        item_id_min = min(original_item_ids)
        item_id_max = max(original_item_ids)
        
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "mappings_created": True,
            # Flat structure for easier access
            "user_id_min": user_id_min,
            "user_id_max": user_id_max,
            "item_id_min": item_id_min,
            "item_id_max": item_id_max,
            "user_idx_min": 0,
            "user_idx_max": self.num_users - 1,
            "item_idx_min": 0,
            "item_idx_max": self.num_items - 1,
            # Nested structure for detailed info
            "user_id_range": {
                "min": user_id_min,
                "max": user_id_max,
                "span": user_id_max - user_id_min
            },
            "product_id_range": {
                "min": item_id_min,
                "max": item_id_max,
                "span": item_id_max - item_id_min
            },
            "u_idx_range": {
                "min": 0,
                "max": self.num_users - 1
            },
            "i_idx_range": {
                "min": 0,
                "max": self.num_items - 1
            },
            "compression_ratio": {
                "users": (max(original_user_ids) - min(original_user_ids)) / self.num_users if self.num_users > 0 else 0,
                "items": (max(original_item_ids) - min(original_item_ids)) / self.num_items if self.num_items > 0 else 0
            }
        }
    
    def reverse_user_mapping(self, u_indices) -> list:
        """
        Convert contiguous indices back to original user IDs.
        
        Args:
            u_indices: Array-like of u_idx values
        
        Returns:
            List of original user_ids
        """
        if isinstance(u_indices, (int, np.integer)):
            return self.idx_to_user[u_indices]
        return [self.idx_to_user[u_idx] for u_idx in u_indices]
    
    def reverse_item_mapping(self, i_indices) -> list:
        """
        Convert contiguous indices back to original item IDs.
        
        Args:
            i_indices: Array-like of i_idx values
        
        Returns:
            List of original product_ids
        """
        if isinstance(i_indices, (int, np.integer)):
            return self.idx_to_item[i_indices]
        return [self.idx_to_item[i_idx] for i_idx in i_indices]
