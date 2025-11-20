"""
Data Saver Module - Step 6: Save Processed Data

This module handles saving all processed data artifacts to disk in optimal formats:
- Parquet for DataFrames (fast I/O, compression, type preservation)
- NPZ for sparse matrices (CSR format)
- JSON for mappings and metadata
- Pickle for Python objects (sets, dicts)
- NumPy for dense arrays

Author: Data Team
Created: 2025-01-15
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import logging

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz

logger = logging.getLogger(__name__)


class DataSaver:
    """
    Save processed data artifacts to disk with comprehensive metadata tracking.
    
    This class handles saving:
    1. Interactions DataFrame (Parquet)
    2. ID mappings (JSON)
    3. Sparse matrices (NPZ)
    4. User positive/negative sets (Pickle)
    5. Item popularity (NumPy)
    6. Statistics summary (JSON)
    
    All saves include metadata for reproducibility and versioning.
    """
    
    def __init__(self, output_dir: str = 'data/processed'):
        """
        Initialize DataSaver.
        
        Args:
            output_dir: Directory to save processed data artifacts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataSaver initialized with output_dir: {self.output_dir}")
    
    def save_interactions_parquet(
        self,
        interactions_df: pd.DataFrame,
        filename: str = 'interactions.parquet'
    ) -> str:
        """
        Save interactions DataFrame to Parquet format.
        
        Expected columns:
        - user_id, product_id, u_idx, i_idx
        - rating, comment_quality, confidence_score
        - is_positive, is_hard_negative, timestamp
        - is_trainable_user, split
        
        Args:
            interactions_df: Full interactions DataFrame with all columns
            filename: Output filename (default: interactions.parquet)
        
        Returns:
            str: Path to saved file
        
        Example:
            >>> saver = DataSaver('data/processed')
            >>> path = saver.save_interactions_parquet(df)
            >>> print(f"Saved to {path}")
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving interactions to Parquet: {output_path}")
        logger.info(f"  Rows: {len(interactions_df):,}")
        logger.info(f"  Columns: {list(interactions_df.columns)}")
        
        # Validate required columns
        required_cols = [
            'user_id', 'product_id', 'u_idx', 'i_idx',
            'rating', 'confidence_score', 'is_positive', 'timestamp', 'split'
        ]
        missing_cols = [col for col in required_cols if col not in interactions_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Save to Parquet (fast I/O, compression, type preservation)
        interactions_df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved {len(interactions_df):,} rows, {file_size_mb:.2f} MB")
        
        return str(output_path)
    
    def save_mappings_json(
        self,
        user_to_idx: Dict[int, int],
        idx_to_user: Dict[int, int],
        item_to_idx: Dict[int, int],
        idx_to_item: Dict[int, int],
        metadata: Optional[Dict[str, Any]] = None,
        filename: str = 'user_item_mappings.json'
    ) -> str:
        """
        Save ID mappings to JSON format with metadata.
        
        Args:
            user_to_idx: Dict mapping original user_id to u_idx
            idx_to_user: Dict mapping u_idx to original user_id
            item_to_idx: Dict mapping original product_id to i_idx
            idx_to_item: Dict mapping i_idx to original product_id
            metadata: Optional metadata dict (created_at, num_users, num_items, etc.)
            filename: Output filename (default: user_item_mappings.json)
        
        Returns:
            str: Path to saved file
        
        Example:
            >>> mappings = {
            ...     "metadata": {"created_at": "2025-01-15T10:30:00", ...},
            ...     "user_to_idx": {...},
            ...     "idx_to_user": {...},
            ...     "item_to_idx": {...},
            ...     "idx_to_item": {...}
            ... }
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving ID mappings to JSON: {output_path}")
        logger.info(f"  Users: {len(user_to_idx):,}")
        logger.info(f"  Items: {len(item_to_idx):,}")
        
        # Build metadata if not provided
        if metadata is None:
            metadata = {}
        
        if 'created_at' not in metadata:
            metadata['created_at'] = datetime.now().isoformat()
        if 'num_users' not in metadata:
            metadata['num_users'] = len(user_to_idx)
        if 'num_items' not in metadata:
            metadata['num_items'] = len(item_to_idx)
        
        # Convert int keys to strings for JSON compatibility
        mappings_data = {
            'metadata': metadata,
            'user_to_idx': {str(k): int(v) for k, v in user_to_idx.items()},
            'idx_to_user': {str(k): int(v) for k, v in idx_to_user.items()},
            'item_to_idx': {str(k): int(v) for k, v in item_to_idx.items()},
            'idx_to_item': {str(k): int(v) for k, v in idx_to_item.items()}
        }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mappings_data, f, indent=2, ensure_ascii=False)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved mappings, {file_size_mb:.2f} MB")
        
        return str(output_path)
    
    def save_csr_matrix(
        self,
        matrix: csr_matrix,
        filename: str
    ) -> str:
        """
        Save sparse CSR matrix to NPZ format.
        
        Args:
            matrix: scipy.sparse.csr_matrix to save
            filename: Output filename (e.g., 'X_train_confidence.npz')
        
        Returns:
            str: Path to saved file
        
        Example:
            >>> saver = DataSaver('data/processed')
            >>> path = saver.save_csr_matrix(X_conf, 'X_train_confidence.npz')
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving CSR matrix to NPZ: {output_path}")
        logger.info(f"  Shape: {matrix.shape}")
        logger.info(f"  Non-zero entries: {matrix.nnz:,}")
        logger.info(f"  Sparsity: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
        
        # Save using scipy's save_npz (compressed format)
        save_npz(output_path, matrix)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved CSR matrix, {file_size_mb:.2f} MB")
        
        return str(output_path)
    
    def save_user_sets(
        self,
        user_sets: Dict[int, set],
        filename: str
    ) -> str:
        """
        Save user item sets (positive or hard negative) to Pickle format.
        
        Args:
            user_sets: Dict mapping u_idx to set of i_idx
            filename: Output filename (e.g., 'user_pos_train.pkl')
        
        Returns:
            str: Path to saved file
        
        Example:
            >>> saver = DataSaver('data/processed')
            >>> path = saver.save_user_sets(user_pos, 'user_pos_train.pkl')
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving user sets to Pickle: {output_path}")
        logger.info(f"  Users: {len(user_sets):,}")
        
        total_items = sum(len(items) for items in user_sets.values())
        avg_items = total_items / len(user_sets) if user_sets else 0
        logger.info(f"  Total items: {total_items:,}")
        logger.info(f"  Avg items per user: {avg_items:.2f}")
        
        # Save to Pickle
        with open(output_path, 'wb') as f:
            pickle.dump(user_sets, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved user sets, {file_size_mb:.2f} MB")
        
        return str(output_path)
    
    def save_item_popularity(
        self,
        popularity: np.ndarray,
        filename: str = 'item_popularity.npy'
    ) -> str:
        """
        Save item popularity array to NumPy format.
        
        Args:
            popularity: np.ndarray of shape (num_items,) with log-transformed popularity
            filename: Output filename (default: item_popularity.npy)
        
        Returns:
            str: Path to saved file
        
        Example:
            >>> saver = DataSaver('data/processed')
            >>> path = saver.save_item_popularity(pop_array)
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving item popularity to NumPy: {output_path}")
        logger.info(f"  Items: {len(popularity):,}")
        logger.info(f"  Range: [{popularity.min():.2f}, {popularity.max():.2f}]")
        logger.info(f"  Mean: {popularity.mean():.2f}")
        
        # Save to NumPy format
        np.save(output_path, popularity)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved popularity array, {file_size_mb:.2f} MB")
        
        return str(output_path)
    
    def save_top_k_popular(
        self,
        top_k_items: List[int],
        filename: str = 'top_k_popular_items.json'
    ) -> str:
        """
        Save top-K popular items to JSON format.
        
        Args:
            top_k_items: List of i_idx for top-K popular items
            filename: Output filename (default: top_k_popular_items.json)
        
        Returns:
            str: Path to saved file
        
        Example:
            >>> saver = DataSaver('data/processed')
            >>> path = saver.save_top_k_popular([0, 10, 20, ...])
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving top-K popular items to JSON: {output_path}")
        logger.info(f"  K: {len(top_k_items)}")
        
        data = {
            'top_k_items': [int(i) for i in top_k_items],
            'k': len(top_k_items),
            'created_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"  Saved top-{len(top_k_items)} items")
        
        return str(output_path)
    
    def save_user_metadata(
        self,
        user_metadata: Dict[str, Any],
        filename: str = 'user_metadata.pkl'
    ) -> str:
        """
        Save user segmentation metadata to Pickle format.
        
        Expected structure:
        {
            "trainable_users": set(u_idx, ...),
            "cold_start_users": set(u_idx, ...),
            "user_interaction_counts": {u_idx: count, ...},
            "stats": {...}
        }
        
        Args:
            user_metadata: Dict with user segmentation data
            filename: Output filename (default: user_metadata.pkl)
        
        Returns:
            str: Path to saved file
        
        Example:
            >>> saver = DataSaver('data/processed')
            >>> metadata = {
            ...     "trainable_users": {0, 1, 2, ...},
            ...     "cold_start_users": {100, 101, ...},
            ...     "stats": {...}
            ... }
            >>> path = saver.save_user_metadata(metadata)
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving user metadata to Pickle: {output_path}")
        
        if 'stats' in user_metadata:
            stats = user_metadata['stats']
            logger.info(f"  Trainable users: {stats.get('num_trainable', 0):,}")
            logger.info(f"  Cold-start users: {stats.get('num_cold_start', 0):,}")
        
        # Save to Pickle
        with open(output_path, 'wb') as f:
            pickle.dump(user_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved user metadata, {file_size_mb:.2f} MB")
        
        return str(output_path)
    
    def save_statistics_summary(
        self,
        stats: Dict[str, Any],
        filename: str = 'data_stats.json'
    ) -> str:
        """
        Save comprehensive statistics summary to JSON format.
        
        Expected structure:
        {
            "train_size": 350000,
            "test_size": 15000,
            "sparsity": 0.0012,
            "trainable_users": {
                "count": 26000,
                "percentage": 8.6,
                "avg_interactions_per_user": 2.5,
                "matrix_density": 0.0011
            },
            "popularity": {
                "min": 0.0, "max": 9.21, "mean": 2.45, "std": 1.83,
                "p01": 0.0, "p50": 2.1, "p99": 7.8
            },
            "quality": {
                "min": 1.0, "max": 5.0, "mean": 4.67, "std": 0.52
            },
            "confidence_score": {
                "min": 1.0, "max": 6.0, "mean": 5.12, "std": 0.68,
                "p01": 3.2, "p99": 6.0
            }
        }
        
        Args:
            stats: Dict with comprehensive statistics
            filename: Output filename (default: data_stats.json)
        
        Returns:
            str: Path to saved file
        
        Example:
            >>> saver = DataSaver('data/processed')
            >>> stats = {
            ...     "train_size": 65000,
            ...     "test_size": 26000,
            ...     "sparsity": 0.0011,
            ...     "trainable_users": {...},
            ...     "popularity": {...},
            ...     "quality": {...},
            ...     "confidence_score": {...}
            ... }
            >>> path = saver.save_statistics_summary(stats)
        """
        output_path = self.output_dir / filename
        
        logger.info(f"Saving statistics summary to JSON: {output_path}")
        logger.info(f"  Train size: {stats.get('train_size', 0):,}")
        logger.info(f"  Test size: {stats.get('test_size', 0):,}")
        logger.info(f"  Sparsity: {stats.get('sparsity', 0):.6f}")
        
        # Add metadata
        if 'created_at' not in stats:
            stats['created_at'] = datetime.now().isoformat()
        
        # Convert numpy types to Python native types for JSON serialization
        stats_serializable = self._convert_to_serializable(stats)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_serializable, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Saved statistics summary")
        
        return str(output_path)
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """
        Convert numpy types to Python native types for JSON serialization.
        
        Args:
            obj: Object to convert (can be nested dict/list)
        
        Returns:
            Serializable version of obj
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def compute_data_hash(self, data_paths: List[str]) -> str:
        """
        Compute MD5 hash of multiple data files for versioning.
        
        Args:
            data_paths: List of file paths to hash
        
        Returns:
            str: MD5 hash string
        
        Example:
            >>> saver = DataSaver('data/processed')
            >>> hash_val = saver.compute_data_hash([
            ...     'data/published_data/data_reviews_purchase.csv',
            ...     'data/published_data/data_product.csv'
            ... ])
            >>> print(f"Data hash: {hash_val}")
        """
        logger.info(f"Computing MD5 hash for {len(data_paths)} files...")
        
        hasher = hashlib.md5()
        
        for path in sorted(data_paths):  # Sort for deterministic hash
            if not os.path.exists(path):
                logger.warning(f"File not found for hashing: {path}")
                continue
            
            with open(path, 'rb') as f:
                # Read in chunks for memory efficiency
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        
        hash_value = hasher.hexdigest()
        logger.info(f"  Data hash: {hash_value}")
        
        return hash_value
    
    def save_all_artifacts(
        self,
        interactions_df: pd.DataFrame,
        user_to_idx: Dict[int, int],
        idx_to_user: Dict[int, int],
        item_to_idx: Dict[int, int],
        idx_to_item: Dict[int, int],
        X_train_confidence: csr_matrix,
        user_pos_train: Dict[int, set],
        item_popularity: np.ndarray,
        top_k_popular: List[int],
        user_metadata: Dict[str, Any],
        stats: Dict[str, Any],
        mappings_metadata: Optional[Dict[str, Any]] = None,
        X_train_binary: Optional[csr_matrix] = None,
        user_hard_neg_train: Optional[Dict[int, Dict[str, set]]] = None
    ) -> Dict[str, str]:
        """
        Save all processed data artifacts at once.
        
        This is a convenience method that saves all Step 6 artifacts:
        1. interactions.parquet
        2. user_item_mappings.json
        3. X_train_confidence.npz
        4. X_train_binary.npz (optional)
        5. user_pos_train.pkl
        6. user_hard_neg_train.pkl (optional)
        7. item_popularity.npy
        8. top_k_popular_items.json
        9. user_metadata.pkl
        10. data_stats.json
        
        Args:
            interactions_df: Full interactions DataFrame
            user_to_idx, idx_to_user: User ID mappings
            item_to_idx, idx_to_item: Item ID mappings
            X_train_confidence: Confidence matrix for ALS
            user_pos_train: User positive item sets
            item_popularity: Log-transformed popularity array
            top_k_popular: Top-K popular item indices
            user_metadata: User segmentation data
            stats: Statistics summary
            mappings_metadata: Optional metadata for mappings JSON
            X_train_binary: Optional binary matrix for BPR
            user_hard_neg_train: Optional user hard negative sets
        
        Returns:
            Dict[str, str]: Mapping of artifact name to saved file path
        
        Example:
            >>> saver = DataSaver('data/processed')
            >>> saved_paths = saver.save_all_artifacts(
            ...     interactions_df=df,
            ...     user_to_idx=user_map,
            ...     idx_to_user=inv_user_map,
            ...     ...
            ... )
            >>> print(f"Saved {len(saved_paths)} artifacts")
        """
        logger.info("="*80)
        logger.info("Saving all processed data artifacts...")
        logger.info("="*80)
        
        saved_paths = {}
        
        # 1. Interactions Parquet
        saved_paths['interactions'] = self.save_interactions_parquet(interactions_df)
        
        # 2. ID Mappings JSON
        saved_paths['mappings'] = self.save_mappings_json(
            user_to_idx, idx_to_user, item_to_idx, idx_to_item,
            metadata=mappings_metadata
        )
        
        # 3. Confidence Matrix NPZ
        saved_paths['X_train_confidence'] = self.save_csr_matrix(
            X_train_confidence, 'X_train_confidence.npz'
        )
        
        # 4. Binary Matrix NPZ (optional)
        if X_train_binary is not None:
            saved_paths['X_train_binary'] = self.save_csr_matrix(
                X_train_binary, 'X_train_binary.npz'
            )
        
        # 5. User Positive Sets
        saved_paths['user_pos_train'] = self.save_user_sets(
            user_pos_train, 'user_pos_train.pkl'
        )
        
        # 6. User Hard Negative Sets (optional)
        if user_hard_neg_train is not None:
            saved_paths['user_hard_neg_train'] = self.save_user_sets(
                user_hard_neg_train, 'user_hard_neg_train.pkl'
            )
        
        # 7. Item Popularity
        saved_paths['item_popularity'] = self.save_item_popularity(item_popularity)
        
        # 8. Top-K Popular Items
        saved_paths['top_k_popular'] = self.save_top_k_popular(top_k_popular)
        
        # 9. User Metadata
        saved_paths['user_metadata'] = self.save_user_metadata(user_metadata)
        
        # 10. Statistics Summary
        saved_paths['data_stats'] = self.save_statistics_summary(stats)
        
        logger.info("="*80)
        logger.info(f"Successfully saved {len(saved_paths)} artifacts to {self.output_dir}")
        logger.info("="*80)
        
        return saved_paths
    
    def get_save_summary(self) -> Dict[str, Any]:
        """
        Get summary of saved artifacts in output directory.
        
        Returns:
            Dict with summary information
        
        Example:
            >>> saver = DataSaver('data/processed')
            >>> summary = saver.get_save_summary()
            >>> print(f"Total files: {summary['num_files']}")
            >>> print(f"Total size: {summary['total_size_mb']:.2f} MB")
        """
        files = list(self.output_dir.glob('*'))
        
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        summary = {
            'output_dir': str(self.output_dir),
            'num_files': len([f for f in files if f.is_file()]),
            'total_size_mb': total_size / (1024 * 1024),
            'files': [f.name for f in files if f.is_file()]
        }
        
        return summary
