"""
Data Layer Module for Collaborative Filtering

This module provides a unified interface for data loading, validation, and cleaning
by orchestrating the DataReader and DataAuditor classes.

Key Features:
- Strict temporal validation (no NaT timestamps)
- Rating range enforcement [1.0, 5.0]
- Comprehensive logging of data quality issues
- Reproducible preprocessing pipeline
"""

import os
import logging
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

from .processing.read_data import DataReader
from .processing.audit_data import DataAuditor
from .processing.feature_engineering import FeatureEngineer
from .processing.als_data import ALSDataPreparer
from .processing.bpr_data import BPRDataPreparer
from .processing.user_filtering import UserFilter
from .processing.id_mapping import IDMapper
from .processing.temporal_split import TemporalSplitter
from .processing.matrix_construction import MatrixBuilder
from .processing.data_saver import DataSaver
from .processing.version_registry import VersionRegistry


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging(log_file: str = "logs/cf/data_processing.log") -> logging.Logger:
    """
    Configure logging for data processing pipeline.
    
    Args:
        log_file: Path to log file
    
    Returns:
        Logger instance
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger("data_layer")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler with rotation
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# Main Data Processing Class
# ============================================================================

class DataProcessor:
    """
    Main class for orchestrating data loading, validation, and cleaning.
    
    This class combines DataReader and DataAuditor to provide a complete
    data preprocessing pipeline.
    
    Example:
        >>> processor = DataProcessor(base_path="data/published_data")
        >>> df_clean, stats = processor.load_and_validate_interactions()
        >>> processor.generate_quality_report(df_clean, "Cleaned Data")
    """
    
    def __init__(
        self,
        base_path: str = "data/published_data",
        rating_min: float = 1.0,
        rating_max: float = 5.0,
        drop_missing_timestamps: bool = True,
        positive_threshold: float = 4.0,
        hard_negative_threshold: float = 3.0
    ):
        """
        Initialize DataProcessor.
        
        Args:
            base_path: Base directory containing raw CSV files
            rating_min: Minimum valid rating value (default: 1.0)
            rating_max: Maximum valid rating value (default: 5.0)
            drop_missing_timestamps: If True, drop rows with NaT timestamps (default: True)
            positive_threshold: Rating threshold for positive interactions (default: 4.0)
            hard_negative_threshold: Rating threshold for hard negatives (default: 3.0)
        """
        self.reader = DataReader(base_path=base_path)
        self.auditor = DataAuditor(
            rating_min=rating_min,
            rating_max=rating_max,
            drop_missing_timestamps=drop_missing_timestamps
        )
        self.feature_engineer = FeatureEngineer(
            positive_threshold=positive_threshold,
            hard_negative_threshold=hard_negative_threshold
        )
        self.als_preparer = ALSDataPreparer(
            normalize_confidence=False,
            min_confidence=1.0,
            max_confidence=6.0
        )
        self.bpr_preparer = BPRDataPreparer(
            positive_threshold=positive_threshold,
            hard_negative_threshold=hard_negative_threshold,
            top_k_popular=50,
            hard_negative_ratio=0.3
        )
        self.user_filter = UserFilter(
            min_user_interactions=2,
            min_user_positives=1,
            min_item_positives=5,
            positive_threshold=positive_threshold
        )
        
        # Step 3: ID Mapping (Contiguous Indexing)
        self.id_mapper = IDMapper()
        
        # Step 4: Temporal Split (Leave-One-Out)
        self.temporal_splitter = TemporalSplitter(positive_threshold=positive_threshold)
        
        # Step 5: Matrix Construction
        self.matrix_builder = MatrixBuilder(
            positive_threshold=positive_threshold,
            hard_negative_threshold=hard_negative_threshold,
            top_k_popular=50
        )
        
        # Step 6: Save Processed Data
        self.data_saver = DataSaver(output_dir='data/processed')
        
        # Step 7: Data Versioning
        self.version_registry = VersionRegistry(registry_path='data/processed/versions.json')
    
    def load_and_validate_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load and validate all data files.
        
        Returns:
            Dictionary with cleaned data: 'interactions', 'products', 'attributes', 'shops'
        """
        # Load all data
        data = self.reader.load_all_data()
        
        # Validate and clean interactions
        data['interactions'], stats = self.auditor.validate_and_clean(data['interactions'])
        
        return data
    
    def load_and_validate_interactions(
        self,
        apply_deduplication: bool = True,
        detect_outliers: bool = True,
        compute_quality_scores: bool = False,
        comment_column: str = 'processed_comment',
        max_user_interactions: int = 500,
        min_item_interactions: int = 3,
        rating_bias_threshold: float = 0.90
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Complete Step 1 + Step 2.0 pipeline: Load, validate, deduplicate, detect outliers, compute quality.
        
        Args:
            apply_deduplication: If True, remove duplicate (user_id, product_id) pairs
            detect_outliers: If True, detect but not filter outliers
            compute_quality_scores: If True, compute comment quality and confidence scores (Step 2.0)
            comment_column: Column name for comment text (default: 'processed_comment')
            max_user_interactions: Threshold for bot detection (default: 500)
            min_item_interactions: Threshold for cold items (default: 3)
            rating_bias_threshold: Threshold for rating bias detection (default: 0.90)
        
        Returns:
            Tuple of (cleaned_df, complete_stats)
            - cleaned_df: Validated, deduplicated, and enriched DataFrame
            - complete_stats: Dictionary containing all statistics from Steps 1.1, 1.2, 1.3, 2.0
        """
        # Step 1.1: Load and validate
        df = self.reader.load_interactions()
        df_clean, validation_stats = self.auditor.validate_and_clean(df)
        
        # Step 1.2: Deduplicate
        dedup_stats = {}
        if apply_deduplication:
            df_clean, dedup_stats = self.auditor.deduplicate_interactions(df_clean)
        
        # Step 1.3: Detect outliers (no filtering)
        outlier_report = {}
        if detect_outliers:
            outlier_report = self.auditor.detect_outliers(
                df_clean,
                max_user_interactions=max_user_interactions,
                min_item_interactions=min_item_interactions,
                rating_bias_threshold=rating_bias_threshold
            )
        
        # Step 2.0: Compute comment quality and confidence scores
        quality_stats = {}
        if compute_quality_scores:
            df_clean, quality_stats = self.feature_engineer.compute_confidence_scores(
                df_clean,
                comment_column=comment_column
            )
        
        # Compile complete statistics
        complete_stats = {
            'step_1_1_validation': validation_stats,
            'step_1_2_deduplication': dedup_stats,
            'step_1_3_outliers': outlier_report,
            'step_2_0_quality': quality_stats
        }
        
        return df_clean, complete_stats
    
    def deduplicate_interactions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Remove duplicate (user_id, product_id) pairs.
        
        Args:
            df: DataFrame with potentially duplicate interactions
        
        Returns:
            Tuple of (deduplicated_df, dedup_stats)
        """
        return self.auditor.deduplicate_interactions(df)
    
    def detect_outliers(
        self,
        df: pd.DataFrame,
        max_user_interactions: int = 500,
        min_item_interactions: int = 3,
        rating_bias_threshold: float = 0.90
    ) -> Dict[str, any]:
        """
        Detect outliers in the data.
        
        Args:
            df: DataFrame with cleaned interactions
            max_user_interactions: Threshold for bot detection (default: 500)
            min_item_interactions: Threshold for cold items (default: 3)
            rating_bias_threshold: Threshold for rating bias detection (default: 0.90)
        
        Returns:
            Dictionary containing outlier analysis results
        """
        return self.auditor.detect_outliers(
            df,
            max_user_interactions=max_user_interactions,
            min_item_interactions=min_item_interactions,
            rating_bias_threshold=rating_bias_threshold
        )
    
    def compute_comment_quality(
        self, 
        df: pd.DataFrame,
        comment_column: str = 'processed_comment'
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Compute comment quality and confidence scores (Step 2.0).
        
        Args:
            df: DataFrame with interactions
            comment_column: Column name for comment text (default: 'processed_comment')
        
        Returns:
            Tuple of (enriched_df, quality_stats)
        """
        return self.feature_engineer.compute_confidence_scores(df, comment_column)
    
    def generate_quality_report(self, df: pd.DataFrame, name: str = "DataFrame") -> None:
        """
        Generate quality report for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            name: Name for logging context
        """
        self.auditor.generate_quality_report(df, name)
    
    def compute_data_hash(self, df: pd.DataFrame) -> str:
        """
        Compute hash for DataFrame versioning.
        
        Args:
            df: DataFrame to hash
        
        Returns:
            MD5 hash string
        """
        return self.auditor.compute_data_hash(df)
    
    def prepare_als_matrix(
        self,
        interactions_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        normalize: bool = False
    ):
        """
        Prepare confidence-weighted matrix for ALS training (Step 2.1).
        
        Args:
            interactions_df: DataFrame with u_idx, i_idx, confidence_score
            num_users: Total number of users
            num_items: Total number of items
            normalize: If True, normalize confidence to [0, 1]
        
        Returns:
            Tuple of (csr_matrix, stats_dict)
        
        Example:
            >>> X_conf, stats = processor.prepare_als_matrix(train_df, 26000, 2231)
            >>> print(X_conf.shape)  # (26000, 2231)
        """
        self.als_preparer.normalize_confidence = normalize
        return self.als_preparer.prepare_confidence_matrix(
            interactions_df, num_users, num_items
        )
    
    def analyze_confidence_distribution(
        self,
        interactions_df: pd.DataFrame
    ):
        """
        Analyze confidence score distribution by rating level.
        
        Args:
            interactions_df: DataFrame with ratings and confidence scores
        
        Returns:
            Dict with distribution statistics
        """
        return self.als_preparer.get_confidence_distribution(interactions_df)
    
    def prepare_bpr_labels(
        self,
        interactions_df: pd.DataFrame,
        products_df: pd.DataFrame = None
    ):
        """
        Create positive labels and mine hard negatives for BPR (Step 2.2).
        
        Args:
            interactions_df: DataFrame with interactions
            products_df: Optional DataFrame with product metadata for popularity mining
        
        Returns:
            Tuple of (labeled_df, hard_negative_sets_dict)
        
        Example:
            >>> df_labeled, hard_neg_sets = processor.prepare_bpr_labels(
            ...     train_df, products_df
            ... )
            >>> print(df_labeled['is_positive'].value_counts())
        """
        # Step 2.2.1: Create positive labels
        interactions_df = self.bpr_preparer.create_positive_labels(interactions_df)
        
        # Step 2.2.2: Mine hard negatives
        interactions_df, hard_neg_sets = self.bpr_preparer.mine_hard_negatives(
            interactions_df,
            products_df,
            user_col='user_id',
            item_col='product_id'
        )
        
        # Validate
        self.bpr_preparer.validate_labels(interactions_df)
        
        return interactions_df, hard_neg_sets
    
    def build_bpr_positive_sets(
        self,
        interactions_df: pd.DataFrame
    ):
        """
        Build user positive item sets for BPR negative sampling.
        
        Args:
            interactions_df: DataFrame with is_positive column
        
        Returns:
            Dict mapping user_idx -> Set of positive item indices
        """
        return self.bpr_preparer.build_positive_sets(interactions_df)
    
    def get_bpr_sampling_strategy(self):
        """
        Get BPR negative sampling strategy information.
        
        Returns:
            Dict with sampling ratios and recommendations
        """
        return self.bpr_preparer.get_sampling_strategy_info()
    
    def segment_users(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        rating_col: str = 'rating'
    ):
        """
        Segment users into trainable vs cold-start (Step 2.3).
        
        Args:
            interactions_df: DataFrame with user interactions
            user_col: User ID column
            rating_col: Rating column
        
        Returns:
            Tuple of (segmented_df, stats_dict)
        
        Example:
            >>> df_segmented, stats = processor.segment_users(df)
            >>> print(stats['trainable_users'])  # ~26,000 users
            >>> print(stats['trainable_percentage'])  # ~8.6%
        """
        return self.user_filter.segment_users(interactions_df, user_col, rating_col)
    
    def apply_complete_filtering(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'product_id',
        rating_col: str = 'rating'
    ):
        """
        Apply complete filtering pipeline: user segmentation + iterative item filtering (Step 2.3).
        
        Args:
            interactions_df: DataFrame with interactions
            user_col: User ID column
            item_col: Item ID column
            rating_col: Rating column
        
        Returns:
            Tuple of (filtered_df, stats_dict)
        
        Workflow:
            1. Segment users (≥2 interactions, ≥1 positive → trainable)
            2. Iteratively filter items (≥5 positive interactions from trainable users)
            3. Return filtered data with segmentation metadata
        
        Example:
            >>> df_filtered, stats = processor.apply_complete_filtering(df)
            >>> print(stats['final_trainable_users'])  # ~26,000
            >>> print(stats['final_matrix_density'])  # ~0.11%
        """
        return self.user_filter.apply_complete_filtering(
            interactions_df, user_col, item_col, rating_col
        )
    
    def filter_to_trainable_only(
        self,
        interactions_df: pd.DataFrame
    ):
        """
        Filter DataFrame to trainable users only (for CF training).
        
        Args:
            interactions_df: DataFrame with 'is_trainable_user' column
        
        Returns:
            Filtered DataFrame (trainable users only)
        """
        return self.user_filter.filter_to_trainable_only(interactions_df)
    
    def get_trainable_user_set(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id'
    ):
        """
        Get set of trainable user IDs.
        
        Args:
            interactions_df: DataFrame with 'is_trainable_user' column
            user_col: User ID column
        
        Returns:
            Set of trainable user IDs
        """
        return self.user_filter.get_trainable_user_set(interactions_df, user_col)
    
    def get_cold_start_user_set(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id'
    ):
        """
        Get set of cold-start user IDs.
        
        Args:
            interactions_df: DataFrame with 'is_trainable_user' column
            user_col: User ID column
        
        Returns:
            Set of cold-start user IDs
        """
        return self.user_filter.get_cold_start_user_set(interactions_df, user_col)
    
    # === Step 3: ID Mapping (Contiguous Indexing) ===
    
    def create_id_mappings(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'product_id'
    ):
        """
        Create bidirectional ID mappings (Step 3).
        
        Args:
            interactions_df: DataFrame with user and item IDs
            user_col: User ID column
            item_col: Item ID column
        
        Returns:
            Tuple of (user_to_idx, idx_to_user, item_to_idx, idx_to_item)
        
        Example:
            >>> u2i, i2u, i2i, ii2i = processor.create_id_mappings(df)
            >>> print(f"Mapped {len(u2i)} users, {len(i2i)} items")
        """
        return self.id_mapper.create_mappings(interactions_df, user_col, item_col)
    
    def apply_id_mappings(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'product_id'
    ):
        """
        Apply ID mappings to interactions DataFrame (Step 3).
        
        Args:
            interactions_df: DataFrame with user_id and product_id
            user_col: User ID column
            item_col: Item ID column
        
        Returns:
            DataFrame with added u_idx and i_idx columns
        
        Example:
            >>> df_mapped = processor.apply_id_mappings(df)
            >>> print(df_mapped[['user_id', 'u_idx', 'product_id', 'i_idx']].head())
        """
        return self.id_mapper.apply_mappings(interactions_df, user_col, item_col)
    
    def save_id_mappings(
        self,
        output_path: str,
        interactions_df: pd.DataFrame
    ):
        """
        Save ID mappings to JSON file (Step 3).
        
        Args:
            output_path: Path to save JSON file
            interactions_df: Original interactions DataFrame for hash
        
        Example:
            >>> processor.save_id_mappings('data/processed/user_item_mappings.json', df)
        """
        self.id_mapper.save_mappings(output_path, interactions_df)
    
    def load_id_mappings(self, input_path: str):
        """
        Load ID mappings from JSON file (Step 3).
        
        Args:
            input_path: Path to JSON file
        
        Example:
            >>> processor.load_id_mappings('data/processed/user_item_mappings.json')
        """
        self.id_mapper.load_mappings(input_path)
    
    def get_mapping_stats(self):
        """
        Get statistics about current ID mappings.
        
        Returns:
            Dict with mapping statistics
        """
        return self.id_mapper.get_mapping_stats()
    
    def reverse_user_mapping(self, u_indices):
        """
        Convert contiguous indices back to original user IDs.
        
        Args:
            u_indices: Array-like of u_idx values
        
        Returns:
            List of original user_ids
        """
        return self.id_mapper.reverse_user_mapping(u_indices)
    
    def reverse_item_mapping(self, i_indices):
        """
        Convert contiguous indices back to original item IDs.
        
        Args:
            i_indices: Array-like of i_idx values
        
        Returns:
            List of original product_ids
        """
        return self.id_mapper.reverse_item_mapping(i_indices)
    
    def create_id_mappings(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'product_id'
    ):
        """
        Create bidirectional ID mappings (Step 3).
        
        Args:
            interactions_df: DataFrame with user and item IDs
            user_col: User ID column
            item_col: Item ID column
        
        Returns:
            Tuple of (user_to_idx, idx_to_user, item_to_idx, idx_to_item)
        
        Example:
            >>> u2i, i2u, i2i, i2i = processor.create_id_mappings(df)
            >>> print(f"Mapped {len(u2i)} users, {len(i2i)} items")
        """
        return self.id_mapper.create_mappings(interactions_df, user_col, item_col)
    
    def apply_id_mappings(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'product_id'
    ):
        """
        Apply ID mappings to interactions DataFrame (Step 3).
        
        Args:
            interactions_df: DataFrame with user_id and product_id
            user_col: User ID column
            item_col: Item ID column
        
        Returns:
            DataFrame with added u_idx and i_idx columns
        
        Example:
            >>> df_mapped = processor.apply_id_mappings(df)
            >>> print(df_mapped[['user_id', 'u_idx', 'product_id', 'i_idx']].head())
        """
        return self.id_mapper.apply_mappings(interactions_df, user_col, item_col)
    
    def save_id_mappings(
        self,
        output_path: str,
        interactions_df: pd.DataFrame
    ):
        """
        Save ID mappings to JSON file (Step 3).
        
        Args:
            output_path: Path to save JSON file
            interactions_df: Original interactions DataFrame for hash
        
        Example:
            >>> processor.save_id_mappings('data/processed/user_item_mappings.json', df)
        """
        self.id_mapper.save_mappings(output_path, interactions_df)
    
    def load_id_mappings(self, input_path: str):
        """
        Load ID mappings from JSON file (Step 3).
        
        Args:
            input_path: Path to JSON file
        
        Example:
            >>> processor.load_id_mappings('data/processed/user_item_mappings.json')
        """
        self.id_mapper.load_mappings(input_path)
    
    def get_mapping_stats(self):
        """
        Get statistics about current ID mappings.
        
        Returns:
            Dict with mapping statistics
        """
        return self.id_mapper.get_mapping_stats()
    
    def reverse_user_mapping(self, u_indices):
        """
        Convert contiguous indices back to original user IDs.
        
        Args:
            u_indices: Array-like of u_idx values
        
        Returns:
            List of original user_ids
        """
        return self.id_mapper.reverse_user_mapping(u_indices)
    
    def reverse_item_mapping(self, i_indices):
        """
        Convert contiguous indices back to original item IDs.
        
        Args:
            i_indices: Array-like of i_idx values
        
        Returns:
            List of original product_ids
        """
        return self.id_mapper.reverse_item_mapping(i_indices)
    
    # ========================================================================
    # Step 4: Temporal Split Methods
    # ========================================================================
    
    def temporal_split(
        self,
        interactions_df: pd.DataFrame,
        method: str = 'leave_one_out',
        use_validation: bool = False,
        timestamp_col: str = 'cmt_date',
        user_col: str = 'u_idx',
        rating_col: str = 'rating'
    ):
        """
        Split interactions into train/test(/val) sets with temporal ordering (Step 4).
        
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
            - test_df: Latest positive interaction per user (rating ≥ threshold)
            - val_df: Optional 2nd latest positive interaction per user (or None)
        
        Example:
            >>> train_df, test_df, val_df = processor.temporal_split(
            ...     df_mapped, 
            ...     method='leave_one_out',
            ...     use_validation=False
            ... )
            >>> print(f"Train: {len(train_df)}, Test: {len(test_df)}")
        """
        return self.temporal_splitter.split(
            interactions_df,
            method=method,
            use_validation=use_validation,
            timestamp_col=timestamp_col,
            user_col=user_col,
            rating_col=rating_col
        )
    
    def get_split_metadata(self):
        """
        Get metadata about the temporal split (Step 4).
        
        Returns:
            Dict with split statistics:
            - created_at: Timestamp
            - positive_threshold: Threshold used
            - train/test/val: num_interactions, num_users, num_positives
        
        Example:
            >>> metadata = processor.get_split_metadata()
            >>> print(f"Test users: {metadata['test']['num_users']}")
        """
        return self.temporal_splitter.get_split_metadata()
    
    def save_split_metadata(self, output_path: str):
        """
        Save split metadata to JSON file (Step 4).
        
        Args:
            output_path: Path to save JSON file
        
        Example:
            >>> processor.save_split_metadata('data/processed/split_metadata.json')
        """
        self.temporal_splitter.save_split_metadata(output_path)
    
    def validate_positive_only_test(self, test_df: pd.DataFrame, rating_col: str = 'rating'):
        """
        Validate that test set only contains positive interactions (Step 4).
        
        Args:
            test_df: Test DataFrame
            rating_col: Name of rating column
            
        Returns:
            True if all test interactions are positive, False otherwise
        
        Example:
            >>> is_valid = processor.validate_positive_only_test(test_df)
            >>> if not is_valid:
            ...     print("ERROR: Test set contains negative interactions!")
        """
        return self.temporal_splitter.validate_positive_only_test(test_df, rating_col)
    
    def get_user_split_summary(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df,
        user_col: str = 'u_idx'
    ):
        """
        Get per-user split summary (Step 4).
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            val_df: Validation DataFrame (optional)
            user_col: Name of user column
            
        Returns:
            DataFrame with columns: u_idx, train_count, test_count, val_count, has_test, has_val
        
        Example:
            >>> summary = processor.get_user_split_summary(train_df, test_df, val_df)
            >>> print(summary[summary['has_test'] == False])  # Users without test data
        """
        return self.temporal_splitter.get_user_split_summary(
            train_df, test_df, val_df, user_col
        )
    
    # ========================================================================
    # Step 5: Matrix Construction Methods
    # ========================================================================
    
    def build_confidence_matrix(
        self,
        interactions_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        value_col: str = 'confidence_score'
    ):
        """
        Build sparse CSR matrix with confidence scores for ALS (Step 5).
        
        Args:
            interactions_df: DataFrame with interactions
            num_users: Total number of users (matrix rows)
            num_items: Total number of items (matrix columns)
            user_col: User index column
            item_col: Item index column
            value_col: Value column (confidence_score or rating)
            
        Returns:
            scipy.sparse.csr_matrix of shape (num_users, num_items)
        
        Example:
            >>> X_train_conf = processor.build_confidence_matrix(
            ...     train_df, num_users=5000, num_items=2000, value_col='confidence_score'
            ... )
            >>> print(f"Matrix shape: {X_train_conf.shape}, Sparsity: {1 - X_train_conf.nnz/(5000*2000):.4f}")
        """
        return self.matrix_builder.build_confidence_matrix(
            interactions_df, num_users, num_items, user_col, item_col, value_col
        )
    
    def build_binary_matrix(
        self,
        interactions_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        positive_only: bool = True
    ):
        """
        Build sparse binary CSR matrix for BPR (Step 5, optional).
        
        Args:
            interactions_df: DataFrame with interactions
            num_users: Total number of users
            num_items: Total number of items
            user_col: User index column
            item_col: Item index column
            positive_only: If True, only include positive interactions
            
        Returns:
            scipy.sparse.csr_matrix with binary values (0 or 1)
        
        Example:
            >>> X_train_bin = processor.build_binary_matrix(
            ...     train_df, num_users=5000, num_items=2000, positive_only=True
            ... )
        """
        return self.matrix_builder.build_binary_matrix(
            interactions_df, num_users, num_items, user_col, item_col, positive_only
        )
    
    def build_user_positive_sets(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx'
    ):
        """
        Build user positive item sets for fast lookup (Step 5).
        
        Args:
            interactions_df: DataFrame with positive interactions
            user_col: User index column
            item_col: Item index column
            
        Returns:
            Dict mapping u_idx -> Set[i_idx] of positive items
        
        Example:
            >>> user_pos_sets = processor.build_user_positive_sets(train_df)
            >>> print(f"User 0 positives: {user_pos_sets[0]}")
        """
        return self.matrix_builder.build_user_positive_sets(
            interactions_df, user_col, item_col
        )
    
    def build_user_hard_negative_sets(
        self,
        interactions_df: pd.DataFrame,
        top_k_popular_items=None,
        user_col: str = 'u_idx',
        item_col: str = 'i_idx',
        rating_col: str = 'rating'
    ):
        """
        Build user hard negative sets (explicit + implicit) (Step 5).
        
        Args:
            interactions_df: DataFrame with interactions
            top_k_popular_items: List of popular item indices for implicit negatives
            user_col: User index column
            item_col: Item index column
            rating_col: Rating column
            
        Returns:
            Dict mapping u_idx -> {"explicit": set(...), "implicit": set(...)}
        
        Example:
            >>> top_k = processor.get_top_k_popular_items(train_df, k=50)
            >>> hard_negs = processor.build_user_hard_negative_sets(train_df, top_k)
            >>> print(f"User 0 explicit hard negs: {hard_negs[0]['explicit']}")
        """
        return self.matrix_builder.build_user_hard_negative_sets(
            interactions_df, top_k_popular_items, user_col, item_col, rating_col
        )
    
    def build_item_popularity(
        self,
        interactions_df: pd.DataFrame,
        num_items: int,
        item_col: str = 'i_idx',
        log_transform: bool = True
    ):
        """
        Build item popularity scores with log-transform (Step 5).
        
        Args:
            interactions_df: DataFrame with interactions
            num_items: Total number of items
            item_col: Item index column
            log_transform: If True, apply log(1 + count)
            
        Returns:
            np.ndarray of shape (num_items,) with popularity scores
        
        Example:
            >>> popularity = processor.build_item_popularity(train_df, num_items=2000)
            >>> print(f"Popularity range: [{popularity.min():.2f}, {popularity.max():.2f}]")
        """
        return self.matrix_builder.build_item_popularity(
            interactions_df, num_items, item_col, log_transform
        )
    
    def get_top_k_popular_items(
        self,
        interactions_df: pd.DataFrame,
        k=None,
        item_col: str = 'i_idx'
    ):
        """
        Get top-K most popular items by interaction count (Step 5).
        
        Args:
            interactions_df: DataFrame with interactions
            k: Number of top items (default: 50)
            item_col: Item index column
            
        Returns:
            List of i_idx for top-K popular items
        
        Example:
            >>> top_50 = processor.get_top_k_popular_items(train_df, k=50)
            >>> print(f"Top 10 popular items: {top_50[:10]}")
        """
        return self.matrix_builder.get_top_k_popular_items(
            interactions_df, k, item_col
        )
    
    def build_user_metadata(
        self,
        interactions_df: pd.DataFrame,
        min_interactions_trainable: int = 2,
        user_col: str = 'u_idx'
    ):
        """
        Build user segmentation metadata (trainable vs cold-start) (Step 5).
        
        Args:
            interactions_df: DataFrame with interactions
            min_interactions_trainable: Minimum interactions for trainable users
            user_col: User index column
            
        Returns:
            Dict with user statistics and segmentation
        
        Example:
            >>> metadata = processor.build_user_metadata(train_df, min_interactions_trainable=2)
            >>> print(f"Trainable users: {metadata['stats']['num_trainable']}")
        """
        return self.matrix_builder.build_user_metadata(
            interactions_df, min_interactions_trainable, user_col
        )
    
    def get_matrix_build_metadata(self):
        """
        Get metadata about all built matrix structures (Step 5).
        
        Returns:
            Dict with comprehensive build statistics
        
        Example:
            >>> metadata = processor.get_matrix_build_metadata()
            >>> print(f"Confidence matrix shape: {metadata['confidence_matrix']['shape']}")
        """
        return self.matrix_builder.get_build_metadata()
    
    def save_matrix_build_metadata(self, output_path: str):
        """
        Save matrix build metadata to JSON file (Step 5).
        
        Args:
            output_path: Path to save JSON file
        
        Example:
            >>> processor.save_matrix_build_metadata('data/processed/matrix_metadata.json')
        """
        self.matrix_builder.save_build_metadata(output_path)
    
    # ========================================================================
    # Step 6: Save Processed Data Methods
    # ========================================================================
    
    def save_interactions_parquet(self, interactions_df, filename='interactions.parquet'):
        """
        Save interactions DataFrame to Parquet format (Step 6).
        
        Args:
            interactions_df: Full interactions DataFrame with all columns
            filename: Output filename
            
        Returns:
            str: Path to saved file
        
        Example:
            >>> path = processor.save_interactions_parquet(df)
            >>> print(f"Saved to {path}")
        """
        return self.data_saver.save_interactions_parquet(interactions_df, filename)
    
    def save_mappings_json(self, user_to_idx, idx_to_user, item_to_idx, idx_to_item, 
                          metadata=None, filename='user_item_mappings.json'):
        """
        Save ID mappings to JSON format with metadata (Step 6).
        
        Args:
            user_to_idx: Dict mapping original user_id to u_idx
            idx_to_user: Dict mapping u_idx to original user_id
            item_to_idx: Dict mapping original product_id to i_idx
            idx_to_item: Dict mapping i_idx to original product_id
            metadata: Optional metadata dict
            filename: Output filename
            
        Returns:
            str: Path to saved file
        
        Example:
            >>> path = processor.save_mappings_json(user_map, inv_user_map, item_map, inv_item_map)
        """
        return self.data_saver.save_mappings_json(
            user_to_idx, idx_to_user, item_to_idx, idx_to_item, metadata, filename
        )
    
    def save_csr_matrix(self, matrix, filename):
        """
        Save sparse CSR matrix to NPZ format (Step 6).
        
        Args:
            matrix: scipy.sparse.csr_matrix to save
            filename: Output filename (e.g., 'X_train_confidence.npz')
            
        Returns:
            str: Path to saved file
        
        Example:
            >>> path = processor.save_csr_matrix(X_conf, 'X_train_confidence.npz')
        """
        return self.data_saver.save_csr_matrix(matrix, filename)
    
    def save_user_sets(self, user_sets, filename):
        """
        Save user item sets (positive or hard negative) to Pickle format (Step 6).
        
        Args:
            user_sets: Dict mapping u_idx to set of i_idx
            filename: Output filename (e.g., 'user_pos_train.pkl')
            
        Returns:
            str: Path to saved file
        
        Example:
            >>> path = processor.save_user_sets(user_pos, 'user_pos_train.pkl')
        """
        return self.data_saver.save_user_sets(user_sets, filename)
    
    def save_item_popularity(self, popularity, filename='item_popularity.npy'):
        """
        Save item popularity array to NumPy format (Step 6).
        
        Args:
            popularity: np.ndarray with log-transformed popularity
            filename: Output filename
            
        Returns:
            str: Path to saved file
        
        Example:
            >>> path = processor.save_item_popularity(pop_array)
        """
        return self.data_saver.save_item_popularity(popularity, filename)
    
    def save_top_k_popular(self, top_k_items, filename='top_k_popular_items.json'):
        """
        Save top-K popular items to JSON format (Step 6).
        
        Args:
            top_k_items: List of i_idx for top-K popular items
            filename: Output filename
            
        Returns:
            str: Path to saved file
        
        Example:
            >>> path = processor.save_top_k_popular([0, 10, 20, ...])
        """
        return self.data_saver.save_top_k_popular(top_k_items, filename)
    
    def save_user_metadata(self, user_metadata, filename='user_metadata.pkl'):
        """
        Save user segmentation metadata to Pickle format (Step 6).
        
        Args:
            user_metadata: Dict with user segmentation data
            filename: Output filename
            
        Returns:
            str: Path to saved file
        
        Example:
            >>> metadata = {"trainable_users": {...}, "stats": {...}}
            >>> path = processor.save_user_metadata(metadata)
        """
        return self.data_saver.save_user_metadata(user_metadata, filename)
    
    def save_statistics_summary(self, stats, filename='data_stats.json'):
        """
        Save comprehensive statistics summary to JSON format (Step 6).
        
        Args:
            stats: Dict with comprehensive statistics
            filename: Output filename
            
        Returns:
            str: Path to saved file
        
        Example:
            >>> stats = {
            ...     "train_size": 65000,
            ...     "trainable_users": {...},
            ...     "popularity": {...},
            ...     "quality": {...}
            ... }
            >>> path = processor.save_statistics_summary(stats)
        """
        return self.data_saver.save_statistics_summary(stats, filename)
    
    def save_all_artifacts(self, interactions_df, user_to_idx, idx_to_user, item_to_idx, 
                          idx_to_item, X_train_confidence, user_pos_train, item_popularity,
                          top_k_popular, user_metadata, stats, mappings_metadata=None,
                          X_train_binary=None, user_hard_neg_train=None):
        """
        Save all processed data artifacts at once (Step 6).
        
        This is a convenience method that saves all Step 6 artifacts in one call.
        
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
            >>> saved_paths = processor.save_all_artifacts(
            ...     interactions_df=df,
            ...     user_to_idx=user_map,
            ...     ...
            ... )
            >>> print(f"Saved {len(saved_paths)} artifacts")
        """
        return self.data_saver.save_all_artifacts(
            interactions_df, user_to_idx, idx_to_user, item_to_idx, idx_to_item,
            X_train_confidence, user_pos_train, item_popularity, top_k_popular,
            user_metadata, stats, mappings_metadata, X_train_binary, user_hard_neg_train
        )
    
    def compute_data_hash(self, data_paths):
        """
        Compute MD5 hash of multiple data files for versioning (Step 6).
        
        Args:
            data_paths: List of file paths to hash
            
        Returns:
            str: MD5 hash string
        
        Example:
            >>> hash_val = processor.compute_data_hash([
            ...     'data/published_data/data_reviews_purchase.csv',
            ...     'data/published_data/data_product.csv'
            ... ])
        """
        return self.data_saver.compute_data_hash(data_paths)
    
    def get_save_summary(self):
        """
        Get summary of saved artifacts in output directory (Step 6).
        
        Returns:
            Dict with summary information
        
        Example:
            >>> summary = processor.get_save_summary()
            >>> print(f"Total files: {summary['num_files']}")
            >>> print(f"Total size: {summary['total_size_mb']:.2f} MB")
        """
        return self.data_saver.get_save_summary()
    
    # ========================================================================
    # Step 7: Data Versioning Methods
    # ========================================================================
    
    def create_data_version(self, data_hash, filters, files, stats=None, 
                           git_commit=None, description=None):
        """
        Create a new data version entry in the registry (Step 7).
        
        Args:
            data_hash: MD5 hash of the raw data files
            filters: Dictionary of filter configurations used
            files: List of artifact filenames created in this version
            stats: Optional statistics summary for this version
            git_commit: Optional git commit hash for code version tracking
            description: Optional human-readable description
            
        Returns:
            str: Version ID
        
        Example:
            >>> version_id = processor.create_data_version(
            ...     data_hash="abc123...",
            ...     filters={"min_user_pos": 2, "min_item_pos": 5},
            ...     files=["interactions.parquet", "mappings.json"],
            ...     stats={"train_size": 65000, "test_size": 26000},
            ...     description="Baseline version with updated filters"
            ... )
        """
        return self.version_registry.create_version(
            data_hash=data_hash,
            filters=filters,
            files=files,
            stats=stats,
            git_commit=git_commit,
            description=description
        )
    
    def get_data_version(self, version_id):
        """
        Get version entry by ID (Step 7).
        
        Args:
            version_id: Version identifier
            
        Returns:
            Dict: Version entry or None if not found
        
        Example:
            >>> version = processor.get_data_version("v1_20250115_103000")
            >>> print(f"Created: {version['timestamp']}")
            >>> print(f"Hash: {version['hash']}")
        """
        return self.version_registry.get_version(version_id)
    
    def get_latest_data_version(self):
        """
        Get the most recent version entry (Step 7).
        
        Returns:
            Tuple: (version_id, version_entry) or None if registry is empty
        
        Example:
            >>> latest = processor.get_latest_data_version()
            >>> if latest:
            ...     version_id, version_data = latest
            ...     print(f"Latest version: {version_id}")
        """
        return self.version_registry.get_latest_version()
    
    def list_data_versions(self, limit=None):
        """
        List all versions in chronological order (Step 7).
        
        Args:
            limit: Optional limit on number of versions to return
            
        Returns:
            List[Tuple]: List of (version_id, version_entry) tuples
        
        Example:
            >>> versions = processor.list_data_versions(limit=5)
            >>> for vid, vdata in versions:
            ...     print(f"{vid}: {vdata['timestamp']}")
        """
        return self.version_registry.list_versions(limit=limit)
    
    def compare_data_versions(self, version_id1, version_id2):
        """
        Compare two versions to detect differences (Step 7).
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Dict: Comparison results with keys:
                - hash_changed: bool
                - filters_changed: bool
                - filter_differences: {...}
                - files_added: [...]
                - files_removed: [...]
                - stats_changed: bool
        
        Example:
            >>> comparison = processor.compare_data_versions(
            ...     "v1_20250115_103000", 
            ...     "v2_20250116_140000"
            ... )
            >>> if comparison['hash_changed']:
            ...     print("Raw data changed - models need retraining")
        """
        return self.version_registry.compare_versions(version_id1, version_id2)
    
    def is_data_version_stale(self, version_id, max_age_hours=24):
        """
        Check if a version is stale based on its timestamp (Step 7).
        
        Args:
            version_id: Version identifier
            max_age_hours: Maximum age in hours before considering stale
            
        Returns:
            bool: True if version is stale, False otherwise
        
        Example:
            >>> if processor.is_data_version_stale("v1_20250115_103000", max_age_hours=24):
            ...     print("Data is over 24 hours old - consider retraining")
        """
        return self.version_registry.is_stale(version_id, max_age_hours)
    
    def find_version_by_hash(self, data_hash):
        """
        Find version ID by data hash (Step 7).
        
        Args:
            data_hash: MD5 hash to search for
            
        Returns:
            str: Version ID or None if not found
        
        Example:
            >>> hash_val = processor.compute_data_hash(['data/published_data/data_reviews_purchase.csv'])
            >>> version_id = processor.find_version_by_hash(hash_val)
        """
        return self.version_registry.find_version_by_hash(data_hash)
    
    def find_versions_by_filters(self, filters):
        """
        Find all versions with matching filter configurations (Step 7).
        
        Args:
            filters: Filter configuration to match
            
        Returns:
            List[str]: List of version IDs with matching filters
        
        Example:
            >>> versions = processor.find_versions_by_filters({
            ...     "min_user_pos": 2,
            ...     "min_item_pos": 5
            ... })
        """
        return self.version_registry.find_versions_by_filters(filters)
    
    def get_version_registry_summary(self):
        """
        Get summary statistics about the version registry (Step 7).
        
        Returns:
            Dict: Summary with keys:
                - num_versions: int
                - oldest_version: str
                - newest_version: str
                - total_files: int
                - unique_hashes: int
        
        Example:
            >>> summary = processor.get_version_registry_summary()
            >>> print(f"Total versions: {summary['num_versions']}")
            >>> print(f"Unique data hashes: {summary['unique_hashes']}")
        """
        return self.version_registry.get_registry_summary()
    
    def delete_data_version(self, version_id):
        """
        Delete a version from the registry (Step 7).
        
        Note: This only removes the version from the registry,
        it does not delete the actual data files.
        
        Args:
            version_id: Version identifier to delete
        
        Example:
            >>> processor.delete_data_version("v1_20250115_103000")
        """
        self.version_registry.delete_version(version_id)
    
    def export_version_registry_to_csv(self, output_path):
        """
        Export version registry to CSV format for analysis (Step 7).
        
        Args:
            output_path: Path to output CSV file
        
        Example:
            >>> processor.export_version_registry_to_csv('reports/version_history.csv')
        """
        self.version_registry.export_to_csv(output_path)


# ============================================================================
# Convenience Functions (Backward Compatibility)
# ============================================================================

def load_raw_data(base_path: str = "data/published_data") -> Dict[str, pd.DataFrame]:
    """
    Load raw CSV files from published_data directory.
    
    Args:
        base_path: Base directory containing raw CSV files
    
    Returns:
        Dictionary with keys: 'interactions', 'products', 'attributes', 'shops'
    """
    reader = DataReader(base_path=base_path)
    return reader.load_all_data()


def validate_and_clean_interactions(
    df: pd.DataFrame,
    rating_min: float = 1.0,
    rating_max: float = 5.0,
    drop_missing_timestamps: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Perform strict validation and cleaning on interactions data.
    
    Args:
        df: Raw interactions DataFrame
        rating_min: Minimum valid rating value (default: 1.0)
        rating_max: Maximum valid rating value (default: 5.0)
        drop_missing_timestamps: If True, drop rows with NaT timestamps (default: True)
    
    Returns:
        Tuple of (cleaned_df, cleaning_stats)
    """
    auditor = DataAuditor(
        rating_min=rating_min,
        rating_max=rating_max,
        drop_missing_timestamps=drop_missing_timestamps
    )
    return auditor.validate_and_clean(df)


def deduplicate_interactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Remove duplicate (user_id, product_id) pairs.
    
    Strategy:
    - Keep interaction with most recent cmt_date
    - If cmt_date is equal, keep highest rating
    
    Args:
        df: DataFrame with potentially duplicate interactions
    
    Returns:
        Tuple of (deduplicated_df, dedup_stats)
    """
    auditor = DataAuditor()
    return auditor.deduplicate_interactions(df)


def detect_outliers(
    df: pd.DataFrame,
    max_user_interactions: int = 500,
    min_item_interactions: int = 3,
    rating_bias_threshold: float = 0.90
) -> Dict[str, any]:
    """
    Detect outliers in the data (users, items, rating bias).
    
    Args:
        df: DataFrame with cleaned interactions
        max_user_interactions: Threshold for bot detection (default: 500)
        min_item_interactions: Threshold for cold items (default: 3)
        rating_bias_threshold: Threshold for rating bias detection (default: 0.90)
    
    Returns:
        Dictionary containing outlier analysis results
    """
    auditor = DataAuditor()
    return auditor.detect_outliers(
        df,
        max_user_interactions=max_user_interactions,
        min_item_interactions=min_item_interactions,
        rating_bias_threshold=rating_bias_threshold
    )


def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute MD5 hash of DataFrame for versioning.
    
    Args:
        df: DataFrame to hash
    
    Returns:
        MD5 hash string
    """
    return DataAuditor.compute_data_hash(df)


def log_data_quality_report(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Log comprehensive data quality metrics.
    
    Args:
        df: DataFrame to analyze
        name: Name for logging context
    """
    auditor = DataAuditor()
    auditor.generate_quality_report(df, name)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    """
    Test the data loading and cleaning pipeline.
    """
    print("Testing Data Layer Module - Step 1.1")
    print("="*80)
    
    try:
        # Create processor
        processor = DataProcessor(
            base_path="data/published_data",
            rating_min=1.0,
            rating_max=5.0,
            drop_missing_timestamps=True
        )
        
        # Load and validate interactions
        df_clean, stats = processor.load_and_validate_interactions()
        
        # Generate quality report
        processor.generate_quality_report(df_clean, "Cleaned Interactions")
        
        # Compute data hash
        data_hash = processor.compute_data_hash(df_clean)
        logger.info(f"Data hash (MD5): {data_hash}")
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        print("\n" + "="*80)
        print("TEST FAILED")
        print("="*80)
        raise
