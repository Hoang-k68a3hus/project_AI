"""
Data Auditor Module for Collaborative Filtering

This module handles data validation, cleaning, and quality auditing
for the Vietnamese cosmetics recommender system.
"""

import logging
import hashlib
from typing import Dict, Tuple

import pandas as pd
import numpy as np


logger = logging.getLogger("data_layer")


class DataAuditor:
    """
    Class for auditing, validating, and cleaning interaction data.
    
    This class handles:
    - Type enforcement (user_id, product_id, rating, timestamps)
    - Missing value detection and handling
    - Temporal validation (no NaT timestamps)
    - Rating range validation [1.0, 5.0]
    - Data quality reporting
    - Data versioning with hash computation
    """
    
    def __init__(
        self, 
        rating_min: float = 1.0,
        rating_max: float = 5.0,
        drop_missing_timestamps: bool = True
    ):
        """
        Initialize DataAuditor.
        
        Args:
            rating_min: Minimum valid rating value (default: 1.0)
            rating_max: Maximum valid rating value (default: 5.0)
            drop_missing_timestamps: If True, drop rows with NaT timestamps (default: True)
        """
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.drop_missing_timestamps = drop_missing_timestamps
        
        # Date formats to try for parsing
        self.date_formats = [
            '%Y-%m-%d %H:%M:%S',  # ISO format with time
            '%d/%m/%Y',           # DD/MM/YYYY
            '%Y-%m-%d',           # ISO date only
            '%d-%m-%Y'            # DD-MM-YYYY
        ]
    
    def validate_and_clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Perform strict validation and cleaning on interactions data.
        
        This implements Step 1.1 of the data preprocessing pipeline:
        - Type enforcement (user_id, product_id as int; rating as float)
        - Missing value handling (drop rows with missing critical fields)
        - Temporal validation (drop rows with NaT/Null timestamps - CRITICAL)
        - Rating range validation (enforce [1.0, 5.0])
        
        Args:
            df: Raw interactions DataFrame
        
        Returns:
            Tuple of (cleaned_df, cleaning_stats)
            - cleaned_df: Validated and cleaned DataFrame
            - cleaning_stats: Dict with counts of dropped rows per reason
        
        Example:
            >>> auditor = DataAuditor()
            >>> df_clean, stats = auditor.validate_and_clean(df_raw)
            >>> print(f"Dropped {stats['missing_timestamp']} rows due to missing timestamps")
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1.1: DATA VALIDATION & CLEANING")
        logger.info("="*80)
        
        initial_count = len(df)
        logger.info(f"Initial row count: {initial_count:,}")
        
        # Initialize cleaning statistics
        stats = {
            'initial_count': initial_count,
            'missing_user_id': 0,
            'missing_product_id': 0,
            'missing_rating': 0,
            'missing_timestamp': 0,
            'invalid_rating_range': 0,
            'type_conversion_errors': 0,
            'final_count': 0
        }
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Step 1: Type Enforcement
        df_clean, stats = self._enforce_types(df_clean, stats)
        
        # Step 2: Missing Values Handling
        df_clean, stats = self._handle_missing_values(df_clean, stats)
        
        # Step 3: Rating Validation
        df_clean, stats = self._validate_ratings(df_clean, stats)
        
        # Step 4: Final Statistics
        stats = self._compute_final_stats(df_clean, stats)
        
        # Step 5: Validate final state
        self._validate_final_state(df_clean)
        
        logger.info("✓ All validation checks passed")
        
        return df_clean, stats
    
    def _enforce_types(self, df: pd.DataFrame, stats: Dict[str, int]) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Enforce correct data types for all columns.
        
        Args:
            df: DataFrame to process
            stats: Statistics dictionary to update
        
        Returns:
            Tuple of (processed_df, updated_stats)
        """
        logger.info("\n" + "-"*80)
        logger.info("1. TYPE ENFORCEMENT")
        logger.info("-"*80)
        
        # 1.1 Enforce user_id as integer
        logger.info("Converting user_id to integer...")
        df, dropped = self._convert_to_int(df, 'user_id')
        stats['type_conversion_errors'] += dropped
        if dropped > 0:
            logger.warning(f"⚠ Dropped {dropped:,} rows due to invalid user_id")
        else:
            logger.info("✓ All user_id values are valid integers")
        
        # 1.2 Enforce product_id as integer
        logger.info("Converting product_id to integer...")
        df, dropped = self._convert_to_int(df, 'product_id')
        stats['type_conversion_errors'] += dropped
        if dropped > 0:
            logger.warning(f"⚠ Dropped {dropped:,} rows due to invalid product_id")
        else:
            logger.info("✓ All product_id values are valid integers")
        
        # 1.3 Enforce rating as float
        logger.info("Converting rating to float...")
        df, dropped = self._convert_to_float(df, 'rating')
        stats['type_conversion_errors'] += dropped
        if dropped > 0:
            logger.warning(f"⚠ Dropped {dropped:,} rows due to invalid rating type")
        else:
            logger.info("✓ All rating values are valid floats")
        
        # 1.4 Parse cmt_date as datetime
        df = self._parse_datetime(df, 'cmt_date')
        
        return df, stats
    
    def _convert_to_int(self, df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, int]:
        """
        Convert column to integer, dropping invalid values.
        
        Args:
            df: DataFrame
            column: Column name to convert
        
        Returns:
            Tuple of (processed_df, dropped_count)
        """
        initial_len = len(df)
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df = df[df[column].notna()]
        df[column] = df[column].astype(int)
        dropped = initial_len - len(df)
        return df, dropped
    
    def _convert_to_float(self, df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, int]:
        """
        Convert column to float, dropping invalid values.
        
        Args:
            df: DataFrame
            column: Column name to convert
        
        Returns:
            Tuple of (processed_df, dropped_count)
        """
        initial_len = len(df)
        df[column] = pd.to_numeric(df[column], errors='coerce')
        dropped = initial_len - len(df)
        return df, dropped
    
    def _parse_datetime(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Parse datetime column with multiple format attempts.
        
        Args:
            df: DataFrame
            column: Column name to parse
        
        Returns:
            DataFrame with parsed datetime column
        """
        logger.info("Parsing cmt_date to datetime...")
        
        parsed_dates = None
        successful_format = None
        
        # Try each date format
        for date_format in self.date_formats:
            try:
                test_parse = pd.to_datetime(
                    df[column], 
                    format=date_format,
                    errors='coerce'
                )
                
                valid_dates = test_parse.notna().sum()
                if valid_dates > 0:
                    parsed_dates = test_parse
                    successful_format = date_format
                    logger.info(f"✓ Parsed {valid_dates:,} dates using format: {date_format}")
                    break
            except Exception as e:
                logger.debug(f"Failed to parse with format {date_format}: {str(e)}")
                continue
        
        # If no specific format worked, try auto-detection
        if parsed_dates is None or parsed_dates.notna().sum() == 0:
            logger.info("Trying auto-detection for date parsing...")
            parsed_dates = pd.to_datetime(df[column], errors='coerce')
            valid_dates = parsed_dates.notna().sum()
            if valid_dates > 0:
                successful_format = 'auto-detect'
                logger.info(f"✓ Parsed {valid_dates:,} dates using auto-detection")
        
        df[column] = parsed_dates
        
        if successful_format:
            logger.info(f"✓ Successfully parsed dates using: {successful_format}")
        else:
            logger.warning("⚠ Failed to parse any dates - all timestamps will be NaT")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, stats: Dict[str, int]) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Handle missing values in critical columns.
        
        Args:
            df: DataFrame to process
            stats: Statistics dictionary to update
        
        Returns:
            Tuple of (processed_df, updated_stats)
        """
        logger.info("\n" + "-"*80)
        logger.info("2. MISSING VALUES HANDLING")
        logger.info("-"*80)
        
        # 2.1 Drop rows with missing user_id
        missing_user = df['user_id'].isna().sum()
        if missing_user > 0:
            df = df[df['user_id'].notna()]
            stats['missing_user_id'] = missing_user
            logger.warning(f"⚠ Dropped {missing_user:,} rows with missing user_id")
        else:
            logger.info("✓ No missing user_id values")
        
        # 2.2 Drop rows with missing product_id
        missing_product = df['product_id'].isna().sum()
        if missing_product > 0:
            df = df[df['product_id'].notna()]
            stats['missing_product_id'] = missing_product
            logger.warning(f"⚠ Dropped {missing_product:,} rows with missing product_id")
        else:
            logger.info("✓ No missing product_id values")
        
        # 2.3 Drop rows with missing rating
        missing_rating = df['rating'].isna().sum()
        if missing_rating > 0:
            df = df[df['rating'].notna()]
            stats['missing_rating'] = missing_rating
            logger.warning(f"⚠ Dropped {missing_rating:,} rows with missing rating")
        else:
            logger.info("✓ No missing rating values")
        
        # 2.4 CRITICAL: Drop rows with missing timestamps (Time Travel Fix)
        missing_timestamp = df['cmt_date'].isna().sum()
        if self.drop_missing_timestamps:
            if missing_timestamp > 0:
                logger.warning("\n" + "!"*80)
                logger.warning("CRITICAL: TEMPORAL VALIDATION FAILURE")
                logger.warning(f"Found {missing_timestamp:,} rows with NaT/Null timestamps")
                logger.warning("Dropping these rows to prevent data leakage in train/test split")
                logger.warning("Reason: Model must not 'see the future' - temporal ordering is essential")
                logger.warning("!"*80 + "\n")
                
                df = df[df['cmt_date'].notna()]
                stats['missing_timestamp'] = missing_timestamp
            else:
                logger.info("✓ No missing timestamp values (temporal validation passed)")
        else:
            if missing_timestamp > 0:
                logger.warning(f"⚠ Found {missing_timestamp:,} rows with missing timestamps (not dropped per config)")
        
        return df, stats
    
    def _validate_ratings(self, df: pd.DataFrame, stats: Dict[str, int]) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Validate rating values are within acceptable range.
        
        Args:
            df: DataFrame to process
            stats: Statistics dictionary to update
        
        Returns:
            Tuple of (processed_df, updated_stats)
        """
        logger.info("\n" + "-"*80)
        logger.info("3. RATING RANGE VALIDATION")
        logger.info("-"*80)
        logger.info(f"Valid rating range: [{self.rating_min}, {self.rating_max}]")
        
        # Check rating distribution before filtering
        logger.info(f"Rating statistics before validation:")
        logger.info(f"  Min: {df['rating'].min():.2f}")
        logger.info(f"  Max: {df['rating'].max():.2f}")
        logger.info(f"  Mean: {df['rating'].mean():.2f}")
        logger.info(f"  Median: {df['rating'].median():.2f}")
        
        # Filter invalid ratings
        invalid_ratings = (df['rating'] < self.rating_min) | (df['rating'] > self.rating_max)
        invalid_count = invalid_ratings.sum()
        
        if invalid_count > 0:
            logger.warning(f"⚠ Found {invalid_count:,} ratings outside valid range")
            logger.warning(f"  Below minimum ({self.rating_min}): {(df['rating'] < self.rating_min).sum():,}")
            logger.warning(f"  Above maximum ({self.rating_max}): {(df['rating'] > self.rating_max).sum():,}")
            
            df = df[~invalid_ratings]
            stats['invalid_rating_range'] = invalid_count
            
            logger.info(f"✓ Dropped {invalid_count:,} rows with invalid ratings")
        else:
            logger.info(f"✓ All ratings within valid range [{self.rating_min}, {self.rating_max}]")
        
        return df, stats
    
    def _compute_final_stats(self, df: pd.DataFrame, stats: Dict[str, int]) -> Dict[str, int]:
        """
        Compute final cleaning statistics.
        
        Args:
            df: Final processed DataFrame
            stats: Statistics dictionary to update
        
        Returns:
            Updated statistics dictionary
        """
        stats['final_count'] = len(df)
        stats['total_dropped'] = stats['initial_count'] - stats['final_count']
        stats['retention_rate'] = (stats['final_count'] / stats['initial_count']) * 100
        
        logger.info("\n" + "="*80)
        logger.info("CLEANING SUMMARY")
        logger.info("="*80)
        logger.info(f"Initial rows:              {stats['initial_count']:>12,}")
        logger.info(f"Missing user_id:           {stats['missing_user_id']:>12,}")
        logger.info(f"Missing product_id:        {stats['missing_product_id']:>12,}")
        logger.info(f"Missing rating:            {stats['missing_rating']:>12,}")
        logger.info(f"Missing timestamp:         {stats['missing_timestamp']:>12,}")
        logger.info(f"Invalid rating range:      {stats['invalid_rating_range']:>12,}")
        logger.info(f"Type conversion errors:    {stats['type_conversion_errors']:>12,}")
        logger.info("-"*80)
        logger.info(f"Total dropped:             {stats['total_dropped']:>12,}")
        logger.info(f"Final rows:                {stats['final_count']:>12,}")
        logger.info(f"Retention rate:            {stats['retention_rate']:>11.2f}%")
        logger.info("="*80 + "\n")
        
        return stats
    
    def _validate_final_state(self, df: pd.DataFrame) -> None:
        """
        Validate final state of cleaned DataFrame.
        
        Args:
            df: Cleaned DataFrame to validate
        
        Raises:
            AssertionError: If validation fails
        """
        assert df['user_id'].notna().all(), "Final data contains null user_id"
        assert df['product_id'].notna().all(), "Final data contains null product_id"
        assert df['rating'].notna().all(), "Final data contains null rating"
        assert (df['rating'] >= self.rating_min).all(), f"Final data contains ratings < {self.rating_min}"
        assert (df['rating'] <= self.rating_max).all(), f"Final data contains ratings > {self.rating_max}"
        
        if self.drop_missing_timestamps:
            assert df['cmt_date'].notna().all(), "Final data contains NaT timestamps despite drop_missing_timestamps=True"
    
    def generate_quality_report(self, df: pd.DataFrame, name: str = "DataFrame") -> None:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            name: Name for logging context
        """
        logger.info("\n" + "="*80)
        logger.info(f"DATA QUALITY REPORT: {name}")
        logger.info("="*80)
        
        logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        logger.info("\nMissing values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            logger.info("  ✓ No missing values")
        else:
            for col, count in missing[missing > 0].items():
                pct = (count / len(df)) * 100
                logger.info(f"  {col}: {count:,} ({pct:.2f}%)")
        
        # Data types
        logger.info("\nData types:")
        for col, dtype in df.dtypes.items():
            logger.info(f"  {col}: {dtype}")
        
        # Rating distribution
        if 'rating' in df.columns:
            logger.info("\nRating distribution:")
            rating_dist = df['rating'].value_counts().sort_index()
            for rating, count in rating_dist.items():
                pct = (count / len(df)) * 100
                logger.info(f"  Rating {rating}: {count:,} ({pct:.2f}%)")
        
        logger.info("="*80 + "\n")
    
    def deduplicate_interactions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Remove duplicate (user_id, product_id) pairs following Step 1.2 strategy.
        
        Strategy:
        - Each (user_id, product_id) pair should have only 1 interaction
        - Keep the interaction with the most recent cmt_date
        - If cmt_date is equal, keep the one with highest rating
        
        Args:
            df: DataFrame with potentially duplicate interactions
        
        Returns:
            Tuple of (deduplicated_df, dedup_stats)
            - deduplicated_df: DataFrame with duplicates removed
            - dedup_stats: Dictionary with deduplication statistics
        
        Example:
            >>> auditor = DataAuditor()
            >>> df_dedup, stats = auditor.deduplicate_interactions(df)
            >>> print(f"Removed {stats['duplicates_removed']} duplicate interactions")
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1.2: DEDUPLICATION")
        logger.info("="*80)
        
        initial_count = len(df)
        logger.info(f"Initial row count: {initial_count:,}")
        
        # Check for duplicates
        duplicate_mask = df.duplicated(subset=['user_id', 'product_id'], keep=False)
        num_duplicates = duplicate_mask.sum()
        num_unique_pairs = df[duplicate_mask].groupby(['user_id', 'product_id']).ngroups
        
        logger.info(f"Found {num_duplicates:,} duplicate rows involving {num_unique_pairs:,} unique (user_id, product_id) pairs")
        
        if num_duplicates == 0:
            logger.info("✓ No duplicates found - data is clean")
            return df, {
                'initial_count': initial_count,
                'duplicates_found': 0,
                'duplicates_removed': 0,
                'final_count': initial_count,
                'unique_pairs_affected': 0
            }
        
        # Sort by cmt_date (descending) then rating (descending)
        # This ensures the "best" row is first for each (user_id, product_id) pair
        logger.info("Sorting by cmt_date (desc) and rating (desc)...")
        df_sorted = df.sort_values(
            by=['user_id', 'product_id', 'cmt_date', 'rating'],
            ascending=[True, True, False, False]
        )
        
        # Keep first occurrence (most recent, highest rating)
        logger.info("Removing duplicates (keeping most recent interaction with highest rating)...")
        df_dedup = df_sorted.drop_duplicates(subset=['user_id', 'product_id'], keep='first')
        
        final_count = len(df_dedup)
        duplicates_removed = initial_count - final_count
        
        stats = {
            'initial_count': initial_count,
            'duplicates_found': num_duplicates,
            'duplicates_removed': duplicates_removed,
            'final_count': final_count,
            'unique_pairs_affected': num_unique_pairs,
            'retention_rate': (final_count / initial_count) * 100
        }
        
        logger.info("\n" + "="*80)
        logger.info("DEDUPLICATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Initial rows:              {stats['initial_count']:>12,}")
        logger.info(f"Duplicate rows found:      {stats['duplicates_found']:>12,}")
        logger.info(f"Unique pairs affected:     {stats['unique_pairs_affected']:>12,}")
        logger.info(f"Duplicates removed:        {stats['duplicates_removed']:>12,}")
        logger.info(f"Final rows:                {stats['final_count']:>12,}")
        logger.info(f"Retention rate:            {stats['retention_rate']:>11.2f}%")
        logger.info("="*80 + "\n")
        
        # Validate: No duplicates remaining
        remaining_duplicates = df_dedup.duplicated(subset=['user_id', 'product_id']).sum()
        assert remaining_duplicates == 0, f"Deduplication failed: {remaining_duplicates} duplicates still exist"
        logger.info("✓ Deduplication validation passed - no duplicates remaining")
        
        return df_dedup, stats
    
    def detect_outliers(
        self, 
        df: pd.DataFrame,
        max_user_interactions: int = 500,
        min_item_interactions: int = 3,
        rating_bias_threshold: float = 0.90
    ) -> Dict[str, any]:
        """
        Detect outliers in the data following Step 1.3 strategy.
        
        This method identifies but does NOT filter outliers:
        - Users with >max_user_interactions (potential bots/scrapers)
        - Items with <min_item_interactions (very cold items)
        - Rating bias (>rating_bias_threshold of ratings = 5)
        
        Args:
            df: DataFrame with cleaned interactions
            max_user_interactions: Threshold for bot detection (default: 500)
            min_item_interactions: Threshold for cold items (default: 3)
            rating_bias_threshold: Threshold for rating bias detection (default: 0.90)
        
        Returns:
            Dictionary containing outlier analysis results:
            - 'bot_users': Set of user_ids with suspicious activity
            - 'cold_items': Set of product_ids with too few interactions
            - 'rating_bias': Dict with rating distribution analysis
            - 'stats': Summary statistics
        
        Example:
            >>> auditor = DataAuditor()
            >>> outliers = auditor.detect_outliers(df)
            >>> print(f"Found {len(outliers['bot_users'])} potential bots")
            >>> print(f"Found {len(outliers['cold_items'])} cold items")
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 1.3: OUTLIER DETECTION")
        logger.info("="*80)
        logger.info("NOTE: This step only IDENTIFIES outliers, does not filter them")
        logger.info(f"Configuration:")
        logger.info(f"  - Max user interactions: {max_user_interactions}")
        logger.info(f"  - Min item interactions: {min_item_interactions}")
        logger.info(f"  - Rating bias threshold: {rating_bias_threshold:.0%}")
        
        # 1. User Activity Analysis - Detect potential bots
        logger.info("\n" + "-"*80)
        logger.info("1. USER ACTIVITY ANALYSIS (Bot Detection)")
        logger.info("-"*80)
        
        user_interaction_counts = df.groupby('user_id').size()
        bot_users = set(user_interaction_counts[user_interaction_counts > max_user_interactions].index)
        
        logger.info(f"User interaction statistics:")
        logger.info(f"  Total users: {len(user_interaction_counts):,}")
        logger.info(f"  Mean interactions per user: {user_interaction_counts.mean():.2f}")
        logger.info(f"  Median interactions per user: {user_interaction_counts.median():.1f}")
        logger.info(f"  Max interactions per user: {user_interaction_counts.max():,}")
        
        if len(bot_users) > 0:
            logger.warning(f"\n⚠ Found {len(bot_users):,} potential bot/scraper users (>{max_user_interactions} interactions)")
            logger.warning(f"  These users account for {df[df['user_id'].isin(bot_users)].shape[0]:,} interactions")
            
            # Show top 5 most active users
            top_bots = user_interaction_counts.nlargest(5)
            logger.warning("  Top 5 most active users:")
            for user_id, count in top_bots.items():
                logger.warning(f"    user_id {user_id}: {count:,} interactions")
        else:
            logger.info(f"✓ No bot users detected (all users have ≤{max_user_interactions} interactions)")
        
        # 2. Item Popularity Analysis - Detect cold items
        logger.info("\n" + "-"*80)
        logger.info("2. ITEM POPULARITY ANALYSIS (Cold Item Detection)")
        logger.info("-"*80)
        
        item_interaction_counts = df.groupby('product_id').size()
        cold_items = set(item_interaction_counts[item_interaction_counts < min_item_interactions].index)
        
        logger.info(f"Item interaction statistics:")
        logger.info(f"  Total items: {len(item_interaction_counts):,}")
        logger.info(f"  Mean interactions per item: {item_interaction_counts.mean():.2f}")
        logger.info(f"  Median interactions per item: {item_interaction_counts.median():.1f}")
        logger.info(f"  Min interactions per item: {item_interaction_counts.min():,}")
        
        if len(cold_items) > 0:
            logger.warning(f"\n⚠ Found {len(cold_items):,} very cold items (<{min_item_interactions} interactions)")
            logger.warning(f"  These items account for {df[df['product_id'].isin(cold_items)].shape[0]:,} interactions")
            
            # Distribution of cold items
            cold_item_dist = item_interaction_counts[item_interaction_counts < min_item_interactions].value_counts().sort_index()
            logger.warning("  Cold item distribution:")
            for interaction_count, num_items in cold_item_dist.items():
                logger.warning(f"    {num_items:,} items with exactly {interaction_count} interaction(s)")
        else:
            logger.info(f"✓ No cold items detected (all items have ≥{min_item_interactions} interactions)")
        
        # 3. Rating Distribution Analysis - Detect bias
        logger.info("\n" + "-"*80)
        logger.info("3. RATING DISTRIBUTION ANALYSIS (Bias Detection)")
        logger.info("-"*80)
        
        rating_counts = df['rating'].value_counts().sort_index()
        total_ratings = len(df)
        rating_distribution = {}
        
        logger.info("Rating distribution:")
        for rating in sorted(rating_counts.index):
            count = rating_counts[rating]
            pct = count / total_ratings
            rating_distribution[float(rating)] = {
                'count': int(count),
                'percentage': float(pct)
            }
            stars = "★" * int(rating)
            logger.info(f"  {stars} ({rating:.1f}): {count:>8,} ({pct:>6.2%})")
        
        # Check for rating bias
        has_bias = False
        bias_details = {}
        
        for rating, stats_item in rating_distribution.items():
            if stats_item['percentage'] > rating_bias_threshold:
                has_bias = True
                bias_details[rating] = stats_item['percentage']
                logger.warning(f"\n⚠ RATING BIAS DETECTED!")
                logger.warning(f"  Rating {rating} accounts for {stats_item['percentage']:.2%} of all ratings")
                logger.warning(f"  This exceeds the bias threshold of {rating_bias_threshold:.0%}")
                logger.warning(f"  This may indicate:")
                logger.warning(f"    - Authentic product quality (if high rating)")
                logger.warning(f"    - Review manipulation")
                logger.warning(f"    - Limited discriminative power for recommendations")
        
        if not has_bias:
            logger.info(f"\n✓ No significant rating bias detected (no rating >={rating_bias_threshold:.0%})")
        
        # Compile results
        outlier_report = {
            'bot_users': bot_users,
            'cold_items': cold_items,
            'rating_bias': {
                'has_bias': has_bias,
                'bias_details': bias_details,
                'distribution': rating_distribution,
                'threshold': rating_bias_threshold
            },
            'stats': {
                'total_interactions': total_ratings,
                'total_users': len(user_interaction_counts),
                'total_items': len(item_interaction_counts),
                'bot_users_count': len(bot_users),
                'bot_interactions_count': df[df['user_id'].isin(bot_users)].shape[0] if len(bot_users) > 0 else 0,
                'cold_items_count': len(cold_items),
                'cold_interactions_count': df[df['product_id'].isin(cold_items)].shape[0] if len(cold_items) > 0 else 0,
                'user_interactions': {
                    'mean': float(user_interaction_counts.mean()),
                    'median': float(user_interaction_counts.median()),
                    'max': int(user_interaction_counts.max()),
                    'min': int(user_interaction_counts.min())
                },
                'item_interactions': {
                    'mean': float(item_interaction_counts.mean()),
                    'median': float(item_interaction_counts.median()),
                    'max': int(item_interaction_counts.max()),
                    'min': int(item_interaction_counts.min())
                }
            }
        }
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("OUTLIER DETECTION SUMMARY")
        logger.info("="*80)
        logger.info(f"Potential bot users:       {len(bot_users):>12,} ({len(bot_users)/len(user_interaction_counts)*100:.2f}%)")
        logger.info(f"Very cold items:           {len(cold_items):>12,} ({len(cold_items)/len(item_interaction_counts)*100:.2f}%)")
        logger.info(f"Rating bias detected:      {str(has_bias):>12}")
        logger.info("="*80 + "\n")
        logger.info("✓ Outlier detection completed (no filtering applied)")
        
        return outlier_report
    
    @staticmethod
    def compute_data_hash(df: pd.DataFrame) -> str:
        """
        Compute MD5 hash of DataFrame for versioning.
        
        Args:
            df: DataFrame to hash
        
        Returns:
            MD5 hash string
        """
        # Sort by columns to ensure consistency
        df_sorted = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        
        # Convert to string and hash
        content = df_sorted.to_csv(index=False)
        hash_obj = hashlib.md5(content.encode('utf-8'))
        
        return hash_obj.hexdigest()
