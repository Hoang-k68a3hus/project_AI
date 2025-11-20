"""
Feature Engineering Module for Collaborative Filtering

This module handles Step 2.0: Comment Quality Analysis and confidence score computation.
Addresses the rating skew problem (95% 5-star ratings) by analyzing comment content
to distinguish high-quality reviews from low-quality ones.

Key Features:
- Comment quality scoring based on length and keyword analysis
- Confidence score computation (rating + comment_quality)
- Vietnamese keyword support for sentiment analysis
- Handles missing/empty comments gracefully
"""

import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np


logger = logging.getLogger("data_layer")


class FeatureEngineer:
    """
    Class for engineering features from interaction data.
    
    This class handles:
    - Comment quality analysis (Step 2.0)
    - Confidence score computation for ALS
    - Positive/negative signal labeling for BPR
    
    The main purpose is to address the 95% 5-star rating skew by extracting
    additional quality signals from review comments.
    """
    
    def __init__(
        self,
        positive_threshold: float = 4.0,
        hard_negative_threshold: float = 3.0
    ):
        """
        Initialize FeatureEngineer.
        
        Args:
            positive_threshold: Rating threshold for positive interactions (default: 4.0)
            hard_negative_threshold: Rating threshold for hard negatives (default: 3.0)
        """
        self.positive_threshold = positive_threshold
        self.hard_negative_threshold = hard_negative_threshold
        
        # Vietnamese positive keywords for cosmetics domain
        self.positive_keywords = [
            'thấm nhanh', 'hiệu quả', 'thơm', 'mịn', 'sáng da',
            'trắng da', 'giảm mụn', 'không kích ứng', 'tốt',
            'rất thích', 'đáng mua', 'chất lượng', 'xuất sắc',
            'hài lòng', 'ưng ý', 'tuyệt vời', 'hoàn hảo',
            'mượt mà', 'tươi sáng', 'ẩm', 'mát', 'dễ chịu'
        ]
    
    def compute_comment_quality_score(self, comment_text: str) -> float:
        """
        Compute quality bonus based on review comment content.
        
        Strategy:
        - Empty/missing comments: 0.0 (no additional signal)
        - Length bonus: +0.1 for ≥5 words, +0.2 for ≥10 words (thoughtful feedback)
        - Keyword bonus: +0.1 per positive keyword, max +0.3 (sentiment signal)
        - Result capped at 1.0
        
        Args:
            comment_text: Review comment text (may be NaN or empty)
        
        Returns:
            float: Quality score in range [0.0, 1.0]
        
        Example:
            >>> fe = FeatureEngineer()
            >>> fe.compute_comment_quality_score("Sản phẩm rất tốt, thấm nhanh, hiệu quả")
            0.5  # 0.2 (length) + 0.3 (3 keywords)
        """
        # Handle missing or empty comments
        if pd.isna(comment_text):
            return 0.0
        
        comment_str = str(comment_text).strip()
        if len(comment_str) == 0:
            return 0.0
        
        score = 0.0
        text_lower = comment_str.lower()
        
        # Length bonus (thoughtful reviews tend to be longer)
        word_count = len(comment_str.split())
        if word_count >= 10:
            score += 0.2  # Detailed review
        elif word_count >= 5:
            score += 0.1  # Moderate length
        
        # Positive keyword bonus (sentiment analysis for Vietnamese)
        keyword_matches = sum(1 for kw in self.positive_keywords if kw in text_lower)
        keyword_score = min(keyword_matches * 0.1, 0.3)  # Max 0.3 from keywords
        score += keyword_score
        
        # Image bonus placeholder (if data becomes available in future)
        # This would require adding has_images column to dataset
        # if has_images:
        #     score += 0.5
        
        # Cap at 1.0 to ensure predictable range
        return min(score, 1.0)
    
    def compute_confidence_scores(
        self, 
        df: pd.DataFrame,
        comment_column: str = 'processed_comment'
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Compute comment quality and confidence scores for all interactions.
        
        This implements Step 2.0 of the data preprocessing pipeline:
        - Computes comment_quality score [0.0, 1.0] for each interaction
        - Computes confidence_score = rating + comment_quality [1.0, 6.0]
        - Rationale: Distinguish "truly loved" products from "just okay" despite 95% 5-star skew
        
        Args:
            df: DataFrame with interactions (must have 'rating' and comment column)
            comment_column: Name of the comment column (default: 'processed_comment')
        
        Returns:
            Tuple of (enriched_df, stats)
            - enriched_df: DataFrame with added 'comment_quality' and 'confidence_score' columns
            - stats: Dictionary with quality score distribution statistics
        
        Example:
            >>> fe = FeatureEngineer()
            >>> df_enriched, stats = fe.compute_confidence_scores(df)
            >>> print(f"Mean confidence score: {stats['confidence_score_mean']:.2f}")
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2.0: COMMENT QUALITY ANALYSIS")
        logger.info("="*80)
        logger.info("Addressing rating skew problem: 95% 5-star ratings")
        logger.info("Strategy: Extract quality signals from review comments")
        
        initial_count = len(df)
        logger.info(f"Processing {initial_count:,} interactions")
        
        # Validate required columns
        if 'rating' not in df.columns:
            raise ValueError("DataFrame must contain 'rating' column")
        
        # Check if comment column exists, use fallback if not
        if comment_column not in df.columns:
            logger.warning(f"Column '{comment_column}' not found, checking alternatives...")
            if 'comment' in df.columns:
                comment_column = 'comment'
                logger.info(f"Using 'comment' column instead")
            else:
                logger.warning("No comment column found - all quality scores will be 0.0")
                df['comment_quality'] = 0.0
                df['confidence_score'] = df['rating']
                return df, {
                    'comment_column_used': None,
                    'rows_with_comments': 0,
                    'rows_without_comments': initial_count
                }
        
        # Compute comment quality scores
        logger.info(f"\nComputing comment quality scores from '{comment_column}'...")
        logger.info("Quality scoring criteria:")
        logger.info("  - Length bonus: +0.1 (≥5 words), +0.2 (≥10 words)")
        logger.info("  - Keyword bonus: +0.1 per positive keyword (max +0.3)")
        logger.info(f"  - Vocabulary: {len(self.positive_keywords)} Vietnamese keywords")
        
        df['comment_quality'] = df[comment_column].apply(self.compute_comment_quality_score)
        
        # Compute confidence scores
        logger.info("\nComputing confidence scores = rating + comment_quality...")
        df['confidence_score'] = df['rating'] + df['comment_quality']
        
        # Compute statistics
        stats = self._compute_quality_stats(df, comment_column)
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("COMMENT QUALITY SUMMARY")
        logger.info("="*80)
        logger.info(f"Total interactions:        {stats['total_interactions']:>12,}")
        logger.info(f"Rows with comments:        {stats['rows_with_comments']:>12,} ({stats['comment_coverage']:.2%})")
        logger.info(f"Rows without comments:     {stats['rows_without_comments']:>12,} ({stats['no_comment_rate']:.2%})")
        logger.info("\nComment Quality Distribution:")
        logger.info(f"  Min:    {stats['quality_min']:.3f}")
        logger.info(f"  Mean:   {stats['quality_mean']:.3f}")
        logger.info(f"  Median: {stats['quality_median']:.3f}")
        logger.info(f"  Max:    {stats['quality_max']:.3f}")
        logger.info(f"  Std:    {stats['quality_std']:.3f}")
        logger.info("\nConfidence Score Distribution:")
        logger.info(f"  Min:    {stats['confidence_score_min']:.3f}")
        logger.info(f"  Mean:   {stats['confidence_score_mean']:.3f}")
        logger.info(f"  Median: {stats['confidence_score_median']:.3f}")
        logger.info(f"  Max:    {stats['confidence_score_max']:.3f}")
        logger.info(f"  Std:    {stats['confidence_score_std']:.3f}")
        logger.info("\nQuality Score Breakdown:")
        logger.info(f"  Zero quality (0.0):        {stats['zero_quality_count']:>12,} ({stats['zero_quality_pct']:.2%})")
        logger.info(f"  Low quality (0.0-0.2):     {stats['low_quality_count']:>12,} ({stats['low_quality_pct']:.2%})")
        logger.info(f"  Medium quality (0.2-0.5):  {stats['medium_quality_count']:>12,} ({stats['medium_quality_pct']:.2%})")
        logger.info(f"  High quality (0.5-1.0):    {stats['high_quality_count']:>12,} ({stats['high_quality_pct']:.2%})")
        logger.info("="*80 + "\n")
        
        # Validate results
        self._validate_scores(df)
        logger.info("✓ All quality and confidence scores validated")
        
        return df, stats
    
    def _compute_quality_stats(self, df: pd.DataFrame, comment_column: str) -> Dict[str, any]:
        """
        Compute comprehensive statistics for quality and confidence scores.
        
        Args:
            df: DataFrame with comment_quality and confidence_score columns
            comment_column: Name of the comment column used
        
        Returns:
            Dictionary with detailed statistics
        """
        # Check for non-empty comments
        has_comment = df[comment_column].notna() & (df[comment_column].astype(str).str.strip() != '')
        rows_with_comments = has_comment.sum()
        rows_without_comments = (~has_comment).sum()
        total = len(df)
        
        # Quality score distribution
        quality_scores = df['comment_quality']
        zero_quality = (quality_scores == 0.0).sum()
        low_quality = ((quality_scores > 0.0) & (quality_scores <= 0.2)).sum()
        medium_quality = ((quality_scores > 0.2) & (quality_scores <= 0.5)).sum()
        high_quality = (quality_scores > 0.5).sum()
        
        # Confidence score distribution
        confidence_scores = df['confidence_score']
        
        stats = {
            'comment_column_used': comment_column,
            'total_interactions': total,
            'rows_with_comments': int(rows_with_comments),
            'rows_without_comments': int(rows_without_comments),
            'comment_coverage': float(rows_with_comments / total),
            'no_comment_rate': float(rows_without_comments / total),
            
            # Quality score statistics
            'quality_min': float(quality_scores.min()),
            'quality_max': float(quality_scores.max()),
            'quality_mean': float(quality_scores.mean()),
            'quality_median': float(quality_scores.median()),
            'quality_std': float(quality_scores.std()),
            'quality_p25': float(quality_scores.quantile(0.25)),
            'quality_p75': float(quality_scores.quantile(0.75)),
            
            # Quality breakdown
            'zero_quality_count': int(zero_quality),
            'zero_quality_pct': float(zero_quality / total),
            'low_quality_count': int(low_quality),
            'low_quality_pct': float(low_quality / total),
            'medium_quality_count': int(medium_quality),
            'medium_quality_pct': float(medium_quality / total),
            'high_quality_count': int(high_quality),
            'high_quality_pct': float(high_quality / total),
            
            # Confidence score statistics
            'confidence_score_min': float(confidence_scores.min()),
            'confidence_score_max': float(confidence_scores.max()),
            'confidence_score_mean': float(confidence_scores.mean()),
            'confidence_score_median': float(confidence_scores.median()),
            'confidence_score_std': float(confidence_scores.std()),
            'confidence_score_p25': float(confidence_scores.quantile(0.25)),
            'confidence_score_p75': float(confidence_scores.quantile(0.75)),
            'confidence_score_p01': float(confidence_scores.quantile(0.01)),
            'confidence_score_p99': float(confidence_scores.quantile(0.99))
        }
        
        return stats
    
    def _validate_scores(self, df: pd.DataFrame) -> None:
        """
        Validate that quality and confidence scores are within expected ranges.
        
        Args:
            df: DataFrame with comment_quality and confidence_score columns
        
        Raises:
            AssertionError: If validation fails
        """
        # Validate comment_quality range [0.0, 1.0]
        assert df['comment_quality'].min() >= 0.0, "comment_quality has values < 0.0"
        assert df['comment_quality'].max() <= 1.0, "comment_quality has values > 1.0"
        assert df['comment_quality'].notna().all(), "comment_quality contains NaN values"
        
        # Validate confidence_score range [1.0, 6.0]
        # Min should be ≥1.0 (min rating=1.0, min quality=0.0)
        # Max should be ≤6.0 (max rating=5.0, max quality=1.0)
        min_confidence = df['confidence_score'].min()
        max_confidence = df['confidence_score'].max()
        
        assert min_confidence >= 1.0, f"confidence_score has values < 1.0: {min_confidence}"
        assert max_confidence <= 6.0, f"confidence_score has values > 6.0: {max_confidence}"
        assert df['confidence_score'].notna().all(), "confidence_score contains NaN values"
        
        # Validate relationship: confidence_score = rating + comment_quality
        computed_confidence = df['rating'] + df['comment_quality']
        diff = (df['confidence_score'] - computed_confidence).abs()
        max_diff = diff.max()
        
        assert max_diff < 1e-6, f"confidence_score != rating + comment_quality (max diff: {max_diff})"
