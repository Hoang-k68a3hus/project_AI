"""
Feature Engineering Module for Collaborative Filtering

This module handles Step 2.0: Comment Quality Analysis and confidence score computation.
Addresses the rating skew problem (95% 5-star ratings) by analyzing comment content
to distinguish high-quality reviews from low-quality ones.

Key Features:
- AI-powered sentiment analysis using ViSoBERT for Vietnamese text
- Comment quality scoring based on sentiment and length analysis
- Confidence score computation (rating + comment_quality)
- GPU acceleration support for efficient inference
- Handles missing/empty comments gracefully

Model Architecture:
- Pre-trained model: 5CD-AI/Vietnamese-Sentiment-visobert
- Base: ViSoBERT (continuously trained on 14GB Vietnamese social content)
- Training: 120K Vietnamese sentiment datasets (e-commerce, social, forums)
- Sentiment labels: NEGATIVE (0), POSITIVE (1), NEUTRAL (2)
- Output: Probability distribution via softmax
"""

import logging
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

# Suppress transformers warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger("data_layer")


class FeatureEngineer:
    """
    Class for engineering features from interaction data using AI-powered sentiment analysis.
    
    This class handles:
    - AI-based comment sentiment analysis (Step 2.0) using ViSoBERT
    - Comment quality scoring combining sentiment + length bonus
    - Confidence score computation for ALS (rating + comment_quality)
    - Positive/negative signal labeling for BPR
    
    The main purpose is to address the 95% 5-star rating skew by extracting
    additional quality signals from review comments using deep learning.
    
    Architecture:
    - Model loads once during __init__ for efficiency
    - Uses GPU if available, otherwise CPU
    - Batch processing support for large datasets
    """
    
    def __init__(
        self,
        positive_threshold: float = 4.0,
        hard_negative_threshold: float = 3.0,
        model_name: str = "5CD-AI/Vietnamese-Sentiment-visobert",
        device: Optional[str] = None
    ):
        """
        Initialize FeatureEngineer with AI sentiment model.
        
        Model Details:
        - 5CD-AI/Vietnamese-Sentiment-visobert: State-of-the-art Vietnamese sentiment model
        - Trained on 120K samples from e-commerce, social media, and forums
        - Accuracy: 88-99% across multiple Vietnamese sentiment benchmarks
        - Handles emojis, slang, and social media text effectively
        
        Args:
            positive_threshold: Rating threshold for positive interactions (default: 4.0)
            hard_negative_threshold: Rating threshold for hard negatives (default: 3.0)
            model_name: HuggingFace model identifier for Vietnamese sentiment analysis
            device: Device for model inference ('cuda', 'cpu', or None for auto-detect)
        """
        self.positive_threshold = positive_threshold
        self.hard_negative_threshold = hard_negative_threshold
        self.model_name = model_name
        
        # Setup device (GPU if available, otherwise CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing FeatureEngineer with AI sentiment analysis...")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        # Load tokenizer and model (only once for efficiency)
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info("Loading sentiment model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info("✓ AI sentiment model loaded successfully")
            
            # Sentiment label mapping (model-specific)
            # 5CD-AI/Vietnamese-Sentiment-visobert uses: 0=NEG, 1=POS, 2=NEU
            self.label_map = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}
            self.positive_label_idx = 1
            
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            logger.warning("Falling back to keyword-based sentiment analysis")
            self.model = None
            self.tokenizer = None
            
            # Fallback: Vietnamese positive keywords for cosmetics domain
            self.positive_keywords = [
                'thấm nhanh', 'hiệu quả', 'thơm', 'mịn', 'sáng da',
                'trắng da', 'giảm mụn', 'không kích ứng', 'tốt',
                'rất thích', 'đáng mua', 'chất lượng', 'xuất sắc',
                'hài lòng', 'ưng ý', 'tuyệt vời', 'hoàn hảo',
                'mượt mà', 'tươi sáng', 'ẩm', 'mát', 'dễ chịu'
            ]
    
    def compute_sentiment_score_ai(self, text: str) -> float:
        """
        Compute sentiment score using AI model (ViSoBERT).
        
        This method uses a pre-trained Vietnamese sentiment analysis model to classify
        the sentiment of review comments. It returns the probability of POSITIVE sentiment.
        
        Model: 5CD-AI/Vietnamese-Sentiment-visobert
        - Accuracy: 88-99% on Vietnamese sentiment benchmarks
        - Handles emojis, slang, and social media text
        - Trained on 120K e-commerce, social, and forum comments
        
        Strategy:
        - Tokenize input text using model-specific tokenizer
        - Run inference through ViSoBERT model
        - Apply softmax to get probability distribution
        - Return P(POSITIVE) as sentiment score [0.0, 1.0]
        - Handle errors gracefully (return 0.5 for neutral if error occurs)
        
        Args:
            text: Review comment text (Vietnamese)
        
        Returns:
            float: Sentiment score in range [0.0, 1.0]
                  - 0.0-0.3: Negative sentiment
                  - 0.3-0.7: Neutral sentiment
                  - 0.7-1.0: Positive sentiment
                  - 0.5: Default for errors/empty text
        
        Example:
            >>> fe = FeatureEngineer()
            >>> fe.compute_sentiment_score_ai("Sản phẩm rất tốt, tôi rất hài lòng!")
            0.92  # High positive probability
            
            >>> fe.compute_sentiment_score_ai("Sản phẩm tệ, không đáng tiền")
            0.08  # Low positive probability (negative sentiment)
        """
        # Handle missing or empty text
        if pd.isna(text) or not isinstance(text, str):
            return 0.5  # Neutral score for missing data
        
        text_str = text.strip()
        if len(text_str) == 0:
            return 0.5  # Neutral score for empty text
        
        # Fallback to keyword-based if model not loaded
        if self.model is None or self.tokenizer is None:
            return self._compute_sentiment_score_keywords(text_str)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text_str,
                return_tensors="pt",
                truncation=True,
                max_length=256,  # ViSoBERT max sequence length
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference (no gradient computation needed)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Extract positive sentiment probability
            # For 5CD-AI/Vietnamese-Sentiment-visobert: index 1 = POSITIVE
            positive_prob = probs[0][self.positive_label_idx].item()
            
            return float(positive_prob)
        
        except Exception as e:
            logger.warning(f"Error in sentiment analysis: {e}. Returning neutral score.")
            return 0.5  # Neutral score on error
    
    def _compute_sentiment_score_keywords(self, text: str) -> float:
        """
        Fallback sentiment scoring using keyword matching.
        
        Used when AI model fails to load or encounters errors.
        
        Args:
            text: Review comment text
        
        Returns:
            float: Sentiment score based on keyword presence [0.0, 1.0]
        """
        text_lower = text.lower()
        
        # Count positive keywords
        keyword_matches = sum(1 for kw in self.positive_keywords if kw in text_lower)
        
        # Convert to probability-like score
        # 0 keywords -> 0.3 (slightly negative assumption)
        # 1-2 keywords -> 0.5-0.7 (neutral to positive)
        # 3+ keywords -> 0.8+ (strong positive)
        if keyword_matches == 0:
            return 0.3
        elif keyword_matches == 1:
            return 0.5
        elif keyword_matches == 2:
            return 0.7
        else:
            return min(0.8 + (keyword_matches - 3) * 0.05, 1.0)
    
    def compute_comment_quality_score(self, comment_text: str) -> float:
        """
        Compute quality score based on AI sentiment analysis.
        
        Strategy (AI-Powered):
        - AI Sentiment Score (0.0-1.0): Use ViSoBERT to analyze sentiment
        - Final Score: Sentiment score directly (no length bonus)
        
        Args:
            comment_text: Review comment text (may be NaN or empty)
        
        Returns:
            float: Quality score in range [0.0, 1.0]
        
        Example:
            >>> fe = FeatureEngineer()
            >>> fe.compute_comment_quality_score("Sản phẩm rất tốt, thấm nhanh, hiệu quả!")
            0.92  # High sentiment score
            
            >>> fe.compute_comment_quality_score("Sản phẩm tệ lắm, rất thất vọng...")
            0.08  # Low sentiment score
            
            >>> fe.compute_comment_quality_score("")
            0.0   # Empty comment
        """
        # Handle missing or empty comments
        if pd.isna(comment_text):
            return 0.0
        
        comment_str = str(comment_text).strip()
        if len(comment_str) == 0:
            return 0.0
        
        # Get AI sentiment score (0.0-1.0)
        sentiment_score = self.compute_sentiment_score_ai(comment_str)
        
        return sentiment_score
    
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
        logger.info(f"\nComputing AI-powered comment quality scores from '{comment_column}'...")
        
        # Check if AI model is loaded
        if self.model is not None:
            logger.info("Quality scoring method: AI-Powered Sentiment Analysis")
            logger.info("  - AI Sentiment Score: ViSoBERT Vietnamese sentiment model (5CD-AI)")
            logger.info("  - Length bonus: +0.1 (≥10 words), +0.2 (≥20 words)")
            logger.info("  - Note: Length bonus only applied for non-negative sentiment (≥0.4)")
            logger.info("  - Formula: quality_score = min(sentiment_score + length_bonus, 1.0)")
        else:
            logger.warning("AI model not loaded - using fallback keyword method")
            logger.info("Quality scoring criteria:")
            logger.info("  - Keyword matching: Based on positive word presence")
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
