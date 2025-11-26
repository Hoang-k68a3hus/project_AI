"""
Feature Engineering Module for Collaborative Filtering

This module handles Step 2.0: Comment Quality Analysis and confidence score computation.
Addresses the rating skew problem (95% 5-star ratings) by analyzing comment content
to distinguish high-quality reviews from low-quality ones while down-weighting likely
fake/spam reviews.

Key Features:
- AI-powered sentiment analysis using ViSoBERT for Vietnamese text
- Batch processing for GPU optimization (up to 10x faster)
- Comment quality scoring based on sentiment analysis
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

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        device: Optional[str] = None,
        batch_size: int = 64,
        use_ai_sentiment: bool = True,
        no_comment_quality: float = 0.5,
        enable_fake_review_checks: bool = True
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
            batch_size: Batch size for GPU processing (default: 64, increase if you have more VRAM)
            use_ai_sentiment: Whether to use AI model or fallback to keywords (default: True)
            no_comment_quality: Default quality score assigned when comment text is missing/empty
        """
        self.positive_threshold = positive_threshold
        self.hard_negative_threshold = hard_negative_threshold
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_ai_sentiment = use_ai_sentiment
        self.no_comment_quality = float(np.clip(no_comment_quality, 0.0, 1.0))
        self.enable_fake_review_checks = enable_fake_review_checks

        # Keyword banks shared by both AI and fallback scoring
        self.positive_keywords = [
            'thấm nhanh', 'hiệu quả', 'thơm', 'mịn', 'sáng da',
            'trắng da', 'giảm mụn', 'không kích ứng', 'tốt',
            'rất thích', 'đáng mua', 'chất lượng', 'xuất sắc',
            'hài lòng', 'ưng ý', 'tuyệt vời', 'hoàn hảo',
            'mượt mà', 'tươi sáng', 'ẩm', 'mát', 'dễ chịu',
            'auth', 'chuẩn auth', 'okela', 'okie', 'xịn sò',
            'êm', 'êm ái', 'ngon lành', 'đáng tiền', 'đẹp mê',
            'ưng bụng', 'phê', 'bao phê', 'yêu thích', 'sạch sâu',
            'căng bóng', 'dễ dùng', 'dịu nhẹ', 'siêu thích', 'ổn áp',
            'xanh mướt', 'mềm mại', 'thơm lâu', 'xinh xắn'
        ]
        self.negative_keywords = [
            'hàng giả', 'fake', 'kém chất lượng', 'bị dị ứng',
            'không giống mô tả', 'vỡ', 'hỏng', 'trễ giao',
            'không nhận được', 'lừa đảo', 'mùi khó chịu', 'rất tệ',
            'bể', 'móp méo', 'bẩn', 'dở tệ', 'đau rát',
            'chậm giao', 'giao thiếu', 'hết hạn', 'mốc', 'nhớt',
            'khó xài', 'nhức', 'rát da', 'khô căng', 'khó chịu',
            'giả mạo', 'không uy tín', 'đòi hoàn', 'bom hàng'
        ]

        # Enrich domain keywords with slang/typo dictionaries if available
        if self.enable_fake_review_checks:
            self._augment_keywords_from_corrections()

        # Load emoji/icon sentiment mappings
        self._load_emoji_sentiment_mappings()

        # Heuristic parameters for fake-review mitigation
        self.long_review_min_words = 25
        self.long_review_bonus_cap = 0.06
        self.long_review_bonus_per_word = 0.002
        self.short_review_max_words = 4
        self.short_review_penalty = 0.08
        self.keyword_bonus = 0.015
        self.keyword_penalty = 0.02
        self.keyword_bonus_cap = 0.05
        self.keyword_penalty_cap = 0.08
        self.recency_half_life_days = 365
        self.recency_floor = 0.65
        self.rating_mismatch_penalty = 0.12
        self.low_sentiment_threshold = 0.3
        self.high_sentiment_threshold = 0.7
        self.repetition_penalty = 0.12
        self.min_unique_char_ratio = 0.35
        
        # Setup device (GPU if available, otherwise CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Skip model loading if not using AI sentiment
        if not self.use_ai_sentiment:
            logger.info(f"Initializing FeatureEngineer with keyword-based sentiment (AI disabled)")
            self.model = None
            self.tokenizer = None
        else:
            logger.info(f"Initializing FeatureEngineer with AI sentiment analysis...")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Batch size: {self.batch_size}")
            
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
    
    def compute_sentiment_scores_batch(self, texts: list) -> np.ndarray:
        """
        Compute sentiment scores for a batch of texts using AI model (GPU-optimized).
        
        This method processes multiple texts at once, maximizing GPU utilization.
        Much faster than processing one by one.
        
        Args:
            texts: List of review comment texts (Vietnamese)
        
        Returns:
            np.ndarray: Array of sentiment scores [0.0, 1.0] for each text
        
        Example:
            >>> fe = FeatureEngineer(batch_size=64)
            >>> texts = ["Sản phẩm tốt", "Sản phẩm tệ", "Sản phẩm bình thường"]
            >>> scores = fe.compute_sentiment_scores_batch(texts)
            >>> print(scores)  # [0.92, 0.08, 0.45]
        """
        # Fallback to keyword-based if model not loaded
        if self.model is None or self.tokenizer is None:
            return np.array([self._compute_sentiment_score_keywords(t) for t in texts])
        
        try:
            # Tokenize all texts at once
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run batch inference (no gradient computation needed)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Extract positive sentiment probabilities for all texts
            positive_probs = probs[:, self.positive_label_idx].cpu().numpy()
            
            return positive_probs
        
        except Exception as e:
            logger.warning(f"Error in batch sentiment analysis: {e}. Returning neutral scores.")
            return np.full(len(texts), 0.5)  # Neutral scores on error
    
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
            return self.no_comment_quality  # Neutral score for missing data
        
        text_str = text.strip()
        if len(text_str) == 0:
            return self.no_comment_quality  # Neutral score for empty text
        
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
    
    def _compute_quality_scores_batch(self, df: pd.DataFrame, comment_column: str) -> pd.Series:
        """
        Compute quality scores for all comments using batch processing (GPU-optimized).
        
        This method processes comments in batches to maximize GPU utilization,
        significantly faster than processing one by one.
        
        Args:
            df: DataFrame with comment column
            comment_column: Name of the comment column
        
        Returns:
            pd.Series: Quality scores for all comments
        """
        from tqdm import tqdm
        
        # Prepare all texts
        comments = df[comment_column].fillna('').astype(str).tolist()
        quality_scores = np.full(len(comments), self.no_comment_quality, dtype=np.float32)
        
        # Process in batches
        num_batches = (len(comments) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(comments), desc="Computing sentiment scores", unit="comments") as pbar:
            for i in range(0, len(comments), self.batch_size):
                batch_texts = comments[i:i+self.batch_size]
                
                # Filter out empty comments
                valid_indices = [j for j, text in enumerate(batch_texts) if text.strip()]
                valid_texts = [batch_texts[j] for j in valid_indices]
                
                if valid_texts:
                    # Compute sentiment scores for valid texts
                    batch_scores = self.compute_sentiment_scores_batch(valid_texts)
                    
                    # Assign scores back to original positions
                    for local_idx, global_idx in enumerate(valid_indices):
                        quality_scores[i + global_idx] = batch_scores[local_idx]
                
                pbar.update(len(batch_texts))
        
        return pd.Series(quality_scores, index=df.index)
    
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
            return self.no_comment_quality
        
        comment_str = str(comment_text).strip()
        if len(comment_str) == 0:
            return self.no_comment_quality
        
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
                logger.warning(
                    "No comment column found - assigning default quality score %.2f",
                    self.no_comment_quality
                )
                df['comment_quality'] = self.no_comment_quality
                df['confidence_score'] = df['rating'] + self.no_comment_quality
                return df, {
                    'comment_column_used': None,
                    'rows_with_comments': 0,
                    'rows_without_comments': initial_count
                }
        
        # Compute comment quality scores
        logger.info(f"\nComputing comment quality scores from '{comment_column}'...")
        
        # Check if AI model is loaded
        if self.model is not None:
            logger.info("Quality scoring method: AI-Powered Sentiment Analysis (Batch Processing)")
            logger.info("  - AI Sentiment Score: ViSoBERT Vietnamese sentiment model (5CD-AI)")
            logger.info(f"  - Batch size: {self.batch_size} (GPU-optimized)")
            logger.info("  - Device: " + str(self.device))
            logger.info("  - Formula: quality_score = sentiment_score [0.0, 1.0]")
            
            # Batch processing for GPU efficiency
            logger.info(f"\nProcessing {initial_count:,} comments in batches...")
            df['comment_quality'] = self._compute_quality_scores_batch(df, comment_column)
            
        else:
            logger.warning("AI model not loaded - using fallback keyword method")
            logger.info("Quality scoring criteria:")
            logger.info("  - Keyword matching: Based on positive word presence")
            logger.info(f"  - Vocabulary: {len(self.positive_keywords)} Vietnamese keywords")
            
            # Sequential processing for keyword-based (fast anyway)
            df['comment_quality'] = df[comment_column].apply(self.compute_comment_quality_score)
        
        # Compute confidence scores
        if self.enable_fake_review_checks:
            logger.info("\nApplying fake-review heuristics (length, keywords, recency, mismatch)...")
            suspicious_pct = self._apply_fake_review_adjustments(df, comment_column)
            logger.info(
                "  ↳ Flagged %.2f%% interactions as suspicious reviews",
                suspicious_pct * 100.0
            )

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

    def _augment_keywords_from_corrections(self) -> None:
        """
        Enrich keyword banks using typo/slang corrections collected from review data.
        """
        corrections_dir = Path("data/content_based_embeddings/processed/corrections_v2")
        if not corrections_dir.exists():
            logger.debug("Corrections directory %s not found; skipping keyword enrichment", corrections_dir)
            self.positive_keywords = sorted(set(kw.lower() for kw in self.positive_keywords))
            self.negative_keywords = sorted(set(kw.lower() for kw in self.negative_keywords))
            return

        positive_roots = {
            'ưng', 'ưng ý', 'hài lòng', 'thích', 'tốt', 'đáng',
            'xịn', 'chuẩn', 'auth', 'ổn', 'ok', 'okela', 'okie',
            'thơm', 'dịu', 'êm', 'đẹp', 'mịn', 'mượt', 'phê',
            'yêu', 'thần thánh', 'căng bóng', 'sáng', 'mát', 'dễ chịu'
        }
        negative_roots = {
            'fake', 'giả', 'kém', 'tệ', 'dở', 'bẩn', 'móp', 'vỡ', 'hỏng',
            'lừa', 'không giống', 'không nhận', 'chậm', 'trễ', 'khó chịu',
            'mùi lạ', 'mùi khó chịu', 'dị ứng', 'rát', 'ngứa', 'hết hạn',
            'mốc', 'bom', 'giao thiếu', 'nhớt', 'khô căng', 'đau', 'nứt'
        }

        pos_set = {kw.lower() for kw in self.positive_keywords}
        neg_set = {kw.lower() for kw in self.negative_keywords}
        pos_added = neg_added = 0

        for file_path in sorted(corrections_dir.glob("corrected_chunk_*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", file_path, exc)
                continue

            for typo, normalized in data.items():
                typo_token = typo.strip().lower()
                normalized_token = str(normalized).strip().lower()
                if not typo_token or not normalized_token:
                    continue

                if any(root in normalized_token for root in positive_roots):
                    if typo_token not in pos_set:
                        pos_set.add(typo_token)
                        pos_added += 1

                if any(root in normalized_token for root in negative_roots):
                    if typo_token not in neg_set:
                        neg_set.add(typo_token)
                        neg_added += 1

        self.positive_keywords = sorted(pos_set)
        self.negative_keywords = sorted(neg_set)

        logger.info(
            "✓ Enriched keyword banks from corrections_v2 ( +%d positive, +%d negative )",
            pos_added,
            neg_added
        )

    def _load_emoji_sentiment_mappings(self) -> None:
        """
        Load emoji/icon sentiment mappings from JSON file.
        
        Emoji are classified into positive, negative, and neutral categories
        based on their meaning. This helps adjust sentiment scores when
        emoji are present in reviews.
        """
        # Positive emoji patterns - indicate satisfaction/happiness
        self.positive_emojis = {
            # Hearts
            'red_heart', 'orange_heart', 'yellow_heart', 'green_heart', 
            'blue_heart', 'purple_heart', 'brown_heart', 'black_heart',
            'white_heart', 'pink_heart', 'two_hearts', 'beating_heart',
            'growing_heart', 'sparkling_heart', 'revolving_hearts',
            'heart_with_arrow', 'heart_with_ribbon', 'heart_decoration',
            'heart_exclamation', 'mending_heart', 'heart_on_fire',
            'heart_hands', 'heart_hands_light_skin_tone', 'heart_hands_medium',
            'anatomical_heart', 'heart_suit',
            
            # Happy faces
            'smiling_face', 'smiling_face_with_hearts', 'smiling_face_with_heart',
            'smiling_face_with_smiling_eyes', 'smiling_face_with_tear',
            'smiling_face_with_open_hands', 'smiling_face_with_sunglasses',
            'smiling_face_with_halo', 'beaming_face_with_smiling_eyes',
            'grinning_face', 'grinning_face_with_smiling_eyes',
            'grinning_face_with_big_eyes', 'grinning_face_with_sweat',
            'grinning_squinting_face', 'face_with_tears_of_joy',
            'slightly_smiling_face', 'winking_face', 'relieved_face',
            'face_savoring_food', 'partying_face', 'zany_face',
            'face_blowing_a_kiss', 'kissing_face', 'kissing_face_with_closed_eyes',
            'kissing_face_with_smiling_eyes', 'smiling_cat_with_heart',
            'grinning_cat', 'grinning_cat_with_smiling_eyes', 'cat_with_tears_of_joy',
            
            # Positive gestures
            'thumbs_up', 'thumbs_up_light_skin_tone', 'thumbs_up_medium',
            'thumbs_up_medium_skin_tone', 'thumbs_up_dark_skin_tone',
            'ok_hand', 'ok_hand_light_skin_tone', 'ok_hand_medium',
            'clapping_hands', 'clapping_hands_light_skin_tone', 'clapping_hands_medium',
            'raising_hands', 'raising_hands_light_skin_tone',
            'folded_hands', 'folded_hands_light_skin_tone',
            'flexed_biceps', 'flexed_biceps_light_skin_tone', 'flexed_biceps_medium',
            'victory_hand', 'love_you_gesture', 'call_me_hand',
            'call_me_hand_light_skin_tone', 'call_me_hand_medium',
            
            # Positive symbols
            'check_mark', 'check_mark_button', 'hundred_points',
            'glowing_star', 'sparkles', 'star', 'shooting_star',
            'fire', 'party_popper', 'confetti_ball', 'wrapped_gift',
            'trophy', 'medal', 'crown', 'gem_stone',
            'four_leaf_clover', 'cherry_blossom', 'rose', 'bouquet',
            
            # Others
            'kiss_mark', 'love_letter', 'smiling_face_with_horns',
        }
        
        # Negative emoji patterns - indicate dissatisfaction/sadness
        self.negative_emojis = {
            # Sad/angry faces
            'crying_face', 'loudly_crying_face', 'face_holding_back_tears',
            'sad_but_relieved_face', 'disappointed_face', 'pensive_face',
            'worried_face', 'confused_face', 'slightly_frowning_face',
            'frowning_face', 'frowning_face_with_open_mouth',
            'anguished_face', 'fearful_face', 'anxious_face_with_sweat',
            'downcast_face_with_sweat', 'weary_face', 'tired_face',
            'persevering_face', 'confounded_face', 'pleading_face',
            'grimacing_face', 'expressionless_face', 'neutral_face',
            'face_without_mouth', 'face_in_clouds', 'dotted_line_face',
            
            # Angry faces
            'angry_face', 'enraged_face', 'face_with_steam_from_nose',
            'pouting_face', 'face_with_symbols_on_mouth',
            'angry_face_with_horns', 'skull', 'skull_and_crossbones',
            
            # Sick/unwell faces
            'nauseated_face', 'face_vomiting', 'sneezing_face',
            'face_with_thermometer', 'face_with_head_bandage',
            'dizzy_face', 'exploding_head', 'cold_face', 'hot_face',
            'woozy_face', 'face_with_crossed_out_eyes',
            
            # Negative gestures
            'thumbs_down', 'thumbs_down_light_skin_tone', 'thumbs_down_medium_skin_tone',
            'middle_finger', 'middle_finger_light_skin_tone',
            
            # Negative symbols  
            'cross_mark', 'broken_heart', 'anger_symbol',
            'collision', 'bomb', 'warning', 'no_entry',
            'prohibited', 'stop_sign',
            
            # Crying cats
            'crying_cat', 'weary_cat', 'pouting_cat',
        }
        
        # Neutral emoji - don't significantly affect sentiment
        self.neutral_emojis = {
            'thinking_face', 'face_with_raised_eyebrow', 'face_with_monocle',
            'face_with_rolling_eyes', 'smirking_face', 'unamused_face',
            'nerd_face', 'face_with_hand_over_mouth', 'shushing_face',
            'lying_face', 'drooling_face', 'sleepy_face', 'sleeping_face',
            'yawning_face', 'face_with_medical_mask', 'disguised_face',
            'face_with_open_mouth', 'hushed_face', 'astonished_face',
            'flushed_face', 'squinting_face_with_tongue', 'winking_face_with_tongue',
            'face_with_tongue', 'money_mouth_face', 'sunglasses',
            
            # Pointing gestures
            'backhand_index_pointing_right', 'backhand_index_pointing_left',
            'backhand_index_pointing_up', 'backhand_index_pointing_down',
            'index_pointing_up', 'backhand_index_pointing_right_light_skin_tone',
            'backhand_index_pointing_left_light_skin_tone',
            
            # Other neutral
            'waving_hand', 'waving_hand_light_skin_tone',
            'raised_hand', 'raised_hand_light_skin_tone',
            'hand_with_fingers_splayed', 'vulcan_salute',
            'open_hands', 'open_hands_light_skin_tone',
            'pinched_fingers', 'pinched_fingers_light_skin_tone',
        }
        
        # Emoji bonus/penalty values
        self.emoji_positive_bonus = 0.03  # Bonus per positive emoji
        self.emoji_negative_penalty = 0.05  # Penalty per negative emoji
        self.emoji_bonus_cap = 0.10  # Max bonus from emoji
        self.emoji_penalty_cap = 0.15  # Max penalty from emoji
        
        logger.info(
            "✓ Loaded emoji sentiment mappings: %d positive, %d negative, %d neutral",
            len(self.positive_emojis), len(self.negative_emojis), len(self.neutral_emojis)
        )

    def _count_emojis_in_text(self, text: str) -> Tuple[int, int, int]:
        """
        Count positive, negative, and neutral emoji in text.
        
        Args:
            text: Review text containing potential emoji patterns like "red_heart"
            
        Returns:
            Tuple of (positive_count, negative_count, neutral_count)
        """
        import re
        
        text_lower = text.lower()
        
        # Find all underscore-connected words (potential emoji)
        pattern = r'\b\w+(?:_\w+)+\b'
        matches = re.findall(pattern, text_lower)
        
        pos_count = 0
        neg_count = 0
        neu_count = 0
        
        for match in matches:
            # Check if it's a known emoji
            if match in self.positive_emojis:
                pos_count += 1
            elif match in self.negative_emojis:
                neg_count += 1
            elif match in self.neutral_emojis:
                neu_count += 1
            else:
                # Check partial matches for emoji patterns with skin tone suffixes
                base_match = match.split('_light_skin_tone')[0].split('_medium')[0].split('_dark_skin_tone')[0]
                if base_match in self.positive_emojis:
                    pos_count += 1
                elif base_match in self.negative_emojis:
                    neg_count += 1
        
        return pos_count, neg_count, neu_count

    def _apply_emoji_adjustment(self, base_score: float, text: str) -> float:
        """
        Adjust sentiment score based on emoji presence in text.
        
        Args:
            base_score: Base sentiment score [0.0, 1.0]
            text: Review text
            
        Returns:
            Adjusted score [0.0, 1.0]
        """
        if not text or not isinstance(text, str):
            return base_score
        
        pos_count, neg_count, neu_count = self._count_emojis_in_text(text)
        
        # Calculate adjustment
        positive_bonus = min(pos_count * self.emoji_positive_bonus, self.emoji_bonus_cap)
        negative_penalty = min(neg_count * self.emoji_negative_penalty, self.emoji_penalty_cap)
        
        adjusted = base_score + positive_bonus - negative_penalty
        
        return float(np.clip(adjusted, 0.0, 1.0))

    def _apply_fake_review_adjustments(self, df: pd.DataFrame, comment_column: str) -> float:
        """
        Apply heuristic adjustments to comment_quality to down-weight suspected fake reviews.

        Args:
            df: DataFrame with comment_quality already computed.
            comment_column: Column containing the original text.

        Returns:
            float: Ratio of rows flagged as suspicious (for logging/monitoring).
        """
        if comment_column not in df.columns or 'comment_quality' not in df.columns:
            return 0.0

        comments = df[comment_column].fillna('').astype(str).to_numpy()
        scores = df['comment_quality'].to_numpy(dtype=np.float32, copy=True)
        ratings = df['rating'].to_numpy(copy=False) if 'rating' in df.columns else None
        timestamps = df['cmt_date'].to_numpy(copy=False) if 'cmt_date' in df.columns else None

        suspicious_mask = np.zeros(len(scores), dtype=bool)
        # Use timezone-naive timestamp to match cmt_date column (timezone-naive)
        current_time = pd.Timestamp.now()

        for idx in range(len(scores)):
            rating_value = ratings[idx] if ratings is not None else None
            ts_value = timestamps[idx] if timestamps is not None else None
            adjusted_score, flagged = self._adjust_quality_score(
                base_score=scores[idx],
                comment_text=comments[idx],
                rating=rating_value,
                timestamp=ts_value,
                current_time=current_time
            )
            scores[idx] = adjusted_score
            suspicious_mask[idx] = flagged

        df['comment_quality'] = scores
        if 'is_suspicious_review' in df.columns:
            df['is_suspicious_review'] = df['is_suspicious_review'] | suspicious_mask
        else:
            df['is_suspicious_review'] = suspicious_mask

        return float(suspicious_mask.mean())

    def _adjust_quality_score(
        self,
        base_score: float,
        comment_text: str,
        rating: Optional[float],
        timestamp,
        current_time: Optional[pd.Timestamp] = None
    ) -> Tuple[float, bool]:
        """
        Apply additional heuristics to detect/penalize likely fake reviews.

        Returns:
            Tuple[float, bool]: (adjusted_score, is_suspicious)
        """
        score = float(np.clip(base_score, 0.0, 1.0))
        text = (comment_text or "").strip()
        if not text:
            return score, False

        suspicious = False
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)

        # Length-based adjustments
        high_rating_context = True
        if rating is not None and not pd.isna(rating):
            high_rating_context = float(rating) >= self.positive_threshold

        if word_count >= self.long_review_min_words:
            extra_words = word_count - self.long_review_min_words
            bonus = 0.02 + extra_words * self.long_review_bonus_per_word
            bonus = min(bonus, self.long_review_bonus_cap)
            score += bonus
        elif word_count <= self.short_review_max_words and high_rating_context:
            score -= self.short_review_penalty
            suspicious = True  # extremely short comments with high ratings are often fake

        # Keyword-based modifiers
        pos_hits = sum(1 for kw in self.positive_keywords if kw in text_lower)
        neg_hits = sum(1 for kw in self.negative_keywords if kw in text_lower)
        if pos_hits:
            score += min(self.keyword_bonus_cap, pos_hits * self.keyword_bonus)
        if neg_hits:
            score -= min(self.keyword_penalty_cap, neg_hits * self.keyword_penalty)

        # Rating vs text mismatch
        if rating is not None and not pd.isna(rating):
            rating_val = float(rating)
            if rating_val >= self.positive_threshold and score < self.low_sentiment_threshold:
                score *= (1 - self.rating_mismatch_penalty)
                suspicious = True
            elif rating_val <= self.hard_negative_threshold and score > self.high_sentiment_threshold:
                score *= (1 - self.rating_mismatch_penalty * 0.6)
                suspicious = True

        # Recency decay (older reviews carry less weight)
        if timestamp is not None and not pd.isna(timestamp):
            ts = pd.to_datetime(timestamp, errors='coerce')
            if pd.notna(ts) and current_time is not None:
                # Handle timezone mismatch: make both timezone-naive if one is naive
                # This ensures compatibility when cmt_date is timezone-naive
                if ts.tz is None and current_time.tz is not None:
                    current_time_naive = current_time.tz_localize(None)
                elif ts.tz is not None and current_time.tz is None:
                    ts = ts.tz_localize(None)
                    current_time_naive = current_time
                else:
                    # Both have same timezone awareness (both naive or both aware)
                    current_time_naive = current_time
                
                age_days = max(0.0, (current_time_naive - ts).days)
                if age_days > 0 and self.recency_half_life_days > 0:
                    decay = np.exp(-age_days / self.recency_half_life_days)
                    recency_factor = self.recency_floor + (1 - self.recency_floor) * decay
                    score *= recency_factor

        # Repetition / low uniqueness penalty (common in fake reviews)
        unique_chars = len(set(text_lower.replace(' ', '')))
        total_chars = max(1, len(text_lower.replace(' ', '')))
        unique_ratio = unique_chars / total_chars
        if unique_ratio < self.min_unique_char_ratio:
            score *= (1 - self.repetition_penalty)
            suspicious = True

        # Emoji-based adjustments
        score = self._apply_emoji_adjustment(score, text)

        return float(np.clip(score, 0.0, 1.0)), suspicious
    
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
    
    def create_explicit_features(
        self,
        df: pd.DataFrame,
        rating_col: str = 'rating'
    ) -> pd.DataFrame:
        """
        Create explicit feedback features for ALS and BPR (Step 2.1-2.2).
        
        This method labels interactions as positive/negative based on rating thresholds
        to support both ALS (confidence-weighted) and BPR (pairwise ranking) training.
        
        Features created:
        - is_positive: Binary flag (1 if rating >= positive_threshold, else 0)
        - is_hard_negative: Binary flag (1 if rating <= hard_negative_threshold, else 0)
        
        Args:
            df: DataFrame with interactions (must have rating column)
            rating_col: Name of the rating column (default: 'rating')
        
        Returns:
            DataFrame with added is_positive and is_hard_negative columns
        
        Example:
            >>> fe = FeatureEngineer(positive_threshold=4, hard_negative_threshold=3)
            >>> df_labeled = fe.create_explicit_features(df)
            >>> print(df_labeled['is_positive'].sum())  # Count positive interactions
        """
        logger.info("\n" + "="*80)
        logger.info("STEP 2.1-2.2: EXPLICIT FEEDBACK FEATURE ENGINEERING")
        logger.info("="*80)
        
        # Validate input
        if rating_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{rating_col}' column")
        
        initial_count = len(df)
        logger.info(f"Processing {initial_count:,} interactions")
        logger.info(f"Positive threshold: rating >= {self.positive_threshold}")
        logger.info(f"Hard negative threshold: rating <= {self.hard_negative_threshold}")
        
        # Create positive labels (Step 2.1 - for ALS and BPR positive samples)
        df['is_positive'] = (df[rating_col] >= self.positive_threshold).astype(int)
        
        # Create hard negative labels (Step 2.2 - for BPR hard negative mining)
        df['is_hard_negative'] = (df[rating_col] <= self.hard_negative_threshold).astype(int)
        
        # Compute statistics
        positive_count = df['is_positive'].sum()
        hard_neg_count = df['is_hard_negative'].sum()
        neutral_count = initial_count - positive_count - hard_neg_count
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("EXPLICIT FEEDBACK SUMMARY")
        logger.info("="*80)
        logger.info(f"Total interactions:        {initial_count:>12,}")
        logger.info(f"Positive (≥{self.positive_threshold}):         {positive_count:>12,} ({positive_count/initial_count:.2%})")
        logger.info(f"Hard Negative (≤{self.hard_negative_threshold}):    {hard_neg_count:>12,} ({hard_neg_count/initial_count:.2%})")
        logger.info(f"Neutral:                   {neutral_count:>12,} ({neutral_count/initial_count:.2%})")
        logger.info("="*80 + "\n")
        
        # Validate results
        assert df['is_positive'].notna().all(), "is_positive contains NaN values"
        assert df['is_hard_negative'].notna().all(), "is_hard_negative contains NaN values"
        assert df['is_positive'].isin([0, 1]).all(), "is_positive contains non-binary values"
        assert df['is_hard_negative'].isin([0, 1]).all(), "is_hard_negative contains non-binary values"
        
        logger.info("✓ All explicit feedback labels validated")
        
        return df
