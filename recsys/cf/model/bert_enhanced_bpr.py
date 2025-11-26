"""
BERT-Enhanced BPR Model (Task 02 - BERT + Sentiment-Aware BPR)

This module implements BPR with:
1. BERT-initialized item factors (transfer semantic knowledge from PhoBERT)
2. Sentiment-aware confidence weighting (distinguish genuine vs fake 5-star reviews)

Similar to BERT-Enhanced ALS but for pairwise ranking:
- Projects PhoBERT embeddings (768-dim) to item factors (64-dim) via SVD
- Uses sentiment-enhanced confidence to weight BPR triplets
- Hard negative mining with confidence-aware sampling

CRITICAL for ≥2 Threshold:
- With ~26k trainable users and high sparsity
- BERT initialization provides semantic priors for sparse items
- Sentiment weighting helps identify truly positive vs spam reviews

Author: VieComRec Team
Date: 2025-11-26
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Set, Any, List
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


@dataclass
class SentimentAwareConfig:
    """
    Configuration for sentiment-aware BPR training.
    
    Attributes:
        use_sentiment_weighting: Enable confidence-based triplet weighting
        confidence_col: Column name for confidence scores in interactions
        confidence_min: Minimum confidence value for normalization
        confidence_max: Maximum confidence value for normalization
        positive_confidence_threshold: Minimum confidence for trusted positives
        suspicious_penalty: Penalty factor for suspicious reviews
        high_confidence_bonus: Bonus factor for high-confidence reviews
    """
    use_sentiment_weighting: bool = True
    confidence_col: str = 'confidence_score'
    confidence_min: float = 1.0  # rating 1 + quality 0
    confidence_max: float = 6.0  # rating 5 + quality 1
    positive_confidence_threshold: float = 4.5  # rating 4 + quality 0.5
    suspicious_penalty: float = 0.5  # Reduce weight for suspicious reviews
    high_confidence_bonus: float = 1.5  # Increase weight for high-confidence reviews


class BERTEnhancedBPR:
    """
    BPR with BERT-initialized item factors and sentiment-aware training.
    
    Key Features:
    1. BERT Initialization:
       - Projects PhoBERT embeddings to target dimension via SVD
       - Aligns embeddings to match item indices in training data
       - Handles missing embeddings with statistical initialization
    
    2. Sentiment-Aware Confidence:
       - Uses confidence_score (rating + comment_quality) for triplet weighting
       - Higher weight for genuine positive reviews (high sentiment score)
       - Lower weight for suspicious reviews (rating/sentiment mismatch)
    
    3. Enhanced Training:
       - Weighted BPR loss based on confidence
       - Adaptive learning rate per triplet confidence
       - Hard negative mining with confidence consideration
    
    Attributes:
        factors (int): Embedding dimension for user/item factors
        bert_embeddings (np.ndarray): BERT embeddings for items
        product_ids (list): Product IDs corresponding to BERT embeddings
        sentiment_config (SentimentAwareConfig): Sentiment weighting configuration
    """
    
    def __init__(
        self,
        bert_embeddings_path: Optional[Union[str, Path]] = None,
        factors: int = 64,
        projection_method: str = "svd",
        learning_rate: float = 0.05,
        regularization: float = 0.0001,
        lr_decay: float = 0.9,
        lr_decay_every: int = 10,
        sentiment_config: Optional[SentimentAwareConfig] = None,
        random_seed: int = 42
    ):
        """
        Initialize BERT-Enhanced BPR model.
        
        Args:
            bert_embeddings_path: Path to BERT embeddings (.pt file)
                Expected format: {
                    'embeddings': torch.Tensor (num_items, bert_dim),
                    'product_ids': list of product IDs
                }
            factors: Embedding dimension (will project BERT embeddings to this)
            projection_method: "svd" (recommended) or "pca" for dimensionality reduction
            learning_rate: Initial learning rate for SGD
            regularization: L2 regularization coefficient
            lr_decay: Learning rate decay factor
            lr_decay_every: Decay LR every N epochs
            sentiment_config: Configuration for sentiment-aware training
            random_seed: Random seed for reproducibility
        """
        self.factors = factors
        self.projection_method = projection_method
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.lr_decay = lr_decay
        self.lr_decay_every = lr_decay_every
        self.random_seed = random_seed
        
        # Sentiment-aware configuration
        self.sentiment_config = sentiment_config or SentimentAwareConfig()
        
        # Initialize random generator
        self.rng = np.random.default_rng(random_seed)
        
        # BERT embeddings (loaded if path provided)
        self.bert_embeddings = None
        self.product_ids = None
        self.bert_dim = None
        self.projection_info = {}
        self.bert_init_used = False
        
        if bert_embeddings_path:
            self._load_bert_embeddings(bert_embeddings_path)
        
        # Model parameters (initialized in fit())
        self.U = None  # User embeddings
        self.V = None  # Item embeddings
        self.num_users = None
        self.num_items = None
        
        # Training tracking
        self.training_start_time = None
        self.training_end_time = None
        self.history = None
        self.is_fitted = False
        
        logger.info(f"BERTEnhancedBPR initialized:")
        logger.info(f"  Factors: {factors}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Regularization: {regularization}")
        logger.info(f"  Sentiment weighting: {self.sentiment_config.use_sentiment_weighting}")
    
    def _load_bert_embeddings(self, bert_embeddings_path: Union[str, Path]) -> None:
        """Load BERT embeddings from file."""
        logger.info(f"Loading BERT embeddings from {bert_embeddings_path}")
        
        bert_data = torch.load(bert_embeddings_path, map_location='cpu')
        
        self.bert_embeddings = bert_data['embeddings'].numpy()
        self.product_ids = bert_data['product_ids']
        self.bert_dim = self.bert_embeddings.shape[1]
        
        logger.info(
            f"Loaded BERT embeddings: shape={self.bert_embeddings.shape}, "
            f"dim={self.bert_dim}, num_products={len(self.product_ids)}"
        )
    
    def project_bert_to_factors(self, target_dim: int) -> np.ndarray:
        """
        Project BERT embeddings to target dimension using SVD or PCA.
        
        Args:
            target_dim: Target embedding dimension (e.g., 64)
        
        Returns:
            projected: Projected embeddings (num_items, target_dim)
        """
        if self.bert_embeddings is None:
            raise ValueError("BERT embeddings not loaded. Provide bert_embeddings_path.")
        
        if self.bert_embeddings.shape[1] == target_dim:
            logger.info("BERT embeddings already at target dimension")
            return self.bert_embeddings.copy()
        
        logger.info(
            f"Projecting BERT embeddings: {self.bert_embeddings.shape[1]} -> {target_dim}"
        )
        
        start_time = time.time()
        
        if self.projection_method == "svd":
            reducer = TruncatedSVD(n_components=target_dim, random_state=self.random_seed)
            projected = reducer.fit_transform(self.bert_embeddings)
            explained_var = reducer.explained_variance_ratio_.sum()
            
        elif self.projection_method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=target_dim, random_state=self.random_seed)
            projected = reducer.fit_transform(self.bert_embeddings)
            explained_var = reducer.explained_variance_ratio_.sum()
            
        else:
            raise ValueError(f"Unknown projection method: {self.projection_method}")
        
        projection_time = time.time() - start_time
        
        # Validate for NaN/Inf
        if np.isnan(projected).any() or np.isinf(projected).any():
            logger.error("CRITICAL: Projection created invalid values! Replacing with random.")
            invalid_mask = np.isnan(projected) | np.isinf(projected)
            projected[invalid_mask] = self.rng.normal(0, 0.01, size=invalid_mask.sum())
        
        # Store projection info
        self.projection_info = {
            "method": self.projection_method,
            "original_dim": self.bert_embeddings.shape[1],
            "target_dim": target_dim,
            "explained_variance": float(explained_var),
            "projection_time_seconds": projection_time,
        }
        
        logger.info(
            f"Projection complete: explained_variance={explained_var:.3f}, "
            f"time={projection_time:.2f}s"
        )
        
        return projected.astype(np.float32)
    
    def align_embeddings_to_matrix(
        self,
        projected_embeddings: np.ndarray,
        item_to_idx: Dict[str, int],
        num_items: int
    ) -> Optional[np.ndarray]:
        """
        Align BERT embeddings to match item ordering in training data.
        
        Args:
            projected_embeddings: Projected BERT embeddings (num_bert_items, factors)
            item_to_idx: Mapping from product_id (str) to item index (int)
            num_items: Total number of items
        
        Returns:
            aligned: Aligned embeddings (num_items, factors) or None if coverage too low
        """
        logger.info("Aligning BERT embeddings to training data ordering")
        
        aligned_embeddings = np.zeros((num_items, self.factors), dtype=np.float32)
        
        matched_count = 0
        
        for i, product_id in enumerate(self.product_ids):
            product_id_str = str(product_id)
            
            if product_id_str in item_to_idx:
                idx = item_to_idx[product_id_str]
                aligned_embeddings[idx] = projected_embeddings[i]
                matched_count += 1
            elif product_id in item_to_idx:
                idx = item_to_idx[product_id]
                aligned_embeddings[idx] = projected_embeddings[i]
                matched_count += 1
        
        coverage = matched_count / num_items if num_items > 0 else 0
        
        logger.info(f"Alignment complete: matched={matched_count}, coverage={coverage:.1%}")
        
        if coverage < 0.05:
            logger.error(f"CRITICAL: BERT embedding coverage too low: {coverage:.1%}")
            return None
        
        # Fill missing embeddings with statistical initialization
        zero_vector_mask = np.all(aligned_embeddings == 0, axis=1)
        zero_vector_count = zero_vector_mask.sum()
        
        if zero_vector_count > 0:
            matched_embeddings = aligned_embeddings[~zero_vector_mask]
            
            if len(matched_embeddings) > 0:
                mean = matched_embeddings.mean(axis=0)
                std = np.maximum(matched_embeddings.std(axis=0), 1e-6)
                std = np.clip(std, 1e-6, 10.0)
                
                random_init = self.rng.normal(
                    mean, std * 0.1, size=(zero_vector_count, self.factors)
                ).astype(np.float32)
                
                random_init = np.clip(random_init, -10.0, 10.0)
                aligned_embeddings[zero_vector_mask] = random_init
                
                logger.info(f"Initialized {zero_vector_count} missing items with statistical values")
        
        return aligned_embeddings
    
    def initialize_embeddings(
        self,
        num_users: int,
        num_items: int,
        item_to_idx: Optional[Dict[str, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize user and item embeddings.
        
        User embeddings: Always random Gaussian (mean=0, std=0.01)
        Item embeddings: BERT-initialized if available, else random
        
        Args:
            num_users: Number of users
            num_items: Number of items
            item_to_idx: Mapping from product_id to item index (for BERT alignment)
        
        Returns:
            Tuple of (U, V) - user and item embeddings
        """
        self.num_users = num_users
        self.num_items = num_items
        
        # Initialize user embeddings (random)
        U = self.rng.normal(
            loc=0.0, scale=0.01, size=(num_users, self.factors)
        ).astype(np.float32)
        
        logger.info(f"User embeddings initialized: {U.shape}")
        
        # Initialize item embeddings (BERT or random)
        if self.bert_embeddings is not None and item_to_idx is not None:
            logger.info("Initializing item embeddings from BERT...")
            
            projected = self.project_bert_to_factors(self.factors)
            aligned = self.align_embeddings_to_matrix(projected, item_to_idx, num_items)
            
            if aligned is not None:
                V = aligned
                self.bert_init_used = True
                logger.info("Item embeddings initialized from BERT")
            else:
                V = self.rng.normal(
                    loc=0.0, scale=0.01, size=(num_items, self.factors)
                ).astype(np.float32)
                self.bert_init_used = False
                logger.warning("BERT alignment failed, using random initialization")
        else:
            V = self.rng.normal(
                loc=0.0, scale=0.01, size=(num_items, self.factors)
            ).astype(np.float32)
            self.bert_init_used = False
            logger.info("Item embeddings initialized randomly (no BERT provided)")
        
        logger.info(f"Item embeddings initialized: {V.shape}")
        
        return U, V
    
    def compute_confidence_weights(
        self,
        confidence_scores: np.ndarray
    ) -> np.ndarray:
        """
        Compute training weights from confidence scores.
        
        Strategy:
        - Normalize confidence to [0, 1] range
        - Apply penalty/bonus based on thresholds
        - High confidence (genuine reviews) -> higher weight
        - Low confidence (suspicious reviews) -> lower weight
        
        Args:
            confidence_scores: Array of confidence scores (rating + comment_quality)
        
        Returns:
            weights: Array of training weights [0.0, 2.0]
        """
        if not self.sentiment_config.use_sentiment_weighting:
            return np.ones(len(confidence_scores), dtype=np.float32)
        
        config = self.sentiment_config
        
        # Normalize confidence to [0, 1]
        normalized = (confidence_scores - config.confidence_min) / (
            config.confidence_max - config.confidence_min
        )
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Base weight = 1.0
        weights = np.ones(len(confidence_scores), dtype=np.float32)
        
        # Apply bonus for high-confidence (genuine) reviews
        high_conf_mask = confidence_scores >= config.positive_confidence_threshold
        weights[high_conf_mask] = 1.0 + (normalized[high_conf_mask] - 0.5) * (
            config.high_confidence_bonus - 1.0
        ) * 2
        
        # Apply penalty for low-confidence (suspicious) reviews
        low_conf_mask = confidence_scores < config.positive_confidence_threshold
        low_conf_penalty = (0.5 - normalized[low_conf_mask]) * (
            1.0 - config.suspicious_penalty
        ) * 2
        weights[low_conf_mask] = np.clip(1.0 - low_conf_penalty, config.suspicious_penalty, 1.0)
        
        return weights.astype(np.float32)
    
    def _sgd_update_weighted(
        self,
        users: np.ndarray,
        pos_items: np.ndarray,
        neg_items: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """
        Perform weighted SGD update for a batch of triplets.
        
        Weighted BPR Loss:
            L = -sum(w_uij * log(sigmoid(x_uij))) + reg * (||U||^2 + ||V||^2)
        
        where w_uij is the confidence-based weight for triplet (u, i, j)
        
        Args:
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative item indices
            weights: Confidence-based weights per triplet
        
        Returns:
            Average weighted batch loss
        """
        # Get embeddings
        user_emb = self.U[users]
        pos_emb = self.V[pos_items]
        neg_emb = self.V[neg_items]
        
        # Compute x_uij = score(u, i_pos) - score(u, i_neg)
        x_uij = np.sum(user_emb * pos_emb, axis=1) - np.sum(user_emb * neg_emb, axis=1)
        
        # Sigmoid and gradient factor
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x_uij, -50, 50)))
        grad_factor = ((1.0 - sigmoid) * weights).reshape(-1, 1)
        
        # Compute weighted gradients
        grad_u = grad_factor * (pos_emb - neg_emb) - self.regularization * user_emb
        grad_v_pos = grad_factor * user_emb - self.regularization * pos_emb
        grad_v_neg = -grad_factor * user_emb - self.regularization * neg_emb
        
        # Update embeddings
        self.U[users] += self.learning_rate * grad_u
        np.add.at(self.V, pos_items, self.learning_rate * grad_v_pos)
        np.add.at(self.V, neg_items, self.learning_rate * grad_v_neg)
        
        # Compute weighted loss
        loss = -np.sum(weights * np.log(np.clip(sigmoid, 1e-10, 1.0))) / len(weights)
        
        return loss
    
    def _update_learning_rate(self, epoch: int) -> None:
        """Apply learning rate decay."""
        if self.lr_decay_every > 0 and epoch > 0 and epoch % self.lr_decay_every == 0:
            self.learning_rate *= self.lr_decay
            logger.info(f"Learning rate decayed to {self.learning_rate:.6f}")
    
    def fit(
        self,
        positive_pairs: np.ndarray,
        user_pos_sets: Dict[int, Set[int]],
        num_users: int,
        num_items: int,
        item_to_idx: Optional[Dict[str, int]] = None,
        hard_neg_sets: Optional[Dict[int, Set[int]]] = None,
        confidence_scores: Optional[np.ndarray] = None,
        epochs: int = 50,
        samples_per_positive: int = 5,
        hard_ratio: float = 0.3,
        batch_size: int = 4096,
        val_user_pos_test: Optional[Dict[int, Set[int]]] = None,
        early_stopping_patience: int = 5,
        early_stopping_metric: str = 'recall@10',
        show_progress: bool = True,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Train BERT-Enhanced BPR model with sentiment-aware weighting.
        
        Training Process:
        1. Initialize embeddings (BERT for items, random for users)
        2. Sample triplets (u, i_pos, i_neg) with hard negative mining
        3. Compute confidence weights for triplets
        4. Update with weighted BPR loss
        5. Early stopping based on validation metrics
        
        Args:
            positive_pairs: Array of (u, i) positive pairs
            user_pos_sets: Dict mapping u -> set of positive items
            num_users: Total number of users
            num_items: Total number of items
            item_to_idx: Mapping from product_id to item index (for BERT)
            hard_neg_sets: Optional hard negative sets per user
            confidence_scores: Optional confidence scores per positive pair
                              (must align with positive_pairs ordering)
            epochs: Number of training epochs
            samples_per_positive: Samples per positive pair per epoch
            hard_ratio: Fraction of hard negatives (default: 0.3)
            batch_size: Mini-batch size for SGD
            val_user_pos_test: Optional validation test sets for early stopping
            early_stopping_patience: Epochs without improvement before stopping
            early_stopping_metric: Metric for early stopping
            show_progress: Whether to show progress logs
            checkpoint_dir: Directory for checkpoints
            checkpoint_interval: Save checkpoint every N epochs
        
        Returns:
            Training summary dictionary
        """
        from .bpr.sampler import TripletSampler
        
        logger.info("="*60)
        logger.info("Starting BERT-Enhanced BPR Training")
        logger.info("="*60)
        
        self.training_start_time = time.time()
        
        # Initialize embeddings
        self.U, self.V = self.initialize_embeddings(num_users, num_items, item_to_idx)
        
        # Setup confidence weights
        if confidence_scores is not None and self.sentiment_config.use_sentiment_weighting:
            # Build mapping from (u, i) to confidence
            pair_to_confidence = {}
            for idx, (u, i) in enumerate(positive_pairs):
                pair_to_confidence[(int(u), int(i))] = confidence_scores[idx]
            logger.info(f"Sentiment weighting enabled with {len(pair_to_confidence)} confidence scores")
        else:
            pair_to_confidence = None
            logger.info("Sentiment weighting disabled (uniform weights)")
        
        # Initialize sampler
        sampler = TripletSampler(
            positive_pairs=positive_pairs,
            user_pos_sets=user_pos_sets,
            num_items=num_items,
            hard_neg_sets=hard_neg_sets or {},
            hard_ratio=hard_ratio,
            samples_per_positive=samples_per_positive,
            random_seed=self.random_seed
        )
        
        samples_per_epoch = sampler.samples_per_epoch
        logger.info(f"Samples per epoch: {samples_per_epoch:,}")
        logger.info(f"Batch size: {batch_size:,}")
        
        # Setup checkpointing
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'epochs': [],
            'losses': [],
            'learning_rates': [],
            'val_metrics': []
        }
        
        # Early stopping
        best_val_metric = -np.inf
        patience_counter = 0
        best_U = None
        best_V = None
        best_epoch = 0
        
        # Training loop
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Sample triplets
            triplets = sampler.sample_epoch()
            self.rng.shuffle(triplets)
            
            # Mini-batch training
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i + batch_size]
                users = batch[:, 0].astype(np.int64)
                pos_items = batch[:, 1].astype(np.int64)
                neg_items = batch[:, 2].astype(np.int64)
                
                # Get confidence weights for this batch
                if pair_to_confidence is not None:
                    weights = np.array([
                        pair_to_confidence.get((int(u), int(p)), 
                                               (self.sentiment_config.confidence_min + 
                                                self.sentiment_config.confidence_max) / 2)
                        for u, p in zip(users, pos_items)
                    ], dtype=np.float32)
                    weights = self.compute_confidence_weights(weights)
                else:
                    weights = np.ones(len(users), dtype=np.float32)
                
                batch_loss = self._sgd_update_weighted(users, pos_items, neg_items, weights)
                epoch_loss += batch_loss
                num_batches += 1
            
            epoch_loss /= num_batches
            epoch_duration = time.time() - epoch_start
            
            # Update learning rate
            self._update_learning_rate(epoch)
            
            # Compute validation metrics
            val_metrics = {}
            if val_user_pos_test:
                val_metrics = self._compute_validation_metrics(
                    user_pos_sets, val_user_pos_test, num_items
                )
            
            # Record history
            self.history['epochs'].append(epoch)
            self.history['losses'].append(float(epoch_loss))
            self.history['learning_rates'].append(float(self.learning_rate))
            self.history['val_metrics'].append(val_metrics)
            
            # Logging
            if show_progress:
                msg = f"Epoch {epoch}/{epochs}: loss={epoch_loss:.4f}, time={epoch_duration:.1f}s"
                if val_metrics:
                    val_str = ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
                    msg += f", {val_str}"
                logger.info(msg)
            
            # Checkpoint
            if checkpoint_dir and epoch % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, epoch)
            
            # Early stopping
            if val_metrics and early_stopping_metric in val_metrics:
                current_val = val_metrics[early_stopping_metric]
                if current_val > best_val_metric:
                    best_val_metric = current_val
                    best_epoch = epoch
                    best_U = self.U.copy()
                    best_V = self.V.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        logger.info(f"Best epoch: {best_epoch} with {early_stopping_metric}={best_val_metric:.4f}")
                        break
        
        # Restore best model
        if best_U is not None:
            self.U = best_U
            self.V = best_V
            logger.info(f"Restored best model from epoch {best_epoch}")
        
        self.is_fitted = True
        self.training_end_time = time.time()
        total_duration = self.training_end_time - self.training_start_time
        
        # Build summary
        summary = {
            'total_duration_seconds': total_duration,
            'epochs_completed': len(self.history['epochs']),
            'best_epoch': best_epoch if best_U is not None else len(self.history['epochs']),
            'final_loss': self.history['losses'][-1] if self.history['losses'] else None,
            'final_learning_rate': self.learning_rate,
            'bert_init_used': self.bert_init_used,
            'sentiment_weighting_enabled': self.sentiment_config.use_sentiment_weighting,
            'U_shape': list(self.U.shape),
            'V_shape': list(self.V.shape),
            'sampling_stats': sampler.get_sampling_stats(),
            'projection_info': self.projection_info,
            'history': self.history
        }
        
        logger.info("="*60)
        logger.info("Training Complete")
        logger.info("="*60)
        logger.info(f"Total time: {total_duration:.1f}s")
        logger.info(f"Epochs: {len(self.history['epochs'])}")
        logger.info(f"Final loss: {summary['final_loss']:.4f}")
        logger.info(f"BERT init used: {self.bert_init_used}")
        
        return summary
    
    def _compute_validation_metrics(
        self,
        user_pos_train: Dict[int, Set[int]],
        user_pos_test: Dict[int, Set[int]],
        num_items: int,
        k_values: List[int] = [10, 20],
        sample_users: int = 1000
    ) -> Dict[str, float]:
        """Compute validation metrics on a sample of users."""
        test_users = list(user_pos_test.keys())
        if len(test_users) > sample_users:
            test_users = self.rng.choice(test_users, sample_users, replace=False)
        
        metrics = {}
        
        for k in k_values:
            recalls = []
            ndcgs = []
            
            for u in test_users:
                test_items = user_pos_test.get(u, set())
                if not test_items:
                    continue
                
                train_items = user_pos_train.get(u, set())
                
                # Score all items
                scores = self.U[u] @ self.V.T
                
                # Mask seen items
                for i in train_items:
                    scores[i] = -np.inf
                
                # Get top-k
                top_k = np.argsort(scores)[-k:][::-1]
                
                # Recall@k
                hits = len(set(top_k) & test_items)
                recalls.append(hits / min(len(test_items), k))
                
                # NDCG@k
                dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(top_k) if item in test_items)
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
                ndcgs.append(dcg / idcg if idcg > 0 else 0)
            
            metrics[f'recall@{k}'] = float(np.mean(recalls)) if recalls else 0
            metrics[f'ndcg@{k}'] = float(np.mean(ndcgs)) if ndcgs else 0
        
        return metrics
    
    def _save_checkpoint(self, checkpoint_dir: Path, epoch: int) -> None:
        """Save checkpoint at current epoch."""
        checkpoint_name = f"epoch{epoch:03d}"
        
        np.save(checkpoint_dir / f"bert_bpr_U_{checkpoint_name}.npy", self.U)
        np.save(checkpoint_dir / f"bert_bpr_V_{checkpoint_name}.npy", self.V)
        
        metadata = {
            'epoch': epoch,
            'learning_rate': float(self.learning_rate),
            'loss': float(self.history['losses'][-1]) if self.history['losses'] else None,
            'bert_init_used': self.bert_init_used,
            'U_shape': list(self.U.shape),
            'V_shape': list(self.V.shape)
        }
        
        with open(checkpoint_dir / f"checkpoint_{checkpoint_name}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved: epoch={epoch}")
    
    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get trained user and item embeddings."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.U, self.V
    
    def recommend(
        self,
        user_id: int,
        user_items: Set[int],
        N: int = 10,
        filter_already_liked_items: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User index (u_idx)
            user_items: Set of items user already interacted with
            N: Number of recommendations
            filter_already_liked_items: Whether to exclude seen items
        
        Returns:
            Tuple of (item_indices, scores)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Compute scores
        scores = self.U[user_id] @ self.V.T
        
        # Filter seen items
        if filter_already_liked_items:
            for i in user_items:
                scores[i] = -np.inf
        
        # Get top-N
        top_indices = np.argsort(scores)[-N:][::-1]
        top_scores = scores[top_indices]
        
        return top_indices, top_scores
    
    def save_artifacts(
        self,
        output_dir: Union[str, Path],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save trained model artifacts.
        
        Saves:
        - bert_bpr_U.npy: User factors
        - bert_bpr_V.npy: Item factors
        - bert_bpr_params.json: Hyperparameters
        - bert_bpr_metadata.json: Training metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving BERT-enhanced BPR artifacts to {output_dir}")
        
        # Save embeddings
        np.save(output_dir / "bert_bpr_U.npy", self.U)
        np.save(output_dir / "bert_bpr_V.npy", self.V)
        
        logger.info(f"Saved embeddings: U={self.U.shape}, V={self.V.shape}")
        
        # Save parameters
        params = {
            "model_type": "bert_enhanced_bpr",
            "factors": self.factors,
            "projection_method": self.projection_method,
            "initial_learning_rate": self.initial_lr,
            "regularization": self.regularization,
            "lr_decay": self.lr_decay,
            "lr_decay_every": self.lr_decay_every,
            "sentiment_weighting": self.sentiment_config.use_sentiment_weighting,
        }
        
        with open(output_dir / "bert_bpr_params.json", "w") as f:
            json.dump(params, f, indent=2)
        
        # Save metadata
        training_time = (
            self.training_end_time - self.training_start_time
            if self.training_start_time and self.training_end_time
            else None
        )
        
        meta = {
            "model_type": "bert_enhanced_bpr",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_time_seconds": training_time,
            "bert_initialization": {
                "enabled": self.bert_init_used,
                "projection_method": self.projection_method,
                **self.projection_info,
            },
            "sentiment_config": {
                "enabled": self.sentiment_config.use_sentiment_weighting,
                "confidence_threshold": self.sentiment_config.positive_confidence_threshold,
                "suspicious_penalty": self.sentiment_config.suspicious_penalty,
                "high_confidence_bonus": self.sentiment_config.high_confidence_bonus,
            },
            "num_users": self.num_users,
            "num_items": self.num_items,
            "factors": self.factors,
        }
        
        if metadata:
            meta.update(metadata)
        
        with open(output_dir / "bert_bpr_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info("Artifacts saved successfully")
    
    @classmethod
    def load_artifacts(
        cls,
        artifact_dir: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load saved BERT-enhanced BPR artifacts.
        
        Args:
            artifact_dir: Directory containing saved artifacts
        
        Returns:
            Tuple of (U, V, metadata)
        """
        artifact_dir = Path(artifact_dir)
        
        logger.info(f"Loading BERT-enhanced BPR artifacts from {artifact_dir}")
        
        U = np.load(artifact_dir / "bert_bpr_U.npy")
        V = np.load(artifact_dir / "bert_bpr_V.npy")
        
        with open(artifact_dir / "bert_bpr_metadata.json", "r") as f:
            metadata = json.load(f)
        
        logger.info(
            f"Loaded artifacts: U={U.shape}, V={V.shape}, "
            f"BERT init={metadata.get('bert_initialization', {}).get('enabled', False)}"
        )
        
        return U, V, metadata


def compute_score_range_bpr(
    U: np.ndarray,
    V: np.ndarray,
    sample_size: int = 1000
) -> Dict:
    """
    Compute BPR score range for global normalization (Task 08).
    
    Args:
        U: User factors (num_users, factors)
        V: Item factors (num_items, factors)
        sample_size: Number of users to sample
    
    Returns:
        score_range: Dictionary with min, max, mean, std, p01, p99
    """
    logger.info(f"Computing BPR score range on sample of {sample_size} users")
    
    num_users = U.shape[0]
    sample_users = np.random.choice(num_users, min(sample_size, num_users), replace=False)
    
    U_sample = U[sample_users]
    scores_sample = U_sample @ V.T
    scores_flat = scores_sample.flatten()
    
    score_range = {
        "method": "sample_users",
        "sample_size": len(sample_users),
        "min": float(np.min(scores_flat)),
        "max": float(np.max(scores_flat)),
        "mean": float(np.mean(scores_flat)),
        "std": float(np.std(scores_flat)),
        "p01": float(np.percentile(scores_flat, 1)),
        "p99": float(np.percentile(scores_flat, 99)),
    }
    
    logger.info(
        f"Score range: min={score_range['min']:.3f}, "
        f"max={score_range['max']:.3f}, mean={score_range['mean']:.3f}"
    )
    
    return score_range
