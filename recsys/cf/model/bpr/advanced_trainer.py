"""
Advanced BPR Trainer with Modern Optimization Techniques

This module provides an enhanced trainer for BPR models with:
1. AdamW / AdaGrad / SGD optimizer options
2. Differential regularization (separate for user/item)
3. Dropout for embeddings
4. Learning rate scheduling with warmup
5. Gradient clipping
6. Early stopping with patience
7. Mixed precision training (optional)

Key Features:
- Multiple optimizer choices with configuration
- Warmup + cosine/linear decay schedules
- Per-parameter regularization
- Embedding dropout for regularization
- Comprehensive logging and checkpointing

Author: VieComRec Team
Date: 2025-11-26
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Supported optimizer types."""
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAM = "adam"
    ADAMW = "adamw"


class SchedulerType(Enum):
    """Learning rate scheduler types."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    WARMUP_COSINE = "warmup_cosine"


@dataclass
class OptimizerConfig:
    """
    Configuration for optimizer and learning rate schedule.
    
    Attributes:
        optimizer_type: Type of optimizer (sgd, adagrad, adam, adamw)
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength (for AdamW)
        user_weight_decay: Separate weight decay for user embeddings
        item_weight_decay: Separate weight decay for item embeddings
        momentum: Momentum for SGD (ignored for other optimizers)
        beta1: First moment decay for Adam/AdamW
        beta2: Second moment decay for Adam/AdamW
        epsilon: Small constant for numerical stability
        scheduler_type: Learning rate schedule type
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate after decay
    """
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    user_weight_decay: Optional[float] = None  # If None, use weight_decay
    item_weight_decay: Optional[float] = None  # If None, use weight_decay
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    scheduler_type: SchedulerType = SchedulerType.WARMUP_COSINE
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    def __post_init__(self):
        if self.user_weight_decay is None:
            self.user_weight_decay = self.weight_decay
        if self.item_weight_decay is None:
            self.item_weight_decay = self.weight_decay


@dataclass
class TrainingConfig:
    """
    Configuration for BPR training process.
    
    Attributes:
        factors: Embedding dimension
        epochs: Maximum training epochs
        samples_per_positive: Negative samples per positive pair per epoch
        batch_size: Mini-batch size for gradient updates
        dropout_rate: Dropout probability for embeddings
        gradient_clip: Maximum gradient norm (0 to disable)
        early_stopping_patience: Epochs without improvement before stopping
        early_stopping_min_delta: Minimum improvement to reset patience
        eval_every: Evaluate metrics every N epochs
        checkpoint_every: Save checkpoint every N epochs
        random_seed: Random seed
    """
    factors: int = 64
    epochs: int = 50
    samples_per_positive: int = 5
    batch_size: int = 1024
    dropout_rate: float = 0.1
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    eval_every: int = 5
    checkpoint_every: int = 10
    random_seed: int = 42


class EmbeddingDropout:
    """
    Dropout for embeddings during training.
    
    Randomly zeros entire embedding dimensions to prevent co-adaptation.
    """
    
    def __init__(self, dropout_rate: float = 0.1, random_seed: int = 42):
        self.dropout_rate = dropout_rate
        self.rng = np.random.default_rng(random_seed)
        self.training = True
    
    def __call__(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply dropout to embeddings.
        
        Args:
            embeddings: Input embeddings (batch, factors)
        
        Returns:
            Tuple of (dropped embeddings, dropout mask)
        """
        if not self.training or self.dropout_rate == 0:
            return embeddings, None
        
        # Create dropout mask (keep probability = 1 - dropout_rate)
        keep_prob = 1.0 - self.dropout_rate
        mask = self.rng.random(embeddings.shape) < keep_prob
        
        # Apply mask and scale
        dropped = embeddings * mask / keep_prob
        
        return dropped, mask
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False


class AdaGradOptimizer:
    """
    AdaGrad optimizer with per-parameter learning rates.
    
    Adapts learning rate based on historical gradient magnitudes.
    Good for sparse data like recommender systems.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # Accumulated squared gradients
        self.cache: Dict[str, np.ndarray] = {}
    
    def update(
        self,
        param_name: str,
        param: np.ndarray,
        grad: np.ndarray,
        indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update parameters using AdaGrad.
        
        Args:
            param_name: Name of parameter for caching
            param: Current parameter values
            grad: Gradient
            indices: Optional sparse indices for update
        
        Returns:
            Updated parameters
        """
        # Initialize cache if needed
        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(param)
        
        if indices is not None:
            # Sparse update
            self.cache[param_name][indices] += grad ** 2
            adapted_lr = self.learning_rate / (np.sqrt(self.cache[param_name][indices]) + self.epsilon)
            param[indices] -= adapted_lr * (grad + self.weight_decay * param[indices])
        else:
            # Dense update
            self.cache[param_name] += grad ** 2
            adapted_lr = self.learning_rate / (np.sqrt(self.cache[param_name]) + self.epsilon)
            param -= adapted_lr * (grad + self.weight_decay * param)
        
        return param


class AdamWOptimizer:
    """
    AdamW optimizer with decoupled weight decay.
    
    Unlike Adam, AdamW applies weight decay directly to parameters
    rather than including it in gradient computation. This leads to
    better generalization.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # Moment estimates
        self.m: Dict[str, np.ndarray] = {}  # First moment
        self.v: Dict[str, np.ndarray] = {}  # Second moment
        self.t = 0  # Timestep
    
    def step(self):
        """Increment timestep."""
        self.t += 1
    
    def update(
        self,
        param_name: str,
        param: np.ndarray,
        grad: np.ndarray,
        indices: Optional[np.ndarray] = None,
        custom_weight_decay: Optional[float] = None
    ) -> np.ndarray:
        """
        Update parameters using AdamW.
        
        Args:
            param_name: Name of parameter for caching
            param: Current parameter values
            grad: Gradient
            indices: Optional sparse indices for update
            custom_weight_decay: Override default weight decay
        
        Returns:
            Updated parameters
        """
        wd = custom_weight_decay if custom_weight_decay is not None else self.weight_decay
        
        # Initialize moments if needed
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
        
        if indices is not None:
            # Sparse update for efficiency
            g = grad
            
            # Update biased moments
            self.m[param_name][indices] = self.beta1 * self.m[param_name][indices] + (1 - self.beta1) * g
            self.v[param_name][indices] = self.beta2 * self.v[param_name][indices] + (1 - self.beta2) * (g ** 2)
            
            # Bias correction
            m_hat = self.m[param_name][indices] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name][indices] / (1 - self.beta2 ** self.t)
            
            # AdamW update: Adam update + decoupled weight decay
            param[indices] -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))
            param[indices] -= self.learning_rate * wd * param[indices]
        else:
            # Dense update
            g = grad
            
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * g
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            
            param -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))
            param -= self.learning_rate * wd * param
        
        return param


class LearningRateScheduler:
    """
    Learning rate scheduler with various decay strategies.
    """
    
    def __init__(
        self,
        initial_lr: float,
        total_epochs: int,
        scheduler_type: SchedulerType = SchedulerType.WARMUP_COSINE,
        warmup_epochs: int = 5,
        min_lr: float = 1e-6
    ):
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.current_lr = initial_lr
    
    def get_lr(self, epoch: int) -> float:
        """
        Get learning rate for current epoch.
        
        Args:
            epoch: Current epoch (0-indexed)
        
        Returns:
            Learning rate for this epoch
        """
        if self.scheduler_type == SchedulerType.CONSTANT:
            return self.initial_lr
        
        # Warmup phase
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            self.current_lr = self.initial_lr * warmup_factor
            return self.current_lr
        
        # Post-warmup decay
        decay_epoch = epoch - self.warmup_epochs
        decay_total = self.total_epochs - self.warmup_epochs
        
        if decay_total <= 0:
            return self.initial_lr
        
        progress = decay_epoch / decay_total
        
        if self.scheduler_type == SchedulerType.LINEAR:
            self.current_lr = self.initial_lr * (1 - progress)
        
        elif self.scheduler_type == SchedulerType.COSINE:
            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress)) / 2
        
        elif self.scheduler_type == SchedulerType.WARMUP_COSINE:
            self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress)) / 2
        
        elif self.scheduler_type == SchedulerType.EXPONENTIAL:
            decay_rate = 0.95
            self.current_lr = self.initial_lr * (decay_rate ** decay_epoch)
        
        self.current_lr = max(self.current_lr, self.min_lr)
        return self.current_lr


class AdvancedBPRTrainer:
    """
    Advanced BPR Trainer with modern optimization techniques.
    
    Features:
    - AdamW / AdaGrad / SGD optimizers
    - Differential regularization (separate for user/item)
    - Embedding dropout
    - Learning rate scheduling with warmup
    - Gradient clipping
    - Early stopping
    
    Usage:
        >>> trainer = AdvancedBPRTrainer(
        ...     num_users=26000,
        ...     num_items=2200,
        ...     training_config=TrainingConfig(factors=64, epochs=50),
        ...     optimizer_config=OptimizerConfig(optimizer_type=OptimizerType.ADAMW)
        ... )
        >>> trainer.fit(sampler, eval_data=test_data)
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        training_config: Optional[TrainingConfig] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        init_user_embeddings: Optional[np.ndarray] = None,
        init_item_embeddings: Optional[np.ndarray] = None,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize advanced BPR trainer.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            training_config: Training configuration
            optimizer_config: Optimizer configuration
            init_user_embeddings: Optional initial user embeddings (BERT)
            init_item_embeddings: Optional initial item embeddings (BERT)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.num_users = num_users
        self.num_items = num_items
        self.config = training_config or TrainingConfig()
        self.opt_config = optimizer_config or OptimizerConfig()
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize random state
        self.rng = np.random.default_rng(self.config.random_seed)
        
        # Initialize embeddings
        self._init_embeddings(init_user_embeddings, init_item_embeddings)
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Initialize scheduler
        self.scheduler = LearningRateScheduler(
            initial_lr=self.opt_config.learning_rate,
            total_epochs=self.config.epochs,
            scheduler_type=self.opt_config.scheduler_type,
            warmup_epochs=self.opt_config.warmup_epochs,
            min_lr=self.opt_config.min_lr
        )
        
        # Initialize dropout
        self.dropout = EmbeddingDropout(
            dropout_rate=self.config.dropout_rate,
            random_seed=self.config.random_seed
        )
        
        # Training state
        self.current_epoch = 0
        self.best_metric = -np.inf
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Metrics history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'eval_metric': [],
            'learning_rate': []
        }
        
        logger.info(f"AdvancedBPRTrainer initialized:")
        logger.info(f"  Users: {num_users:,}, Items: {num_items:,}")
        logger.info(f"  Factors: {self.config.factors}")
        logger.info(f"  Optimizer: {self.opt_config.optimizer_type.value}")
        logger.info(f"  LR: {self.opt_config.learning_rate}, Schedule: {self.opt_config.scheduler_type.value}")
        logger.info(f"  Dropout: {self.config.dropout_rate}")
    
    def _init_embeddings(
        self,
        init_user: Optional[np.ndarray],
        init_item: Optional[np.ndarray]
    ):
        """Initialize user and item embeddings."""
        factors = self.config.factors
        scale = 0.01
        
        # User embeddings
        if init_user is not None:
            if init_user.shape != (self.num_users, factors):
                raise ValueError(f"User embeddings shape {init_user.shape} != ({self.num_users}, {factors})")
            self.U = init_user.copy()
            logger.info(f"  Initialized user embeddings from provided values")
        else:
            self.U = self.rng.normal(0, scale, (self.num_users, factors))
            logger.info(f"  Initialized user embeddings randomly (scale={scale})")
        
        # Item embeddings
        if init_item is not None:
            if init_item.shape != (self.num_items, factors):
                raise ValueError(f"Item embeddings shape {init_item.shape} != ({self.num_items}, {factors})")
            self.V = init_item.copy()
            logger.info(f"  Initialized item embeddings from provided values")
        else:
            self.V = self.rng.normal(0, scale, (self.num_items, factors))
            logger.info(f"  Initialized item embeddings randomly (scale={scale})")
        
        # Store best embeddings
        self.best_U = self.U.copy()
        self.best_V = self.V.copy()
    
    def _init_optimizer(self):
        """Initialize optimizer based on configuration."""
        if self.opt_config.optimizer_type == OptimizerType.ADAMW:
            self.optimizer = AdamWOptimizer(
                learning_rate=self.opt_config.learning_rate,
                beta1=self.opt_config.beta1,
                beta2=self.opt_config.beta2,
                epsilon=self.opt_config.epsilon,
                weight_decay=self.opt_config.weight_decay
            )
        elif self.opt_config.optimizer_type == OptimizerType.ADAGRAD:
            self.optimizer = AdaGradOptimizer(
                learning_rate=self.opt_config.learning_rate,
                epsilon=self.opt_config.epsilon,
                weight_decay=self.opt_config.weight_decay
            )
        else:
            # SGD (manual updates in _sgd_update)
            self.optimizer = None
    
    def _compute_bpr_loss(
        self,
        u_idx: np.ndarray,
        i_pos_idx: np.ndarray,
        i_neg_idx: np.ndarray,
        apply_dropout: bool = True
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute BPR loss and gradients for a batch.
        
        BPR Loss: -log(sigmoid(x_uij)) where x_uij = <u, i+> - <u, i->
        
        Args:
            u_idx: User indices (batch_size,)
            i_pos_idx: Positive item indices (batch_size,)
            i_neg_idx: Negative item indices (batch_size,)
            apply_dropout: Whether to apply embedding dropout
        
        Returns:
            Tuple of (loss, grad_u, grad_i_pos, grad_i_neg, x_uij, unique indices)
        """
        # Get embeddings
        u_emb = self.U[u_idx]
        i_pos_emb = self.V[i_pos_idx]
        i_neg_emb = self.V[i_neg_idx]
        
        # Apply dropout during training
        if apply_dropout:
            u_emb, _ = self.dropout(u_emb)
            i_pos_emb, _ = self.dropout(i_pos_emb)
            i_neg_emb, _ = self.dropout(i_neg_emb)
        
        # Compute preference difference: x_uij = <u, i+> - <u, i->
        x_uij = np.sum(u_emb * i_pos_emb, axis=1) - np.sum(u_emb * i_neg_emb, axis=1)
        
        # Sigmoid and loss
        sigmoid_x = 1.0 / (1.0 + np.exp(-np.clip(x_uij, -30, 30)))
        loss = -np.mean(np.log(sigmoid_x + 1e-10))
        
        # Gradient: d_loss/d_x = sigmoid(-x) = 1 - sigmoid(x)
        d_loss = (1.0 - sigmoid_x).reshape(-1, 1)
        
        # Gradients for embeddings
        grad_u = -d_loss * (i_pos_emb - i_neg_emb)
        grad_i_pos = -d_loss * u_emb
        grad_i_neg = d_loss * u_emb
        
        return loss, grad_u, grad_i_pos, grad_i_neg, x_uij
    
    def _clip_gradients(
        self,
        *gradients: np.ndarray,
        max_norm: float
    ) -> List[np.ndarray]:
        """
        Clip gradients by global norm.
        
        Args:
            gradients: Gradient arrays
            max_norm: Maximum allowed norm
        
        Returns:
            Clipped gradients
        """
        if max_norm <= 0:
            return list(gradients)
        
        # Compute global norm
        total_norm = 0.0
        for g in gradients:
            total_norm += np.sum(g ** 2)
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            return [g * clip_coef for g in gradients]
        
        return list(gradients)
    
    def _update_embeddings(
        self,
        u_idx: np.ndarray,
        i_pos_idx: np.ndarray,
        i_neg_idx: np.ndarray,
        grad_u: np.ndarray,
        grad_i_pos: np.ndarray,
        grad_i_neg: np.ndarray,
        lr: float
    ):
        """
        Update embeddings using configured optimizer.
        
        Args:
            u_idx: User indices
            i_pos_idx: Positive item indices
            i_neg_idx: Negative item indices
            grad_u: User gradients
            grad_i_pos: Positive item gradients
            grad_i_neg: Negative item gradients
            lr: Current learning rate
        """
        if isinstance(self.optimizer, AdamWOptimizer):
            self.optimizer.step()
            
            # Aggregate gradients for unique indices
            unique_u, inv_u = np.unique(u_idx, return_inverse=True)
            agg_grad_u = np.zeros((len(unique_u), self.config.factors))
            np.add.at(agg_grad_u, inv_u, grad_u)
            
            unique_i_pos, inv_pos = np.unique(i_pos_idx, return_inverse=True)
            agg_grad_i_pos = np.zeros((len(unique_i_pos), self.config.factors))
            np.add.at(agg_grad_i_pos, inv_pos, grad_i_pos)
            
            unique_i_neg, inv_neg = np.unique(i_neg_idx, return_inverse=True)
            agg_grad_i_neg = np.zeros((len(unique_i_neg), self.config.factors))
            np.add.at(agg_grad_i_neg, inv_neg, grad_i_neg)
            
            # Update with AdamW (different weight decay for user/item)
            self.U = self.optimizer.update(
                'U', self.U, agg_grad_u, unique_u,
                custom_weight_decay=self.opt_config.user_weight_decay
            )
            self.V = self.optimizer.update(
                'V_pos', self.V, agg_grad_i_pos, unique_i_pos,
                custom_weight_decay=self.opt_config.item_weight_decay
            )
            self.V = self.optimizer.update(
                'V_neg', self.V, agg_grad_i_neg, unique_i_neg,
                custom_weight_decay=self.opt_config.item_weight_decay
            )
        
        elif isinstance(self.optimizer, AdaGradOptimizer):
            # Similar aggregation
            unique_u, inv_u = np.unique(u_idx, return_inverse=True)
            agg_grad_u = np.zeros((len(unique_u), self.config.factors))
            np.add.at(agg_grad_u, inv_u, grad_u)
            
            unique_i_pos, inv_pos = np.unique(i_pos_idx, return_inverse=True)
            agg_grad_i_pos = np.zeros((len(unique_i_pos), self.config.factors))
            np.add.at(agg_grad_i_pos, inv_pos, grad_i_pos)
            
            unique_i_neg, inv_neg = np.unique(i_neg_idx, return_inverse=True)
            agg_grad_i_neg = np.zeros((len(unique_i_neg), self.config.factors))
            np.add.at(agg_grad_i_neg, inv_neg, grad_i_neg)
            
            self.U = self.optimizer.update('U', self.U, agg_grad_u, unique_u)
            self.V = self.optimizer.update('V_pos', self.V, agg_grad_i_pos, unique_i_pos)
            self.V = self.optimizer.update('V_neg', self.V, agg_grad_i_neg, unique_i_neg)
        
        else:
            # SGD update
            self._sgd_update(
                u_idx, i_pos_idx, i_neg_idx,
                grad_u, grad_i_pos, grad_i_neg, lr
            )
    
    def _sgd_update(
        self,
        u_idx: np.ndarray,
        i_pos_idx: np.ndarray,
        i_neg_idx: np.ndarray,
        grad_u: np.ndarray,
        grad_i_pos: np.ndarray,
        grad_i_neg: np.ndarray,
        lr: float
    ):
        """
        Vanilla SGD update with L2 regularization.
        
        Args:
            u_idx: User indices
            i_pos_idx: Positive item indices
            i_neg_idx: Negative item indices
            grad_u: User gradients
            grad_i_pos: Positive item gradients
            grad_i_neg: Negative item gradients
            lr: Learning rate
        """
        reg_u = self.opt_config.user_weight_decay
        reg_i = self.opt_config.item_weight_decay
        
        # Update with regularization
        for i in range(len(u_idx)):
            u = u_idx[i]
            i_pos = i_pos_idx[i]
            i_neg = i_neg_idx[i]
            
            self.U[u] -= lr * (grad_u[i] + reg_u * self.U[u])
            self.V[i_pos] -= lr * (grad_i_pos[i] + reg_i * self.V[i_pos])
            self.V[i_neg] -= lr * (grad_i_neg[i] + reg_i * self.V[i_neg])
    
    def train_epoch(
        self,
        triplets: np.ndarray,
        verbose: bool = False
    ) -> float:
        """
        Train one epoch.
        
        Args:
            triplets: Array of (user, pos_item, neg_item) triplets
            verbose: Print progress
        
        Returns:
            Average loss for this epoch
        """
        self.dropout.train()
        
        # Get current learning rate
        lr = self.scheduler.get_lr(self.current_epoch)
        
        # Shuffle triplets
        self.rng.shuffle(triplets)
        
        num_samples = len(triplets)
        batch_size = self.config.batch_size
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        total_loss = 0.0
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)
            
            batch = triplets[start:end]
            u_idx = batch[:, 0].astype(np.int64)
            i_pos_idx = batch[:, 1].astype(np.int64)
            i_neg_idx = batch[:, 2].astype(np.int64)
            
            # Compute loss and gradients
            loss, grad_u, grad_i_pos, grad_i_neg, _ = self._compute_bpr_loss(
                u_idx, i_pos_idx, i_neg_idx, apply_dropout=True
            )
            
            total_loss += loss * len(batch)
            
            # Clip gradients
            if self.config.gradient_clip > 0:
                grad_u, grad_i_pos, grad_i_neg = self._clip_gradients(
                    grad_u, grad_i_pos, grad_i_neg,
                    max_norm=self.config.gradient_clip
                )
            
            # Update embeddings
            self._update_embeddings(
                u_idx, i_pos_idx, i_neg_idx,
                grad_u, grad_i_pos, grad_i_neg, lr
            )
            
            if verbose and batch_idx % 100 == 0:
                logger.info(f"  Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}, LR: {lr:.6f}")
        
        avg_loss = total_loss / num_samples
        return avg_loss
    
    def evaluate(
        self,
        user_pos_test: Dict[int, set],
        user_pos_train: Dict[int, set],
        k_values: List[int] = [10, 20],
        num_eval_users: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            user_pos_test: Dict of user -> set of positive test items
            user_pos_train: Dict of user -> set of positive train items
            k_values: Values of K for Recall@K, NDCG@K
            num_eval_users: Maximum users to evaluate (for speed)
        
        Returns:
            Dict of metric_name -> value
        """
        self.dropout.eval()
        
        metrics = {}
        
        # Sample users for evaluation
        eval_users = list(user_pos_test.keys())
        if len(eval_users) > num_eval_users:
            eval_users = self.rng.choice(eval_users, num_eval_users, replace=False)
        
        for k in k_values:
            recalls = []
            ndcgs = []
            
            for u_idx in eval_users:
                test_items = user_pos_test.get(u_idx, set())
                train_items = user_pos_train.get(u_idx, set())
                
                if len(test_items) == 0:
                    continue
                
                # Score all items
                scores = self.U[u_idx] @ self.V.T
                
                # Mask train items
                for item in train_items:
                    scores[item] = -np.inf
                
                # Get top-k items
                top_k_items = np.argsort(scores)[-k:][::-1]
                
                # Compute Recall@k
                hits = len(set(top_k_items) & test_items)
                recall = hits / min(k, len(test_items))
                recalls.append(recall)
                
                # Compute NDCG@k
                dcg = 0.0
                for rank, item in enumerate(top_k_items):
                    if item in test_items:
                        dcg += 1.0 / np.log2(rank + 2)
                
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(test_items))))
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcgs.append(ndcg)
            
            metrics[f'Recall@{k}'] = np.mean(recalls) if recalls else 0.0
            metrics[f'NDCG@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
        
        return metrics
    
    def fit(
        self,
        sampler,
        user_pos_test: Optional[Dict[int, set]] = None,
        user_pos_train: Optional[Dict[int, set]] = None,
        eval_metric: str = 'Recall@10',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            sampler: Triplet sampler (should have sample_epoch() method)
            user_pos_test: Test data for evaluation
            user_pos_train: Train data (for filtering in evaluation)
            eval_metric: Metric to use for early stopping
            verbose: Print progress
        
        Returns:
            Training history
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Advanced BPR Training")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Update sampler epoch (for dynamic sampling)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
            
            # Sample triplets
            # Pass model embeddings for dynamic resampling if supported
            if hasattr(sampler, 'sample_epoch'):
                triplets = sampler.sample_epoch(
                    model_scores=(self.U, self.V) if epoch > 0 else None
                )
            else:
                triplets = sampler.sample_epoch()
            
            # Train epoch
            epoch_loss = self.train_epoch(triplets, verbose=False)
            
            # Get current LR
            current_lr = self.scheduler.get_lr(epoch)
            
            # Log
            self.history['train_loss'].append(epoch_loss)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            if verbose:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs} - "
                          f"Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}, "
                          f"Time: {epoch_time:.1f}s")
            
            # Evaluation
            if user_pos_test is not None and (epoch + 1) % self.config.eval_every == 0:
                metrics = self.evaluate(user_pos_test, user_pos_train or {})
                
                metric_val = metrics.get(eval_metric, 0.0)
                self.history['eval_metric'].append(metric_val)
                
                if verbose:
                    metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    logger.info(f"  Eval: {metrics_str}")
                
                # Early stopping check
                if metric_val > self.best_metric + self.config.early_stopping_min_delta:
                    self.best_metric = metric_val
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    self.best_U = self.U.copy()
                    self.best_V = self.V.copy()
                    logger.info(f"  New best {eval_metric}: {metric_val:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Checkpoint
            if self.checkpoint_dir and (epoch + 1) % self.config.checkpoint_every == 0:
                self._save_checkpoint(epoch)
        
        # Restore best embeddings
        self.U = self.best_U
        self.V = self.best_V
        
        total_time = time.time() - start_time
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Complete!")
        logger.info(f"  Total time: {total_time/60:.1f} minutes")
        logger.info(f"  Best {eval_metric}: {self.best_metric:.4f} at epoch {self.best_epoch+1}")
        logger.info(f"{'='*60}")
        
        return {
            'history': self.history,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'total_time': total_time
        }
    
    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'U': self.U,
            'V': self.V,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'history': self.history
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.npz"
        np.savez(checkpoint_path, **checkpoint)
        logger.info(f"  Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint."""
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        self.current_epoch = int(checkpoint['epoch'])
        self.U = checkpoint['U']
        self.V = checkpoint['V']
        self.best_metric = float(checkpoint['best_metric'])
        self.best_epoch = int(checkpoint['best_epoch'])
        self.history = checkpoint['history'].item()
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get user and item embeddings."""
        return self.U.copy(), self.V.copy()
    
    def predict_score(self, user_idx: int, item_idx: int) -> float:
        """Predict score for a user-item pair."""
        return float(np.dot(self.U[user_idx], self.V[item_idx]))
    
    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_items: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """
        Get top-k recommendations for a user.
        
        Args:
            user_idx: User index
            k: Number of recommendations
            exclude_items: Items to exclude (e.g., already purchased)
        
        Returns:
            List of (item_idx, score) tuples
        """
        scores = self.U[user_idx] @ self.V.T
        
        if exclude_items:
            for item in exclude_items:
                scores[item] = -np.inf
        
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        return [(int(idx), float(scores[idx])) for idx in top_k_indices]
