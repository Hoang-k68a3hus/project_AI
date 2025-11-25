"""
BPR Model Training Module (Task 02 - Step 4)

This module implements the BPR (Bayesian Personalized Ranking) training loop:
- SGD updates with BPR loss
- Learning rate decay
- Early stopping based on validation metrics
- Progress tracking and checkpointing

BPR Loss:
    L = -log(sigmoid(x_uij)) + reg * (||U[u]||^2 + ||V[i]||^2 + ||V[j]||^2)
    where x_uij = U[u] @ V[i] - U[u] @ V[j]

Update Rules (SGD):
    U[u]   += lr * ((1 - sigmoid) * (V[i] - V[j]) - reg * U[u])
    V[i]   += lr * ((1 - sigmoid) * U[u] - reg * V[i])
    V[j]   += lr * (-(1 - sigmoid) * U[u] - reg * V[j])
"""

import logging
import time
import json
from typing import Dict, Set, Tuple, Optional, Any, List
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from .sampler import TripletSampler

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """
    Track training metrics across epochs.
    
    Attributes:
        epochs: List of epoch numbers
        losses: List of average losses per epoch
        val_metrics: List of validation metric dicts per epoch
        learning_rates: List of learning rates per epoch
        durations: List of epoch durations in seconds
    """
    epochs: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    val_metrics: List[Dict[str, float]] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
    
    def add_epoch(
        self,
        epoch: int,
        loss: float,
        duration: float,
        learning_rate: float,
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Add epoch metrics."""
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.durations.append(duration)
        self.learning_rates.append(learning_rate)
        self.val_metrics.append(val_metrics or {})
    
    def get_best_epoch(self, metric: str = 'recall@10', higher_is_better: bool = True) -> int:
        """Get epoch with best validation metric."""
        if not self.val_metrics or metric not in self.val_metrics[0]:
            return self.epochs[-1] if self.epochs else 0
        
        values = [m.get(metric, 0) for m in self.val_metrics]
        if higher_is_better:
            return self.epochs[np.argmax(values)]
        else:
            return self.epochs[np.argmin(values)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'epochs': self.epochs,
            'losses': self.losses,
            'val_metrics': self.val_metrics,
            'learning_rates': self.learning_rates,
            'durations': self.durations
        }


class BPRTrainer:
    """
    Train BPR model with SGD.
    
    Implements:
    - Mini-batch SGD with BPR loss
    - Hard negative mining (30% hard + 70% random)
    - Learning rate decay
    - Early stopping
    - Checkpoint saving
    
    Attributes:
        U: User embedding matrix (num_users, factors)
        V: Item embedding matrix (num_items, factors)
        learning_rate: Current learning rate
        regularization: L2 regularization coefficient
        history: Training history tracker
    """
    
    def __init__(
        self,
        U: np.ndarray,
        V: np.ndarray,
        learning_rate: float = 0.05,
        regularization: float = 0.0001,
        lr_decay: float = 0.9,
        lr_decay_every: int = 10,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = 5,
        random_seed: int = 42
    ):
        """
        Initialize BPR trainer.
        
        Args:
            U: User embedding matrix (num_users, factors)
            V: Item embedding matrix (num_items, factors)
            learning_rate: Initial learning rate (default: 0.05)
            regularization: L2 regularization (default: 0.0001)
            lr_decay: Learning rate decay factor (default: 0.9)
            lr_decay_every: Decay LR every N epochs (default: 10)
            checkpoint_dir: Directory for checkpoints (None = no checkpointing)
            checkpoint_interval: Save checkpoint every N epochs
            random_seed: Random seed
        
        Example:
            >>> trainer = BPRTrainer(
            ...     U=np.random.randn(26000, 64) * 0.01,
            ...     V=np.random.randn(2200, 64) * 0.01,
            ...     learning_rate=0.05
            ... )
        """
        self.U = U.astype(np.float32)
        self.V = V.astype(np.float32)
        
        self.initial_lr = learning_rate
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.lr_decay = lr_decay
        self.lr_decay_every = lr_decay_every
        
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_interval = checkpoint_interval
        self.random_seed = random_seed
        
        self.history = TrainingHistory()
        self.best_U = None
        self.best_V = None
        self.best_epoch = 0
        self.is_fitted = False
        
        # Create checkpoint directory
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"BPRTrainer initialized:")
        logger.info(f"  U: {U.shape}, V: {V.shape}")
        logger.info(f"  LR: {learning_rate}, Reg: {regularization}")
        logger.info(f"  LR decay: {lr_decay} every {lr_decay_every} epochs")
    
    def _compute_bpr_loss_batch(
        self,
        users: np.ndarray,
        pos_items: np.ndarray,
        neg_items: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute BPR loss for a batch.
        
        Args:
            users: User indices (batch_size,)
            pos_items: Positive item indices (batch_size,)
            neg_items: Negative item indices (batch_size,)
        
        Returns:
            Tuple of (average_loss, x_uij scores)
        """
        # Compute scores
        user_emb = self.U[users]  # (batch, factors)
        pos_emb = self.V[pos_items]  # (batch, factors)
        neg_emb = self.V[neg_items]  # (batch, factors)
        
        # x_uij = score_pos - score_neg
        x_uij = np.sum(user_emb * pos_emb, axis=1) - np.sum(user_emb * neg_emb, axis=1)
        
        # BPR loss = -log(sigmoid(x_uij))
        # Numerical stability: clip to prevent log(0)
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x_uij, -50, 50)))
        loss = -np.log(np.clip(sigmoid, 1e-10, 1.0))
        
        return loss.mean(), x_uij
    
    def _sgd_update_batch(
        self,
        users: np.ndarray,
        pos_items: np.ndarray,
        neg_items: np.ndarray
    ) -> float:
        """
        Perform SGD update for a batch of triplets.
        
        Args:
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative item indices
        
        Returns:
            Average batch loss
        """
        batch_size = len(users)
        
        # Get embeddings
        user_emb = self.U[users]  # (batch, factors)
        pos_emb = self.V[pos_items]  # (batch, factors)
        neg_emb = self.V[neg_items]  # (batch, factors)
        
        # Compute x_uij
        x_uij = np.sum(user_emb * pos_emb, axis=1) - np.sum(user_emb * neg_emb, axis=1)
        
        # Sigmoid and gradient factor
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x_uij, -50, 50)))
        grad_factor = (1.0 - sigmoid).reshape(-1, 1)  # (batch, 1)
        
        # Compute gradients
        grad_u = grad_factor * (pos_emb - neg_emb) - self.regularization * user_emb
        grad_v_pos = grad_factor * user_emb - self.regularization * pos_emb
        grad_v_neg = -grad_factor * user_emb - self.regularization * neg_emb
        
        # Update embeddings
        self.U[users] += self.learning_rate * grad_u
        
        # Aggregate item gradients (items may repeat)
        np.add.at(self.V, pos_items, self.learning_rate * grad_v_pos)
        np.add.at(self.V, neg_items, self.learning_rate * grad_v_neg)
        
        # Compute loss for logging
        loss = -np.log(np.clip(sigmoid, 1e-10, 1.0)).mean()
        
        return loss
    
    def _update_learning_rate(self, epoch: int):
        """Apply learning rate decay."""
        if self.lr_decay_every > 0 and epoch > 0 and epoch % self.lr_decay_every == 0:
            self.learning_rate *= self.lr_decay
            logger.info(f"Learning rate decayed to {self.learning_rate:.6f}")
    
    def _save_checkpoint(self, epoch: int):
        """Save checkpoint at current epoch."""
        if not self.checkpoint_dir:
            return
        
        checkpoint_name = f"epoch{epoch:03d}"
        
        # Save embeddings
        u_path = self.checkpoint_dir / f"bpr_U_{checkpoint_name}.npy"
        v_path = self.checkpoint_dir / f"bpr_V_{checkpoint_name}.npy"
        
        np.save(u_path, self.U)
        np.save(v_path, self.V)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            # Cast numpy scalars to native Python floats for JSON serialization
            'learning_rate': float(self.learning_rate),
            'loss': float(self.history.losses[-1]) if self.history.losses else None,
            'U_shape': list(self.U.shape),
            'V_shape': list(self.V.shape)
        }
        
        meta_path = self.checkpoint_dir / f"checkpoint_{checkpoint_name}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved: epoch={epoch}")
    
    def fit(
        self,
        positive_pairs: np.ndarray,
        user_pos_sets: Dict[int, Set[int]],
        num_items: int,
        hard_neg_sets: Optional[Dict[int, Set[int]]] = None,
        epochs: int = 50,
        samples_per_positive: int = 5,
        hard_ratio: float = 0.3,
        batch_size: int = 4096,
        val_user_pos_test: Optional[Dict[int, Set[int]]] = None,
        early_stopping_patience: int = 5,
        early_stopping_metric: str = 'recall@10',
        show_progress: bool = True,
        enable_early_stopping: bool = True
    ) -> Dict[str, Any]:
        """
        Train BPR model.
        
        Args:
            positive_pairs: Array of (u, i) positive pairs
            user_pos_sets: Dict mapping u -> set of positive items
            num_items: Total number of items
            hard_neg_sets: Optional hard negative sets
            epochs: Number of training epochs
            samples_per_positive: Samples per positive pair per epoch
            hard_ratio: Fraction of hard negatives
            batch_size: Mini-batch size for SGD
            val_user_pos_test: Optional validation test sets for early stopping
            early_stopping_patience: Epochs without improvement before stopping
            early_stopping_metric: Metric for early stopping
            show_progress: Whether to show progress logs
            enable_early_stopping: Toggle early stopping logic
        
        Returns:
            Training summary dictionary
        
        Example:
            >>> results = trainer.fit(
            ...     positive_pairs=pairs,
            ...     user_pos_sets=pos_sets,
            ...     num_items=2200,
            ...     epochs=50
            ... )
        """
        logger.info("="*60)
        logger.info("Starting BPR Training")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Initialize sampler
        sampler = TripletSampler(
            positive_pairs=positive_pairs,
            user_pos_sets=user_pos_sets,
            num_items=num_items,
            hard_neg_sets=hard_neg_sets,
            hard_ratio=hard_ratio,
            samples_per_positive=samples_per_positive,
            random_seed=self.random_seed
        )
        
        samples_per_epoch = sampler.samples_per_epoch
        logger.info(f"Samples per epoch: {samples_per_epoch:,}")
        logger.info(f"Batch size: {batch_size:,}")
        logger.info(f"Batches per epoch: {samples_per_epoch // batch_size + 1}")
        
        # Early stopping setup
        best_val_metric = -np.inf
        patience_counter = 0
        
        # Training loop
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Sample triplets for this epoch
            triplets = sampler.sample_epoch()
            
            # Shuffle triplets
            np.random.shuffle(triplets)
            
            # Mini-batch training
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i + batch_size]
                users = batch[:, 0].astype(np.int64)
                pos_items = batch[:, 1].astype(np.int64)
                neg_items = batch[:, 2].astype(np.int64)
                
                batch_loss = self._sgd_update_batch(users, pos_items, neg_items)
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
            self.history.add_epoch(
                epoch=epoch,
                loss=epoch_loss,
                duration=epoch_duration,
                learning_rate=self.learning_rate,
                val_metrics=val_metrics
            )
            
            # Logging
            if show_progress:
                msg = f"Epoch {epoch}/{epochs}: loss={epoch_loss:.4f}, time={epoch_duration:.1f}s"
                if val_metrics:
                    val_str = ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
                    msg += f", {val_str}"
                logger.info(msg)
            
            # Checkpoint
            if self.checkpoint_dir and epoch % self.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping
            if (
                enable_early_stopping
                and val_metrics
                and early_stopping_metric in val_metrics
            ):
                current_val = val_metrics[early_stopping_metric]
                if current_val > best_val_metric:
                    best_val_metric = current_val
                    self.best_epoch = epoch
                    self.best_U = self.U.copy()
                    self.best_V = self.V.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        logger.info(f"Best epoch: {self.best_epoch} with {early_stopping_metric}={best_val_metric:.4f}")
                        break
        
        # Restore best model if early stopping was used
        if self.best_U is not None:
            self.U = self.best_U
            self.V = self.best_V
            logger.info(f"Restored best model from epoch {self.best_epoch}")
        
        self.is_fitted = True
        total_duration = time.time() - start_time
        
        # Build summary
        summary = {
            'total_duration_seconds': total_duration,
            'epochs_completed': len(self.history.epochs),
            'best_epoch': self.best_epoch if self.best_U is not None else len(self.history.epochs),
            'final_loss': self.history.losses[-1] if self.history.losses else None,
            'final_learning_rate': self.learning_rate,
            'U_shape': list(self.U.shape),
            'V_shape': list(self.V.shape),
            'sampling_stats': sampler.get_sampling_stats(),
            'history': self.history.to_dict()
        }
        
        logger.info("="*60)
        logger.info("Training Complete")
        logger.info("="*60)
        logger.info(f"Total time: {total_duration:.1f}s")
        logger.info(f"Epochs: {len(self.history.epochs)}")
        logger.info(f"Final loss: {summary['final_loss']:.4f}")
        
        return summary
    
    def _compute_validation_metrics(
        self,
        user_pos_train: Dict[int, Set[int]],
        user_pos_test: Dict[int, Set[int]],
        num_items: int,
        k_values: List[int] = [10, 20],
        sample_users: int = 1000
    ) -> Dict[str, float]:
        """
        Compute validation metrics on a sample of users.
        
        Args:
            user_pos_train: Training positive sets
            user_pos_test: Test positive sets
            num_items: Total items
            k_values: K values for metrics
            sample_users: Number of users to sample for evaluation
        
        Returns:
            Dictionary with recall@k and ndcg@k metrics
        """
        # Sample users for fast evaluation
        test_users = list(user_pos_test.keys())
        if len(test_users) > sample_users:
            test_users = np.random.choice(test_users, sample_users, replace=False)
        
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
                
                # Compute recall@k
                hits = len(set(top_k) & test_items)
                recalls.append(hits / min(len(test_items), k))
                
                # Compute NDCG@k
                dcg = 0.0
                for rank, item in enumerate(top_k):
                    if item in test_items:
                        dcg += 1.0 / np.log2(rank + 2)
                
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_items), k)))
                ndcgs.append(dcg / idcg if idcg > 0 else 0)
            
            metrics[f'recall@{k}'] = np.mean(recalls) if recalls else 0
            metrics[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0
        
        return metrics
    
    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get trained embeddings.
        
        Returns:
            Tuple of (U, V)
        """
        return self.U, self.V
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get training parameters.
        
        Returns:
            Parameter dictionary
        """
        return {
            'initial_learning_rate': self.initial_lr,
            'final_learning_rate': self.learning_rate,
            'regularization': self.regularization,
            'lr_decay': self.lr_decay,
            'lr_decay_every': self.lr_decay_every,
            'factors': self.U.shape[1],
            'num_users': self.U.shape[0],
            'num_items': self.V.shape[0]
        }


def train_bpr_model(
    positive_pairs: np.ndarray,
    user_pos_sets: Dict[int, Set[int]],
    num_users: int,
    num_items: int,
    hard_neg_sets: Optional[Dict[int, Set[int]]] = None,
    factors: int = 64,
    learning_rate: float = 0.05,
    regularization: float = 0.0001,
    epochs: int = 50,
    samples_per_positive: int = 5,
    hard_ratio: float = 0.3,
    random_seed: int = 42,
    checkpoint_dir: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convenience function for end-to-end BPR training.
    
    Args:
        positive_pairs: Array of (u, i) positive pairs
        user_pos_sets: Dict mapping u -> set of positive items
        num_users: Total number of users
        num_items: Total number of items
        hard_neg_sets: Optional hard negative sets
        factors: Embedding dimension
        learning_rate: Initial learning rate
        regularization: L2 regularization
        epochs: Number of epochs
        samples_per_positive: Samples per positive per epoch
        hard_ratio: Fraction of hard negatives
        random_seed: Random seed
        checkpoint_dir: Optional checkpoint directory
    
    Returns:
        Tuple of (U, V, summary)
    
    Example:
        >>> U, V, summary = train_bpr_model(
        ...     positive_pairs=pairs,
        ...     user_pos_sets=pos_sets,
        ...     num_users=26000,
        ...     num_items=2200,
        ...     epochs=50
        ... )
    """
    from .model_init import BPRModelInitializer
    
    # Initialize embeddings
    initializer = BPRModelInitializer(
        num_users=num_users,
        num_items=num_items,
        factors=factors,
        random_seed=random_seed
    )
    U, V = initializer.initialize_embeddings()
    
    # Create trainer
    trainer = BPRTrainer(
        U=U,
        V=V,
        learning_rate=learning_rate,
        regularization=regularization,
        checkpoint_dir=checkpoint_dir,
        random_seed=random_seed
    )
    
    # Train
    summary = trainer.fit(
        positive_pairs=positive_pairs,
        user_pos_sets=user_pos_sets,
        num_items=num_items,
        hard_neg_sets=hard_neg_sets,
        epochs=epochs,
        samples_per_positive=samples_per_positive,
        hard_ratio=hard_ratio
    )
    
    return trainer.U, trainer.V, summary


# Main execution example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("BPR Trainer Demo")
    print("="*60)
    
    # Create synthetic data
    np.random.seed(42)
    num_users, num_items = 1000, 500
    num_pairs = 5000
    
    # Generate random positive pairs
    positive_pairs = np.column_stack([
        np.random.randint(0, num_users, num_pairs),
        np.random.randint(0, num_items, num_pairs)
    ])
    
    # Build user positive sets
    user_pos_sets = {}
    for u, i in positive_pairs:
        if u not in user_pos_sets:
            user_pos_sets[u] = set()
        user_pos_sets[u].add(i)
    
    # Generate hard negatives
    top_popular = set(range(50))
    hard_neg_sets = {}
    for u in user_pos_sets:
        hard_neg_sets[u] = top_popular - user_pos_sets[u]
    
    print(f"\nData summary:")
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Positive pairs: {num_pairs}")
    
    # Train using convenience function
    print("\n" + "-"*60)
    print("Training BPR model")
    print("-"*60)
    
    U, V, summary = train_bpr_model(
        positive_pairs=positive_pairs,
        user_pos_sets=user_pos_sets,
        num_users=num_users,
        num_items=num_items,
        hard_neg_sets=hard_neg_sets,
        factors=32,
        epochs=10,
        samples_per_positive=3
    )
    
    print(f"\nTraining summary:")
    print(f"  Duration: {summary['total_duration_seconds']:.1f}s")
    print(f"  Epochs: {summary['epochs_completed']}")
    print(f"  Final loss: {summary['final_loss']:.4f}")
    print(f"  U shape: {U.shape}")
    print(f"  V shape: {V.shape}")
    
    # Verify embeddings
    print("\n" + "-"*60)
    print("Verifying embeddings")
    print("-"*60)
    
    # Sample recommendations
    user_idx = 0
    scores = U[user_idx] @ V.T
    train_items = user_pos_sets.get(user_idx, set())
    for i in train_items:
        scores[i] = -np.inf
    top_5 = np.argsort(scores)[-5:][::-1]
    
    print(f"Top 5 recommendations for user {user_idx}: {top_5}")
    print(f"User's training items: {sorted(train_items)[:5]}...")
    
    print("\n" + "="*60)
    print("Demo complete!")
