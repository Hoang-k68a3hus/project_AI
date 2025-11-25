"""
BPR Triplet Sampling Module (Task 02 - Step 2)

This module handles negative sampling for BPR training:
- Dual-strategy hard negative mining (explicit + implicit)
- Mixed sampling: 30% hard negatives + 70% random
- Efficient batch sampling for training

Sampling Strategy:
1. For each positive (u, i_pos), sample a negative i_neg
2. 30% of negatives from hard_neg_sets (explicit + implicit)
3. 70% of negatives uniformly random from unseen items
4. Fallback to pure random if user has no hard negatives
"""

import logging
from typing import Dict, Set, Tuple, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class HardNegativeMixer:
    """
    Mix hard and random negatives for BPR training.
    
    Implements the dual hard negative strategy:
    - Explicit: Items user rated <= 3 (bought but disliked)
    - Implicit: Top-K popular items user didn't interact with
    
    Sampling ratio: hard_ratio% hard + (1-hard_ratio)% random
    """
    
    def __init__(
        self,
        hard_neg_sets: Dict[int, Set[int]],
        hard_ratio: float = 0.3,
        random_seed: int = 42
    ):
        """
        Initialize hard negative mixer.
        
        Args:
            hard_neg_sets: Dict mapping u_idx -> Set of hard negative i_idx
            hard_ratio: Fraction of samples from hard negatives (default: 0.3)
            random_seed: Random seed for reproducibility
        
        Example:
            >>> mixer = HardNegativeMixer(hard_neg_sets, hard_ratio=0.3)
            >>> neg_idx = mixer.sample_negative(
            ...     user_idx=42,
            ...     positive_set={10, 20, 30},
            ...     num_items=2000
            ... )
        """
        self.hard_neg_sets = hard_neg_sets
        self.hard_ratio = hard_ratio
        self.rng = np.random.default_rng(random_seed)
        
        # Pre-convert sets to arrays for faster sampling
        self.hard_neg_arrays = {
            u: np.array(list(items)) for u, items in hard_neg_sets.items()
            if len(items) > 0
        }
        
        # Statistics tracking
        self.stats = {
            'hard_samples': 0,
            'random_samples': 0,
            'fallback_to_random': 0
        }
        
        logger.info(f"HardNegativeMixer initialized: hard_ratio={hard_ratio:.1%}")
        logger.info(f"Users with hard negatives: {len(self.hard_neg_arrays):,}")
    
    def sample_negative(
        self,
        user_idx: int,
        positive_set: Set[int],
        num_items: int,
        use_hard: Optional[bool] = None,
        max_attempts: int = 100
    ) -> int:
        """
        Sample a negative item for a user.
        
        Args:
            user_idx: User index
            positive_set: Set of positive item indices to exclude
            num_items: Total number of items
            use_hard: Override random selection (True=force hard, False=force random)
            max_attempts: Maximum sampling attempts before raising error
        
        Returns:
            Sampled negative item index
        """
        # Decide whether to use hard negative
        if use_hard is None:
            use_hard = self.rng.random() < self.hard_ratio
        
        # Try hard negative sampling
        if use_hard and user_idx in self.hard_neg_arrays:
            hard_arr = self.hard_neg_arrays[user_idx]
            
            # Filter out positive items from hard negatives
            valid_hard = hard_arr[~np.isin(hard_arr, list(positive_set))]
            
            if len(valid_hard) > 0:
                neg_idx = self.rng.choice(valid_hard)
                self.stats['hard_samples'] += 1
                return int(neg_idx)
            else:
                self.stats['fallback_to_random'] += 1
        
        # Random negative sampling
        for _ in range(max_attempts):
            neg_idx = self.rng.integers(0, num_items)
            if neg_idx not in positive_set:
                self.stats['random_samples'] += 1
                return int(neg_idx)
        
        # Final fallback: find any valid negative
        all_items = set(range(num_items))
        valid_items = list(all_items - positive_set)
        if valid_items:
            self.stats['random_samples'] += 1
            return int(self.rng.choice(valid_items))
        
        raise ValueError(f"Cannot sample negative for user {user_idx}: no valid items")
    
    def sample_negatives_batch(
        self,
        user_indices: np.ndarray,
        user_pos_sets: Dict[int, Set[int]],
        num_items: int,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample negative items for a batch of users.
        
        Args:
            user_indices: Array of user indices
            user_pos_sets: Dict mapping u_idx -> Set of positive items
            num_items: Total number of items
            batch_size: Process in chunks (memory optimization)
        
        Returns:
            Array of sampled negative item indices
        """
        n_samples = len(user_indices)
        negatives = np.zeros(n_samples, dtype=np.int64)
        
        # Decide hard vs random for entire batch upfront
        use_hard_mask = self.rng.random(n_samples) < self.hard_ratio
        
        for i, user_idx in enumerate(user_indices):
            user_idx = int(user_idx)
            pos_set = user_pos_sets.get(user_idx, set())
            negatives[i] = self.sample_negative(
                user_idx=user_idx,
                positive_set=pos_set,
                num_items=num_items,
                use_hard=use_hard_mask[i]
            )
        
        return negatives
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get sampling statistics.
        
        Returns:
            Dictionary with sampling counts and ratios
        """
        total = self.stats['hard_samples'] + self.stats['random_samples']
        hard_pct = self.stats['hard_samples'] / total * 100 if total > 0 else 0
        
        return {
            **self.stats,
            'total_samples': total,
            'actual_hard_ratio': hard_pct,
            'target_hard_ratio': self.hard_ratio * 100
        }
    
    def reset_stats(self):
        """Reset sampling statistics."""
        self.stats = {
            'hard_samples': 0,
            'random_samples': 0,
            'fallback_to_random': 0
        }


class TripletSampler:
    """
    Sample (user, positive, negative) triplets for BPR training.
    
    Generates triplets efficiently for mini-batch SGD:
    - Each epoch samples `samples_per_epoch` triplets
    - Uses HardNegativeMixer for intelligent negative sampling
    - Shuffles triplets for better gradient estimation
    """
    
    def __init__(
        self,
        positive_pairs: np.ndarray,
        user_pos_sets: Dict[int, Set[int]],
        num_items: int,
        hard_neg_sets: Optional[Dict[int, Set[int]]] = None,
        hard_ratio: float = 0.3,
        samples_per_positive: int = 5,
        random_seed: int = 42
    ):
        """
        Initialize triplet sampler.
        
        Args:
            positive_pairs: Array of shape (N, 2) with [u_idx, i_idx]
            user_pos_sets: Dict mapping u_idx -> Set of positive items
            num_items: Total number of items
            hard_neg_sets: Optional hard negative sets
            hard_ratio: Fraction of hard negatives (default: 0.3)
            samples_per_positive: Multiplier for samples per epoch
            random_seed: Random seed
        
        Example:
            >>> sampler = TripletSampler(
            ...     positive_pairs=pairs,
            ...     user_pos_sets=pos_sets,
            ...     num_items=2000,
            ...     hard_neg_sets=hard_negs
            ... )
            >>> triplets = sampler.sample_epoch()
            >>> print(f"Sampled {len(triplets)} triplets")
        """
        self.positive_pairs = positive_pairs
        self.user_pos_sets = user_pos_sets
        self.num_items = num_items
        self.samples_per_positive = samples_per_positive
        self.rng = np.random.default_rng(random_seed)
        
        # Setup hard negative mixer
        self.mixer = HardNegativeMixer(
            hard_neg_sets=hard_neg_sets or {},
            hard_ratio=hard_ratio,
            random_seed=random_seed
        )
        
        # Compute samples per epoch
        self.num_positives = len(positive_pairs)
        self.samples_per_epoch = self.num_positives * samples_per_positive
        
        logger.info(f"TripletSampler initialized:")
        logger.info(f"  Positive pairs: {self.num_positives:,}")
        logger.info(f"  Samples per epoch: {self.samples_per_epoch:,}")
        logger.info(f"  Items: {num_items:,}")
        logger.info(f"  Hard ratio: {hard_ratio:.1%}")
    
    def sample_epoch(self, shuffle: bool = True) -> np.ndarray:
        """
        Sample triplets for one epoch.
        
        Args:
            shuffle: Whether to shuffle triplets
        
        Returns:
            Array of shape (samples_per_epoch, 3) with [u, i_pos, i_neg]
        """
        # Sample positive pairs with replacement
        pair_indices = self.rng.choice(
            self.num_positives,
            size=self.samples_per_epoch,
            replace=True
        )
        
        sampled_pairs = self.positive_pairs[pair_indices]
        users = sampled_pairs[:, 0]
        positives = sampled_pairs[:, 1]
        
        # Sample negatives
        negatives = self.mixer.sample_negatives_batch(
            user_indices=users,
            user_pos_sets=self.user_pos_sets,
            num_items=self.num_items
        )
        
        # Stack triplets
        triplets = np.column_stack([users, positives, negatives])
        
        # Shuffle
        if shuffle:
            self.rng.shuffle(triplets)
        
        return triplets
    
    def sample_batch(self, batch_size: int) -> np.ndarray:
        """
        Sample a single batch of triplets.
        
        Args:
            batch_size: Number of triplets to sample
        
        Returns:
            Array of shape (batch_size, 3) with [u, i_pos, i_neg]
        """
        # Sample positive pairs
        pair_indices = self.rng.choice(
            self.num_positives,
            size=batch_size,
            replace=True
        )
        
        sampled_pairs = self.positive_pairs[pair_indices]
        users = sampled_pairs[:, 0]
        positives = sampled_pairs[:, 1]
        
        # Sample negatives
        negatives = self.mixer.sample_negatives_batch(
            user_indices=users,
            user_pos_sets=self.user_pos_sets,
            num_items=self.num_items
        )
        
        return np.column_stack([users, positives, negatives])
    
    def get_sampling_stats(self) -> Dict[str, any]:
        """
        Get sampling statistics.
        
        Returns:
            Dictionary with sampler configuration and stats
        """
        return {
            'num_positives': self.num_positives,
            'samples_per_epoch': self.samples_per_epoch,
            'samples_per_positive': self.samples_per_positive,
            'num_items': self.num_items,
            **self.mixer.get_stats()
        }
    
    def reset_stats(self):
        """Reset sampling statistics."""
        self.mixer.reset_stats()


def sample_triplets(
    positive_pairs: np.ndarray,
    user_pos_sets: Dict[int, Set[int]],
    num_items: int,
    num_samples: int,
    hard_neg_sets: Optional[Dict[int, Set[int]]] = None,
    hard_ratio: float = 0.3,
    random_seed: int = 42
) -> np.ndarray:
    """
    Convenience function to sample triplets.
    
    Args:
        positive_pairs: Array of shape (N, 2) with [u_idx, i_idx]
        user_pos_sets: Dict mapping u_idx -> Set of positive items
        num_items: Total number of items
        num_samples: Number of triplets to sample
        hard_neg_sets: Optional hard negative sets
        hard_ratio: Fraction of hard negatives
        random_seed: Random seed
    
    Returns:
        Array of shape (num_samples, 3) with [u, i_pos, i_neg]
    
    Example:
        >>> triplets = sample_triplets(
        ...     positive_pairs=pairs,
        ...     user_pos_sets=pos_sets,
        ...     num_items=2000,
        ...     num_samples=10000
        ... )
        >>> print(f"Sampled {len(triplets)} triplets")
    """
    sampler = TripletSampler(
        positive_pairs=positive_pairs,
        user_pos_sets=user_pos_sets,
        num_items=num_items,
        hard_neg_sets=hard_neg_sets,
        hard_ratio=hard_ratio,
        samples_per_positive=1,  # Will sample exactly num_samples
        random_seed=random_seed
    )
    
    # Override samples_per_epoch to get exact count
    sampler.samples_per_epoch = num_samples
    
    return sampler.sample_epoch()


# Main execution example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("Triplet Sampler Demo")
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
    
    # Generate hard negatives (top-50 popular items not in user history)
    top_popular = set(range(50))
    hard_neg_sets = {}
    for u in user_pos_sets:
        hard_neg_sets[u] = top_popular - user_pos_sets[u]
    
    print(f"\nData summary:")
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Positive pairs: {num_pairs}")
    print(f"  Users with hard negatives: {len(hard_neg_sets)}")
    
    # Test TripletSampler
    print("\n" + "-"*60)
    print("Testing TripletSampler")
    print("-"*60)
    
    sampler = TripletSampler(
        positive_pairs=positive_pairs,
        user_pos_sets=user_pos_sets,
        num_items=num_items,
        hard_neg_sets=hard_neg_sets,
        hard_ratio=0.3,
        samples_per_positive=5
    )
    
    # Sample epoch
    triplets = sampler.sample_epoch()
    print(f"\nSampled triplets: {triplets.shape}")
    print(f"First 5 triplets:\n{triplets[:5]}")
    
    # Get stats
    stats = sampler.get_sampling_stats()
    print(f"\nSampling stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Verify no positive in negatives
    print("\nValidating triplets...")
    invalid_count = 0
    for u, i_pos, i_neg in triplets:
        if i_neg in user_pos_sets.get(int(u), set()):
            invalid_count += 1
    print(f"  Invalid triplets (neg in pos set): {invalid_count}")
    
    print("\n" + "="*60)
    print("Demo complete!")
