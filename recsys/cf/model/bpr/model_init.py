"""
BPR Model Initialization Module (Task 02 - Step 3)

This module handles embedding initialization for BPR:
- Random Gaussian initialization (default)
- Optional BERT-based initialization for items
- Hyperparameter configuration

Initialization Strategy:
- Small random values (mean=0, std=0.01) to prevent gradient explosion
- Same dimension for both user and item embeddings
- Optional BERT projection for semantic transfer
"""

import logging
from typing import Dict, Tuple, Optional, Any
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# Default BPR configurations
BPR_PRESETS = {
    'default': {
        'factors': 64,
        'learning_rate': 0.05,
        'regularization': 0.0001,
        'epochs': 50,
        'samples_per_positive': 5,
        'hard_ratio': 0.3,
        'init_std': 0.01,
        'random_seed': 42
    },
    'fast': {
        'factors': 32,
        'learning_rate': 0.1,
        'regularization': 0.001,
        'epochs': 30,
        'samples_per_positive': 3,
        'hard_ratio': 0.3,
        'init_std': 0.01,
        'random_seed': 42
    },
    'accurate': {
        'factors': 128,
        'learning_rate': 0.01,
        'regularization': 0.00001,
        'epochs': 100,
        'samples_per_positive': 10,
        'hard_ratio': 0.3,
        'init_std': 0.005,
        'random_seed': 42
    },
    'sparse': {
        # Tuned for sparse data (≥2 threshold)
        'factors': 64,
        'learning_rate': 0.05,
        'regularization': 0.0001,
        'epochs': 50,
        'samples_per_positive': 5,
        'hard_ratio': 0.3,  # Higher hard ratio helps with sparsity
        'init_std': 0.01,
        'random_seed': 42
    }
}


class BPRModelInitializer:
    """
    Initialize BPR model embeddings and configuration.
    
    Handles:
    - User embedding matrix U (num_users, factors)
    - Item embedding matrix V (num_items, factors)
    - Optional BERT-based item initialization
    - Hyperparameter configuration
    
    Attributes:
        num_users: Number of users
        num_items: Number of items
        factors: Embedding dimension
        config: Full configuration dictionary
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        factors: int = 64,
        init_std: float = 0.01,
        random_seed: int = 42,
        preset: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize BPR model initializer.
        
        Args:
            num_users: Number of users in the system
            num_items: Number of items in the system
            factors: Embedding dimension (default: 64)
            init_std: Standard deviation for random init (default: 0.01)
            random_seed: Random seed for reproducibility
            preset: Use preset configuration ('default', 'fast', 'accurate', 'sparse')
            config: Override configuration dictionary
        
        Example:
            >>> initializer = BPRModelInitializer(
            ...     num_users=26000,
            ...     num_items=2200,
            ...     factors=64,
            ...     preset='sparse'
            ... )
            >>> U, V = initializer.initialize_embeddings()
        """
        self.num_users = num_users
        self.num_items = num_items
        
        # Load preset config
        if preset and preset in BPR_PRESETS:
            self.config = BPR_PRESETS[preset].copy()
            logger.info(f"Using preset configuration: {preset}")
        else:
            self.config = BPR_PRESETS['default'].copy()
        
        # Override with explicit parameters
        if factors:
            self.config['factors'] = factors
        if init_std:
            self.config['init_std'] = init_std
        if random_seed:
            self.config['random_seed'] = random_seed
        
        # Override with custom config
        if config:
            self.config.update(config)
        
        self.factors = self.config['factors']
        self.init_std = self.config['init_std']
        self.random_seed = self.config['random_seed']
        
        # Initialize random generator
        self.rng = np.random.default_rng(self.random_seed)
        
        logger.info(f"BPRModelInitializer:")
        logger.info(f"  Users: {num_users:,}")
        logger.info(f"  Items: {num_items:,}")
        logger.info(f"  Factors: {self.factors}")
        logger.info(f"  Init std: {self.init_std}")
    
    def initialize_embeddings(
        self,
        bert_embeddings: Optional[np.ndarray] = None,
        item_mapping: Optional[Dict[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize user and item embedding matrices.
        
        Args:
            bert_embeddings: Optional BERT embeddings for items (num_items, bert_dim)
            item_mapping: Optional mapping from BERT indices to BPR indices
        
        Returns:
            Tuple of (U, V) where:
                - U: User embeddings (num_users, factors)
                - V: Item embeddings (num_items, factors)
        
        Example:
            >>> U, V = initializer.initialize_embeddings()
            >>> print(f"U: {U.shape}, V: {V.shape}")
        """
        logger.info("Initializing BPR embeddings...")
        
        # Initialize user embeddings (always random)
        U = self.rng.normal(
            loc=0.0,
            scale=self.init_std,
            size=(self.num_users, self.factors)
        ).astype(np.float32)
        
        logger.info(f"User embeddings initialized: {U.shape}")
        logger.info(f"  Mean: {U.mean():.6f}, Std: {U.std():.6f}")
        
        # Initialize item embeddings
        if bert_embeddings is not None:
            V = self._initialize_from_bert(bert_embeddings, item_mapping)
        else:
            V = self.rng.normal(
                loc=0.0,
                scale=self.init_std,
                size=(self.num_items, self.factors)
            ).astype(np.float32)
        
        logger.info(f"Item embeddings initialized: {V.shape}")
        logger.info(f"  Mean: {V.mean():.6f}, Std: {V.std():.6f}")
        
        return U, V
    
    def _initialize_from_bert(
        self,
        bert_embeddings: np.ndarray,
        item_mapping: Optional[Dict[int, int]] = None
    ) -> np.ndarray:
        """
        Initialize item embeddings from BERT with SVD projection.
        
        Args:
            bert_embeddings: BERT embeddings (num_bert_items, bert_dim)
            item_mapping: Optional mapping from BERT indices to BPR indices
        
        Returns:
            Item embeddings (num_items, factors)
        """
        logger.info("Initializing item embeddings from BERT...")
        
        from sklearn.decomposition import TruncatedSVD
        
        bert_dim = bert_embeddings.shape[1]
        
        # Project BERT embeddings to target dimension
        if bert_dim > self.factors:
            svd = TruncatedSVD(n_components=self.factors, random_state=self.random_seed)
            projected = svd.fit_transform(bert_embeddings)
            explained_var = svd.explained_variance_ratio_.sum()
            logger.info(f"BERT projection: {bert_dim} -> {self.factors}")
            logger.info(f"Explained variance: {explained_var:.3f}")
        else:
            # Pad if BERT dim < factors (unlikely)
            projected = np.zeros((bert_embeddings.shape[0], self.factors), dtype=np.float32)
            projected[:, :bert_dim] = bert_embeddings
            logger.info(f"BERT embeddings padded: {bert_dim} -> {self.factors}")
        
        # Normalize scale
        scale_factor = self.init_std / projected.std()
        projected = (projected * scale_factor).astype(np.float32)
        
        # Create V matrix
        V = self.rng.normal(
            loc=0.0,
            scale=self.init_std,
            size=(self.num_items, self.factors)
        ).astype(np.float32)
        
        # Map BERT embeddings to correct positions
        if item_mapping:
            for bert_idx, bpr_idx in item_mapping.items():
                if bert_idx < len(projected) and bpr_idx < self.num_items:
                    V[bpr_idx] = projected[bert_idx]
        else:
            # Direct mapping (assume same ordering)
            n_copy = min(len(projected), self.num_items)
            V[:n_copy] = projected[:n_copy]
        
        logger.info(f"BERT initialization complete: {V.shape}")
        
        return V
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get full configuration dictionary.
        
        Returns:
            Configuration dictionary with all hyperparameters
        """
        return {
            **self.config,
            'num_users': self.num_users,
            'num_items': self.num_items
        }
    
    def validate_config(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.num_users <= 0:
            raise ValueError(f"Invalid num_users: {self.num_users}")
        
        if self.num_items <= 0:
            raise ValueError(f"Invalid num_items: {self.num_items}")
        
        if self.factors <= 0:
            raise ValueError(f"Invalid factors: {self.factors}")
        
        if self.init_std <= 0:
            raise ValueError(f"Invalid init_std: {self.init_std}")
        
        if self.config['learning_rate'] <= 0:
            raise ValueError(f"Invalid learning_rate: {self.config['learning_rate']}")
        
        if self.config['epochs'] <= 0:
            raise ValueError(f"Invalid epochs: {self.config['epochs']}")
        
        logger.info("✓ Configuration validation passed")
        return True


def initialize_bpr_model(
    num_users: int,
    num_items: int,
    factors: int = 64,
    preset: str = 'default',
    bert_embeddings: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convenience function to initialize BPR model.
    
    Args:
        num_users: Number of users
        num_items: Number of items
        factors: Embedding dimension
        preset: Configuration preset
        bert_embeddings: Optional BERT embeddings for items
    
    Returns:
        Tuple of (U, V, config) where:
            - U: User embeddings
            - V: Item embeddings
            - config: Full configuration dictionary
    
    Example:
        >>> U, V, config = initialize_bpr_model(
        ...     num_users=26000,
        ...     num_items=2200,
        ...     preset='sparse'
        ... )
    """
    initializer = BPRModelInitializer(
        num_users=num_users,
        num_items=num_items,
        factors=factors,
        preset=preset
    )
    
    U, V = initializer.initialize_embeddings(bert_embeddings=bert_embeddings)
    config = initializer.get_config()
    
    return U, V, config


def get_bpr_preset_config(preset: str = 'default') -> Dict[str, Any]:
    """
    Get BPR preset configuration.
    
    Args:
        preset: Preset name ('default', 'fast', 'accurate', 'sparse')
    
    Returns:
        Configuration dictionary
    
    Example:
        >>> config = get_bpr_preset_config('sparse')
        >>> print(f"Factors: {config['factors']}")
    """
    if preset not in BPR_PRESETS:
        available = list(BPR_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset}. Available: {available}")
    
    return BPR_PRESETS[preset].copy()


# Main execution example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("="*60)
    print("BPR Model Initialization Demo")
    print("="*60)
    
    # Example 1: Default initialization
    print("\nExample 1: Default initialization")
    initializer = BPRModelInitializer(
        num_users=26000,
        num_items=2200,
        preset='default'
    )
    U, V = initializer.initialize_embeddings()
    print(f"U shape: {U.shape}, V shape: {V.shape}")
    print(f"Config: {initializer.get_config()}")
    
    # Example 2: Sparse preset
    print("\n" + "-"*60)
    print("Example 2: Sparse preset (for ≥2 threshold data)")
    U2, V2, config2 = initialize_bpr_model(
        num_users=26000,
        num_items=2200,
        preset='sparse'
    )
    print(f"Config: {config2}")
    
    # Example 3: Available presets
    print("\n" + "-"*60)
    print("Example 3: Available presets")
    for preset_name in BPR_PRESETS:
        config = get_bpr_preset_config(preset_name)
        print(f"\n{preset_name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("Demo complete!")
