"""
ALS Model Initialization Module (Task 02 - Step 2)

This module handles initialization of ALS models using the `implicit` library.
Provides flexible configuration for hyperparameters with sensible defaults
aligned with the project's high sparsity and rating skew characteristics.

Key Features:
- Wrapper around implicit.als.AlternatingLeastSquares
- Hyperparameter validation and recommendations
- GPU/CPU support detection
- Configuration presets for different scenarios
- Integration with Step 1 matrix preparation outputs

Author: Copilot AI Assistant
Date: November 23, 2025
"""

import logging
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import warnings
from pathlib import Path

try:
    from implicit.als import AlternatingLeastSquares
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False
    AlternatingLeastSquares = None  # For type hints
    warnings.warn(
        "implicit library not installed. Run: pip install implicit\n"
        "ALS training will not be available."
    )

import numpy as np
from scipy.sparse import csr_matrix

# Type hint support
if TYPE_CHECKING:
    from implicit.als import AlternatingLeastSquares

# Configure logging
logger = logging.getLogger(__name__)


class ALSModelInitializer:
    """
    Initialize and configure ALS models for collaborative filtering.
    
    This class provides:
    1. Model initialization with validated hyperparameters
    2. GPU availability detection and configuration
    3. Configuration presets for common scenarios
    4. Hyperparameter recommendations based on data characteristics
    5. Model inspection and metadata generation
    
    Attributes:
        model: Initialized AlternatingLeastSquares instance (None if not initialized)
        config: Current hyperparameter configuration
        gpu_available: Whether GPU acceleration is available
    """
    
    # Default hyperparameters aligned with Task 02 specs
    DEFAULT_CONFIG = {
        'factors': 64,
        'regularization': 0.01,
        'iterations': 15,
        'alpha': 10,  # Lower for sentiment-enhanced confidence (range 1-6)
        'random_state': 42,
        'use_gpu': False,
        'dtype': np.float32,
        'num_threads': 0  # 0 = use all available
    }
    
    # Configuration presets for different scenarios
    PRESETS = {
        'default': {
            'factors': 64,
            'regularization': 0.01,
            'iterations': 15,
            'alpha': 10,
            'description': 'Standard config for sentiment-enhanced confidence (1-6 range)'
        },
        'normalized': {
            'factors': 64,
            'regularization': 0.01,
            'iterations': 15,
            'alpha': 40,
            'description': 'Config for normalized confidence (0-1 range)'
        },
        'high_quality': {
            'factors': 128,
            'regularization': 0.05,
            'iterations': 20,
            'alpha': 10,
            'description': 'Higher quality embeddings, longer training'
        },
        'fast': {
            'factors': 32,
            'regularization': 0.01,
            'iterations': 10,
            'alpha': 10,
            'description': 'Faster training for experimentation'
        },
        'sparse_data': {
            'factors': 64,
            'regularization': 0.1,
            'iterations': 15,
            'alpha': 5,
            'description': 'Higher regularization for very sparse data (≥2 threshold)'
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 preset: Optional[str] = None,
                 auto_detect_gpu: bool = True):
        """
        Initialize ALS model configurator.
        
        Args:
            config: Custom hyperparameter configuration (overrides preset)
            preset: Named preset configuration ('default', 'normalized', 'high_quality', 'fast', 'sparse_data')
            auto_detect_gpu: Whether to automatically detect and enable GPU if available
        
        Raises:
            ImportError: If implicit library not installed
            ValueError: If invalid preset name provided
        """
        if not IMPLICIT_AVAILABLE:
            raise ImportError(
                "implicit library required for ALS training. "
                "Install with: pip install implicit"
            )
        
        self.model = None
        self.config = self.DEFAULT_CONFIG.copy()
        self.gpu_available = False
        
        # Apply preset if specified
        if preset:
            if preset not in self.PRESETS:
                raise ValueError(
                    f"Invalid preset '{preset}'. "
                    f"Available: {list(self.PRESETS.keys())}"
                )
            preset_config = self.PRESETS[preset].copy()
            preset_config.pop('description', None)
            self.config.update(preset_config)
            logger.info(f"Applied preset '{preset}': {self.PRESETS[preset]['description']}")
        
        # Apply custom config (overrides preset)
        if config:
            self.config.update(config)
            logger.info(f"Applied custom config: {config}")
        
        # Detect GPU availability
        if auto_detect_gpu:
            self.gpu_available = self._detect_gpu()
            if self.gpu_available and not self.config.get('use_gpu', False):
                logger.info(
                    "GPU detected but not enabled in config. "
                    "Set use_gpu=True to enable GPU acceleration."
                )
        
        # Validate configuration
        self._validate_config()
    
    def _detect_gpu(self) -> bool:
        """
        Detect if GPU acceleration is available for implicit library.
        
        Returns:
            True if GPU available, False otherwise
        """
        try:
            import cupy
            logger.info("GPU support detected (cupy available)")
            return True
        except ImportError:
            logger.debug("GPU support not available (cupy not installed)")
            return False
    
    def _validate_config(self) -> None:
        """
        Validate hyperparameter configuration and log warnings for unusual values.
        
        Raises:
            ValueError: If configuration contains invalid values
        """
        config = self.config
        
        # Validate factors
        if config['factors'] <= 0:
            raise ValueError(f"factors must be positive, got {config['factors']}")
        if config['factors'] > 256:
            logger.warning(
                f"factors={config['factors']} is very high. "
                "This may cause overfitting and slow training."
            )
        
        # Validate regularization
        if config['regularization'] < 0:
            raise ValueError(f"regularization must be non-negative, got {config['regularization']}")
        if config['regularization'] > 1.0:
            logger.warning(
                f"regularization={config['regularization']} is very high. "
                "This may cause underfitting."
            )
        
        # Validate iterations
        if config['iterations'] <= 0:
            raise ValueError(f"iterations must be positive, got {config['iterations']}")
        if config['iterations'] > 50:
            logger.warning(
                f"iterations={config['iterations']} is very high. "
                "Training may be slow with diminishing returns."
            )
        
        # Validate alpha
        if config['alpha'] <= 0:
            raise ValueError(f"alpha must be positive, got {config['alpha']}")
        if config['alpha'] > 100:
            logger.warning(
                f"alpha={config['alpha']} is very high. "
                "This may cause the model to overweight high-confidence items."
            )
        
        # Validate GPU setting
        if config['use_gpu'] and not self.gpu_available:
            logger.warning(
                "use_gpu=True but GPU not available. "
                "Falling back to CPU. Install cupy for GPU support."
            )
            config['use_gpu'] = False
    
    def initialize_model(self):
        """
        Initialize ALS model with current configuration.
        
        Returns:
            AlternatingLeastSquares: Initialized AlternatingLeastSquares model instance
        
        Example:
            >>> initializer = ALSModelInitializer(preset='default')
            >>> model = initializer.initialize_model()
            >>> print(f"Model factors: {model.factors}")
            Model factors: 64
        """
        logger.info("Initializing ALS model...")
        logger.info(f"Configuration: {self.config}")
        
        # Create model instance
        self.model = AlternatingLeastSquares(
            factors=self.config['factors'],
            regularization=self.config['regularization'],
            iterations=self.config['iterations'],
            alpha=self.config['alpha'],
            random_state=self.config['random_state'],
            use_gpu=self.config['use_gpu'],
            dtype=self.config['dtype'],
            num_threads=self.config['num_threads']
        )
        
        logger.info(
            f"ALS model initialized: "
            f"factors={self.model.factors}, "
            f"reg={self.model.regularization}, "
            f"iters={self.model.iterations}, "
            f"alpha={self.model.alpha}, "
            f"GPU={self.config['use_gpu']}"
        )
        
        return self.model
    
    def get_model(self):
        """
        Get initialized model, creating it if necessary.
        
        Returns:
            AlternatingLeastSquares: AlternatingLeastSquares model instance
        """
        if self.model is None:
            self.initialize_model()
        return self.model
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration and reinitialize model.
        
        Args:
            **kwargs: Hyperparameters to update
        
        Example:
            >>> initializer = ALSModelInitializer()
            >>> initializer.update_config(factors=128, regularization=0.05)
            >>> model = initializer.get_model()
        """
        self.config.update(kwargs)
        self._validate_config()
        logger.info(f"Configuration updated: {kwargs}")
        
        # Reinitialize model with new config
        if self.model is not None:
            logger.info("Reinitializing model with updated config...")
            self.initialize_model()
    
    def get_alpha_recommendation(self, confidence_range: Tuple[float, float]) -> int:
        """
        Recommend alpha value based on confidence score range.
        
        Args:
            confidence_range: (min, max) tuple of confidence scores
        
        Returns:
            Recommended alpha value
        
        Logic:
            - Raw scores [1-6]: alpha = 5-10 (lower due to higher range)
            - Normalized [0-1]: alpha = 20-40 (standard scaling)
            - Binary [0-1]: alpha = 40-80 (higher scaling needed)
        
        Example:
            >>> initializer = ALSModelInitializer()
            >>> alpha = initializer.get_alpha_recommendation((1.0, 6.0))
            >>> print(f"Recommended alpha: {alpha}")
            Recommended alpha: 10
        """
        min_conf, max_conf = confidence_range
        conf_span = max_conf - min_conf
        
        if conf_span >= 3.0:
            # Raw sentiment-enhanced confidence (1-6 range)
            recommended = 10
            reason = "Raw confidence scores (1-6 range)"
        elif conf_span >= 0.8 and max_conf <= 1.1:
            # Normalized continuous (0-1 range)
            recommended = 40
            reason = "Normalized confidence scores (0-1 range)"
        elif conf_span < 0.2:
            # Binary or near-binary
            recommended = 80
            reason = "Binary or near-binary confidence"
        else:
            # Intermediate range
            recommended = 20
            reason = "Intermediate confidence range"
        
        logger.info(
            f"Alpha recommendation: {recommended} "
            f"(Confidence range: [{min_conf:.2f}, {max_conf:.2f}], {reason})"
        )
        
        return recommended
    
    def recommend_config_for_data(self, num_users: int, num_items: int, 
                                   nnz: int, confidence_range: Tuple[float, float]) -> Dict[str, Any]:
        """
        Recommend configuration based on data characteristics.
        
        Args:
            num_users: Number of users in training matrix
            num_items: Number of items in training matrix
            nnz: Number of non-zero entries (interactions)
            confidence_range: (min, max) tuple of confidence scores
        
        Returns:
            Recommended configuration dictionary
        
        Example:
            >>> initializer = ALSModelInitializer()
            >>> config = initializer.recommend_config_for_data(
            ...     num_users=26000, num_items=2231, nnz=65000, 
            ...     confidence_range=(1.0, 6.0)
            ... )
            >>> print(config)
            {'factors': 64, 'regularization': 0.1, 'alpha': 10, ...}
        """
        density = nnz / (num_users * num_items)
        avg_interactions_per_user = nnz / num_users
        
        logger.info(f"Data characteristics:")
        logger.info(f"  Users: {num_users:,}, Items: {num_items:,}")
        logger.info(f"  Interactions: {nnz:,}")
        logger.info(f"  Density: {density:.6f}")
        logger.info(f"  Avg interactions/user: {avg_interactions_per_user:.2f}")
        
        recommended = self.DEFAULT_CONFIG.copy()
        
        # Recommend factors based on data size
        if num_items < 1000:
            recommended['factors'] = 32
        elif num_items < 5000:
            recommended['factors'] = 64
        else:
            recommended['factors'] = 128
        
        # Recommend regularization based on sparsity
        if density < 0.001:  # Very sparse (< 0.1%)
            recommended['regularization'] = 0.1
        elif density < 0.01:  # Sparse (< 1%)
            recommended['regularization'] = 0.05
        else:
            recommended['regularization'] = 0.01
        
        # Recommend alpha based on confidence range
        recommended['alpha'] = self.get_alpha_recommendation(confidence_range)
        
        # Recommend iterations based on avg interactions
        if avg_interactions_per_user < 3:
            recommended['iterations'] = 20  # More iterations for sparse users
        else:
            recommended['iterations'] = 15
        
        logger.info(f"Recommended config: {recommended}")
        
        return recommended
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about initialized model.
        
        Returns:
            Dictionary with model information
        
        Raises:
            RuntimeError: If model not initialized
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        info = {
            'model_type': 'ALS',
            'library': 'implicit',
            'factors': self.model.factors,
            'regularization': self.model.regularization,
            'iterations': self.model.iterations,
            'alpha': self.model.alpha,
            'random_state': self.model.random_state,
            'use_gpu': self.config['use_gpu'],
            'dtype': str(self.model.dtype),
            'num_threads': self.model.num_threads,
            'is_fitted': hasattr(self.model, 'user_factors') and self.model.user_factors is not None
        }
        
        # Add embedding info if model is fitted
        if info['is_fitted']:
            info['user_factors_shape'] = self.model.user_factors.shape
            info['item_factors_shape'] = self.model.item_factors.shape
        
        return info
    
    def get_config_summary(self) -> str:
        """
        Get human-readable configuration summary.
        
        Returns:
            Multi-line string summarizing configuration
        """
        lines = [
            "=== ALS Model Configuration ===",
            f"Embedding dimension: {self.config['factors']}",
            f"Regularization (λ): {self.config['regularization']}",
            f"Training iterations: {self.config['iterations']}",
            f"Confidence scaling (α): {self.config['alpha']}",
            f"Random seed: {self.config['random_state']}",
            f"GPU acceleration: {self.config['use_gpu']}",
            f"Data type: {self.config['dtype']}",
            f"CPU threads: {self.config['num_threads'] or 'all'}"
        ]
        
        if self.model is not None:
            lines.append(f"Model status: Initialized")
            if hasattr(self.model, 'user_factors') and self.model.user_factors is not None:
                lines.append(f"Training status: Fitted")
                lines.append(f"User factors: {self.model.user_factors.shape}")
                lines.append(f"Item factors: {self.model.item_factors.shape}")
            else:
                lines.append(f"Training status: Not fitted")
        else:
            lines.append(f"Model status: Not initialized")
        
        lines.append("=" * 31)
        
        return "\n".join(lines)
    
    def export_config(self) -> Dict[str, Any]:
        """
        Export current configuration for saving to JSON/YAML.
        
        Returns:
            Configuration dictionary ready for serialization
        """
        config = self.config.copy()
        config['dtype'] = str(config['dtype'])  # Convert numpy dtype to string
        config['gpu_available'] = self.gpu_available
        
        return config
    
    @staticmethod
    def list_presets() -> Dict[str, str]:
        """
        List all available configuration presets.
        
        Returns:
            Dictionary mapping preset names to descriptions
        """
        return {
            name: preset['description'] 
            for name, preset in ALSModelInitializer.PRESETS.items()
        }
    
    def __repr__(self) -> str:
        """String representation of initializer."""
        status = "initialized" if self.model is not None else "not initialized"
        return (
            f"ALSModelInitializer(factors={self.config['factors']}, "
            f"reg={self.config['regularization']}, "
            f"alpha={self.config['alpha']}, "
            f"model={status})"
        )


def quick_initialize_als(factors: int = 64, regularization: float = 0.01,
                        iterations: int = 15, alpha: int = 10,
                        use_gpu: bool = False, random_state: int = 42):
    """
    Quick convenience function to initialize ALS model with minimal setup.
    
    Args:
        factors: Embedding dimension
        regularization: L2 regularization penalty
        iterations: Number of ALS iterations
        alpha: Confidence scaling factor
        use_gpu: Whether to use GPU acceleration
        random_state: Random seed for reproducibility
    
    Returns:
        AlternatingLeastSquares: Initialized AlternatingLeastSquares model
    
    Example:
        >>> model = quick_initialize_als(factors=64, alpha=10)
        >>> # Ready to fit: model.fit(X_train)
    """
    if not IMPLICIT_AVAILABLE:
        raise ImportError(
            "implicit library required. Install with: pip install implicit"
        )
    
    logger.info(f"Quick initializing ALS: factors={factors}, alpha={alpha}")
    
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha,
        random_state=random_state,
        use_gpu=use_gpu,
        dtype=np.float32,
        num_threads=0
    )
    
    return model


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a named preset.
    
    Args:
        preset_name: Name of preset ('default', 'normalized', etc.)
    
    Returns:
        Configuration dictionary
    
    Raises:
        ValueError: If preset name not found
    
    Example:
        >>> config = get_preset_config('sparse_data')
        >>> print(config)
        {'factors': 64, 'regularization': 0.1, ...}
    """
    if preset_name not in ALSModelInitializer.PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(ALSModelInitializer.PRESETS.keys())}"
        )
    
    preset = ALSModelInitializer.PRESETS[preset_name].copy()
    preset.pop('description', None)
    
    return preset


# Main execution example
if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("ALS Model Initialization Demo")
    print("=" * 60)
    
    # Example 1: Default initialization
    print("\n1. Default Initialization:")
    initializer = ALSModelInitializer()
    model = initializer.initialize_model()
    print(initializer.get_config_summary())
    
    # Example 2: Using preset
    print("\n2. Preset Initialization (sparse_data):")
    initializer_sparse = ALSModelInitializer(preset='sparse_data')
    print(initializer_sparse.get_config_summary())
    
    # Example 3: Custom config
    print("\n3. Custom Configuration:")
    custom_config = {
        'factors': 128,
        'regularization': 0.05,
        'alpha': 20
    }
    initializer_custom = ALSModelInitializer(config=custom_config)
    print(initializer_custom.get_config_summary())
    
    # Example 4: Data-driven recommendations
    print("\n4. Data-Driven Config Recommendation:")
    recommended = initializer.recommend_config_for_data(
        num_users=26000,
        num_items=2231,
        nnz=65000,
        confidence_range=(1.0, 6.0)
    )
    print(f"Recommended config: {recommended}")
    
    # Example 5: Quick initialization
    print("\n5. Quick Initialization:")
    quick_model = quick_initialize_als(factors=64, alpha=10)
    print(f"Quick model: factors={quick_model.factors}, alpha={quick_model.alpha}")
    
    # List all presets
    print("\n6. Available Presets:")
    for name, desc in ALSModelInitializer.list_presets().items():
        print(f"  - {name}: {desc}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
