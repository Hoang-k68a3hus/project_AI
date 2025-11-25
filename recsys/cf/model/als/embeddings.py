"""
ALS Embedding Extraction Module (Task 02 - Step 4)

This module handles extraction and processing of trained ALS embeddings:
- User factor extraction (U matrix)
- Item factor extraction (V matrix)
- Optional L2 normalization for cosine similarity
- Embedding statistics and quality checks
- Support for partial extraction (subset of users/items)

Key Features:
- Extract embeddings from trained implicit ALS models
- L2 normalization for cosine similarity computation
- Embedding quality metrics (norm distribution, sparsity)
- Efficient handling of large embedding matrices
- Export to various formats (NumPy, PyTorch, etc.)

Author: Copilot AI Assistant
Date: November 23, 2025
"""

import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse import csr_matrix

try:
    from implicit.als import AlternatingLeastSquares
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False
    AlternatingLeastSquares = None
    warnings.warn(
        "implicit library not installed. Run: pip install implicit\n"
        "Embedding extraction from implicit models will not be available."
    )

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extract and process embeddings from trained ALS models.
    
    This class provides:
    1. User and item embedding extraction
    2. Optional L2 normalization for cosine similarity
    3. Embedding quality statistics
    4. Partial extraction for specific users/items
    5. Export to various formats
    
    Attributes:
        model: Trained AlternatingLeastSquares model
        user_factors: Extracted user embeddings (U matrix)
        item_factors: Extracted item embeddings (V matrix)
        is_normalized: Whether embeddings are L2 normalized
    """
    
    def __init__(self, model: 'AlternatingLeastSquares', 
                 normalize: bool = False,
                 normalization_axis: int = 1):
        """
        Initialize embedding extractor.
        
        Args:
            model: Trained AlternatingLeastSquares model
            normalize: Whether to L2 normalize embeddings
            normalization_axis: Axis for normalization (1 = row-wise)
        
        Raises:
            RuntimeError: If model not fitted
            ImportError: If implicit library not available
        
        Example:
            >>> from recsys.cf.model.als import ALSTrainer, EmbeddingExtractor
            >>> trainer = ALSTrainer(model)
            >>> trainer.fit(X_train)
            >>> extractor = EmbeddingExtractor(trainer.model, normalize=True)
            >>> U, V = extractor.get_embeddings()
        """
        if not IMPLICIT_AVAILABLE:
            raise ImportError(
                "implicit library required. Install with: pip install implicit"
            )
        
        if not hasattr(model, 'user_factors') or model.user_factors is None:
            raise RuntimeError(
                "Model not fitted. Train the model first before extracting embeddings."
            )
        
        self.model = model
        self.normalize = normalize
        self.normalization_axis = normalization_axis
        
        # Extract embeddings
        self.user_factors = model.user_factors.copy()
        self.item_factors = model.item_factors.copy()
        
        logger.info(
            f"Embeddings extracted: U={self.user_factors.shape}, V={self.item_factors.shape}"
        )
        
        # Apply normalization if requested
        self.is_normalized = False
        if normalize:
            self._normalize_embeddings()
    
    def _normalize_embeddings(self) -> None:
        """
        L2 normalize embeddings for cosine similarity.
        
        Normalization formula:
            normalized_vector = vector / ||vector||_2
        
        Benefits:
            - Enables direct dot product for cosine similarity
            - Removes magnitude bias (focuses on direction)
            - Stabilizes recommendation scores
        """
        logger.info("Normalizing embeddings (L2 norm)...")
        
        # Normalize user factors
        u_norms = np.linalg.norm(self.user_factors, axis=self.normalization_axis, keepdims=True)
        u_norms = np.where(u_norms == 0, 1, u_norms)  # Avoid division by zero
        self.user_factors = self.user_factors / u_norms
        
        # Normalize item factors
        v_norms = np.linalg.norm(self.item_factors, axis=self.normalization_axis, keepdims=True)
        v_norms = np.where(v_norms == 0, 1, v_norms)  # Avoid division by zero
        self.item_factors = self.item_factors / v_norms
        
        self.is_normalized = True
        
        logger.info(
            f"Normalization complete: "
            f"U mean norm={np.linalg.norm(self.user_factors, axis=1).mean():.6f}, "
            f"V mean norm={np.linalg.norm(self.item_factors, axis=1).mean():.6f}"
        )
    
    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get extracted user and item embeddings.
        
        Returns:
            Tuple of (U, V) where:
                - U: User factors matrix (num_users, factors)
                - V: Item factors matrix (num_items, factors)
        
        Example:
            >>> U, V = extractor.get_embeddings()
            >>> scores = U @ V.T  # Compute all scores (cosine similarity if normalized)
        """
        return self.user_factors, self.item_factors
    
    def get_user_embedding(self, user_idx: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Get embedding(s) for specific user(s).
        
        Args:
            user_idx: User index or array of indices
        
        Returns:
            User embedding(s) of shape (factors,) or (n_users, factors)
        
        Example:
            >>> u_emb = extractor.get_user_embedding(42)
            >>> print(u_emb.shape)
            (64,)
        """
        return self.user_factors[user_idx]
    
    def get_item_embedding(self, item_idx: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Get embedding(s) for specific item(s).
        
        Args:
            item_idx: Item index or array of indices
        
        Returns:
            Item embedding(s) of shape (factors,) or (n_items, factors)
        
        Example:
            >>> v_emb = extractor.get_item_embedding([10, 20, 30])
            >>> print(v_emb.shape)
            (3, 64)
        """
        return self.item_factors[item_idx]
    
    def compute_user_item_score(self, user_idx: int, item_idx: int) -> float:
        """
        Compute score for specific user-item pair.
        
        Args:
            user_idx: User index
            item_idx: Item index
        
        Returns:
            Predicted score (cosine similarity if normalized)
        
        Example:
            >>> score = extractor.compute_user_item_score(user_idx=42, item_idx=100)
            >>> print(f"Predicted score: {score:.3f}")
        """
        return float(self.user_factors[user_idx] @ self.item_factors[item_idx])
    
    def compute_user_scores(self, user_idx: int) -> np.ndarray:
        """
        Compute scores for all items for a specific user.
        
        Args:
            user_idx: User index
        
        Returns:
            Array of scores (num_items,)
        
        Example:
            >>> scores = extractor.compute_user_scores(user_idx=42)
            >>> top_items = np.argsort(scores)[-10:][::-1]  # Top 10 items
        """
        return self.user_factors[user_idx] @ self.item_factors.T
    
    def compute_item_similarity(self, item_idx: int, top_k: Optional[int] = None) -> np.ndarray:
        """
        Compute similarity between an item and all other items.
        
        Args:
            item_idx: Item index
            top_k: Return only top-K similar items (None = all)
        
        Returns:
            Array of similarity scores (num_items,) or indices of top-K items
        
        Example:
            >>> similar_items = extractor.compute_item_similarity(item_idx=100, top_k=10)
            >>> print(f"Top 10 similar items: {similar_items}")
        """
        similarities = self.item_factors[item_idx] @ self.item_factors.T
        
        if top_k is not None:
            # Get top-K indices (excluding self)
            top_indices = np.argsort(similarities)[::-1]
            top_indices = top_indices[top_indices != item_idx][:top_k]
            return top_indices
        
        return similarities
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics about extracted embeddings.
        
        Returns:
            Dictionary with embedding statistics:
                - shapes: Embedding dimensions
                - norms: L2 norm statistics
                - sparsity: Percentage of near-zero values
                - value_ranges: Min/max/mean values per dimension
        
        Example:
            >>> stats = extractor.get_embedding_statistics()
            >>> print(f"User embedding sparsity: {stats['user_sparsity']:.2%}")
        """
        stats = {
            'user_factors_shape': tuple(self.user_factors.shape),
            'item_factors_shape': tuple(self.item_factors.shape),
            'is_normalized': self.is_normalized,
            'factors': self.user_factors.shape[1]
        }
        
        # Compute norms
        u_norms = np.linalg.norm(self.user_factors, axis=1)
        v_norms = np.linalg.norm(self.item_factors, axis=1)
        
        stats['user_norms'] = {
            'mean': float(u_norms.mean()),
            'std': float(u_norms.std()),
            'min': float(u_norms.min()),
            'max': float(u_norms.max())
        }
        
        stats['item_norms'] = {
            'mean': float(v_norms.mean()),
            'std': float(v_norms.std()),
            'min': float(v_norms.min()),
            'max': float(v_norms.max())
        }
        
        # Compute sparsity (percentage of values close to zero)
        threshold = 1e-6
        user_sparsity = np.mean(np.abs(self.user_factors) < threshold)
        item_sparsity = np.mean(np.abs(self.item_factors) < threshold)
        
        stats['user_sparsity'] = float(user_sparsity)
        stats['item_sparsity'] = float(item_sparsity)
        
        # Value ranges per dimension
        stats['user_value_range'] = {
            'min_per_dim': self.user_factors.min(axis=0).mean(),
            'max_per_dim': self.user_factors.max(axis=0).mean(),
            'mean_per_dim': self.user_factors.mean(axis=0).mean(),
            'std_per_dim': self.user_factors.std(axis=0).mean()
        }
        
        stats['item_value_range'] = {
            'min_per_dim': self.item_factors.min(axis=0).mean(),
            'max_per_dim': self.item_factors.max(axis=0).mean(),
            'mean_per_dim': self.item_factors.mean(axis=0).mean(),
            'std_per_dim': self.item_factors.std(axis=0).mean()
        }
        
        return stats
    
    def save_embeddings(self, output_dir: Path, prefix: str = "als") -> Dict[str, Path]:
        """
        Save embeddings to disk in NumPy format.
        
        Args:
            output_dir: Directory to save embeddings
            prefix: Filename prefix
        
        Returns:
            Dictionary with paths to saved files
        
        Example:
            >>> paths = extractor.save_embeddings(Path('artifacts/cf/als'))
            >>> print(f"Saved to: {paths['user_factors']}")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add normalization suffix if normalized
        suffix = "_normalized" if self.is_normalized else ""
        
        # Save user factors
        u_path = output_dir / f"{prefix}_U{suffix}.npy"
        np.save(u_path, self.user_factors)
        logger.info(f"User factors saved: {u_path}")
        
        # Save item factors
        v_path = output_dir / f"{prefix}_V{suffix}.npy"
        np.save(v_path, self.item_factors)
        logger.info(f"Item factors saved: {v_path}")
        
        return {
            'user_factors': u_path,
            'item_factors': v_path
        }
    
    def export_to_pytorch(self) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Export embeddings to PyTorch tensors.
        
        Returns:
            Tuple of (U_tensor, V_tensor)
        
        Raises:
            ImportError: If PyTorch not installed
        
        Example:
            >>> U_tensor, V_tensor = extractor.export_to_pytorch()
            >>> print(f"User tensor: {U_tensor.shape}, device: {U_tensor.device}")
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch required for tensor export. Install with: pip install torch"
            )
        
        U_tensor = torch.from_numpy(self.user_factors).float()
        V_tensor = torch.from_numpy(self.item_factors).float()
        
        logger.info(f"Exported to PyTorch tensors: U={U_tensor.shape}, V={V_tensor.shape}")
        
        return U_tensor, V_tensor
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of extracted embeddings.
        
        Returns:
            Multi-line string with embedding information
        """
        stats = self.get_embedding_statistics()
        
        lines = [
            "=== ALS Embedding Summary ===",
            f"User factors: {stats['user_factors_shape']}",
            f"Item factors: {stats['item_factors_shape']}",
            f"Normalized: {stats['is_normalized']}",
            "",
            "User Embeddings:",
            f"  Mean norm: {stats['user_norms']['mean']:.4f}",
            f"  Std norm: {stats['user_norms']['std']:.4f}",
            f"  Sparsity: {stats['user_sparsity']:.2%}",
            "",
            "Item Embeddings:",
            f"  Mean norm: {stats['item_norms']['mean']:.4f}",
            f"  Std norm: {stats['item_norms']['std']:.4f}",
            f"  Sparsity: {stats['item_sparsity']:.2%}",
            "=" * 29
        ]
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation of extractor."""
        return (
            f"EmbeddingExtractor(U={self.user_factors.shape}, "
            f"V={self.item_factors.shape}, "
            f"normalized={self.is_normalized})"
        )


def extract_embeddings(model: 'AlternatingLeastSquares',
                      normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to extract embeddings from trained ALS model.
    
    Args:
        model: Trained AlternatingLeastSquares model
        normalize: Whether to L2 normalize embeddings
    
    Returns:
        Tuple of (U, V) embeddings
    
    Example:
        >>> from recsys.cf.model.als import train_als_model, extract_embeddings
        >>> model, summary = train_als_model(X_train)
        >>> U, V = extract_embeddings(model, normalize=True)
        >>> print(f"Embeddings: U={U.shape}, V={V.shape}")
    """
    extractor = EmbeddingExtractor(model, normalize=normalize)
    return extractor.get_embeddings()


def normalize_embeddings(U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    L2 normalize existing embeddings.
    
    Args:
        U: User embeddings (num_users, factors)
        V: Item embeddings (num_items, factors)
    
    Returns:
        Tuple of normalized (U, V)
    
    Example:
        >>> U_norm, V_norm = normalize_embeddings(U, V)
        >>> # Now U_norm @ V_norm.T gives cosine similarities
    """
    # Normalize U
    u_norms = np.linalg.norm(U, axis=1, keepdims=True)
    u_norms = np.where(u_norms == 0, 1, u_norms)
    U_norm = U / u_norms
    
    # Normalize V
    v_norms = np.linalg.norm(V, axis=1, keepdims=True)
    v_norms = np.where(v_norms == 0, 1, v_norms)
    V_norm = V / v_norms
    
    logger.info(
        f"Embeddings normalized: "
        f"U mean norm={np.linalg.norm(U_norm, axis=1).mean():.6f}, "
        f"V mean norm={np.linalg.norm(V_norm, axis=1).mean():.6f}"
    )
    
    return U_norm, V_norm


def compute_embedding_quality_score(U: np.ndarray, V: np.ndarray) -> Dict[str, float]:
    """
    Compute quality metrics for embeddings.
    
    Metrics:
        - Norm stability: Low variance in L2 norms indicates stable embeddings
        - Orthogonality: High average inner product suggests redundant dimensions
        - Coverage: Percentage of non-zero dimensions
    
    Args:
        U: User embeddings
        V: Item embeddings
    
    Returns:
        Dictionary with quality scores
    
    Example:
        >>> quality = compute_embedding_quality_score(U, V)
        >>> print(f"Norm stability (user): {quality['user_norm_stability']:.3f}")
    """
    # Norm statistics
    u_norms = np.linalg.norm(U, axis=1)
    v_norms = np.linalg.norm(V, axis=1)
    
    # Coefficient of variation (lower = more stable)
    user_norm_cv = u_norms.std() / u_norms.mean() if u_norms.mean() > 0 else 0
    item_norm_cv = v_norms.std() / v_norms.mean() if v_norms.mean() > 0 else 0
    
    # Orthogonality (sample-based for efficiency)
    sample_size = min(100, U.shape[0])
    u_sample = U[np.random.choice(U.shape[0], sample_size, replace=False)]
    v_sample = V[np.random.choice(V.shape[0], sample_size, replace=False)]
    
    # Normalize samples
    u_sample_norm = u_sample / (np.linalg.norm(u_sample, axis=1, keepdims=True) + 1e-8)
    v_sample_norm = v_sample / (np.linalg.norm(v_sample, axis=1, keepdims=True) + 1e-8)
    
    # Average absolute inner product (lower = more orthogonal)
    user_orthogonality = np.abs(u_sample_norm @ u_sample_norm.T).mean()
    item_orthogonality = np.abs(v_sample_norm @ v_sample_norm.T).mean()
    
    # Coverage (percentage of non-trivial values)
    threshold = 1e-6
    user_coverage = 1 - np.mean(np.abs(U) < threshold)
    item_coverage = 1 - np.mean(np.abs(V) < threshold)
    
    return {
        'user_norm_stability': 1.0 / (1.0 + user_norm_cv),  # Higher = better
        'item_norm_stability': 1.0 / (1.0 + item_norm_cv),
        'user_orthogonality': float(user_orthogonality),  # Lower = better
        'item_orthogonality': float(item_orthogonality),
        'user_coverage': float(user_coverage),  # Higher = better
        'item_coverage': float(item_coverage)
    }


# Main execution example
if __name__ == "__main__":
    import traceback
    
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("ALS Embedding Extraction Demo")
    print("=" * 60)
    
    # Create mock trained model
    print("\nCreating mock trained model...")
    
    if IMPLICIT_AVAILABLE:
        from implicit.als import AlternatingLeastSquares
        
        # Initialize and "train" a mock model
        num_users, num_items = 1000, 500
        factors = 64
        
        model = AlternatingLeastSquares(factors=factors, iterations=1)
        
        # Create synthetic training data
        np.random.seed(42)
        rows = np.random.randint(0, num_users, 5000)
        cols = np.random.randint(0, num_items, 5000)
        data = np.random.uniform(1.0, 6.0, 5000)
        X_train = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
        
        # Fit model
        print("Training model...")
        model.fit(X_train.T.tocsr(), show_progress=False)
        print("Model trained!")
        
        # Example 1: Extract embeddings without normalization
        print("\n" + "=" * 60)
        print("Example 1: Extract Embeddings (No Normalization)")
        print("=" * 60)
        
        try:
            extractor = EmbeddingExtractor(model, normalize=False)
            U, V = extractor.get_embeddings()
            
            print(f"\nExtracted embeddings:")
            print(f"  U: {U.shape}, dtype: {U.dtype}")
            print(f"  V: {V.shape}, dtype: {V.dtype}")
            print(f"  Normalized: {extractor.is_normalized}")
            
            print(extractor.get_summary())
            
        except Exception as e:
            print(f"Example 1 failed: {e}")
            traceback.print_exc()
        
        # Example 2: Extract with normalization
        print("\n" + "=" * 60)
        print("Example 2: Extract Embeddings (With Normalization)")
        print("=" * 60)
        
        try:
            extractor_norm = EmbeddingExtractor(model, normalize=True)
            U_norm, V_norm = extractor_norm.get_embeddings()
            
            print(f"\nNormalized embeddings:")
            print(f"  U: {U_norm.shape}")
            print(f"  V: {V_norm.shape}")
            
            # Check norms
            u_norms = np.linalg.norm(U_norm, axis=1)
            v_norms = np.linalg.norm(V_norm, axis=1)
            print(f"\nNorm verification:")
            print(f"  User norms - mean: {u_norms.mean():.6f}, std: {u_norms.std():.6f}")
            print(f"  Item norms - mean: {v_norms.mean():.6f}, std: {v_norms.std():.6f}")
            
            print(extractor_norm.get_summary())
            
        except Exception as e:
            print(f"Example 2 failed: {e}")
            traceback.print_exc()
        
        # Example 3: Compute scores
        print("\n" + "=" * 60)
        print("Example 3: Compute Scores")
        print("=" * 60)
        
        try:
            user_idx = 42
            item_idx = 100
            
            # Single score
            score = extractor_norm.compute_user_item_score(user_idx, item_idx)
            print(f"Score for user {user_idx}, item {item_idx}: {score:.4f}")
            
            # All scores for a user
            scores = extractor_norm.compute_user_scores(user_idx)
            top_items = np.argsort(scores)[-10:][::-1]
            print(f"\nTop 10 items for user {user_idx}:")
            for rank, item_id in enumerate(top_items, 1):
                print(f"  {rank}. Item {item_id}: score={scores[item_id]:.4f}")
            
        except Exception as e:
            print(f"Example 3 failed: {e}")
            traceback.print_exc()
        
        # Example 4: Embedding quality
        print("\n" + "=" * 60)
        print("Example 4: Embedding Quality Metrics")
        print("=" * 60)
        
        try:
            quality = compute_embedding_quality_score(U, V)
            print("\nQuality metrics:")
            for metric, value in quality.items():
                print(f"  {metric}: {value:.4f}")
            
        except Exception as e:
            print(f"Example 4 failed: {e}")
            traceback.print_exc()
        
        # Example 5: Convenience function
        print("\n" + "=" * 60)
        print("Example 5: Convenience Function")
        print("=" * 60)
        
        try:
            U_quick, V_quick = extract_embeddings(model, normalize=True)
            print(f"Quick extraction: U={U_quick.shape}, V={V_quick.shape}")
            print(f"Mean user norm: {np.linalg.norm(U_quick, axis=1).mean():.6f}")
            
        except Exception as e:
            print(f"Example 5 failed: {e}")
            traceback.print_exc()
    
    else:
        print("\nSkipping demo - implicit library not installed")
        print("Install with: pip install implicit")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
