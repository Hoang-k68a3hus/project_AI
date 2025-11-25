"""
BERT-Enhanced ALS Model (Task 02 - BERT Initialization Strategy)

This module implements ALS with BERT-initialized item factors to transfer
semantic knowledge from PhoBERT embeddings into collaborative filtering.

CRITICAL for ≥2 Threshold:
- With ~26k trainable users (≥2 interactions) and matrix density ~0.11%
- BERT initialization prevents random drift for sparse items
- Higher regularization (λ=0.1) anchors user vectors to BERT semantic space
- Especially important for users with exactly 2 interactions

Author: VieComRec Team
Date: 2025-11-24
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from implicit.als import AlternatingLeastSquares
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


class BERTEnhancedALS:
    """
    ALS với item factors initialized từ BERT embeddings.
    
    This class wraps the implicit.als.AlternatingLeastSquares model and
    initializes item factors using projected PhoBERT embeddings to transfer
    semantic knowledge into the collaborative filtering space.
    
    Attributes:
        factors (int): Embedding dimension for user/item factors
        bert_embeddings (np.ndarray): BERT embeddings for items (num_items, 768)
        product_ids (list): Product IDs corresponding to BERT embeddings
        model (AlternatingLeastSquares): Underlying ALS model from implicit library
        projection_method (str): Method to project BERT embeddings ("svd" or "pca")
    """
    
    def __init__(
        self,
        bert_embeddings_path: Union[str, Path],
        factors: int = 64,
        projection_method: str = "svd",
        **als_params
    ):
        """
        Initialize BERT-Enhanced ALS model.
        
        Args:
            bert_embeddings_path: Path to BERT embeddings (.pt file)
                Expected format: {
                    'embeddings': torch.Tensor (num_items, bert_dim),
                    'product_ids': list of product IDs
                }
                Supports various BERT dimensions: 768 (PhoBERT), 1024 (BERT-Large), etc.
            factors: Embedding dimension (will project BERT embeddings to this)
            projection_method: "svd" (recommended) or "pca" for dimensionality reduction
            **als_params: Additional parameters for AlternatingLeastSquares
                - regularization: L2 penalty (0.01-0.1, higher for sparse data)
                - iterations: Number of ALS iterations (10-20)
                - alpha: Confidence scaling (5-10 for sentiment-enhanced)
                - use_gpu: Enable GPU acceleration (requires cupy)
                - random_state: Random seed for reproducibility
        """
        self.factors = factors
        self.projection_method = projection_method
        self.als_params = als_params
        
        # Load BERT embeddings
        logger.info(f"Loading BERT embeddings from {bert_embeddings_path}")
        bert_data = torch.load(bert_embeddings_path, map_location='cpu')
        
        self.bert_embeddings = bert_data['embeddings'].numpy()  # (num_items, bert_dim)
        self.product_ids = bert_data['product_ids']
        self.bert_dim = self.bert_embeddings.shape[1]  # Auto-detect dimension
        
        logger.info(
            f"Loaded BERT embeddings: shape={self.bert_embeddings.shape}, "
            f"dim={self.bert_dim}, num_products={len(self.product_ids)}"
        )
        
        # Initialize ALS model (will set item_factors later)
        self.model = None
        
        # Tracking
        self.training_start_time = None
        self.training_end_time = None
        self.projection_info = {}
    
    def project_bert_to_factors(self, target_dim: int) -> np.ndarray:
        """
        Project BERT embeddings (auto-detected dim) to target_dim using SVD or PCA.
        
        This dimensionality reduction transfers semantic information from
        BERT's embedding space (768 for PhoBERT, 1024 for BERT-Large, etc.)
        into the collaborative filtering embedding space (typically 32, 64, or 128 dimensions).
        
        Args:
            target_dim: Target embedding dimension (e.g., 64)
        
        Returns:
            projected: Projected embeddings (num_items, target_dim)
        """
        if self.bert_embeddings.shape[1] == target_dim:
            logger.info("BERT embeddings already at target dimension")
            return self.bert_embeddings
        
        logger.info(
            f"Projecting BERT embeddings: {self.bert_embeddings.shape[1]} -> {target_dim}"
        )
        
        start_time = time.time()
        
        if self.projection_method == "svd":
            # TruncatedSVD is faster and works well for this use case
            reducer = TruncatedSVD(n_components=target_dim, random_state=42)
            projected = reducer.fit_transform(self.bert_embeddings)
            explained_var = reducer.explained_variance_ratio_.sum()
            
        elif self.projection_method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=target_dim, random_state=42)
            projected = reducer.fit_transform(self.bert_embeddings)
            explained_var = reducer.explained_variance_ratio_.sum()
            
        else:
            raise ValueError(
                f"Unknown projection method: {self.projection_method}. "
                "Choose 'svd' or 'pca'."
            )
        
        projection_time = time.time() - start_time
        
        # CRITICAL: Validate projected embeddings for NaN/Inf
        has_nan = np.isnan(projected).any()
        has_inf = np.isinf(projected).any()
        
        if has_nan or has_inf:
            nan_count = np.isnan(projected).sum() if has_nan else 0
            inf_count = np.isinf(projected).sum() if has_inf else 0
            logger.error(
                f"CRITICAL: Projection created invalid values! "
                f"NaN: {nan_count}, Inf: {inf_count}. "
                f"This will cause training to fail. "
                f"Replacing with small random values..."
            )
            # Replace NaN/Inf with small random values
            invalid_mask = np.isnan(projected) | np.isinf(projected)
            np.random.seed(42)
            projected[invalid_mask] = np.random.normal(
                0, 0.01, size=invalid_mask.sum()
            ).astype(np.float32)
            logger.warning("Replaced NaN/Inf with random values")
        
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
    ) -> np.ndarray:
        """
        Align BERT embeddings to match item ordering in CSR matrix.
        
        The BERT embeddings may have different product ordering than the
        CSR matrix used for training. This function creates an aligned
        matrix where row i corresponds to item index i in the CSR matrix.
        
        Args:
            projected_embeddings: Projected BERT embeddings (num_bert_items, factors)
            item_to_idx: Mapping from product_id (str) to item index (int)
            num_items: Total number of items in CSR matrix
        
        Returns:
            aligned: Aligned embeddings (num_items, factors)
                Items without BERT embeddings are initialized with random vectors
                (sampled from distribution of matched embeddings) to prevent NaN
        """
        logger.info(f"Aligning BERT embeddings to CSR matrix ordering")
        
        # DEBUG: Show sample IDs for troubleshooting
        sample_bert_ids = self.product_ids[:5]
        sample_mapping_keys = list(item_to_idx.keys())[:5]
        logger.info(
            f"Sample BERT IDs: {sample_bert_ids} (type: {type(self.product_ids[0])})\n"
            f"Sample mapping keys: {sample_mapping_keys} (type: {type(sample_mapping_keys[0])})"
        )
        
        aligned_embeddings = np.zeros((num_items, self.factors), dtype=np.float32)
        
        matched_count = 0
        missing_count = 0
        
        # Try both string and int matching
        for i, product_id in enumerate(self.product_ids):
            product_id_str = str(product_id)
            
            if product_id_str in item_to_idx:
                idx = item_to_idx[product_id_str]
                aligned_embeddings[idx] = projected_embeddings[i]
                matched_count += 1
            elif product_id in item_to_idx:  # Try int key
                idx = item_to_idx[product_id]
                aligned_embeddings[idx] = projected_embeddings[i]
                matched_count += 1
            else:
                missing_count += 1
        
        coverage = matched_count / num_items if num_items > 0 else 0
        
        logger.info(
            f"Alignment complete: matched={matched_count}, "
            f"missing={missing_count}, coverage={coverage:.1%}"
        )
        
        if coverage < 0.05:  # Less than 5% coverage
            logger.error(
                f"CRITICAL: BERT embedding coverage too low: {coverage:.1%}. "
                f"Product IDs in BERT embeddings don't match item_to_idx mapping. \n"
                f"BERT IDs sample: {sample_bert_ids}\n"
                f"Mapping keys sample: {sample_mapping_keys}\n"
                f"This likely means BERT embeddings were generated from different data. \n"
                f"FALLBACK: Will use random initialization instead."
            )
            return None  # Signal to use random init
        elif coverage < 0.8:
            logger.warning(
                f"Low BERT embedding coverage: {coverage:.1%}. "
                "This may reduce initialization effectiveness."
            )
        
        # FIX: Replace zero vectors with random initialization to prevent NaN
        zero_vector_mask = np.all(aligned_embeddings == 0, axis=1)
        zero_vector_count = zero_vector_mask.sum()
        
        if zero_vector_count > 0:
            logger.warning(
                f"Found {zero_vector_count} items ({zero_vector_count/num_items:.1%}) without BERT embeddings. "
                f"Replacing zero vectors with random initialization to prevent NaN during training."
            )
            
            # Get statistics from matched embeddings for initialization
            matched_embeddings = aligned_embeddings[~zero_vector_mask]
            
            if len(matched_embeddings) > 0:
                # Use mean and std from matched embeddings
                mean = matched_embeddings.mean(axis=0)
                std = matched_embeddings.std(axis=0)
                
                # CRITICAL FIX: Validate mean and std for NaN/Inf
                if np.isnan(mean).any() or np.isinf(mean).any():
                    logger.error("Mean of matched embeddings contains NaN/Inf! Using safe fallback.")
                    mean = np.zeros(self.factors, dtype=np.float32)
                    std = np.ones(self.factors, dtype=np.float32) * 0.01
                elif np.isnan(std).any() or np.isinf(std).any():
                    logger.error("Std of matched embeddings contains NaN/Inf! Using safe fallback.")
                    std = np.ones(self.factors, dtype=np.float32) * 0.01
                
                # Avoid zero std (add small epsilon)
                std = np.maximum(std, 1e-6)
                
                # Clip std to reasonable range to prevent extreme values
                std = np.clip(std, 1e-6, 10.0)
                
                # Initialize missing items with small random values around mean
                np.random.seed(42)  # For reproducibility
                random_init = np.random.normal(
                    mean, 
                    std * 0.1,  # Small variance to stay close to BERT space
                    size=(zero_vector_count, self.factors)
                ).astype(np.float32)
                
                # CRITICAL: Validate random_init for NaN/Inf
                if np.isnan(random_init).any() or np.isinf(random_init).any():
                    logger.error("Random initialization created NaN/Inf! Using safe fallback.")
                    np.random.seed(42)
                    random_init = np.random.normal(
                        0, 0.01, size=(zero_vector_count, self.factors)
                    ).astype(np.float32)
                
                # Clip to reasonable range to prevent overflow
                random_init = np.clip(random_init, -10.0, 10.0)
                
                aligned_embeddings[zero_vector_mask] = random_init
                
                logger.info(
                    f"Initialized {zero_vector_count} missing items with random vectors "
                    f"(mean norm: {np.linalg.norm(mean):.4f}, std: {np.linalg.norm(std):.4f})"
                )
            else:
                # All embeddings are zero - this shouldn't happen if coverage > 5%
                logger.error(
                    "All embeddings are zero! This indicates a critical alignment issue. "
                    "Falling back to standard random initialization."
                )
                # Use standard small random initialization
                np.random.seed(42)
                aligned_embeddings[zero_vector_mask] = np.random.normal(
                    0, 0.01, size=(zero_vector_count, self.factors)
                ).astype(np.float32)
        
        # FINAL VALIDATION: Ensure no NaN/Inf before returning
        if np.isnan(aligned_embeddings).any() or np.isinf(aligned_embeddings).any():
            logger.error(
                "CRITICAL: NaN/Inf detected in final aligned embeddings! "
                "This should not happen. Replacing all with safe random initialization."
            )
            np.random.seed(42)
            aligned_embeddings = np.random.normal(
                0, 0.01, size=aligned_embeddings.shape
            ).astype(np.float32)
        
        # Final clip to prevent extreme values
        aligned_embeddings = np.clip(aligned_embeddings, -10.0, 10.0)
        
        return aligned_embeddings
    
    def fit(
        self,
        X_train,
        item_to_idx: Dict[str, int],
        freeze_first_iteration: bool = False
    ):
        """
        Train ALS model with BERT-initialized item factors.
        
        Training Process:
        1. Initialize ALS model with parameters
        2. Project BERT embeddings to target dimension
        3. Align embeddings to CSR matrix ordering
        4. Set as initial item_factors
        5. Train ALS (will fine-tune from BERT initialization)
        
        Args:
            X_train: Confidence CSR matrix (num_users, num_items)
                Values should be sentiment-enhanced confidence (1-6 range)
            item_to_idx: Mapping from product_id to item index
            freeze_first_iteration: If True, keep item factors frozen for
                first iteration (experimental feature)
        
        Returns:
            self: Trained model
        """
        self.training_start_time = time.time()
        
        # Initialize ALS model
        logger.info("Initializing AlternatingLeastSquares model")
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            **self.als_params
        )
        
        # Project BERT embeddings
        projected_embeddings = self.project_bert_to_factors(self.factors)
        
        # Align to CSR matrix ordering
        num_items = X_train.shape[1]
        aligned_embeddings = self.align_embeddings_to_matrix(
            projected_embeddings, item_to_idx, num_items
        )
        
        # CRITICAL FIX: Auto-adjust regularization if we have zero vectors
        # Higher regularization helps stabilize training with mixed init
        original_reg = self.als_params.get('regularization', 0.01)
        if aligned_embeddings is not None:
            zero_count = np.sum(np.all(aligned_embeddings == 0, axis=1))
            if zero_count > 0:
                zero_ratio = zero_count / num_items
                # Increase regularization if >5% items have zero vectors
                if zero_ratio > 0.05:
                    adjusted_reg = max(original_reg, 0.05)  # At least 0.05
                    if adjusted_reg != original_reg:
                        logger.warning(
                            f"Auto-adjusting regularization: {original_reg} → {adjusted_reg} "
                            f"(due to {zero_ratio:.1%} zero vectors for stability)"
                        )
                        self.als_params['regularization'] = adjusted_reg
        
        # Check if alignment succeeded
        if aligned_embeddings is None:
            logger.warning(
                "BERT alignment failed due to low coverage. "
                "Training with RANDOM initialization instead (standard ALS)."
            )
            # Don't set item_factors - let ALS use random init
            self.bert_init_used = False
        else:
            # FIX: Validate aligned embeddings before setting (double-check for zero vectors)
            zero_vector_count = np.sum(np.all(aligned_embeddings == 0, axis=1))
            
            if zero_vector_count > 0:
                logger.warning(
                    f"CRITICAL: Found {zero_vector_count} zero vectors in aligned embeddings "
                    f"after alignment. This should not happen if fix is working correctly. "
                    f"Replacing with random initialization as fallback..."
                )
                
                # Replace zero vectors with random init
                zero_mask = np.all(aligned_embeddings == 0, axis=1)
                matched_embeddings = aligned_embeddings[~zero_mask]
                
                if len(matched_embeddings) > 0:
                    mean = matched_embeddings.mean(axis=0)
                    std = np.maximum(matched_embeddings.std(axis=0), 1e-6)
                    np.random.seed(42)
                    aligned_embeddings[zero_mask] = np.random.normal(
                        mean, std * 0.1, size=(zero_vector_count, self.factors)
                    ).astype(np.float32)
                else:
                    logger.error("All embeddings are zero! Using standard random initialization.")
                    aligned_embeddings = None
            
            if aligned_embeddings is not None:
                # CRITICAL FIX: Validate and normalize embeddings BEFORE setting
                # to prevent NaN during training
                
                # 1. Check for NaN/Inf
                has_nan = np.isnan(aligned_embeddings).any()
                has_inf = np.isinf(aligned_embeddings).any()
                
                if has_nan or has_inf:
                    logger.error(
                        f"CRITICAL: Found {'NaN' if has_nan else ''} {'Inf' if has_inf else ''} "
                        f"in aligned embeddings! This will cause training failure. "
                        f"Replacing with safe random initialization..."
                    )
                    # Replace all with safe random init
                    np.random.seed(42)
                    aligned_embeddings = np.random.normal(
                        0, 0.01, size=aligned_embeddings.shape
                    ).astype(np.float32)
                    self.bert_init_used = False
                else:
                    # 2. Normalize to prevent overflow (clip extreme values)
                    # ALS works best with embeddings in reasonable range
                    max_val = np.abs(aligned_embeddings).max()
                    if max_val > 10.0:  # If values too large, normalize
                        logger.warning(
                            f"Large embedding values detected (max={max_val:.2f}). "
                            f"Normalizing to prevent overflow..."
                        )
                        # Normalize to [-1, 1] range while preserving relative magnitudes
                        aligned_embeddings = aligned_embeddings / max_val
                    
                    # 3. Final validation
                    if np.isnan(aligned_embeddings).any() or np.isinf(aligned_embeddings).any():
                        logger.error("Normalization created NaN/Inf! Using random init.")
                        np.random.seed(42)
                        aligned_embeddings = np.random.normal(
                            0, 0.01, size=aligned_embeddings.shape
                        ).astype(np.float32)
                        self.bert_init_used = False
                    else:
                        self.bert_init_used = True
                
                # Set item factors (now guaranteed to be safe)
                logger.info("Setting item factors from BERT embeddings (validated)")
                self.model.item_factors = aligned_embeddings.copy()  # Use copy to avoid reference issues
                
                # 4. Post-set validation
                if hasattr(self.model, 'item_factors'):
                    final_check = self.model.item_factors
                    if np.isnan(final_check).any() or np.isinf(final_check).any():
                        logger.error("CRITICAL: NaN/Inf detected AFTER setting item_factors!")
                        # Emergency fallback
                        np.random.seed(42)
                        self.model.item_factors = np.random.normal(
                            0, 0.01, size=final_check.shape
                        ).astype(np.float32)
                        self.bert_init_used = False
            else:
                # Fallback to random init
                self.bert_init_used = False
        
        # Handle freeze_first_iteration (experimental)
        if freeze_first_iteration:
            logger.warning(
                "freeze_first_iteration=True: Item factors will be frozen for "
                "first iteration. This is experimental and may not be supported "
                "by all implicit library versions."
            )
            # Note: implicit library doesn't directly support freezing factors
            # This would require a custom training loop or model modification
        
        # Train (ALS will fine-tune from BERT initialization)
        logger.info(
            f"Starting ALS training with BERT-initialized item factors: "
            f"factors={self.factors}, iterations={self.als_params.get('iterations', 15)}"
        )
        
        # implicit library expects item-user matrix (transpose)
        X_train_T = X_train.T.tocsr()
        
        # CRITICAL FIX: Final validation before training
        # 1. Validate matrix
        if np.isnan(X_train_T.data).any() or np.isinf(X_train_T.data).any():
            logger.error("CRITICAL: Training matrix contains NaN/Inf! Cannot train.")
            raise ValueError("Training matrix contains invalid values (NaN/Inf)")
        
        # 2. Validate item_factors if set
        if hasattr(self.model, 'item_factors') and self.model.item_factors is not None:
            if np.isnan(self.model.item_factors).any() or np.isinf(self.model.item_factors).any():
                logger.error(
                    "CRITICAL: item_factors contains NaN/Inf before training! "
                    "Clearing and using random initialization."
                )
                self.model.item_factors = None
                self.bert_init_used = False
        
        # 3. Validate regularization is reasonable
        reg = self.als_params.get('regularization', 0.01)
        if reg <= 0 or reg > 1.0:
            logger.warning(f"Unusual regularization value: {reg}. Clipping to [0.001, 1.0]")
            self.als_params['regularization'] = np.clip(reg, 0.001, 1.0)
        
        # FIX: Add error handling for NaN errors with fallback
        try:
            self.model.fit(X_train_T)
        except Exception as e:
            error_msg = str(e).lower()
            if "nan" in error_msg or "nan encountered" in error_msg.lower():
                logger.error(
                    f"Training failed with NaN error: {e}\n"
                    f"This is likely caused by zero vectors or degenerate gradients. "
                    f"Attempting fallback to random initialization..."
                )
                
                # Fallback: Clear item factors and let ALS use random init
                if hasattr(self.model, 'item_factors'):
                    self.model.item_factors = None
                self.bert_init_used = False
                
                # Re-initialize model with random factors
                self.model = AlternatingLeastSquares(
                    factors=self.factors,
                    **self.als_params
                )
                
                # Retry training with random initialization
                logger.info("Retrying training with random initialization (standard ALS)...")
                self.model.fit(X_train_T)
                logger.info("Training succeeded with random initialization")
            else:
                # Re-raise if it's not a NaN error
                raise
        
        self.training_end_time = time.time()
        training_time = self.training_end_time - self.training_start_time
        
        logger.info(f"Training complete: time={training_time:.2f}s")
        
        return self
    
    def get_user_factors(self) -> np.ndarray:
        """
        Get trained user embeddings.
        
        Returns:
            U: User factors matrix (num_users, factors)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.user_factors
    
    def get_item_factors(self) -> np.ndarray:
        """
        Get trained item embeddings.
        
        Returns:
            V: Item factors matrix (num_items, factors)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.item_factors
    
    def recommend(
        self,
        user_id: int,
        user_items,
        N: int = 10,
        filter_already_liked_items: bool = True
    ):
        """
        Generate recommendations for a user.
        
        Args:
            user_id: User index (u_idx, not original user_id)
            user_items: CSR matrix of user-item interactions (for filtering)
            N: Number of recommendations
            filter_already_liked_items: Whether to exclude seen items
        
        Returns:
            items: Recommended item indices
            scores: Recommendation scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        return self.model.recommend(
            userid=user_id,
            user_items=user_items[user_id],
            N=N,
            filter_already_liked_items=filter_already_liked_items
        )
    
    def save_artifacts(
        self,
        output_dir: Union[str, Path],
        metadata: Optional[Dict] = None
    ):
        """
        Save trained model artifacts.
        
        Saves:
        - bert_als_U.npy: User factors
        - bert_als_V.npy: Item factors
        - bert_als_params.json: Hyperparameters
        - bert_als_metadata.json: Training metadata + BERT initialization info
        
        Args:
            output_dir: Directory to save artifacts
            metadata: Additional metadata to include (e.g., data_hash, git_commit)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving BERT-enhanced ALS artifacts to {output_dir}")
        
        # Save embeddings
        U = self.get_user_factors()
        V = self.get_item_factors()
        
        np.save(output_dir / "bert_als_U.npy", U)
        np.save(output_dir / "bert_als_V.npy", V)
        
        logger.info(f"Saved embeddings: U shape={U.shape}, V shape={V.shape}")
        
        # Save parameters
        params = {
            "model_type": "bert_enhanced_als",
            "factors": self.factors,
            "projection_method": self.projection_method,
            **self.als_params,
        }
        
        with open(output_dir / "bert_als_params.json", "w") as f:
            json.dump(params, f, indent=2)
        
        # Save metadata
        training_time = (
            self.training_end_time - self.training_start_time
            if self.training_start_time and self.training_end_time
            else None
        )
        
        meta = {
            "model_type": "bert_enhanced_als",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training_time_seconds": training_time,
            "bert_initialization": {
                "enabled": True,
                "embeddings_path": str(self.als_params.get("bert_embeddings_path", "")),
                "projection_method": self.projection_method,
                **self.projection_info,
            },
            "num_users": U.shape[0],
            "num_items": V.shape[0],
            "factors": self.factors,
        }
        
        # Merge with additional metadata
        if metadata:
            meta.update(metadata)
        
        with open(output_dir / "bert_als_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info("Artifacts saved successfully")
    
    @classmethod
    def load_artifacts(cls, artifact_dir: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load saved BERT-enhanced ALS artifacts.
        
        Args:
            artifact_dir: Directory containing saved artifacts
        
        Returns:
            U: User factors (num_users, factors)
            V: Item factors (num_items, factors)
            metadata: Training metadata dictionary
        """
        artifact_dir = Path(artifact_dir)
        
        logger.info(f"Loading BERT-enhanced ALS artifacts from {artifact_dir}")
        
        # Load embeddings
        U = np.load(artifact_dir / "bert_als_U.npy")
        V = np.load(artifact_dir / "bert_als_V.npy")
        
        # Load metadata
        with open(artifact_dir / "bert_als_metadata.json", "r") as f:
            metadata = json.load(f)
        
        logger.info(
            f"Loaded artifacts: U shape={U.shape}, V shape={V.shape}, "
            f"BERT init={metadata.get('bert_initialization', {}).get('enabled', False)}"
        )
        
        return U, V, metadata


def compute_score_range(
    U: np.ndarray,
    V: np.ndarray,
    sample_users: Optional[np.ndarray] = None,
    sample_size: int = 1000
) -> Dict:
    """
    Compute CF score range for global normalization (Task 08).
    
    This function computes statistics of CF scores (U @ V.T) on a sample
    of users to determine the range of scores for normalization in hybrid
    reranking.
    
    Args:
        U: User factors (num_users, factors)
        V: Item factors (num_items, factors)
        sample_users: Optional array of user indices to sample (if None, random sample)
        sample_size: Number of users to sample (default: 1000)
    
    Returns:
        score_range: Dictionary with min, max, mean, std, p01, p99
    """
    logger.info(f"Computing CF score range on sample of {sample_size} users")
    
    # Sample users
    num_users = U.shape[0]
    if sample_users is None:
        if num_users <= sample_size:
            sample_users = np.arange(num_users)
        else:
            sample_users = np.random.choice(num_users, sample_size, replace=False)
    
    # Compute scores for sampled users
    U_sample = U[sample_users]
    scores_sample = U_sample @ V.T  # (sample_size, num_items)
    
    # Flatten to compute global statistics
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
        f"Score range computed: min={score_range['min']:.3f}, "
        f"max={score_range['max']:.3f}, mean={score_range['mean']:.3f}"
    )
    
    return score_range
