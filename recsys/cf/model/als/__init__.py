"""
ALS (Alternating Least Squares) Model Module

This module contains all components for the ALS training pipeline:
- Step 1: Matrix Preparation (pre_data.py)
- Step 2: Model Initialization (model_init.py)
- Step 3: Model Training (training.py)
- Step 4: Extract Embeddings (training.py)
- Step 5: Recommendation Generation (recommender.py)
- Step 6: Evaluation (evaluation.py)
- Step 7: Save Artifacts (artifact_saver.py) - Including score ranges for Task 08

Usage:
    >>> from recsys.cf.model.als import (
    ...     ALSMatrixPreparer, quick_prepare_als_matrix,
    ...     ALSModelInitializer, quick_initialize_als,
    ...     ALSTrainer, quick_train_als,
    ...     ALSRecommender, quick_recommend,
    ...     ALSEvaluator, evaluate_als_complete,
    ...     save_als_complete
    ... )
    >>> 
    >>> # Step 1: Prepare matrices
    >>> preparer = ALSMatrixPreparer(base_path='data/processed')
    >>> data = preparer.prepare_complete_als_data()
    >>> X_train = data['X_train_implicit']
    >>> 
    >>> # Step 2: Initialize model
    >>> initializer = ALSModelInitializer(preset='default')
    >>> model = initializer.initialize_model()
    >>> 
    >>> # Step 3: Train model
    >>> trainer = ALSTrainer(model=model)
    >>> results = trainer.fit(X_train)
    >>> 
    >>> # Step 4: Extract embeddings
    >>> extractor = EmbeddingExtractor(trainer.model, normalize=True)
    >>> U, V = extractor.get_embeddings()
    >>> print(f"Embeddings: U={U.shape}, V={V.shape}, normalized={extractor.is_normalized}")
    >>> 
    >>> # Or use convenience function
    >>> U, V = extract_embeddings(trainer.model, normalize=True)
    >>> 
    >>> # Step 5: Generate recommendations
    >>> recommender = ALSRecommender(
    ...     user_factors=U,
    ...     item_factors=V,
    ...     user_to_idx=mappings['user_to_idx'],
    ...     idx_to_user=mappings['idx_to_user'],
    ...     item_to_idx=mappings['item_to_idx'],
    ...     idx_to_item=mappings['idx_to_item'],
    ...     user_pos_train=user_pos_train
    ... )
    >>> result = recommender.recommend(user_id='12345', k=10, filter_seen=True)
    >>> print(f"Top items: {result.item_ids[:5]}")
    >>> 
    >>> # Batch recommendations
    >>> results = recommender.recommend_batch(user_ids=test_users, k=10)
    >>> 
    >>> # Or use convenience function
    >>> quick_recs = quick_recommend(U, V, user_ids=test_users, k=10, mappings=mappings)
    >>> 
    >>> # Step 6: Evaluate
    >>> evaluator = ALSEvaluator(
    ...     user_factors=U,
    ...     item_factors=V,
    ...     user_to_idx=mappings['user_to_idx'],
    ...     idx_to_user=mappings['idx_to_user'],
    ...     item_to_idx=mappings['item_to_idx'],
    ...     idx_to_item=mappings['idx_to_item'],
    ...     user_pos_train=user_pos_train,
    ...     user_pos_test=user_pos_test
    ... )
    >>> results = evaluator.evaluate(k_values=[10, 20], compare_baseline=True)
    >>> results.print_summary()
    >>> print(f"Recall@10: {results.metrics['recall@10']:.3f}")
    >>> print(f"Improvement: {results.improvement['recall@10']}")
    >>> 
    >>> # Or use quick function
    >>> metrics = quick_evaluate(U, V, user_pos_test, user_pos_train, k_values=[10, 20])
    >>> 
    >>> # Step 7: Save artifacts with score range for Task 08
    >>> artifacts = save_als_complete(
    ...     user_embeddings=U,
    ...     item_embeddings=V,
    ...     params={'factors': 64, 'regularization': 0.01, 'iterations': 15},
    ...     metrics=results.metrics,
    ...     output_dir='artifacts/cf/als',
    ...     validation_user_indices=[10, 25, 42, 67],  # Critical for score range
    ...     data_version_hash='abc123def456'
    ... )
    >>> print(artifacts.summary())
    >>> # Score range: [0.012, 1.123] for Task 08 normalization
"""

from .pre_data import (
    ALSMatrixPreparer,
    quick_prepare_als_matrix
)

from .model_init import (
    ALSModelInitializer,
    quick_initialize_als,
    get_preset_config
)

from .trainer import (
    ALSTrainer,
    train_als_model
)

from .embeddings import (
    EmbeddingExtractor,
    extract_embeddings,
    normalize_embeddings,
    compute_embedding_quality_score
)

from .recommender import (
    ALSRecommender,
    RecommendationResult,
    quick_recommend
)

from .evaluation import (
    ALSEvaluator,
    EvaluationResult,
    PopularityBaseline,
    recall_at_k,
    ndcg_at_k,
    compute_metrics_single_user,
    quick_evaluate
)

from .artifact_saver import (
    ALSArtifacts,
    ScoreRange,
    compute_score_range,
    save_embeddings,
    save_params,
    save_metrics,
    save_metadata,
    save_model_object,
    save_als_complete,
    load_als_artifacts
)

__all__ = [
    # Step 1: Matrix Preparation
    'ALSMatrixPreparer',
    'quick_prepare_als_matrix',
    
    # Step 2: Model Initialization
    'ALSModelInitializer',
    'quick_initialize_als',
    'get_preset_config',
    
    # Step 3: Model Training
    'ALSTrainer',
    'train_als_model',
    
    # Step 4: Embedding Extraction
    'EmbeddingExtractor',
    'extract_embeddings',
    'normalize_embeddings',
    'compute_embedding_quality_score',
    
    # Step 5: Recommendation Generation
    'ALSRecommender',
    'RecommendationResult',
    'quick_recommend',
    
    # Step 6: Evaluation
    'ALSEvaluator',
    'EvaluationResult',
    'PopularityBaseline',
    'recall_at_k',
    'ndcg_at_k',
    'compute_metrics_single_user',
    'quick_evaluate',
    
    # Step 7: Artifact Saving
    'ALSArtifacts',
    'ScoreRange',
    'compute_score_range',
    'save_embeddings',
    'save_params',
    'save_metrics',
    'save_metadata',
    'save_model_object',
    'save_als_complete',
    'load_als_artifacts'
]
