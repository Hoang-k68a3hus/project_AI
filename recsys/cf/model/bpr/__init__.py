"""
BPR (Bayesian Personalized Ranking) Model Module

This module contains all components for the BPR training pipeline:
- Step 1: Data Preparation (pre_data.py) - Load positive sets, pairs, hard negatives
- Step 2: Hard Negative Sampling (sampler.py) - Dual strategy: explicit + implicit
- Step 2+: Advanced Sampling (advanced_sampler.py) - Contextual, sentiment-contrast, dynamic
- Step 3: Model Initialization (model_init.py) - Initialize embeddings
- Step 4: Model Training (trainer.py) - SGD with BPR loss
- Step 4+: Advanced Training (advanced_trainer.py) - AdamW, dropout, scheduling
- Step 5: Recommendation Generation (recommender.py)
- Step 6: Evaluation (evaluation.py)
- Step 7: Save Artifacts (artifact_saver.py) - Including score ranges for Task 08

Enhanced Features (2025-11):
- BERT-Enhanced BPR: Initialize item factors from PhoBERT embeddings
- Sentiment-Aware Training: Weight triplets by confidence scores (rating + comment_quality)
- Fake Review Detection: Down-weight suspicious reviews with high ratings but low sentiment
- Advanced Negative Sampling: Contextual (BERT-similar), sentiment-contrast, dynamic difficulty
- Modern Optimizers: AdamW with differential regularization, embedding dropout, LR scheduling

Usage:
    >>> from recsys.cf.model.bpr import (
    ...     BPRDataPreparer, prepare_bpr_data,
    ...     TripletSampler, sample_triplets,
    ...     BPRModelInitializer, initialize_bpr_model,
    ...     BPRTrainer, train_bpr_model,
    ...     BPRRecommender, quick_recommend,
    ...     BPREvaluator, evaluate_bpr,
    ...     save_bpr_complete, load_bpr_artifacts
    ... )
    >>> 
    >>> # Step 1: Prepare data
    >>> preparer = BPRDataPreparer(base_path='data/processed')
    >>> data = preparer.load_bpr_data()
    >>> 
    >>> # Step 2: Initialize model
    >>> initializer = BPRModelInitializer(
    ...     num_users=data['num_users'],
    ...     num_items=data['num_items'],
    ...     factors=64
    ... )
    >>> U, V = initializer.initialize_embeddings()
    >>> 
    >>> # Step 3 & 4: Train model
    >>> trainer = BPRTrainer(
    ...     U=U, V=V,
    ...     learning_rate=0.05,
    ...     regularization=0.0001
    ... )
    >>> results = trainer.fit(
    ...     positive_pairs=data['positive_pairs'],
    ...     user_pos_sets=data['user_pos_sets'],
    ...     hard_neg_sets=data['hard_neg_sets'],
    ...     num_items=data['num_items'],
    ...     epochs=50
    ... )
    >>> 
    >>> # Step 5: Generate recommendations
    >>> recommender = BPRRecommender(
    ...     user_factors=trainer.U,
    ...     item_factors=trainer.V,
    ...     user_pos_train=data['user_pos_sets']
    ... )
    >>> recs = recommender.recommend(user_idx=42, k=10)
    >>> 
    >>> # Step 6: Evaluate
    >>> evaluator = BPREvaluator(
    ...     user_factors=trainer.U,
    ...     item_factors=trainer.V,
    ...     user_pos_train=data['user_pos_sets'],
    ...     user_pos_test=test_pos_sets
    ... )
    >>> metrics = evaluator.evaluate(k_values=[10, 20])
    >>> 
    >>> # Step 7: Save artifacts
    >>> artifacts = save_bpr_complete(
    ...     user_embeddings=trainer.U,
    ...     item_embeddings=trainer.V,
    ...     params=trainer.get_params(),
    ...     metrics=metrics,
    ...     output_dir='artifacts/cf/bpr'
    ... )
    >>>
    >>> # === ENHANCED: BERT-Enhanced BPR with Sentiment-Aware Training ===
    >>> from recsys.cf.model.bert_enhanced_bpr import BERTEnhancedBPR, SentimentAwareConfig
    >>> 
    >>> # Configure sentiment weighting
    >>> sentiment_config = SentimentAwareConfig(
    ...     use_sentiment_weighting=True,
    ...     positive_confidence_threshold=4.5,
    ...     suspicious_penalty=0.5,
    ...     high_confidence_bonus=1.5
    ... )
    >>> 
    >>> # Initialize BERT-Enhanced BPR
    >>> model = BERTEnhancedBPR(
    ...     bert_embeddings_path='data/processed/content_based_embeddings/product_embeddings.pt',
    ...     factors=64,
    ...     sentiment_config=sentiment_config
    ... )
    >>> 
    >>> # Train with confidence-weighted loss
    >>> summary = model.fit(
    ...     positive_pairs=data['positive_pairs'],
    ...     confidence_scores=data['confidence_scores'],  # From BPRDataPreparer
    ...     user_pos_sets=data['user_pos_sets'],
    ...     hard_neg_sets=data['hard_neg_sets'],
    ...     ...
    ... )
"""

from .pre_data import (
    BPRDataLoader,
    load_bpr_data,
    prepare_bpr_data
)

from .sampler import (
    TripletSampler,
    sample_triplets,
    HardNegativeMixer
)

from .model_init import (
    BPRModelInitializer,
    initialize_bpr_model,
    get_bpr_preset_config
)

from .trainer import (
    BPRTrainer,
    train_bpr_model,
    TrainingHistory
)

from .artifact_saver import (
    BPRArtifacts,
    save_bpr_complete,
    load_bpr_artifacts,
    compute_bpr_score_range
)

from .advanced_sampler import (
    AdvancedTripletSampler,
    SamplingStrategy,
    DynamicSamplingConfig,
    ContextualNegativeSampler,
    SentimentContrastSampler,
    PopularNegativeSampler,
    create_advanced_sampler
)

from .advanced_trainer import (
    AdvancedBPRTrainer,
    OptimizerType,
    OptimizerConfig,
    TrainingConfig,
    SchedulerType,
    LearningRateScheduler,
    AdamWOptimizer,
    AdaGradOptimizer,
    EmbeddingDropout
)

__all__ = [
    # Step 1: Data Preparation
    'BPRDataLoader',
    'load_bpr_data',
    'prepare_bpr_data',
    
    # Step 2: Triplet Sampling (Basic)
    'TripletSampler',
    'sample_triplets',
    'HardNegativeMixer',
    
    # Step 2+: Advanced Sampling
    'AdvancedTripletSampler',
    'SamplingStrategy',
    'DynamicSamplingConfig',
    'ContextualNegativeSampler',
    'SentimentContrastSampler',
    'PopularNegativeSampler',
    'create_advanced_sampler',
    
    # Step 3: Model Initialization
    'BPRModelInitializer',
    'initialize_bpr_model',
    'get_bpr_preset_config',
    
    # Step 4: Model Training (Basic)
    'BPRTrainer',
    'train_bpr_model',
    'TrainingHistory',
    
    # Step 4+: Advanced Training
    'AdvancedBPRTrainer',
    'OptimizerType',
    'OptimizerConfig',
    'TrainingConfig',
    'SchedulerType',
    'LearningRateScheduler',
    'AdamWOptimizer',
    'AdaGradOptimizer',
    'EmbeddingDropout',
    
    # Step 7: Artifact Saving
    'BPRArtifacts',
    'save_bpr_complete',
    'load_bpr_artifacts',
    'compute_bpr_score_range'
]
