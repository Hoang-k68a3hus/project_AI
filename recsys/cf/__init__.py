"""
CF (Collaborative Filtering) Module.

This module provides the complete CF recommendation system:
- Data processing for training data preparation
- Evaluation metrics and model comparison
- Model registry for version management
- Model loading for serving
- Logging and metrics tracking
- Drift detection and alerting
- Monitoring utilities

Submodules:
    data: Data processing pipeline (Task 01)
    evaluation: Evaluation metrics and comparison (Task 03)
    registry: Model versioning and management (Task 04)
    logging_utils: Training and service logging (Task 06)
    drift_detection: Data drift detection (Task 06)
    alerting: Alert management system (Task 06)

Example:
    >>> from recsys.cf.evaluation import ModelEvaluator
    >>> from recsys.cf.registry import ModelRegistry, ModelLoader
    >>> from recsys.cf.logging_utils import TrainingMetricsDB, setup_training_logger
    >>> from recsys.cf.drift_detection import detect_rating_drift
    >>> from recsys.cf.alerting import AlertManager
    >>> 
    >>> # Register and load models
    >>> registry = ModelRegistry()
    >>> loader = ModelLoader()
    >>> U, V, metadata = loader.load_current_best()
    >>> 
    >>> # Evaluate
    >>> evaluator = ModelEvaluator(U, V)
    >>> metrics = evaluator.evaluate(test_data)
    >>>
    >>> # Monitor
    >>> db = TrainingMetricsDB()
    >>> db.log_training_complete(run_id, metrics)
"""

# Note: We don't import submodules here to avoid circular imports
# Users should import from specific submodules:
#   from recsys.cf.data import DataProcessor
#   from recsys.cf.evaluation import ModelEvaluator
#   from recsys.cf.registry import ModelRegistry, ModelLoader
#   from recsys.cf.logging_utils import TrainingMetricsDB, ServiceMetricsDB
#   from recsys.cf.drift_detection import detect_rating_drift
#   from recsys.cf.alerting import AlertManager

__all__ = [
    'data', 
    'evaluation', 
    'registry', 
    'model',
    'logging_utils',
    'drift_detection',
    'alerting'
]
