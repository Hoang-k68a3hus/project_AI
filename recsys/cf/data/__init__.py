"""
Data package for collaborative filtering.

This package provides classes and utilities for:
- Loading raw data (DataReader)
- Validating and cleaning data (DataAuditor)
- Complete data processing pipeline (DataProcessor)
"""

from .processing.read_data import DataReader
from .processing.audit_data import DataAuditor
from .processing.feature_engineering import FeatureEngineer
from .processing.als_data import ALSDataPreparer
from .processing.bpr_data import BPRDataPreparer
from .processing.user_filtering import UserFilter
from .processing.id_mapping import IDMapper
from .processing.temporal_split import TemporalSplitter
from .processing.matrix_construction import MatrixBuilder
from .processing.data_saver import DataSaver
from .processing.version_registry import VersionRegistry
from .data import (
    DataProcessor,
    setup_logging,
    load_raw_data,
    validate_and_clean_interactions,
    deduplicate_interactions,
    detect_outliers,
    compute_data_hash,
    log_data_quality_report
)

__all__ = [
    'DataReader',
    'DataAuditor',
    'FeatureEngineer',
    'ALSDataPreparer',
    'BPRDataPreparer',
    'UserFilter',
    'IDMapper',
    'TemporalSplitter',
    'MatrixBuilder',
    'DataSaver',
    'VersionRegistry',
    'DataProcessor',
    'setup_logging',
    'load_raw_data',
    'validate_and_clean_interactions',
    'deduplicate_interactions',
    'detect_outliers',
    'compute_data_hash',
    'log_data_quality_report'
]
