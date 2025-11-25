"""
Logging Utilities for CF Training and Service.

This module provides structured logging for:
- Training runs (ALS, BPR, BERT-ALS)
- Service requests
- Metrics tracking to SQLite

Example:
    >>> from recsys.cf.logging_utils import setup_training_logger, TrainingMetricsDB
    >>> logger = setup_training_logger('als', 'run_001')
    >>> db = TrainingMetricsDB()
    >>> db.log_training_start('run_001', 'als', params)
"""

import logging
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import threading

# ============================================================================
# Constants
# ============================================================================

DEFAULT_LOG_DIR = "logs"
TRAINING_LOG_DIR = "logs/cf"
SERVICE_LOG_DIR = "logs/service"
TRAINING_DB_PATH = "logs/training_metrics.db"
SERVICE_DB_PATH = "logs/service_metrics.db"

LOG_FORMAT = '%(asctime)s | %(levelname)s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


# ============================================================================
# Directory Setup
# ============================================================================

def ensure_log_dirs():
    """Create log directories if they don't exist."""
    for dir_path in [DEFAULT_LOG_DIR, TRAINING_LOG_DIR, SERVICE_LOG_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Training Logger Setup
# ============================================================================

def setup_training_logger(
    model_type: str,
    run_id: str,
    log_dir: str = TRAINING_LOG_DIR,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger for a training run.
    
    Args:
        model_type: 'als', 'bpr', or 'bert_als'
        run_id: Unique run identifier
        log_dir: Directory for log files
        console: Whether to also log to console
    
    Returns:
        Configured logger
    
    Example:
        >>> logger = setup_training_logger('als', 'als_20251125_103000')
        >>> logger.info("Training started | factors=64, reg=0.01")
    """
    ensure_log_dirs()
    
    logger_name = f'cf.{model_type}.{run_id}'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler - append to model-specific log
    log_file = Path(log_dir) / f'{model_type}.log'
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(fh)
    
    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(ch)
    
    return logger


def setup_service_logger(
    name: str = 'recommender',
    log_dir: str = SERVICE_LOG_DIR,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger for recommendation service.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        console: Whether to also log to console
    
    Returns:
        Configured logger
    """
    ensure_log_dirs()
    
    logger = logging.getLogger(f'service.{name}')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler
    log_file = Path(log_dir) / f'{name}.log'
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(fh)
    
    # Error file handler
    error_file = Path(log_dir) / 'error.log'
    eh = logging.FileHandler(error_file, encoding='utf-8')
    eh.setLevel(logging.ERROR)
    eh.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(eh)
    
    # Console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(ch)
    
    return logger


# ============================================================================
# Format Helpers
# ============================================================================

def format_params(params: Dict[str, Any]) -> str:
    """Format parameters for logging."""
    items = []
    for k, v in params.items():
        if isinstance(v, float):
            items.append(f"{k}={v:.4g}")
        else:
            items.append(f"{k}={v}")
    return ", ".join(items)


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for logging."""
    items = []
    for k, v in metrics.items():
        if v is not None:
            items.append(f"{k}={v:.4f}")
    return ", ".join(items)


# ============================================================================
# Training Metrics Database
# ============================================================================

class TrainingMetricsDB:
    """
    SQLite database for training metrics.
    
    Tables:
    - training_runs: One row per training run
    - iteration_metrics: Detailed iteration/epoch metrics
    
    Example:
        >>> db = TrainingMetricsDB()
        >>> db.log_training_start('run_001', 'als', {'factors': 64})
        >>> db.log_iteration('run_001', 1, loss=0.5, time=1.2)
        >>> db.log_training_complete('run_001', {'recall@10': 0.15})
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str = TRAINING_DB_PATH):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, db_path: str = TRAINING_DB_PATH):
        if self._initialized:
            return
        
        self.db_path = db_path
        ensure_log_dirs()
        self._create_tables()
        self._initialized = True
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            # Training runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id TEXT PRIMARY KEY,
                    model_type TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT,
                    
                    -- Hyperparameters (JSON)
                    hyperparameters TEXT,
                    
                    -- Metrics
                    recall_at_10 REAL,
                    recall_at_20 REAL,
                    ndcg_at_10 REAL,
                    ndcg_at_20 REAL,
                    coverage REAL,
                    
                    -- Comparison
                    baseline_recall_at_10 REAL,
                    improvement_pct REAL,
                    
                    -- System
                    training_time_seconds REAL,
                    data_version TEXT,
                    git_commit TEXT,
                    
                    -- Artifacts
                    artifacts_path TEXT
                )
            """)
            
            # Iteration metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS iteration_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    iteration INT,
                    timestamp TIMESTAMP,
                    loss REAL,
                    validation_recall REAL,
                    validation_ndcg REAL,
                    wall_time_seconds REAL,
                    
                    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
                )
            """)
            
            # Create indices
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_iteration_run_id 
                ON iteration_metrics(run_id)
            """)
            
            conn.commit()
    
    def log_training_start(
        self,
        run_id: str,
        model_type: str,
        params: Dict[str, Any],
        data_version: Optional[str] = None,
        git_commit: Optional[str] = None
    ):
        """
        Log training start.
        
        Args:
            run_id: Unique run identifier
            model_type: 'als', 'bpr', 'bert_als'
            params: Hyperparameters dict
            data_version: Data hash/version
            git_commit: Git commit hash
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO training_runs 
                (run_id, model_type, started_at, status, hyperparameters, 
                 data_version, git_commit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                model_type,
                datetime.now().isoformat(),
                'running',
                json.dumps(params),
                data_version,
                git_commit
            ))
            conn.commit()
    
    def log_iteration(
        self,
        run_id: str,
        iteration: int,
        loss: Optional[float] = None,
        validation_recall: Optional[float] = None,
        validation_ndcg: Optional[float] = None,
        wall_time_seconds: Optional[float] = None
    ):
        """
        Log iteration/epoch metrics.
        
        Args:
            run_id: Run identifier
            iteration: Iteration or epoch number
            loss: Training loss
            validation_recall: Validation Recall@K
            validation_ndcg: Validation NDCG@K
            wall_time_seconds: Wall time for this iteration
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO iteration_metrics 
                (run_id, iteration, timestamp, loss, validation_recall, 
                 validation_ndcg, wall_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                iteration,
                datetime.now().isoformat(),
                loss,
                validation_recall,
                validation_ndcg,
                wall_time_seconds
            ))
            conn.commit()
    
    def log_training_complete(
        self,
        run_id: str,
        metrics: Dict[str, float],
        artifacts_path: Optional[str] = None,
        training_time_seconds: Optional[float] = None,
        baseline_recall: Optional[float] = None
    ):
        """
        Log training completion.
        
        Args:
            run_id: Run identifier
            metrics: Dict with recall@10, ndcg@10, etc.
            artifacts_path: Path to saved artifacts
            training_time_seconds: Total training time
            baseline_recall: Baseline Recall@10 for comparison
        """
        recall_10 = metrics.get('recall@10') or metrics.get('recall_at_10')
        ndcg_10 = metrics.get('ndcg@10') or metrics.get('ndcg_at_10')
        
        improvement = None
        if baseline_recall and recall_10:
            improvement = ((recall_10 - baseline_recall) / baseline_recall) * 100
        
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE training_runs
                SET completed_at = ?,
                    status = 'completed',
                    recall_at_10 = ?,
                    recall_at_20 = ?,
                    ndcg_at_10 = ?,
                    ndcg_at_20 = ?,
                    coverage = ?,
                    training_time_seconds = ?,
                    artifacts_path = ?,
                    baseline_recall_at_10 = ?,
                    improvement_pct = ?
                WHERE run_id = ?
            """, (
                datetime.now().isoformat(),
                recall_10,
                metrics.get('recall@20') or metrics.get('recall_at_20'),
                ndcg_10,
                metrics.get('ndcg@20') or metrics.get('ndcg_at_20'),
                metrics.get('coverage'),
                training_time_seconds,
                artifacts_path,
                baseline_recall,
                improvement,
                run_id
            ))
            conn.commit()
    
    def log_training_failed(self, run_id: str, error_message: str):
        """Log training failure."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE training_runs
                SET completed_at = ?,
                    status = 'failed',
                    artifacts_path = ?
                WHERE run_id = ?
            """, (
                datetime.now().isoformat(),
                f"ERROR: {error_message}",
                run_id
            ))
            conn.commit()
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get training run by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM training_runs WHERE run_id = ?",
                (run_id,)
            ).fetchone()
            return dict(row) if row else None
    
    def get_recent_runs(
        self,
        model_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent training runs."""
        with self._get_connection() as conn:
            if model_type:
                rows = conn.execute("""
                    SELECT * FROM training_runs 
                    WHERE model_type = ?
                    ORDER BY started_at DESC
                    LIMIT ?
                """, (model_type, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM training_runs 
                    ORDER BY started_at DESC
                    LIMIT ?
                """, (limit,)).fetchall()
            return [dict(row) for row in rows]
    
    def get_iteration_metrics(self, run_id: str) -> List[Dict]:
        """Get all iteration metrics for a run."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM iteration_metrics 
                WHERE run_id = ?
                ORDER BY iteration
            """, (run_id,)).fetchall()
            return [dict(row) for row in rows]
    
    def get_best_run(self, model_type: str) -> Optional[Dict]:
        """Get the best run by Recall@10."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM training_runs 
                WHERE model_type = ? AND status = 'completed'
                ORDER BY recall_at_10 DESC
                LIMIT 1
            """, (model_type,)).fetchone()
            return dict(row) if row else None


# ============================================================================
# Service Metrics Database
# ============================================================================

class ServiceMetricsDB:
    """
    SQLite database for service metrics.
    
    Tables:
    - requests: Individual request logs
    - service_health: Aggregated health metrics
    - reranking_metrics: Reranking performance
    
    Example:
        >>> db = ServiceMetricsDB()
        >>> db.log_request(user_id=123, topk=10, latency_ms=50, fallback=False)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: str = SERVICE_DB_PATH):
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, db_path: str = SERVICE_DB_PATH):
        if self._initialized:
            return
        
        self.db_path = db_path
        ensure_log_dirs()
        self._create_tables()
        self._initialized = True
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        with self._get_connection() as conn:
            # Requests table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    user_id INTEGER,
                    topk INTEGER,
                    exclude_seen BOOLEAN,
                    filter_params TEXT,
                    
                    latency_ms REAL,
                    num_recommendations INT,
                    fallback BOOLEAN,
                    fallback_method TEXT,
                    rerank_enabled BOOLEAN,
                    
                    error TEXT,
                    model_id TEXT
                )
            """)
            
            # Service health table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS service_health (
                    timestamp TIMESTAMP PRIMARY KEY,
                    requests_per_minute REAL,
                    avg_latency_ms REAL,
                    p50_latency_ms REAL,
                    p95_latency_ms REAL,
                    p99_latency_ms REAL,
                    fallback_rate REAL,
                    error_rate REAL,
                    active_model_id TEXT
                )
            """)
            
            # Reranking metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reranking_metrics (
                    timestamp TIMESTAMP PRIMARY KEY,
                    requests_with_rerank INTEGER,
                    requests_without_rerank INTEGER,
                    
                    avg_latency_rerank_ms REAL,
                    avg_latency_cf_only_ms REAL,
                    latency_overhead_pct REAL,
                    
                    avg_content_score REAL,
                    avg_diversity_rerank REAL,
                    avg_diversity_cf_only REAL
                )
            """)
            
            # Create indices
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_requests_timestamp 
                ON requests(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_requests_user 
                ON requests(user_id)
            """)
            
            conn.commit()
    
    def log_request(
        self,
        user_id: int,
        topk: int,
        latency_ms: float,
        num_recommendations: int,
        fallback: bool,
        model_id: Optional[str] = None,
        fallback_method: Optional[str] = None,
        rerank_enabled: bool = False,
        exclude_seen: bool = True,
        filter_params: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """
        Log a recommendation request.
        
        Args:
            user_id: User ID
            topk: Requested number of recommendations
            latency_ms: Request latency in milliseconds
            num_recommendations: Actual number returned
            fallback: Whether fallback was used
            model_id: Active model ID
            fallback_method: Fallback method used (if any)
            rerank_enabled: Whether reranking was applied
            exclude_seen: Whether seen items were excluded
            filter_params: Filter parameters (JSON)
            error: Error message if failed
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO requests 
                (timestamp, user_id, topk, exclude_seen, filter_params,
                 latency_ms, num_recommendations, fallback, fallback_method,
                 rerank_enabled, error, model_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                user_id,
                topk,
                exclude_seen,
                json.dumps(filter_params) if filter_params else None,
                latency_ms,
                num_recommendations,
                fallback,
                fallback_method,
                rerank_enabled,
                error,
                model_id
            ))
            conn.commit()
    
    def aggregate_health_metrics(self, model_id: Optional[str] = None):
        """
        Aggregate health metrics for the last minute.
        
        Args:
            model_id: Current active model ID
        """
        with self._get_connection() as conn:
            # Get last minute stats
            stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(latency_ms) as avg_latency,
                    AVG(CASE WHEN fallback = 1 THEN 1.0 ELSE 0.0 END) as fallback_rate,
                    AVG(CASE WHEN error IS NOT NULL THEN 1.0 ELSE 0.0 END) as error_rate
                FROM requests
                WHERE timestamp > datetime('now', '-1 minute')
            """).fetchone()
            
            # Get percentiles
            latencies = conn.execute("""
                SELECT latency_ms FROM requests
                WHERE timestamp > datetime('now', '-1 minute')
                ORDER BY latency_ms
            """).fetchall()
            
            if latencies:
                latency_values = [r[0] for r in latencies]
                n = len(latency_values)
                p50 = latency_values[int(n * 0.5)] if n > 0 else 0
                p95 = latency_values[int(n * 0.95)] if n > 0 else 0
                p99 = latency_values[int(n * 0.99)] if n > 0 else 0
            else:
                p50 = p95 = p99 = 0
            
            # Insert aggregated metrics
            conn.execute("""
                INSERT OR REPLACE INTO service_health
                (timestamp, requests_per_minute, avg_latency_ms, 
                 p50_latency_ms, p95_latency_ms, p99_latency_ms,
                 fallback_rate, error_rate, active_model_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                stats['total_requests'] or 0,
                stats['avg_latency'] or 0,
                p50,
                p95,
                p99,
                stats['fallback_rate'] or 0,
                stats['error_rate'] or 0,
                model_id
            ))
            conn.commit()
    
    def get_recent_health(self, minutes: int = 60) -> List[Dict]:
        """Get health metrics for the last N minutes."""
        with self._get_connection() as conn:
            rows = conn.execute(f"""
                SELECT * FROM service_health
                WHERE timestamp > datetime('now', '-{minutes} minutes')
                ORDER BY timestamp DESC
            """).fetchall()
            return [dict(row) for row in rows]
    
    def get_request_stats(
        self,
        minutes: int = 60,
        user_id: Optional[int] = None
    ) -> Dict:
        """Get request statistics."""
        with self._get_connection() as conn:
            where_clause = f"timestamp > datetime('now', '-{minutes} minutes')"
            if user_id:
                where_clause += f" AND user_id = {user_id}"
            
            stats = conn.execute(f"""
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(latency_ms) as avg_latency,
                    MIN(latency_ms) as min_latency,
                    MAX(latency_ms) as max_latency,
                    SUM(CASE WHEN fallback = 1 THEN 1 ELSE 0 END) as fallback_count,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_count
                FROM requests
                WHERE {where_clause}
            """).fetchone()
            
            return dict(stats) if stats else {}


# ============================================================================
# Convenience Functions
# ============================================================================

def get_training_db() -> TrainingMetricsDB:
    """Get singleton TrainingMetricsDB instance."""
    return TrainingMetricsDB()


def get_service_db() -> ServiceMetricsDB:
    """Get singleton ServiceMetricsDB instance."""
    return ServiceMetricsDB()


def generate_run_id(model_type: str) -> str:
    """
    Generate unique run ID.
    
    Args:
        model_type: 'als', 'bpr', 'bert_als'
    
    Returns:
        Run ID like 'als_20251125_103000'
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{model_type}_{timestamp}"
