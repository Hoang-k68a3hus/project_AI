"""
Common utilities for automation scripts.
Provides retry logic, error handling, and pipeline helpers.
"""

import os
import sys
import time
import json
import logging
import sqlite3
import hashlib
import functools
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass, asdict, field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Retry Decorator
# =============================================================================

T = TypeVar('T')


def retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,),
    on_failure: Optional[Callable[[Exception, int], None]] = None
) -> Callable:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        base_delay: Initial delay in seconds
        exceptions: Tuple of exceptions to catch
        on_failure: Callback function(exception, attempt) called on each failure
        
    Returns:
        Decorated function
        
    Example:
        @retry(max_attempts=3, backoff_factor=2.0)
        def fetch_data():
            # May fail temporarily
            pass
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if on_failure:
                        on_failure(e, attempt)
                    
                    if attempt < max_attempts:
                        delay = base_delay * (backoff_factor ** (attempt - 1))
                        logging.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logging.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )
            
            # Re-raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


# =============================================================================
# Pipeline Status Tracking
# =============================================================================

@dataclass
class PipelineRun:
    """Represents a single pipeline execution."""
    run_id: str
    pipeline_name: str
    status: str  # 'running', 'success', 'failed', 'cancelled'
    started_at: str
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PipelineTracker:
    """
    Track pipeline execution status in SQLite database.
    
    Usage:
        tracker = PipelineTracker()
        run_id = tracker.start_run("data_refresh")
        try:
            # ... pipeline logic ...
            tracker.complete_run(run_id, metadata={"records_processed": 1000})
        except Exception as e:
            tracker.fail_run(run_id, str(e))
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize tracker with database path."""
        self.db_path = db_path or (PROJECT_ROOT / "logs" / "pipeline_metrics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id TEXT PRIMARY KEY,
                    pipeline_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    duration_seconds REAL,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            
            # Index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_name_status
                ON pipeline_runs(pipeline_name, status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_started_at
                ON pipeline_runs(started_at)
            """)
            
            conn.commit()
    
    def start_run(self, pipeline_name: str, metadata: Optional[Dict] = None) -> str:
        """
        Start tracking a new pipeline run.
        
        Args:
            pipeline_name: Name of the pipeline
            metadata: Optional initial metadata
            
        Returns:
            Unique run_id for tracking
        """
        run_id = f"{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        started_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs 
                (run_id, pipeline_name, status, started_at, metadata)
                VALUES (?, ?, 'running', ?, ?)
                """,
                (run_id, pipeline_name, started_at, json.dumps(metadata or {}))
            )
            conn.commit()
        
        logging.info(f"Started pipeline run: {run_id}")
        return run_id
    
    def complete_run(
        self, 
        run_id: str, 
        metadata: Optional[Dict] = None
    ) -> None:
        """Mark a pipeline run as completed successfully."""
        self._finish_run(run_id, 'success', metadata=metadata)
    
    def fail_run(
        self, 
        run_id: str, 
        error_message: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Mark a pipeline run as failed."""
        self._finish_run(run_id, 'failed', error_message, metadata)
    
    def cancel_run(self, run_id: str) -> None:
        """Mark a pipeline run as cancelled."""
        self._finish_run(run_id, 'cancelled')
    
    def _finish_run(
        self, 
        run_id: str, 
        status: str,
        error_message: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Internal method to finish a run."""
        finished_at = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get start time to calculate duration
            cursor = conn.execute(
                "SELECT started_at, metadata FROM pipeline_runs WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            
            if row:
                started_at = datetime.fromisoformat(row[0])
                duration = (datetime.now() - started_at).total_seconds()
                
                # Merge metadata
                existing_metadata = json.loads(row[1]) if row[1] else {}
                if metadata:
                    existing_metadata.update(metadata)
                
                conn.execute(
                    """
                    UPDATE pipeline_runs 
                    SET status = ?, finished_at = ?, duration_seconds = ?,
                        error_message = ?, metadata = ?
                    WHERE run_id = ?
                    """,
                    (status, finished_at, duration, error_message,
                     json.dumps(existing_metadata), run_id)
                )
                conn.commit()
                
                logging.info(f"Finished pipeline run: {run_id} ({status})")
    
    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get details of a specific run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM pipeline_runs WHERE run_id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return PipelineRun(
                    run_id=row['run_id'],
                    pipeline_name=row['pipeline_name'],
                    status=row['status'],
                    started_at=row['started_at'],
                    finished_at=row['finished_at'],
                    duration_seconds=row['duration_seconds'],
                    error_message=row['error_message'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
        return None
    
    def get_recent_runs(
        self, 
        pipeline_name: Optional[str] = None,
        limit: int = 10,
        status: Optional[str] = None
    ) -> List[PipelineRun]:
        """Get recent pipeline runs."""
        query = "SELECT * FROM pipeline_runs WHERE 1=1"
        params = []
        
        if pipeline_name:
            query += " AND pipeline_name = ?"
            params.append(pipeline_name)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        
        runs = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            for row in cursor:
                runs.append(PipelineRun(
                    run_id=row['run_id'],
                    pipeline_name=row['pipeline_name'],
                    status=row['status'],
                    started_at=row['started_at'],
                    finished_at=row['finished_at'],
                    duration_seconds=row['duration_seconds'],
                    error_message=row['error_message'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                ))
        
        return runs
    
    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get pipeline statistics for the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Count by status
            cursor = conn.execute(
                """
                SELECT pipeline_name, status, COUNT(*) as count
                FROM pipeline_runs
                WHERE started_at >= ?
                GROUP BY pipeline_name, status
                """,
                (cutoff,)
            )
            
            stats_by_pipeline = {}
            for row in cursor:
                name, status, count = row
                if name not in stats_by_pipeline:
                    stats_by_pipeline[name] = {'success': 0, 'failed': 0, 'running': 0}
                stats_by_pipeline[name][status] = count
            
            # Average duration for successful runs
            cursor = conn.execute(
                """
                SELECT pipeline_name, AVG(duration_seconds) as avg_duration
                FROM pipeline_runs
                WHERE started_at >= ? AND status = 'success'
                GROUP BY pipeline_name
                """,
                (cutoff,)
            )
            
            for row in cursor:
                name, avg_duration = row
                if name in stats_by_pipeline:
                    stats_by_pipeline[name]['avg_duration_seconds'] = avg_duration
            
            # Calculate success rate
            for name, stats in stats_by_pipeline.items():
                total = stats.get('success', 0) + stats.get('failed', 0)
                if total > 0:
                    stats['success_rate'] = stats.get('success', 0) / total
                else:
                    stats['success_rate'] = None
            
            return {
                'period_days': days,
                'stats_by_pipeline': stats_by_pipeline
            }
    
    def is_pipeline_running(self, pipeline_name: str) -> bool:
        """Check if a pipeline is currently running (prevent concurrent runs)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM pipeline_runs
                WHERE pipeline_name = ? AND status = 'running'
                """,
                (pipeline_name,)
            )
            count = cursor.fetchone()[0]
            return count > 0
    
    def cleanup_stale_runs(self, max_running_hours: int = 24) -> int:
        """Mark runs that have been 'running' for too long as failed."""
        cutoff = (datetime.now() - timedelta(hours=max_running_hours)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE pipeline_runs
                SET status = 'failed', 
                    error_message = 'Marked as stale - exceeded max running time',
                    finished_at = ?
                WHERE status = 'running' AND started_at < ?
                """,
                (datetime.now().isoformat(), cutoff)
            )
            conn.commit()
            
            return cursor.rowcount


# =============================================================================
# Alert Helper
# =============================================================================

def send_pipeline_alert(
    pipeline_name: str,
    status: str,
    message: str,
    severity: str = "warning",
    metadata: Optional[Dict] = None
) -> None:
    """
    Send an alert for pipeline events.
    
    Args:
        pipeline_name: Name of the pipeline
        status: Current status (success, failed, etc.)
        message: Alert message
        severity: Alert severity (info, warning, error, critical)
        metadata: Additional context
    """
    try:
        from recsys.cf.alerting import AlertManager
        
        alert_manager = AlertManager()
        
        # Format the message with context
        full_message = f"""Pipeline: {pipeline_name}
Status: {status}
Time: {datetime.now().isoformat()}
Message: {message}
"""
        if metadata:
            full_message += f"\nMetadata: {json.dumps(metadata, indent=2, default=str)}"
        
        alert_manager.send_alert(
            subject=f"[Pipeline] {pipeline_name}: {status}",
            message=full_message,
            severity=severity
        )
        
    except ImportError:
        logging.warning("AlertManager not available, logging alert instead")
        logging.log(
            logging.ERROR if severity in ('error', 'critical') else logging.WARNING,
            f"PIPELINE ALERT [{severity.upper()}] {pipeline_name}: {message}"
        )
    except Exception as e:
        logging.error(f"Failed to send alert: {e}")


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    log_name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up logging for a script.
    
    Args:
        log_name: Name for the log file
        log_dir: Directory for log files (default: logs/)
        level: Logging level
        console: Whether to also log to console
        
    Returns:
        Configured logger
    """
    log_dir = log_dir or (PROJECT_ROOT / "logs" / "pipelines")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{log_name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(file_format)
        logger.addHandler(console_handler)
    
    return logger


# =============================================================================
# Data Hashing
# =============================================================================

def compute_file_hash(file_path: Path, algorithm: str = 'md5') -> str:
    """
    Compute hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256')
        
    Returns:
        Hex digest of hash
    """
    hasher = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def compute_data_hash(data_dir: Path, file_patterns: List[str]) -> str:
    """
    Compute combined hash of multiple data files.
    
    Args:
        data_dir: Directory containing files
        file_patterns: List of file patterns to include
        
    Returns:
        Combined hex digest
    """
    import glob
    
    file_hashes = []
    
    for pattern in file_patterns:
        for file_path in glob.glob(str(data_dir / pattern)):
            path = Path(file_path)
            if path.is_file():
                file_hash = compute_file_hash(path)
                file_hashes.append(f"{path.name}:{file_hash}")
    
    # Sort for consistency
    file_hashes.sort()
    
    # Combine hashes
    combined = hashlib.md5()
    for h in file_hashes:
        combined.update(h.encode())
    
    return combined.hexdigest()


# =============================================================================
# Lock File Management
# =============================================================================

class PipelineLock:
    """
    File-based lock to prevent concurrent pipeline runs.
    
    Usage:
        with PipelineLock("data_refresh") as lock:
            if lock.acquired:
                # ... run pipeline ...
            else:
                print("Pipeline already running")
    """
    
    def __init__(self, pipeline_name: str, lock_dir: Optional[Path] = None):
        self.pipeline_name = pipeline_name
        self.lock_dir = lock_dir or (PROJECT_ROOT / "logs" / "locks")
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.lock_dir / f"{pipeline_name}.lock"
        self.acquired = False
    
    def __enter__(self) -> 'PipelineLock':
        if self.lock_file.exists():
            # Check if lock is stale (>24 hours)
            mtime = datetime.fromtimestamp(self.lock_file.stat().st_mtime)
            if datetime.now() - mtime > timedelta(hours=24):
                logging.warning(f"Removing stale lock: {self.lock_file}")
                self.lock_file.unlink()
            else:
                self.acquired = False
                return self
        
        # Create lock file
        with open(self.lock_file, 'w') as f:
            f.write(json.dumps({
                'pipeline': self.pipeline_name,
                'pid': os.getpid(),
                'started_at': datetime.now().isoformat()
            }))
        
        self.acquired = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired and self.lock_file.exists():
            self.lock_file.unlink()
        return False


# =============================================================================
# Git Utilities
# =============================================================================

def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def get_git_branch() -> Optional[str]:
    """Get current git branch name."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test utilities
    print("Testing automation utilities...")
    
    # Test logging
    logger = setup_logging("test_utils")
    logger.info("Logging test successful")
    
    # Test pipeline tracker
    tracker = PipelineTracker()
    run_id = tracker.start_run("test_pipeline", {"test": True})
    print(f"Started run: {run_id}")
    
    time.sleep(0.5)
    tracker.complete_run(run_id, {"records": 100})
    
    run = tracker.get_run(run_id)
    print(f"Run details: {run}")
    
    # Test retry decorator
    @retry(max_attempts=3, base_delay=0.1)
    def flaky_function(fail_times: int):
        flaky_function.attempts = getattr(flaky_function, 'attempts', 0) + 1
        if flaky_function.attempts <= fail_times:
            raise ValueError(f"Simulated failure {flaky_function.attempts}")
        return "success"
    
    result = flaky_function(2)
    print(f"Retry test result: {result}")
    
    # Test lock
    with PipelineLock("test_pipeline") as lock:
        print(f"Lock acquired: {lock.acquired}")
    
    # Test git
    print(f"Git commit: {get_git_commit()}")
    print(f"Git branch: {get_git_branch()}")
    
    print("\nAll utils tests passed!")
