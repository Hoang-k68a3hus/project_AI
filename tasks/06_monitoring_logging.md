# Task 06: Monitoring & Logging

## Mục Tiêu

Xây dựng hệ thống monitoring và logging toàn diện để theo dõi training performance, service health, data quality, và trigger retraining khi cần. System phải hỗ trợ debugging, alerting, và continuous improvement.

## Monitoring Architecture

```
Data Sources
    ↓
├─ Training Logs (cf/als.log, cf/bpr.log)
├─ Service Logs (service/recommender.log)
├─ Registry Audit Logs (logs/registry_audit.log)
├─ Metrics DB (SQLite/Prometheus)
└─ Data Quality Checks
    ↓
Aggregation & Analysis
    ↓
├─ Dashboards (Grafana/custom)
├─ Alerts (email/Slack)
├─ Registry Health Monitoring
└─ Drift Detection
    ↓
Actions
    ↓
├─ Trigger Retrain
├─ Rollback Model
├─ Model Hot Reload
└─ Notify Team
```

## Component 1: Training Monitoring

### Log File Structure

#### File: `logs/cf/als.log`
```
2025-01-15 10:30:00 | INFO | Training started | factors=64, reg=0.01, alpha=40
2025-01-15 10:30:15 | INFO | Iteration 1/15 | loss=125.34, time=1.2s
2025-01-15 10:30:30 | INFO | Iteration 2/15 | loss=98.12, time=1.1s
...
2025-01-15 10:32:00 | INFO | Training completed | total_time=45.2s
2025-01-15 10:32:05 | INFO | Evaluation | recall@10=0.234, ndcg@10=0.189
2025-01-15 10:32:10 | INFO | Artifacts saved | path=artifacts/cf/als/v1_20250115_103000
```

#### File: `logs/cf/bpr.log`
```
2025-01-15 12:00:00 | INFO | Training started | factors=64, lr=0.05, epochs=50
2025-01-15 12:00:30 | INFO | Epoch 1/50 | avg_loss=0.523, samples=1600000, time=28.5s
2025-01-15 12:01:00 | INFO | Epoch 2/50 | avg_loss=0.412, samples=1600000, time=27.8s
2025-01-15 12:02:30 | INFO | Validation | recall@10=0.198, ndcg@10=0.153
...
2025-01-15 12:30:00 | INFO | Early stopping | best_epoch=35, best_ndcg@10=0.192
2025-01-15 12:30:05 | INFO | Artifacts saved | path=artifacts/cf/bpr/v1_20250115_120000
```

### Training Metrics Database

#### Schema: `logs/training_metrics.db` (SQLite)

##### Table: `training_runs`
```sql
CREATE TABLE training_runs (
    run_id TEXT PRIMARY KEY,
    model_type TEXT,  -- 'als' or 'bpr'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT,  -- 'running', 'completed', 'failed'
    
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
    artifacts_path TEXT,
    
    -- Registry Integration
    model_id TEXT,  -- Registry model_id
    registered_at TIMESTAMP,
    registry_status TEXT  -- 'active', 'archived', 'failed'
);
```

##### Table: `iteration_metrics` (Detailed tracking)
```sql
CREATE TABLE iteration_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INT,  -- ALS iteration hoặc BPR epoch
    timestamp TIMESTAMP,
    loss REAL,
    validation_recall REAL,
    validation_ndcg REAL,
    wall_time_seconds REAL,
    
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
);
```

### Logging Implementation

#### Module: `recsys/cf/logging_utils.py`

##### Function: `setup_training_logger(model_type, run_id)`
```python
import logging
from datetime import datetime

def setup_training_logger(model_type, run_id):
    """
    Setup logger cho training run.
    
    Args:
        model_type: 'als' or 'bpr'
        run_id: Unique run identifier
    
    Returns:
        logging.Logger
    """
    logger = logging.getLogger(f'cf.{model_type}.{run_id}')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(f'logs/cf/{model_type}.log', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger
```

##### Function: `log_training_start(logger, params, db_conn, run_id)`
```python
def log_training_start(logger, params, db_conn, run_id):
    """Log training start và insert vào DB."""
    logger.info(f"Training started | {format_params(params)}")
    
    db_conn.execute("""
        INSERT INTO training_runs 
        (run_id, model_type, started_at, status, hyperparameters, data_version)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        params['model_type'],
        datetime.now(),
        'running',
        json.dumps(params),
        params.get('data_version', 'unknown')
    ))
    db_conn.commit()
```

##### Function: `log_iteration(logger, iteration, loss, time, db_conn, run_id)`
```python
def log_iteration(logger, iteration, loss, time, db_conn, run_id):
    """Log iteration metrics."""
    logger.info(f"Iteration {iteration} | loss={loss:.4f}, time={time:.1f}s")
    
    db_conn.execute("""
        INSERT INTO iteration_metrics 
        (run_id, iteration, timestamp, loss, wall_time_seconds)
        VALUES (?, ?, ?, ?, ?)
    """, (run_id, iteration, datetime.now(), loss, time))
    db_conn.commit()
```

##### Function: `log_training_complete(logger, metrics, artifacts_path, db_conn, run_id)`
```python
from recsys.cf.registry import ModelRegistry

def log_training_complete(logger, metrics, artifacts_path, db_conn, run_id, model_type):
    """Log completion, update DB, và register model."""
    logger.info(f"Training completed | total_time={metrics['training_time']}s")
    logger.info(f"Evaluation | recall@10={metrics['recall@10']:.3f}, ndcg@10={metrics['ndcg@10']:.3f}")
    
    # Register model in registry
    registry = ModelRegistry()
    model_id = registry.register_model(
        artifacts_path=artifacts_path,
        model_type=model_type,
        hyperparameters=metrics.get('hyperparameters', {}),
        metrics={
            'recall@10': metrics['recall@10'],
            'recall@20': metrics.get('recall@20', 0),
            'ndcg@10': metrics['ndcg@10'],
            'ndcg@20': metrics.get('ndcg@20', 0),
            'coverage': metrics.get('coverage', 0)
        },
        training_info={
            'training_time_seconds': metrics['training_time'],
            'run_id': run_id
        },
        data_version=metrics.get('data_version'),
        baseline_comparison=metrics.get('baseline_comparison')
    )
    
    logger.info(f"Model registered: {model_id}")
    
    # Update DB with registry info
    db_conn.execute("""
        UPDATE training_runs
        SET completed_at = ?,
            status = 'completed',
            recall_at_10 = ?,
            ndcg_at_10 = ?,
            coverage = ?,
            training_time_seconds = ?,
            artifacts_path = ?,
            model_id = ?,
            registered_at = ?,
            registry_status = 'active'
        WHERE run_id = ?
    """, (
        datetime.now(),
        metrics['recall@10'],
        metrics['ndcg@10'],
        metrics['coverage'],
        metrics['training_time'],
        artifacts_path,
        model_id,
        datetime.now(),
        run_id
    ))
    db_conn.commit()
    
    # Auto-select best model if improved
    best = registry.select_best_model(metric='ndcg@10')
    if best and best['model_id'] == model_id:
        logger.info(f"New best model selected: {model_id} (ndcg@10={best['value']:.4f})")
```

### Training Progress Visualization

#### Script: `scripts/plot_training_progress.py`
```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_curves(run_id):
    """Plot loss và validation metrics over iterations."""
    conn = sqlite3.connect('logs/training_metrics.db')
    
    df = pd.read_sql(f"""
        SELECT iteration, loss, validation_ndcg 
        FROM iteration_metrics 
        WHERE run_id = '{run_id}'
        ORDER BY iteration
    """, conn)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    ax1.plot(df['iteration'], df['loss'])
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss - {run_id}')
    
    # Validation metric
    ax2.plot(df['iteration'], df['validation_ndcg'])
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('Validation NDCG@10')
    
    plt.tight_layout()
    plt.savefig(f'reports/training_curves_{run_id}.png')
    plt.close()
```

## Component 2: Registry Monitoring

### Registry Audit Log

#### File: `logs/registry_audit.log`
```
2025-01-15 10:32:15 | REGISTER | als_v1_20250115_103000 | ndcg@10=0.189
2025-01-15 10:32:20 | SELECT_BEST | als_v1_20250115_103000 | ndcg@10=0.189 improvement=+5.2%
2025-01-15 12:30:10 | REGISTER | bpr_v1_20250115_120000 | ndcg@10=0.192
2025-01-15 12:30:15 | SELECT_BEST | bpr_v1_20250115_120000 | ndcg@10=0.192 improvement=+1.6%
2025-01-15 15:00:00 | ARCHIVE | als_v1_20250110_090000 | reason=manual
2025-01-15 16:00:00 | UPDATE_STATUS | bpr_v1_20250115_120000 | status=failed
```

### Registry Operations Database

#### Schema: `logs/registry_operations.db` (SQLite)

##### Table: `registry_operations`
```sql
CREATE TABLE registry_operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP,
    action TEXT,  -- 'REGISTER', 'SELECT_BEST', 'ARCHIVE', 'DELETE', 'UPDATE_STATUS'
    model_id TEXT,
    model_type TEXT,
    details TEXT,  -- JSON with action-specific details
    
    -- For SELECT_BEST
    metric TEXT,
    metric_value REAL,
    improvement_pct REAL,
    previous_best_id TEXT,
    
    -- For REGISTER
    artifacts_path TEXT,
    data_version TEXT,
    git_commit TEXT
);
```

##### Table: `model_versions`
```sql
CREATE TABLE model_versions (
    model_id TEXT PRIMARY KEY,
    model_type TEXT,
    version TEXT,
    created_at TIMESTAMP,
    registered_at TIMESTAMP,
    
    -- Metrics snapshot
    recall_at_10 REAL,
    ndcg_at_10 REAL,
    coverage REAL,
    
    -- Status tracking
    status TEXT,  -- 'active', 'archived', 'failed'
    is_current_best BOOLEAN,
    selected_at TIMESTAMP,
    
    -- Lineage
    data_version TEXT,
    git_commit TEXT,
    artifacts_path TEXT
);
```

### Registry Monitoring Implementation

#### Module: `recsys/cf/monitoring/registry_monitor.py`

##### Function: `setup_registry_monitoring(registry_path)`
```python
from recsys.cf.registry import ModelRegistry
import sqlite3
import json
from datetime import datetime

def setup_registry_monitoring(registry_path='artifacts/cf/registry.json'):
    """
    Setup registry monitoring với database tracking.
    
    Returns:
        RegistryMonitor instance
    """
    return RegistryMonitor(registry_path)

class RegistryMonitor:
    """Monitor registry operations và track changes."""
    
    def __init__(self, registry_path):
        self.registry_path = registry_path
        self.registry = ModelRegistry(registry_path)
        self.db_path = 'logs/registry_operations.db'
        self._init_db()
    
    def _init_db(self):
        """Initialize monitoring database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS registry_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                action TEXT,
                model_id TEXT,
                model_type TEXT,
                details TEXT,
                metric TEXT,
                metric_value REAL,
                improvement_pct REAL,
                previous_best_id TEXT,
                artifacts_path TEXT,
                data_version TEXT,
                git_commit TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                model_id TEXT PRIMARY KEY,
                model_type TEXT,
                version TEXT,
                created_at TIMESTAMP,
                registered_at TIMESTAMP,
                recall_at_10 REAL,
                ndcg_at_10 REAL,
                coverage REAL,
                status TEXT,
                is_current_best BOOLEAN,
                selected_at TIMESTAMP,
                data_version TEXT,
                git_commit TEXT,
                artifacts_path TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_operation(self, action, model_id, details=None):
        """Log registry operation to database."""
        conn = sqlite3.connect(self.db_path)
        
        model_info = self.registry.get_model(model_id) if model_id else None
        
        conn.execute("""
            INSERT INTO registry_operations
            (timestamp, action, model_id, model_type, details, artifacts_path, data_version, git_commit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(),
            action,
            model_id,
            model_info['model_type'] if model_info else None,
            json.dumps(details) if details else None,
            model_info['path'] if model_info else None,
            model_info.get('data_version') if model_info else None,
            model_info.get('git_commit') if model_info else None
        ))
        
        conn.commit()
        conn.close()
    
    def sync_model_versions(self):
        """Sync current registry state to model_versions table."""
        conn = sqlite3.connect(self.db_path)
        
        # Get current best
        current_best = self.registry.get_current_best()
        current_best_id = current_best['model_id'] if current_best else None
        
        # Sync all models
        for model_id, model in self.registry._registry['models'].items():
            is_best = (model_id == current_best_id)
            
            conn.execute("""
                INSERT OR REPLACE INTO model_versions
                (model_id, model_type, version, created_at, registered_at,
                 recall_at_10, ndcg_at_10, coverage, status, is_current_best,
                 selected_at, data_version, git_commit, artifacts_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                model['model_type'],
                model['version'],
                model['created_at'],
                model['created_at'],  # registered_at
                model['metrics'].get('recall@10'),
                model['metrics'].get('ndcg@10'),
                model['metrics'].get('coverage'),
                model['status'],
                is_best,
                current_best['selected_at'] if is_best and current_best else None,
                model.get('data_version'),
                model.get('git_commit'),
                model['path']
            ))
        
        conn.commit()
        conn.close()
    
    def get_registry_health(self):
        """Get registry health metrics."""
        stats = self.registry.get_registry_stats()
        current_best = self.registry.get_current_best()
        
        return {
            'total_models': stats['total_models'],
            'active_models': stats['active_models'],
            'archived_models': stats['archived_models'],
            'current_best': current_best['model_id'] if current_best else None,
            'current_best_ndcg': (
                current_best['model_info']['metrics']['ndcg@10']
                if current_best and current_best.get('model_info') else None
            ),
            'by_type': stats['by_type'],
            'last_updated': stats['last_updated']
        }
```

### Model Loader Monitoring

#### Table: `model_loader_stats`
```sql
CREATE TABLE model_loader_stats (
    timestamp TIMESTAMP PRIMARY KEY,
    model_id TEXT,
    total_loads INTEGER,
    cache_hits INTEGER,
    cache_misses INTEGER,
    cache_hit_rate REAL,
    reload_count INTEGER,
    last_load_time_ms REAL,
    last_reload_at TIMESTAMP
);
```

#### Function: `log_loader_stats(loader)`
```python
from recsys.cf.registry import ModelLoader

def log_loader_stats(loader: ModelLoader, db_path='logs/service_metrics.db'):
    """Log model loader statistics."""
    import sqlite3
    
    stats = loader.get_stats()
    model_info = loader.get_model_info()
    
    conn = sqlite3.connect(db_path)
    
    conn.execute("""
        INSERT INTO model_loader_stats
        (timestamp, model_id, total_loads, cache_hits, cache_misses,
         cache_hit_rate, reload_count, last_load_time_ms, last_reload_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(),
        model_info.get('model_id'),
        stats['total_loads'],
        stats['cache_hits'],
        stats['cache_misses'],
        stats['cache_hit_rate'],
        stats['reload_count'],
        stats['last_load_time_ms'],
        stats['last_reload_at']
    ))
    
    conn.commit()
    conn.close()
```

### Registry Health Checks

#### Function: `check_registry_health()`
```python
def check_registry_health(registry_path='artifacts/cf/registry.json'):
    """
    Check registry health và alert nếu có vấn đề.
    
    Returns:
        dict: Health status
    """
    from recsys.cf.registry import ModelRegistry
    
    registry = ModelRegistry(registry_path)
    health = {
        'status': 'healthy',
        'issues': [],
        'warnings': []
    }
    
    # Check 1: Has current best model
    current_best = registry.get_current_best()
    if not current_best:
        health['status'] = 'unhealthy'
        health['issues'].append('No current best model selected')
    
    # Check 2: Model files exist
    stats = registry.get_registry_stats()
    for model_id, model in registry._registry['models'].items():
        if model['status'] == 'active':
            import os
            if not os.path.exists(model['path']):
                health['status'] = 'unhealthy'
                health['issues'].append(f"Model {model_id} artifacts missing: {model['path']}")
    
    # Check 3: Too many archived models
    if stats['archived_models'] > 50:
        health['warnings'].append(f"Too many archived models: {stats['archived_models']}")
    
    # Check 4: Registry file corruption
    try:
        registry._load_registry()
    except Exception as e:
        health['status'] = 'unhealthy'
        health['issues'].append(f"Registry file corruption: {e}")
    
    return health
```

## Component 3: Service Monitoring

### Log File Structure

#### File: `logs/service/recommender.log`
```
2025-01-15 15:30:00 | INFO | Service started | model_id=als_v2_20250116_141500
2025-01-15 15:30:15 | INFO | Request | user_id=12345, topk=10, latency=0.082s, fallback=False
2025-01-15 15:30:16 | INFO | Request | user_id=67890, topk=10, latency=0.095s, fallback=False
2025-01-15 15:30:20 | WARNING | Cold-start | user_id=99999, fallback=True
2025-01-15 15:35:00 | ERROR | Recommendation failed | user_id=11111, error=KeyError
2025-01-15 16:00:00 | INFO | Model reloaded | old=als_v2, new=als_v3
```

### Request Metrics Database

#### Schema: `logs/service_metrics.db` (SQLite)

##### Table: `requests`
```sql
CREATE TABLE requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP,
    user_id INTEGER,
    topk INTEGER,
    exclude_seen BOOLEAN,
    filter_params TEXT,  -- JSON
    
    latency_ms REAL,
    num_recommendations INT,
    fallback BOOLEAN,
    fallback_method TEXT,
    
    error TEXT,  -- NULL if success
    model_id TEXT,
    
    -- Registry Integration
    model_version TEXT,
    loader_cache_hit BOOLEAN,
    reload_triggered BOOLEAN
);
```

##### Table: `service_health`
```sql
CREATE TABLE service_health (
    timestamp TIMESTAMP PRIMARY KEY,
    requests_per_minute REAL,
    avg_latency_ms REAL,
    p95_latency_ms REAL,
    fallback_rate REAL,
    error_rate REAL,
    active_model_id TEXT
);
```

### Service Logging Implementation

#### Middleware: Log Requests
```python
from fastapi import Request
import time
import sqlite3

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    latency = (time.time() - start_time) * 1000  # ms
    
    # Log to file
    logger.info(f"Request | path={request.url.path}, latency={latency:.1f}ms, status={response.status_code}")
    
    # Log to DB (async)
    if request.url.path == "/recommend":
        body = await request.json()
        log_request_to_db(body, latency, response.status_code)
    
    return response
```

#### Function: `log_request_to_db(body, latency, status)`
```python
from recsys.cf.registry import ModelLoader

def log_request_to_db(body, latency, status, loader: ModelLoader = None):
    """Log request metrics to SQLite với registry info."""
    conn = sqlite3.connect('logs/service_metrics.db')
    
    # Get model info from loader
    model_info = loader.get_model_info() if loader else {}
    loader_stats = loader.get_stats() if loader else {}
    
    conn.execute("""
        INSERT INTO requests 
        (timestamp, user_id, topk, exclude_seen, latency_ms, model_id,
         model_version, loader_cache_hit, reload_triggered)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(),
        body.get('user_id'),
        body.get('topk', 10),
        body.get('exclude_seen', True),
        latency,
        model_info.get('model_id'),
        model_info.get('version'),
        loader_stats.get('cache_hit_rate', 0) > 0.5,  # Approximate cache hit
        False  # reload_triggered - set by reload logic
    ))
    conn.commit()
    conn.close()
```

### Real-Time Metrics Aggregation

#### Background Task: Aggregate Metrics
```python
from fastapi_utils.tasks import repeat_every

@app.on_event("startup")
@repeat_every(seconds=60)  # Every minute
def aggregate_metrics():
    """Compute service health metrics."""
    conn = sqlite3.connect('logs/service_metrics.db')
    
    # Last 1 minute stats
    stats = pd.read_sql("""
        SELECT 
            COUNT(*) as requests,
            AVG(latency_ms) as avg_latency,
            PERCENTILE(latency_ms, 0.95) as p95_latency,
            AVG(CAST(fallback AS FLOAT)) as fallback_rate,
            AVG(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as error_rate
        FROM requests
        WHERE timestamp > datetime('now', '-1 minute')
    """, conn)
    
    # Insert vào service_health
    conn.execute("""
        INSERT INTO service_health
        (timestamp, requests_per_minute, avg_latency_ms, p95_latency_ms, fallback_rate, error_rate, active_model_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(),
        stats['requests'][0],
        stats['avg_latency'][0],
        stats['p95_latency'][0],
        stats['fallback_rate'][0],
        stats['error_rate'][0],
        recommender.model['model_id']
    ))
    conn.commit()
    conn.close()
```

### Metrics Dashboard

#### Streamlit Dashboard: `service/dashboard.py`
```python
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

st.title("CF Recommendation Service Dashboard")

# Load data
conn = sqlite3.connect('logs/service_metrics.db')

# Time series: Requests per minute
health = pd.read_sql("""
    SELECT timestamp, requests_per_minute, avg_latency_ms, fallback_rate
    FROM service_health
    WHERE timestamp > datetime('now', '-1 hour')
    ORDER BY timestamp
""", conn)

# Plot
fig1 = px.line(health, x='timestamp', y='requests_per_minute', title='Requests per Minute')
st.plotly_chart(fig1)

fig2 = px.line(health, x='timestamp', y='avg_latency_ms', title='Average Latency (ms)')
st.plotly_chart(fig2)

fig3 = px.line(health, x='timestamp', y='fallback_rate', title='Fallback Rate')
st.plotly_chart(fig3)

# Summary stats
st.subheader("Last Hour Summary")
st.metric("Total Requests", health['requests_per_minute'].sum())
st.metric("Avg Latency", f"{health['avg_latency_ms'].mean():.1f} ms")
st.metric("Fallback Rate", f"{health['fallback_rate'].mean():.1%}")
```

## Component 4: Data Drift Detection

### Drift Monitoring

#### Function: `detect_rating_distribution_drift()`
```python
import numpy as np
from scipy.stats import ks_2samp

def detect_rating_distribution_drift(historical_data, new_data, threshold=0.05):
    """
    Detect drift trong rating distribution sử dụng Kolmogorov-Smirnov test.
    
    Args:
        historical_data: DataFrame với ratings từ train data
        new_data: DataFrame với ratings mới (last week)
        threshold: p-value threshold (default 0.05)
    
    Returns:
        dict: {
            'drift_detected': bool,
            'p_value': float,
            'statistic': float,
            'recommendation': str
        }
    """
    # Extract ratings
    hist_ratings = historical_data['rating'].values
    new_ratings = new_data['rating'].values
    
    # KS test
    statistic, p_value = ks_2samp(hist_ratings, new_ratings)
    
    drift_detected = p_value < threshold
    
    return {
        'drift_detected': drift_detected,
        'p_value': p_value,
        'statistic': statistic,
        'recommendation': 'Retrain model' if drift_detected else 'No action needed'
    }
```

#### Function: `detect_popularity_shift()`
```python
from scipy.stats import spearmanr

def detect_popularity_shift(old_popularity, new_popularity, threshold=0.8):
    """
    Detect shift trong item popularity ranking.
    
    Args:
        old_popularity: Array of item popularities (train data)
        new_popularity: Array of item popularities (recent data)
        threshold: Spearman correlation threshold (default 0.8)
    
    Returns:
        dict: {
            'shift_detected': bool,
            'correlation': float,
            'recommendation': str
        }
    """
    # Spearman rank correlation
    correlation, p_value = spearmanr(old_popularity, new_popularity)
    
    shift_detected = correlation < threshold
    
    return {
        'shift_detected': shift_detected,
        'correlation': correlation,
        'p_value': p_value,
        'recommendation': 'Retrain with updated popularity' if shift_detected else 'No action'
    }
```

#### Scheduled Drift Check
```python
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

@scheduler.scheduled_job('cron', day_of_week='mon', hour=9)  # Every Monday 9am
def weekly_drift_check():
    """Run drift detection và alert nếu cần."""
    logger.info("Running weekly drift detection")
    
    # Load data
    historical = pd.read_parquet('data/processed/interactions.parquet')
    recent = load_recent_interactions(days=7)  # Last week
    
    # Rating drift
    rating_drift = detect_rating_distribution_drift(historical, recent)
    logger.info(f"Rating drift: {rating_drift}")
    
    if rating_drift['drift_detected']:
        send_alert(
            subject="Data Drift Detected",
            message=f"Rating distribution has shifted (p={rating_drift['p_value']:.4f}). Consider retraining."
        )
    
    # Popularity drift
    old_pop = compute_item_popularity(historical)
    new_pop = compute_item_popularity(recent)
    pop_shift = detect_popularity_shift(old_pop, new_pop)
    logger.info(f"Popularity shift: {pop_shift}")
    
    if pop_shift['shift_detected']:
        send_alert(
            subject="Popularity Shift Detected",
            message=f"Item popularity ranking changed (corr={pop_shift['correlation']:.3f}). Update baseline."
        )

scheduler.start()
```

## Component 5: Alerting System

### Alert Configuration

#### File: `config/alerts_config.yaml`
```yaml
alerts:
  - name: "high_latency"
    metric: "avg_latency_ms"
    threshold: 200
    window: "5min"
    severity: "warning"
    action: "email"
  
  - name: "high_error_rate"
    metric: "error_rate"
    threshold: 0.05  # 5%
    window: "5min"
    severity: "critical"
    action: "email+slack"
  
  - name: "high_fallback_rate"
    metric: "fallback_rate"
    threshold: 0.3  # 30%
    window: "1hour"
    severity: "warning"
    action: "email"
  
  - name: "data_drift"
    metric: "drift_detected"
    threshold: true
    window: "weekly"
    severity: "info"
    action: "email"
  
  - name: "registry_health"
    metric: "registry_status"
    threshold: "unhealthy"
    window: "5min"
    severity: "critical"
    action: "email+slack"
  
  - name: "model_reload_failed"
    metric: "reload_error"
    threshold: true
    window: "1min"
    severity: "warning"
    action: "email"
  
  - name: "no_best_model"
    metric: "current_best"
    threshold: null
    window: "5min"
    severity: "critical"
    action: "email+slack"

email:
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  sender: "alerts@example.com"
  recipients:
    - "team@example.com"

slack:
  webhook_url: "https://hooks.slack.com/services/..."
```

### Alert Implementation

#### Function: `send_alert(subject, message, severity='info')`
```python
import smtplib
from email.mime.text import MIMEText
import requests

def send_alert(subject, message, severity='info'):
    """
    Send alert via email và/hoặc Slack.
    
    Args:
        subject: Alert subject
        message: Alert message
        severity: 'info', 'warning', 'critical'
    """
    config = load_config('config/alerts_config.yaml')
    
    # Email
    if 'email' in config:
        send_email_alert(subject, message, config['email'])
    
    # Slack
    if 'slack' in config and severity in ['warning', 'critical']:
        send_slack_alert(subject, message, config['slack'])

def send_email_alert(subject, message, email_config):
    """Send email alert."""
    msg = MIMEText(message)
    msg['Subject'] = f"[{severity.upper()}] {subject}"
    msg['From'] = email_config['sender']
    msg['To'] = ', '.join(email_config['recipients'])
    
    with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
        server.starttls()
        server.login(email_config['sender'], os.getenv('EMAIL_PASSWORD'))
        server.send_message(msg)

def send_slack_alert(subject, message, slack_config):
    """Send Slack alert."""
    payload = {
        'text': f"*{subject}*\n{message}"
    }
    requests.post(slack_config['webhook_url'], json=payload)
```

### Alert Monitoring Task

#### Background Task: Check Thresholds
```python
@repeat_every(seconds=300)  # Every 5 minutes
def check_alert_conditions():
    """Check alert conditions và trigger nếu vượt threshold."""
    conn = sqlite3.connect('logs/service_metrics.db')
    
    # Load alert config
    alerts = load_config('config/alerts_config.yaml')['alerts']
    
    for alert in alerts:
        # Query recent metrics
        if alert['window'] == '5min':
            query = f"""
                SELECT AVG({alert['metric']}) as value
                FROM service_health
                WHERE timestamp > datetime('now', '-5 minutes')
            """
        elif alert['window'] == '1hour':
            query = f"""
                SELECT AVG({alert['metric']}) as value
                FROM service_health
                WHERE timestamp > datetime('now', '-1 hour')
            """
        
        result = pd.read_sql(query, conn)
        value = result['value'][0]
        
        # Check threshold
        if value > alert['threshold']:
            send_alert(
                subject=alert['name'],
                message=f"{alert['metric']} = {value:.3f} exceeds threshold {alert['threshold']}",
                severity=alert['severity']
            )
            
            logger.warning(f"Alert triggered: {alert['name']} | value={value:.3f}, threshold={alert['threshold']}")
```

## Component 6: Retrain Trigger

### Automatic Retrain Logic

#### Function: `should_retrain()`
```python
def should_retrain():
    """
    Determine nếu cần retrain model dựa trên drift, performance degradation.
    
    Returns:
        dict: {
            'should_retrain': bool,
            'reasons': list[str]
        }
    """
    reasons = []
    
    # Check data drift
    drift_result = detect_rating_distribution_drift(...)
    if drift_result['drift_detected']:
        reasons.append(f"Rating distribution drift (p={drift_result['p_value']:.4f})")
    
    # Check online performance (if có A/B test metrics)
    recent_ctr = get_recent_ctr(days=7)
    baseline_ctr = get_baseline_ctr()
    if recent_ctr < baseline_ctr * 0.9:  # >10% drop
        reasons.append(f"CTR degraded {(1 - recent_ctr/baseline_ctr):.1%}")
    
    # Check data freshness
    data_age_days = get_data_age()
    if data_age_days > 30:
        reasons.append(f"Training data is {data_age_days} days old")
    
    return {
        'should_retrain': len(reasons) > 0,
        'reasons': reasons
    }
```

#### Scheduled Retrain Check
```python
@scheduler.scheduled_job('cron', day_of_week='sun', hour=2)  # Every Sunday 2am
def check_retrain_trigger():
    """Check nếu cần retrain và trigger job."""
    decision = should_retrain()
    
    if decision['should_retrain']:
        logger.info(f"Retraining triggered: {decision['reasons']}")
        
        send_alert(
            subject="Model Retrain Triggered",
            message=f"Reasons:\n" + "\n".join(decision['reasons'])
        )
        
        # Trigger retrain script
        subprocess.run([
            'python', 'scripts/train_cf.py',
            '--model', 'als',
            '--auto-select'
        ])
        
        subprocess.run([
            'python', 'scripts/train_cf.py',
            '--model', 'bpr',
            '--auto-select'
        ])
    else:
        logger.info("No retrain needed")
```

## Component 7: BERT Embeddings Monitoring

### Embedding Freshness Tracking

#### Metric: Embedding Age
```python
def check_embedding_freshness(embeddings_path):
    """
    Check embedding age và alert nếu stale.
    """
    metadata = torch.load(embeddings_path)
    created_at = datetime.fromisoformat(metadata['created_at'])
    age_days = (datetime.now() - created_at).days
    
    if age_days > 30:
        send_alert(
            subject="BERT Embeddings Stale",
            message=f"Embeddings are {age_days} days old. Consider regenerating.",
            severity="warning"
        )
    
    return age_days
```

### Content Similarity Drift Detection

#### Function: `detect_semantic_drift()`
```python
def detect_semantic_drift(old_embeddings, new_embeddings, product_ids, threshold=0.95):
    """
    Detect drift trong BERT embeddings (e.g., sau khi re-train PhoBERT).
    
    Args:
        old_embeddings: np.array (N, 768)
        new_embeddings: np.array (N, 768)
        product_ids: List of product IDs
        threshold: Correlation threshold
    
    Returns:
        dict: Drift detection results
    """
    # Normalize
    old_norm = old_embeddings / np.linalg.norm(old_embeddings, axis=1, keepdims=True)
    new_norm = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    
    # Cosine similarities (diagonal = self-similarity)
    similarities = np.diag(old_norm @ new_norm.T)
    
    # Statistics
    mean_sim = similarities.mean()
    std_sim = similarities.std()
    min_sim = similarities.min()
    
    # Identify drifted items
    drifted_mask = similarities < threshold
    drifted_products = [product_ids[i] for i in np.where(drifted_mask)[0]]
    
    drift_detected = mean_sim < threshold
    
    return {
        'drift_detected': drift_detected,
        'mean_similarity': mean_sim,
        'std_similarity': std_sim,
        'min_similarity': min_sim,
        'num_drifted': len(drifted_products),
        'drifted_products': drifted_products[:10],  # Top 10
        'recommendation': 'Regenerate embeddings' if drift_detected else 'No action'
    }
```

### Reranking Performance Tracking

#### Table: `reranking_metrics`
```sql
CREATE TABLE reranking_metrics (
    timestamp TIMESTAMP,
    requests_with_rerank INTEGER,
    requests_without_rerank INTEGER,
    
    -- Latency
    avg_latency_rerank_ms REAL,
    avg_latency_cf_only_ms REAL,
    latency_overhead_pct REAL,
    
    -- Content scores
    avg_content_score REAL,
    avg_semantic_alignment REAL,
    
    -- Diversity
    avg_diversity_rerank REAL,
    avg_diversity_cf_only REAL
);
```

#### Aggregation Function
```python
@scheduler.scheduled_job('cron', hour='*')  # Hourly
def aggregate_reranking_metrics():
    """
    Aggregate reranking performance metrics.
    """
    conn = sqlite3.connect('logs/service_metrics.db')
    
    # Last hour stats
    rerank_requests = pd.read_sql("""
        SELECT 
            COUNT(*) as num_requests,
            AVG(latency_ms) as avg_latency,
            AVG(content_score) as avg_content_score,
            AVG(diversity) as avg_diversity
        FROM requests
        WHERE timestamp > datetime('now', '-1 hour')
          AND rerank_enabled = 1
    """, conn)
    
    cf_only_requests = pd.read_sql("""
        SELECT 
            COUNT(*) as num_requests,
            AVG(latency_ms) as avg_latency,
            AVG(diversity) as avg_diversity
        FROM requests
        WHERE timestamp > datetime('now', '-1 hour')
          AND rerank_enabled = 0
    """, conn)
    
    # Insert aggregated metrics
    conn.execute("""
        INSERT INTO reranking_metrics
        (timestamp, requests_with_rerank, requests_without_rerank,
         avg_latency_rerank_ms, avg_latency_cf_only_ms, latency_overhead_pct,
         avg_content_score, avg_diversity_rerank, avg_diversity_cf_only)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(),
        rerank_requests['num_requests'][0],
        cf_only_requests['num_requests'][0],
        rerank_requests['avg_latency'][0],
        cf_only_requests['avg_latency'][0],
        (rerank_requests['avg_latency'][0] / cf_only_requests['avg_latency'][0] - 1) * 100 if cf_only_requests['avg_latency'][0] > 0 else 0,
        rerank_requests['avg_content_score'][0],
        rerank_requests['avg_diversity'][0],
        cf_only_requests['avg_diversity'][0]
    ))
    conn.commit()
    conn.close()
```

## Dependencies

```python
# requirements_monitoring.txt
streamlit>=1.20.0  # Dashboard
plotly>=5.13.0  # Plots
apscheduler>=3.10.0  # Scheduling
requests>=2.28.0  # Slack webhooks

# BERT monitoring
torch>=1.13.0
```

## Timeline Estimate

- **Training logging**: 1 day
- **Service logging**: 1 day
- **Drift detection**: 1 day
- **BERT monitoring**: 1 day
- **Alerting system**: 1 day
- **Dashboard**: 1 day
- **Integration & testing**: 1 day
- **Total**: ~7 days

## Component 8: Registry Integration Summary

### Key Integrations

1. **Training → Registry**: Auto-register models sau training completion
2. **Registry → Service**: ModelLoader stats tracked trong service metrics
3. **Registry Health**: Continuous monitoring của registry state
4. **Model Changes**: Alert khi best model changes hoặc reload fails
5. **Audit Trail**: All registry operations logged to database

### Registry Monitoring Dashboard

#### Streamlit Dashboard: `service/registry_dashboard.py`
```python
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from recsys.cf.registry import ModelRegistry

st.title("Model Registry Dashboard")

# Load registry
registry = ModelRegistry()
health = check_registry_health()

# Health status
st.subheader("Registry Health")
if health['status'] == 'healthy':
    st.success("✓ Registry is healthy")
else:
    st.error(f"✗ Registry issues: {health['issues']}")

# Current best model
current_best = registry.get_current_best()
if current_best:
    st.metric("Current Best Model", current_best['model_id'])
    st.metric("NDCG@10", f"{current_best['model_info']['metrics']['ndcg@10']:.4f}")
else:
    st.warning("No current best model selected")

# Model versions timeline
conn = sqlite3.connect('logs/registry_operations.db')
versions = pd.read_sql("""
    SELECT timestamp, model_id, model_type, metric_value, improvement_pct
    FROM registry_operations
    WHERE action = 'SELECT_BEST'
    ORDER BY timestamp DESC
    LIMIT 20
""", conn)

if not versions.empty:
    fig = px.line(versions, x='timestamp', y='metric_value', 
                  color='model_type', title='Best Model NDCG@10 Over Time')
    st.plotly_chart(fig)

# Registry stats
stats = registry.get_registry_stats()
col1, col2, col3 = st.columns(3)
col1.metric("Total Models", stats['total_models'])
col2.metric("Active Models", stats['active_models'])
col3.metric("Archived Models", stats['archived_models'])

# Recent operations
operations = pd.read_sql("""
    SELECT timestamp, action, model_id, details
    FROM registry_operations
    ORDER BY timestamp DESC
    LIMIT 50
""", conn)
st.subheader("Recent Registry Operations")
st.dataframe(operations)
```

## Success Criteria

- [ ] Training runs logged với metrics và auto-registered
- [ ] Registry operations tracked trong database
- [ ] Model loader stats monitored
- [ ] Registry health checks run continuously
- [ ] Service requests tracked với model_id và version
- [ ] Drift detection runs weekly (CF + BERT)
- [ ] BERT embedding freshness monitored
- [ ] Alerts sent cho critical issues (including registry)
- [ ] Dashboard visualizes metrics (training + registry + service)
- [ ] Retrain triggered automatically khi needed
- [ ] Model hot-reload monitored và logged
- [ ] Logs retained với rotation policy
