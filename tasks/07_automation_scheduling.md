# Task 07: Automation & Scheduling

## Mục Tiêu

Xây dựng hệ thống tự động hóa cho toàn bộ ML pipeline: data refresh, model training, evaluation, deployment, và monitoring. Hệ thống phải chạy ổn định, có error handling, và dễ dàng maintain.

## Automation Architecture

```
Scheduler (Cron/Airflow)
    ↓
├─ Daily: Data Refresh
│   - Ingest new interactions
│   - Update preprocessed data
│   - Check data quality
│
├─ Weekly: Drift Detection
│   - Rating distribution drift
│   - Popularity shift
│   - Alert if needed
│
├─ Triggered: Model Retraining
│   - ALS pipeline
│   - BPR pipeline
│   - Evaluate both
│   - Select best
│   - Update registry
│
├─ Daily: Model Deployment
│   - Check registry updates
│   - Hot-reload service
│   - Validate serving
│
└─ Hourly: Health Checks
    - Service status
    - Metrics aggregation
    - Alert on issues
```

## Component 1: Orchestration Scripts

### Script 1: Data Refresh

#### File: `scripts/refresh_data.py`
```python
"""
Daily data refresh pipeline.

Usage:
    python scripts/refresh_data.py --config config/data_config.yaml
"""

import argparse
import logging
from datetime import datetime
import sys

from recsys.cf.data import (
    load_raw_data,
    preprocess_interactions,
    create_mappings,
    temporal_split,
    build_csr_matrix,
    save_processed_data
)
from recsys.cf.logging_utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Refresh processed data")
    parser.add_argument('--config', default='config/data_config.yaml')
    parser.add_argument('--incremental', action='store_true', help='Incremental update (not full refresh)')
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('data_refresh', 'logs/data_refresh.log')
    logger.info("=" * 50)
    logger.info(f"Data refresh started | config={args.config}")
    
    try:
        # Load config
        config = load_config(args.config)
        
        # Load raw data
        logger.info("Loading raw CSV files...")
        raw_data = load_raw_data(config)
        logger.info(f"Loaded {len(raw_data['interactions'])} interactions")
        
        # Preprocess
        logger.info("Preprocessing interactions...")
        cleaned = preprocess_interactions(raw_data['interactions'], config)
        logger.info(f"After preprocessing: {len(cleaned)} interactions")
        
        # Create mappings
        logger.info("Creating ID mappings...")
        mappings = create_mappings(cleaned, 'user_id', 'product_id')
        logger.info(f"Mapped {len(mappings['user_to_idx'])} users, {len(mappings['item_to_idx'])} items")
        
        # Temporal split
        logger.info("Splitting train/test...")
        train_df, test_df = temporal_split(cleaned, method='leave_one_out')
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Build matrix
        logger.info("Building CSR matrix...")
        X_train = build_csr_matrix(train_df, num_users=len(mappings['user_to_idx']), num_items=len(mappings['item_to_idx']))
        logger.info(f"Matrix shape: {X_train.shape}, sparsity: {1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]):.4f}")
        
        # Save
        logger.info("Saving processed data...")
        artifacts = {
            'interactions': cleaned,
            'train': train_df,
            'test': test_df,
            'X_train': X_train,
            'mappings': mappings
        }
        save_processed_data(artifacts, 'data/processed')
        logger.info(f"Data refresh completed successfully")
        
        return 0
    
    except Exception as e:
        logger.error(f"Data refresh failed: {str(e)}", exc_info=True)
        send_alert(
            subject="Data Refresh Failed",
            message=f"Error: {str(e)}",
            severity="critical"
        )
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

### Script 2: Train Both Models

#### File: `scripts/train_both_models.py`
```python
"""
Train both ALS and BPR models, evaluate, select best.

Usage:
    python scripts/train_both_models.py --config config/training_config.yaml
"""

import argparse
import logging
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Train ALS and BPR models")
    parser.add_argument('--config', default='config/training_config.yaml')
    parser.add_argument('--parallel', action='store_true', help='Train in parallel')
    parser.add_argument('--auto-select', action='store_true', help='Auto-select best model')
    args = parser.parse_args()
    
    logger = setup_logger('train_both', 'logs/train_both.log')
    logger.info("Training both models started")
    
    try:
        if args.parallel:
            # Parallel training (requires enough resources)
            logger.info("Starting parallel training...")
            
            import multiprocessing
            
            def train_als():
                subprocess.run([
                    'python', 'scripts/train_cf.py',
                    '--model', 'als',
                    '--config', args.config,
                    '--auto-select' if args.auto_select else ''
                ])
            
            def train_bpr():
                subprocess.run([
                    'python', 'scripts/train_cf.py',
                    '--model', 'bpr',
                    '--config', args.config,
                    '--auto-select' if args.auto_select else ''
                ])
            
            p1 = multiprocessing.Process(target=train_als)
            p2 = multiprocessing.Process(target=train_bpr)
            
            p1.start()
            p2.start()
            
            p1.join()
            p2.join()
        
        else:
            # Sequential training
            logger.info("Training ALS...")
            result_als = subprocess.run([
                'python', 'scripts/train_cf.py',
                '--model', 'als',
                '--config', args.config
            ], capture_output=True, text=True)
            
            if result_als.returncode != 0:
                raise Exception(f"ALS training failed: {result_als.stderr}")
            
            logger.info("Training BPR...")
            result_bpr = subprocess.run([
                'python', 'scripts/train_cf.py',
                '--model', 'bpr',
                '--config', args.config
            ], capture_output=True, text=True)
            
            if result_bpr.returncode != 0:
                raise Exception(f"BPR training failed: {result_bpr.stderr}")
        
        # Select best model
        if args.auto_select:
            logger.info("Selecting best model...")
            subprocess.run([
                'python', 'scripts/update_registry.py',
                '--auto-select',
                '--metric', 'ndcg@10'
            ])
        
        logger.info("Training completed successfully")
        
        # Send success notification
        send_alert(
            subject="Model Training Completed",
            message="Both ALS and BPR models trained successfully. Check registry for best model.",
            severity="info"
        )
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        send_alert(
            subject="Model Training Failed",
            message=f"Error: {str(e)}",
            severity="critical"
        )
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

### Script 3: Deploy Model Update

#### File: `scripts/deploy_model_update.py`
```python
"""
Check registry for updates và deploy to service.

Usage:
    python scripts/deploy_model_update.py --service-url http://localhost:8000
"""

import argparse
import requests
import logging
import sys

def main():
    parser = argparse.ArgumentParser(description="Deploy model update")
    parser.add_argument('--service-url', default='http://localhost:8000')
    parser.add_argument('--dry-run', action='store_true', help='Check only, no reload')
    args = parser.parse_args()
    
    logger = setup_logger('deploy', 'logs/deploy.log')
    logger.info("Checking for model updates...")
    
    try:
        # Load registry
        registry = load_registry_json('artifacts/cf/registry.json')
        current_best = registry['current_best']['model_id']
        logger.info(f"Current best model in registry: {current_best}")
        
        # Check service status
        health = requests.get(f"{args.service_url}/health").json()
        service_model = health['model_id']
        logger.info(f"Service currently serving: {service_model}")
        
        if current_best == service_model:
            logger.info("Service already serving latest model")
            return 0
        
        # Model update needed
        logger.info(f"Update needed: {service_model} → {current_best}")
        
        if args.dry_run:
            logger.info("Dry-run mode, skipping reload")
            return 0
        
        # Trigger reload
        logger.info("Triggering model reload...")
        response = requests.post(f"{args.service_url}/reload_model")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Reload successful: {result}")
            
            # Verify
            health_after = requests.get(f"{args.service_url}/health").json()
            if health_after['model_id'] == current_best:
                logger.info("Deployment verified successfully")
                
                send_alert(
                    subject="Model Deployed",
                    message=f"Successfully deployed {current_best} to production",
                    severity="info"
                )
                return 0
            else:
                raise Exception(f"Verification failed: expected {current_best}, got {health_after['model_id']}")
        else:
            raise Exception(f"Reload failed: {response.status_code} {response.text}")
    
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}", exc_info=True)
        send_alert(
            subject="Model Deployment Failed",
            message=f"Error: {str(e)}",
            severity="critical"
        )
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

### Script 4: Health Check

#### File: `scripts/health_check.py`
```python
"""
Periodic health check cho service.

Usage:
    python scripts/health_check.py --service-url http://localhost:8000
"""

import argparse
import requests
import logging
import sys
import pandas as pd
import sqlite3

def main():
    parser = argparse.ArgumentParser(description="Service health check")
    parser.add_argument('--service-url', default='http://localhost:8000')
    parser.add_argument('--alert-threshold', type=float, default=0.1, help='Error rate threshold')
    args = parser.parse_args()
    
    logger = setup_logger('health_check', 'logs/health_check.log')
    
    try:
        # Check service alive
        health = requests.get(f"{args.service_url}/health", timeout=5).json()
        logger.info(f"Service healthy: {health}")
        
        # Check recent metrics
        conn = sqlite3.connect('logs/service_metrics.db')
        recent_metrics = pd.read_sql("""
            SELECT 
                AVG(error_rate) as avg_error_rate,
                AVG(avg_latency_ms) as avg_latency,
                AVG(fallback_rate) as avg_fallback_rate
            FROM service_health
            WHERE timestamp > datetime('now', '-1 hour')
        """, conn)
        conn.close()
        
        logger.info(f"Recent metrics: {recent_metrics.to_dict('records')[0]}")
        
        # Check thresholds
        error_rate = recent_metrics['avg_error_rate'][0]
        if error_rate > args.alert_threshold:
            send_alert(
                subject="High Error Rate Detected",
                message=f"Error rate: {error_rate:.2%} (threshold: {args.alert_threshold:.2%})",
                severity="critical"
            )
            return 1
        
        logger.info("Health check passed")
        return 0
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        send_alert(
            subject="Service Health Check Failed",
            message=f"Service may be down: {str(e)}",
            severity="critical"
        )
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

## Component 2: Scheduling

### Option 1: Cron Jobs (Linux/Mac)

#### File: `cron_schedule.sh`
```bash
#!/bin/bash

# Edit crontab: crontab -e
# Add these lines:

# Data refresh - Daily at 2am
0 2 * * * cd /path/to/project && python scripts/refresh_data.py >> logs/cron_data_refresh.log 2>&1

# Drift detection - Weekly on Monday 9am
0 9 * * 1 cd /path/to/project && python scripts/detect_drift.py >> logs/cron_drift.log 2>&1

# Model training - Weekly on Sunday 3am (after data refresh)
0 3 * * 0 cd /path/to/project && python scripts/train_both_models.py --auto-select >> logs/cron_training.log 2>&1

# Deploy updates - Daily at 5am (after training)
0 5 * * * cd /path/to/project && python scripts/deploy_model_update.py >> logs/cron_deploy.log 2>&1

# Health check - Every hour
0 * * * * cd /path/to/project && python scripts/health_check.py >> logs/cron_health.log 2>&1

# Cleanup old logs - Monthly on 1st at midnight
0 0 1 * * cd /path/to/project && find logs/ -name "*.log" -mtime +30 -delete
```

### Option 2: Task Scheduler (Windows)

#### PowerShell Script: `setup_windows_tasks.ps1`
```powershell
# Data Refresh - Daily at 2am
$action = New-ScheduledTaskAction -Execute "python" -Argument "scripts/refresh_data.py" -WorkingDirectory "D:\app\IAI\project"
$trigger = New-ScheduledTaskTrigger -Daily -At 2am
Register-ScheduledTask -TaskName "CF_DataRefresh" -Action $action -Trigger $trigger

# Model Training - Weekly Sunday 3am
$action = New-ScheduledTaskAction -Execute "python" -Argument "scripts/train_both_models.py --auto-select" -WorkingDirectory "D:\app\IAI\project"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3am
Register-ScheduledTask -TaskName "CF_Training" -Action $action -Trigger $trigger

# Deploy Updates - Daily at 5am
$action = New-ScheduledTaskAction -Execute "python" -Argument "scripts/deploy_model_update.py" -WorkingDirectory "D:\app\IAI\project"
$trigger = New-ScheduledTaskTrigger -Daily -At 5am
Register-ScheduledTask -TaskName "CF_Deploy" -Action $action -Trigger $trigger

# Health Check - Hourly
$action = New-ScheduledTaskAction -Execute "python" -Argument "scripts/health_check.py" -WorkingDirectory "D:\app\IAI\project"
$trigger = New-ScheduledTaskTrigger -Once -At 12am -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration ([TimeSpan]::MaxValue)
Register-ScheduledTask -TaskName "CF_HealthCheck" -Action $action -Trigger $trigger
```

### Option 3: Airflow DAG (Advanced)

#### File: `airflow/dags/cf_pipeline_dag.py`
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['team@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'cf_recommendation_pipeline',
    default_args=default_args,
    description='CF model training and deployment pipeline',
    schedule_interval='0 2 * * 0',  # Weekly Sunday 2am
    start_date=datetime(2025, 1, 1),
    catchup=False
)

# Task 1: Data Refresh
task_data_refresh = BashOperator(
    task_id='data_refresh',
    bash_command='cd /path/to/project && python scripts/refresh_data.py',
    dag=dag
)

# Task 2: Drift Detection
task_drift_detection = BashOperator(
    task_id='drift_detection',
    bash_command='cd /path/to/project && python scripts/detect_drift.py',
    dag=dag
)

# Task 3: Train ALS
task_train_als = BashOperator(
    task_id='train_als',
    bash_command='cd /path/to/project && python scripts/train_cf.py --model als',
    dag=dag
)

# Task 4: Train BPR
task_train_bpr = BashOperator(
    task_id='train_bpr',
    bash_command='cd /path/to/project && python scripts/train_cf.py --model bpr',
    dag=dag
)

# Task 5: Select Best Model
task_select_best = BashOperator(
    task_id='select_best_model',
    bash_command='cd /path/to/project && python scripts/update_registry.py --auto-select',
    dag=dag
)

# Task 6: Deploy to Production
task_deploy = BashOperator(
    task_id='deploy_model',
    bash_command='cd /path/to/project && python scripts/deploy_model_update.py',
    dag=dag
)

# Task 7: Validate Deployment
task_validate = BashOperator(
    task_id='validate_deployment',
    bash_command='cd /path/to/project && python scripts/health_check.py',
    dag=dag
)

# Define dependencies
task_data_refresh >> task_drift_detection >> [task_train_als, task_train_bpr]
[task_train_als, task_train_bpr] >> task_select_best >> task_deploy >> task_validate
```

## Component 3: Error Handling & Retry Logic

### Retry Decorator

#### Module: `scripts/utils.py`
```python
import time
import logging
from functools import wraps

def retry(max_attempts=3, delay=60, backoff=2, exceptions=(Exception,)):
    """
    Retry decorator với exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logging.error(f"{func.__name__} failed after {max_attempts} attempts: {str(e)}")
                        raise
                    
                    logging.warning(f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {str(e)}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
        
        return wrapper
    return decorator

# Example usage
@retry(max_attempts=3, delay=30)
def train_model_with_retry():
    subprocess.run(['python', 'scripts/train_cf.py', '--model', 'als'], check=True)
```

### Error Notification

#### Function: `handle_pipeline_error(stage, error)`
```python
def handle_pipeline_error(stage, error):
    """
    Handle pipeline errors với logging và alerting.
    
    Args:
        stage: Pipeline stage name (e.g., 'data_refresh', 'training')
        error: Exception object
    """
    # Log error
    logger = logging.getLogger(f'pipeline.{stage}')
    logger.error(f"Pipeline failed at stage {stage}: {str(error)}", exc_info=True)
    
    # Send alert
    send_alert(
        subject=f"Pipeline Failure: {stage}",
        message=f"Stage: {stage}\nError: {str(error)}\nTimestamp: {datetime.now()}",
        severity="critical"
    )
    
    # Save error report
    error_report = {
        'stage': stage,
        'error': str(error),
        'traceback': traceback.format_exc(),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f'logs/errors/pipeline_error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(error_report, f, indent=2)
```

## Component 4: Pipeline Monitoring

### Pipeline Run Tracker

#### File: `logs/pipeline_runs.db` (SQLite)
```sql
CREATE TABLE pipeline_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_type TEXT,  -- 'data_refresh', 'training', 'deployment'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT,  -- 'running', 'completed', 'failed'
    duration_seconds REAL,
    error_message TEXT,
    
    -- Outputs
    data_version TEXT,
    models_trained TEXT,  -- JSON list
    best_model_selected TEXT
);
```

### Dashboard: Pipeline Status

#### Streamlit: `scripts/pipeline_dashboard.py`
```python
import streamlit as st
import sqlite3
import pandas as pd

st.title("CF Pipeline Dashboard")

conn = sqlite3.connect('logs/pipeline_runs.db')

# Recent runs
recent_runs = pd.read_sql("""
    SELECT run_id, pipeline_type, started_at, status, duration_seconds
    FROM pipeline_runs
    ORDER BY started_at DESC
    LIMIT 20
""", conn)

st.subheader("Recent Pipeline Runs")
st.dataframe(recent_runs)

# Success rate
success_rate = pd.read_sql("""
    SELECT 
        pipeline_type,
        COUNT(*) as total,
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
        ROUND(100.0 * SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate
    FROM pipeline_runs
    WHERE started_at > datetime('now', '-30 days')
    GROUP BY pipeline_type
""", conn)

st.subheader("30-Day Success Rate")
st.bar_chart(success_rate.set_index('pipeline_type')['success_rate'])

# Average duration
avg_duration = pd.read_sql("""
    SELECT 
        pipeline_type,
        AVG(duration_seconds) / 60 as avg_duration_minutes
    FROM pipeline_runs
    WHERE status = 'completed'
    GROUP BY pipeline_type
""", conn)

st.subheader("Average Pipeline Duration")
st.bar_chart(avg_duration.set_index('pipeline_type')['avg_duration_minutes'])

conn.close()
```

## Component 5: Cleanup & Maintenance

### Log Rotation

#### Script: `scripts/cleanup_logs.py`
```python
"""
Cleanup old logs và artifacts.

Usage:
    python scripts/cleanup_logs.py --days 30
"""

import argparse
import os
from datetime import datetime, timedelta
import shutil

def main():
    parser = argparse.ArgumentParser(description="Cleanup old logs")
    parser.add_argument('--days', type=int, default=30, help='Keep logs from last N days')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    cutoff_date = datetime.now() - timedelta(days=args.days)
    
    # Cleanup log files
    log_dirs = ['logs/cf', 'logs/service', 'logs/errors']
    total_deleted = 0
    
    for log_dir in log_dirs:
        for filename in os.listdir(log_dir):
            filepath = os.path.join(log_dir, filename)
            
            if os.path.isfile(filepath):
                mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if mod_time < cutoff_date:
                    if args.dry_run:
                        print(f"Would delete: {filepath}")
                    else:
                        os.remove(filepath)
                        total_deleted += 1
    
    print(f"Deleted {total_deleted} old log files (older than {args.days} days)")
    
    # Archive old models
    # Keep only last 5 versions per model type
    # ... (implementation)

if __name__ == '__main__':
    main()
```

## Full Automation Checklist

### Daily Tasks
- [ ] Data refresh (2am)
- [ ] Deploy model updates (5am)
- [ ] Health checks (hourly)
- [ ] Metrics aggregation (continuous)

### Weekly Tasks
- [ ] Drift detection (Monday 9am)
- [ ] Model retraining (Sunday 3am)
- [ ] Performance review
- [ ] Log cleanup

### Monthly Tasks
- [ ] Archive old models
- [ ] Database maintenance
- [ ] Dependency updates
- [ ] Security audit

## Dependencies

```python
# requirements_automation.txt
apscheduler>=3.10.0
apache-airflow>=2.5.0  # Optional
streamlit>=1.20.0  # Dashboard
```

## Timeline Estimate

- **Scripts implementation**: 2 days
- **Scheduler setup**: 1 day
- **Error handling**: 1 day
- **Monitoring dashboard**: 1 day
- **Testing**: 1 day
- **Documentation**: 0.5 day
- **Total**: ~6.5 days

## Success Criteria

- [ ] All scripts run end-to-end
- [ ] Scheduler executes tasks correctly
- [ ] Errors handled với retry logic
- [ ] Alerts sent on failures
- [ ] Pipeline dashboard functional
- [ ] Logs rotated automatically
- [ ] Documentation complete
