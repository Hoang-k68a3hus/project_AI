"""
Health Check Script.
Performs comprehensive health checks on all system components.

Usage:
    python scripts/health_check.py [--component all|service|models|data]
    
Options:
    --component     Which component to check (default: all)
    --json          Output results as JSON
    --alert         Send alerts for failures
"""

import os
import sys
import json
import argparse
import logging
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    PipelineTracker, setup_logging, send_pipeline_alert
)


# =============================================================================
# Configuration
# =============================================================================

HEALTH_CONFIG = {
    'service_url': 'http://localhost:8000',
    'processed_dir': PROJECT_ROOT / 'data' / 'processed',
    'artifacts_dir': PROJECT_ROOT / 'artifacts' / 'cf',
    'registry_path': PROJECT_ROOT / 'artifacts' / 'cf' / 'registry.json',
    'embeddings_path': PROJECT_ROOT / 'data' / 'processed' / 'content_based_embeddings' / 'product_embeddings.pt',
    
    # Thresholds
    'max_data_age_days': 7,
    'max_model_age_days': 30,
    'min_recall_threshold': 0.10,
    'service_timeout': 10
}


# =============================================================================
# Health Check Functions
# =============================================================================

def check_service_health(logger: logging.Logger) -> Dict[str, Any]:
    """
    Check recommendation service health.
    
    Returns:
        Health check result
    """
    result = {
        'component': 'service',
        'status': 'unknown',
        'checks': []
    }
    
    url = f"{HEALTH_CONFIG['service_url']}/health"
    
    try:
        response = requests.get(url, timeout=HEALTH_CONFIG['service_timeout'])
        response.raise_for_status()
        
        health_data = response.json()
        result['status'] = 'healthy'
        result['checks'].append({
            'name': 'api_reachable',
            'passed': True,
            'message': 'Service is reachable'
        })
        result['service_info'] = health_data
        
        # Check model loaded
        if health_data.get('model_loaded'):
            result['checks'].append({
                'name': 'model_loaded',
                'passed': True,
                'message': f"Model {health_data.get('model_id')} is loaded"
            })
        else:
            result['status'] = 'degraded'
            result['checks'].append({
                'name': 'model_loaded',
                'passed': False,
                'message': 'No model loaded'
            })
        
    except requests.exceptions.ConnectionError:
        result['status'] = 'offline'
        result['checks'].append({
            'name': 'api_reachable',
            'passed': False,
            'message': 'Service is not running'
        })
    except requests.exceptions.Timeout:
        result['status'] = 'degraded'
        result['checks'].append({
            'name': 'api_reachable',
            'passed': False,
            'message': 'Service timeout'
        })
    except Exception as e:
        result['status'] = 'error'
        result['checks'].append({
            'name': 'api_reachable',
            'passed': False,
            'message': f'Error: {str(e)}'
        })
    
    return result


def check_data_health(logger: logging.Logger) -> Dict[str, Any]:
    """
    Check processed data health.
    
    Returns:
        Health check result
    """
    result = {
        'component': 'data',
        'status': 'healthy',
        'checks': []
    }
    
    processed_dir = HEALTH_CONFIG['processed_dir']
    
    # Check required files exist
    required_files = [
        'interactions.parquet',
        'X_train_confidence.npz',
        'X_train_binary.npz',
        'user_item_mappings.json',
        'user_metadata.pkl',
        'user_pos_train.pkl',
        'data_stats.json'
    ]
    
    missing_files = []
    file_ages = {}
    
    for filename in required_files:
        file_path = processed_dir / filename
        if file_path.exists():
            # Check age
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            file_ages[filename] = age_days
        else:
            missing_files.append(filename)
    
    if missing_files:
        result['status'] = 'critical'
        result['checks'].append({
            'name': 'files_exist',
            'passed': False,
            'message': f"Missing files: {missing_files}"
        })
    else:
        result['checks'].append({
            'name': 'files_exist',
            'passed': True,
            'message': 'All required files present'
        })
    
    # Check data age
    if file_ages:
        max_age = max(file_ages.values())
        is_stale = max_age > HEALTH_CONFIG['max_data_age_days']
        
        if is_stale:
            result['status'] = 'warning'
        
        result['checks'].append({
            'name': 'data_freshness',
            'passed': not is_stale,
            'message': f"Data age: {max_age} days (threshold: {HEALTH_CONFIG['max_data_age_days']})",
            'details': file_ages
        })
    
    # Validate data stats
    stats_file = processed_dir / 'data_stats.json'
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # Check for reasonable values
            num_users = stats.get('num_users', 0)
            num_items = stats.get('num_items', 0)
            num_interactions = stats.get('num_interactions', 0)
            
            is_valid = num_users > 0 and num_items > 0 and num_interactions > 0
            
            result['checks'].append({
                'name': 'data_integrity',
                'passed': is_valid,
                'message': f"Users: {num_users}, Items: {num_items}, Interactions: {num_interactions}"
            })
            
            result['data_stats'] = stats
            
        except Exception as e:
            result['checks'].append({
                'name': 'data_integrity',
                'passed': False,
                'message': f'Failed to read stats: {e}'
            })
    
    # Check embeddings
    embeddings_path = HEALTH_CONFIG['embeddings_path']
    if embeddings_path.exists():
        result['checks'].append({
            'name': 'embeddings_exist',
            'passed': True,
            'message': 'PhoBERT embeddings available'
        })
    else:
        result['status'] = 'warning' if result['status'] != 'critical' else 'critical'
        result['checks'].append({
            'name': 'embeddings_exist',
            'passed': False,
            'message': 'PhoBERT embeddings not found'
        })
    
    return result


def check_model_health(logger: logging.Logger) -> Dict[str, Any]:
    """
    Check model artifacts health.
    
    Returns:
        Health check result
    """
    result = {
        'component': 'models',
        'status': 'healthy',
        'checks': []
    }
    
    registry_path = HEALTH_CONFIG['registry_path']
    
    # Check registry exists
    if not registry_path.exists():
        result['status'] = 'critical'
        result['checks'].append({
            'name': 'registry_exists',
            'passed': False,
            'message': 'Model registry not found'
        })
        return result
    
    result['checks'].append({
        'name': 'registry_exists',
        'passed': True,
        'message': 'Registry file exists'
    })
    
    # Load registry
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except Exception as e:
        result['status'] = 'critical'
        result['checks'].append({
            'name': 'registry_valid',
            'passed': False,
            'message': f'Failed to load registry: {e}'
        })
        return result
    
    result['checks'].append({
        'name': 'registry_valid',
        'passed': True,
        'message': 'Registry is valid JSON'
    })
    
    # Check current_best
    current_best_data = registry.get('current_best')
    if not current_best_data:
        result['status'] = 'critical'
        result['checks'].append({
            'name': 'current_best',
            'passed': False,
            'message': 'No current_best model defined'
        })
        return result
    
    # Handle both dict (new format) and string (old format)
    if isinstance(current_best_data, dict):
        current_best = current_best_data.get('model_id')
    else:
        current_best = current_best_data
    
    # Find best model info (models can be dict or list)
    models_data = registry.get('models', {})
    if isinstance(models_data, dict):
        best_model = models_data.get(current_best)
        if best_model:
            best_model['model_id'] = current_best  # Ensure model_id is set
    else:
        # Old list format
        best_model = None
        for model in models_data:
            if model.get('model_id') == current_best:
                best_model = model
                break
    
    if not best_model:
        result['status'] = 'critical'
        result['checks'].append({
            'name': 'current_best',
            'passed': False,
            'message': f'current_best {current_best} not found in models'
        })
        return result
    
    result['checks'].append({
        'name': 'current_best',
        'passed': True,
        'message': f'Current best: {current_best}'
    })
    
    result['current_model'] = best_model
    
    # Check model files exist
    model_path = Path(best_model.get('path', ''))
    if model_path.exists():
        result['checks'].append({
            'name': 'model_files',
            'passed': True,
            'message': 'Model files exist'
        })
    else:
        result['status'] = 'critical'
        result['checks'].append({
            'name': 'model_files',
            'passed': False,
            'message': f'Model path not found: {model_path}'
        })
    
    # Check model age
    registered_at = best_model.get('registered_at')
    if registered_at:
        try:
            reg_date = datetime.fromisoformat(registered_at)
            age_days = (datetime.now() - reg_date).days
            is_stale = age_days > HEALTH_CONFIG['max_model_age_days']
            
            if is_stale:
                result['status'] = 'warning' if result['status'] != 'critical' else 'critical'
            
            result['checks'].append({
                'name': 'model_freshness',
                'passed': not is_stale,
                'message': f'Model age: {age_days} days (threshold: {HEALTH_CONFIG["max_model_age_days"]})'
            })
        except Exception:
            pass
    
    # Check model performance
    metrics = best_model.get('metrics', {})
    recall_10 = metrics.get('recall@10', 0)
    
    is_performing = recall_10 >= HEALTH_CONFIG['min_recall_threshold']
    
    if not is_performing:
        result['status'] = 'warning' if result['status'] != 'critical' else 'critical'
    
    result['checks'].append({
        'name': 'model_performance',
        'passed': is_performing,
        'message': f'Recall@10: {recall_10:.4f} (threshold: {HEALTH_CONFIG["min_recall_threshold"]})'
    })
    
    return result


def check_pipeline_health(logger: logging.Logger) -> Dict[str, Any]:
    """
    Check pipeline execution health.
    
    Returns:
        Health check result
    """
    result = {
        'component': 'pipelines',
        'status': 'healthy',
        'checks': []
    }
    
    try:
        tracker = PipelineTracker()
        stats = tracker.get_stats(days=7)
        
        result['stats'] = stats
        
        # Check for failed pipelines
        for pipeline_name, pipeline_stats in stats.get('stats_by_pipeline', {}).items():
            failed_count = pipeline_stats.get('failed', 0)
            success_rate = pipeline_stats.get('success_rate')
            
            if failed_count > 0:
                if success_rate is not None and success_rate < 0.5:
                    result['status'] = 'warning'
                    result['checks'].append({
                        'name': f'{pipeline_name}_failures',
                        'passed': False,
                        'message': f'{pipeline_name}: {failed_count} failures, {success_rate:.0%} success rate'
                    })
                else:
                    result['checks'].append({
                        'name': f'{pipeline_name}_status',
                        'passed': True,
                        'message': f'{pipeline_name}: {success_rate:.0%} success rate'
                    })
        
        # Check for stale runs
        stale_count = tracker.cleanup_stale_runs(max_running_hours=24)
        if stale_count > 0:
            result['status'] = 'warning'
            result['checks'].append({
                'name': 'stale_runs',
                'passed': False,
                'message': f'Cleaned up {stale_count} stale pipeline runs'
            })
        
    except Exception as e:
        result['checks'].append({
            'name': 'pipeline_tracker',
            'passed': False,
            'message': f'Failed to check pipelines: {e}'
        })
    
    return result


# =============================================================================
# Main Health Check
# =============================================================================

def run_health_check(
    components: List[str] = None,
    send_alerts: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run comprehensive health check.
    
    Args:
        components: List of components to check (None = all)
        send_alerts: Send alerts for failures
        logger: Logger instance
        
    Returns:
        Health check results
    """
    if logger is None:
        logger = setup_logging("health_check")
    
    if components is None:
        components = ['service', 'data', 'models', 'pipelines']
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy',
        'components': {}
    }
    
    check_functions = {
        'service': check_service_health,
        'data': check_data_health,
        'models': check_model_health,
        'pipelines': check_pipeline_health
    }
    
    status_priority = {'critical': 3, 'warning': 2, 'degraded': 1, 'healthy': 0}
    max_severity = 0
    
    for component in components:
        if component in check_functions:
            logger.info(f"Checking {component}...")
            
            try:
                check_result = check_functions[component](logger)
                result['components'][component] = check_result
                
                component_status = check_result.get('status', 'unknown')
                severity = status_priority.get(component_status, 0)
                
                if severity > max_severity:
                    max_severity = severity
                    result['overall_status'] = component_status
                
                # Log results
                for check in check_result.get('checks', []):
                    status = "✓" if check.get('passed') else "✗"
                    logger.info(f"  {status} {check['name']}: {check['message']}")
                
            except Exception as e:
                logger.error(f"Failed to check {component}: {e}")
                result['components'][component] = {
                    'status': 'error',
                    'error': str(e)
                }
                max_severity = max(max_severity, 3)
                result['overall_status'] = 'critical'
    
    # Send alerts for failures
    if send_alerts and result['overall_status'] in ('warning', 'critical'):
        failed_components = [
            name for name, data in result['components'].items()
            if data.get('status') in ('warning', 'critical', 'error')
        ]
        
        send_pipeline_alert(
            'health_check',
            result['overall_status'],
            f"Health check {result['overall_status']}: {', '.join(failed_components)}",
            severity='error' if result['overall_status'] == 'critical' else 'warning',
            metadata=result
        )
    
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run health checks on system components"
    )
    parser.add_argument(
        '--component', '-c',
        choices=['all', 'service', 'data', 'models', 'pipelines'],
        default='all',
        help='Which component to check'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--alert',
        action='store_true',
        help='Send alerts for failures'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Determine components
    if args.component == 'all':
        components = None
    else:
        components = [args.component]
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging("health_check", level=level, console=not args.json)
    
    result = run_health_check(
        components=components,
        send_alerts=args.alert,
        logger=logger
    )
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Health Check: {result['overall_status'].upper()}")
        print(f"{'='*60}")
        
        for component, data in result['components'].items():
            status_icon = {
                'healthy': '✓',
                'warning': '⚠',
                'degraded': '⚡',
                'critical': '✗',
                'offline': '○',
                'error': '✗'
            }.get(data.get('status'), '?')
            
            print(f"\n{status_icon} {component.upper()}: {data.get('status', 'unknown')}")
            
            for check in data.get('checks', []):
                icon = "  ✓" if check.get('passed') else "  ✗"
                print(f"{icon} {check['name']}: {check['message']}")
    
    # Exit code based on status
    exit_codes = {
        'healthy': 0,
        'warning': 1,
        'degraded': 1,
        'critical': 2,
        'error': 2
    }
    sys.exit(exit_codes.get(result['overall_status'], 1))


if __name__ == "__main__":
    main()
