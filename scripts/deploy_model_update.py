"""
Model Deployment Script.
Deploys new models to production service with zero-downtime reload.

Usage:
    python scripts/deploy_model_update.py [--model-id MODEL_ID] [--rollback]
    
Options:
    --model-id      Specific model ID to deploy (default: current_best)
    --rollback      Rollback to previous model
    --dry-run       Check deployment without applying
"""

import os
import sys
import json
import argparse
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    retry, PipelineTracker, PipelineLock,
    setup_logging, send_pipeline_alert, get_git_commit
)


# =============================================================================
# Configuration
# =============================================================================

DEPLOY_CONFIG = {
    'registry_path': PROJECT_ROOT / 'artifacts' / 'cf' / 'registry.json',
    'service_url': 'http://localhost:8000',
    'health_check_timeout': 30,
    'reload_timeout': 60,
    'deployment_history_path': PROJECT_ROOT / 'logs' / 'deployment_history.json'
}


# =============================================================================
# Deployment Functions
# =============================================================================

def load_registry() -> Dict[str, Any]:
    """Load model registry."""
    registry_path = DEPLOY_CONFIG['registry_path']
    
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found at {registry_path}")
    
    with open(registry_path, 'r') as f:
        return json.load(f)


def get_model_info(model_id: str, registry: Dict) -> Optional[Dict]:
    """Get model info from registry."""
    models_data = registry.get('models', {})
    
    # Handle dict format (new)
    if isinstance(models_data, dict):
        model = models_data.get(model_id)
        if model:
            model['model_id'] = model_id
            return model
    else:
        # Handle list format (old)
        for model in models_data:
            if model.get('model_id') == model_id:
                return model
    return None


def get_previous_model(registry: Dict) -> Optional[str]:
    """Get the previous active model for rollback."""
    models_data = registry.get('models', {})
    
    # Get current best
    current_best_data = registry.get('current_best')
    if isinstance(current_best_data, dict):
        current_best = current_best_data.get('model_id')
    else:
        current_best = current_best_data
    
    # Handle dict format (new)
    if isinstance(models_data, dict):
        # Sort by created_at descending
        sorted_models = sorted(
            models_data.items(),
            key=lambda x: x[1].get('created_at', ''),
            reverse=True
        )
        for model_id, model_info in sorted_models:
            if model_id != current_best:
                return model_id
    else:
        # Handle list format (old)
        sorted_models = sorted(
            models_data,
            key=lambda m: m.get('registered_at', ''),
            reverse=True
        )
        for model in sorted_models:
            if model['model_id'] != current_best:
                return model['model_id']
    
    return None


@retry(max_attempts=3, backoff_factor=2.0)
def check_service_health(logger: logging.Logger) -> Dict[str, Any]:
    """
    Check if the recommendation service is healthy.
    
    Returns:
        Service health info
    """
    url = f"{DEPLOY_CONFIG['service_url']}/health"
    
    try:
        response = requests.get(url, timeout=DEPLOY_CONFIG['health_check_timeout'])
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        logger.warning("Service not reachable - may not be running")
        return {'status': 'offline', 'error': 'Connection refused'}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {'status': 'error', 'error': str(e)}


@retry(max_attempts=3, backoff_factor=2.0)
def trigger_model_reload(
    model_id: Optional[str],
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Trigger model reload on the service.
    
    Args:
        model_id: Specific model to load (None = use registry best)
        
    Returns:
        Reload response
    """
    url = f"{DEPLOY_CONFIG['service_url']}/reload_model"
    
    payload = {}
    if model_id:
        payload['model_id'] = model_id
    
    response = requests.post(
        url,
        json=payload,
        timeout=DEPLOY_CONFIG['reload_timeout']
    )
    response.raise_for_status()
    
    return response.json()


def verify_deployment(
    expected_model_id: str,
    logger: logging.Logger
) -> bool:
    """
    Verify the correct model is loaded.
    
    Returns:
        True if verification passed
    """
    url = f"{DEPLOY_CONFIG['service_url']}/model_info"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        info = response.json()
        loaded_model = info.get('model_id')
        
        if loaded_model == expected_model_id:
            logger.info(f"Verification passed: {loaded_model}")
            return True
        else:
            logger.error(f"Model mismatch: expected {expected_model_id}, got {loaded_model}")
            return False
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def update_registry_active_status(
    model_id: str,
    logger: logging.Logger
) -> None:
    """Update registry to mark model as active."""
    registry_path = DEPLOY_CONFIG['registry_path']
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    # Update active status
    for model in registry['models']:
        model['is_active'] = (model['model_id'] == model_id)
    
    registry['current_best'] = model_id
    
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
    
    logger.info(f"Registry updated: {model_id} is now active")


def record_deployment(
    model_id: str,
    status: str,
    metadata: Optional[Dict] = None
) -> None:
    """Record deployment in history file."""
    history_path = DEPLOY_CONFIG['deployment_history_path']
    
    # Load existing history
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = {'deployments': []}
    
    # Add new deployment
    history['deployments'].append({
        'model_id': model_id,
        'deployed_at': datetime.now().isoformat(),
        'status': status,
        'git_commit': get_git_commit(),
        'metadata': metadata or {}
    })
    
    # Keep last 100 deployments
    history['deployments'] = history['deployments'][-100:]
    
    # Save
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


# =============================================================================
# Main Deployment Pipeline
# =============================================================================

def deploy_model(
    model_id: Optional[str] = None,
    rollback: bool = False,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Main deployment function.
    
    Args:
        model_id: Specific model to deploy (None = current_best)
        rollback: Rollback to previous model
        dry_run: Check without applying
        logger: Logger instance
        
    Returns:
        Deployment result
    """
    if logger is None:
        logger = setup_logging("model_deployment")
    
    tracker = PipelineTracker()
    result = {
        'pipeline': 'model_deployment',
        'started_at': datetime.now().isoformat()
    }
    
    with PipelineLock("model_deployment") as lock:
        if not lock.acquired:
            msg = "Deployment already in progress"
            logger.warning(msg)
            result['status'] = 'skipped'
            result['message'] = msg
            return result
        
        run_id = tracker.start_run("model_deployment", {
            'model_id': model_id,
            'rollback': rollback,
            'dry_run': dry_run
        })
        
        try:
            # Load registry
            logger.info("Loading model registry...")
            registry = load_registry()
            
            # Determine model to deploy
            if rollback:
                model_id = get_previous_model(registry)
                if not model_id:
                    raise ValueError("No previous model available for rollback")
                logger.info(f"Rollback mode: deploying previous model {model_id}")
            elif not model_id:
                current_best_data = registry.get('current_best')
                if isinstance(current_best_data, dict):
                    model_id = current_best_data.get('model_id')
                else:
                    model_id = current_best_data
                if not model_id:
                    raise ValueError("No current_best model in registry")
                logger.info(f"Deploying current_best: {model_id}")
            
            result['model_id'] = model_id
            
            # Get model info
            model_info = get_model_info(model_id, registry)
            if not model_info:
                raise ValueError(f"Model not found in registry: {model_id}")
            
            result['model_info'] = {
                'model_type': model_info.get('model_type'),
                'metrics': model_info.get('metrics', {}),
                'path': model_info.get('path')
            }
            
            # Check service health
            logger.info("Checking service health...")
            health = check_service_health(logger)
            result['service_health'] = health
            
            if health.get('status') == 'offline':
                msg = "Service is offline - deployment will take effect on next startup"
                logger.warning(msg)
                
                if dry_run:
                    result['status'] = 'dry_run'
                    result['message'] = msg
                else:
                    # Just update registry for offline service
                    update_registry_active_status(model_id, logger)
                    record_deployment(model_id, 'pending_restart')
                    result['status'] = 'pending'
                    result['message'] = msg
                
                tracker.complete_run(run_id, {'status': result['status']})
                return result
            
            if dry_run:
                logger.info("Dry run - would deploy model")
                result['status'] = 'dry_run'
                result['message'] = f"Would deploy {model_id}"
                tracker.complete_run(run_id, {'status': 'dry_run'})
                return result
            
            # Trigger reload
            logger.info(f"Triggering reload for model: {model_id}")
            reload_result = trigger_model_reload(model_id, logger)
            result['reload_result'] = reload_result
            
            # Verify deployment
            logger.info("Verifying deployment...")
            verified = verify_deployment(model_id, logger)
            
            if not verified:
                raise RuntimeError("Deployment verification failed")
            
            # Update registry
            update_registry_active_status(model_id, logger)
            
            # Record deployment
            record_deployment(model_id, 'success', {
                'metrics': model_info.get('metrics'),
                'reload_response': reload_result
            })
            
            # Success
            result['status'] = 'success'
            result['finished_at'] = datetime.now().isoformat()
            
            tracker.complete_run(run_id, {
                'status': 'success',
                'model_id': model_id,
                'verified': True
            })
            
            logger.info(f"Deployment successful: {model_id}")
            
            # Send alert
            send_pipeline_alert(
                'model_deployment',
                'success',
                f"Deployed model: {model_id}",
                severity='info'
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Deployment failed: {error_msg}")
            
            result['status'] = 'failed'
            result['error'] = error_msg
            
            tracker.fail_run(run_id, error_msg)
            record_deployment(model_id or 'unknown', 'failed', {'error': error_msg})
            
            send_pipeline_alert(
                'model_deployment',
                'failed',
                f"Deployment failed: {error_msg}",
                severity='error'
            )
            
            raise
    
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Deploy model to production service"
    )
    parser.add_argument(
        '--model-id', '-m',
        type=str,
        default=None,
        help='Specific model ID to deploy (default: current_best)'
    )
    parser.add_argument(
        '--rollback', '-r',
        action='store_true',
        help='Rollback to previous model'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Check deployment without applying'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging("model_deployment", level=level)
    
    try:
        result = deploy_model(
            model_id=args.model_id,
            rollback=args.rollback,
            dry_run=args.dry_run,
            logger=logger
        )
        
        print(f"\n{'='*60}")
        print(f"Deployment Result: {result['status'].upper()}")
        print(f"{'='*60}")
        
        if result.get('model_id'):
            print(f"  Model ID: {result['model_id']}")
        
        if result.get('model_info'):
            info = result['model_info']
            print(f"  Model Type: {info.get('model_type', 'unknown')}")
            if info.get('metrics'):
                print(f"  Recall@10: {info['metrics'].get('recall@10', 'N/A')}")
        
        if result.get('message'):
            print(f"  Message: {result['message']}")
        
        sys.exit(0 if result['status'] in ('success', 'pending', 'dry_run') else 1)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
