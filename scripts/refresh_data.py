"""
Data Refresh Pipeline Script.
Refreshes processed data from raw sources and prepares for model training.

Usage:
    python scripts/refresh_data.py [--force] [--dry-run]
    
Options:
    --force     Force refresh even if data unchanged
    --dry-run   Check for changes without processing
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    retry, PipelineTracker, PipelineLock, 
    setup_logging, compute_data_hash, send_pipeline_alert,
    get_git_commit
)


# =============================================================================
# Configuration
# =============================================================================

DATA_CONFIG = {
    'raw_data_dir': PROJECT_ROOT / 'data' / 'published_data',
    'processed_dir': PROJECT_ROOT / 'data' / 'processed',
    'raw_files': [
        'data_reviews_purchase.csv',
        'data_product.csv',
        'data_product_attribute.csv'
    ],
    'output_files': [
        'interactions.parquet',
        'X_train_confidence.npz',
        'X_train_binary.npz',
        'user_item_mappings.json',
        'user_metadata.pkl',
        'user_pos_train.pkl',
        'user_hard_neg_train.pkl',
        'data_stats.json'
    ]
}


# =============================================================================
# Data Refresh Logic
# =============================================================================

def check_data_changed(logger: logging.Logger) -> Dict[str, Any]:
    """
    Check if raw data has changed since last processing.
    
    Returns:
        Dict with 'changed' flag and hash values
    """
    raw_dir = DATA_CONFIG['raw_data_dir']
    processed_dir = DATA_CONFIG['processed_dir']
    
    # Compute current raw data hash
    current_hash = compute_data_hash(raw_dir, DATA_CONFIG['raw_files'])
    logger.info(f"Current raw data hash: {current_hash}")
    
    # Check previous hash
    mappings_file = processed_dir / 'user_item_mappings.json'
    previous_hash = None
    
    if mappings_file.exists():
        try:
            with open(mappings_file, 'r') as f:
                mappings = json.load(f)
                previous_hash = mappings.get('data_hash')
                logger.info(f"Previous data hash: {previous_hash}")
        except Exception as e:
            logger.warning(f"Could not read previous hash: {e}")
    
    changed = current_hash != previous_hash
    
    return {
        'changed': changed,
        'current_hash': current_hash,
        'previous_hash': previous_hash
    }


@retry(max_attempts=3, backoff_factor=2.0)
def run_data_pipeline(logger: logging.Logger) -> Dict[str, Any]:
    """
    Run the Task 01 data processing pipeline.
    
    Returns:
        Dict with processing results
    """
    from recsys.cf.data import DataProcessor
    
    logger.info("Initializing DataProcessor...")
    processor = DataProcessor(
        reviews_path=str(DATA_CONFIG['raw_data_dir'] / 'data_reviews_purchase.csv'),
        products_path=str(DATA_CONFIG['raw_data_dir'] / 'data_product.csv'),
        output_dir=str(DATA_CONFIG['processed_dir'])
    )
    
    logger.info("Running complete data pipeline...")
    start_time = datetime.now()
    
    # Run pipeline
    result = processor.run_complete_pipeline()
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Data pipeline completed in {duration:.1f}s")
    
    return {
        'success': True,
        'duration_seconds': duration,
        'stats': result.get('stats', {}),
        'files_created': DATA_CONFIG['output_files']
    }


def verify_outputs(logger: logging.Logger) -> Dict[str, Any]:
    """
    Verify all expected output files exist and are valid.
    
    Returns:
        Dict with verification results
    """
    processed_dir = DATA_CONFIG['processed_dir']
    missing_files = []
    file_sizes = {}
    
    for filename in DATA_CONFIG['output_files']:
        file_path = processed_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
        else:
            file_sizes[filename] = file_path.stat().st_size
    
    # Verify data_stats.json has required fields
    stats_valid = False
    stats_file = processed_dir / 'data_stats.json'
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                required_fields = ['num_users', 'num_items', 'num_interactions']
                stats_valid = all(field in stats for field in required_fields)
        except Exception as e:
            logger.warning(f"Could not validate stats file: {e}")
    
    return {
        'success': len(missing_files) == 0 and stats_valid,
        'missing_files': missing_files,
        'file_sizes': file_sizes,
        'stats_valid': stats_valid
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def refresh_data(
    force: bool = False,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Main data refresh function.
    
    Args:
        force: Force refresh even if data unchanged
        dry_run: Check for changes without processing
        logger: Logger instance
        
    Returns:
        Dict with refresh results
    """
    if logger is None:
        logger = setup_logging("data_refresh")
    
    tracker = PipelineTracker()
    result = {
        'pipeline': 'data_refresh',
        'started_at': datetime.now().isoformat(),
        'git_commit': get_git_commit()
    }
    
    # Check for concurrent runs
    with PipelineLock("data_refresh") as lock:
        if not lock.acquired:
            msg = "Data refresh already running"
            logger.warning(msg)
            result['status'] = 'skipped'
            result['message'] = msg
            return result
        
        # Start tracking
        run_id = tracker.start_run("data_refresh", {'force': force, 'dry_run': dry_run})
        
        try:
            # Step 1: Check if data changed
            logger.info("Checking for data changes...")
            change_check = check_data_changed(logger)
            result['change_check'] = change_check
            
            if not change_check['changed'] and not force:
                msg = "Raw data unchanged, skipping refresh"
                logger.info(msg)
                result['status'] = 'skipped'
                result['message'] = msg
                tracker.complete_run(run_id, {'status': 'skipped', 'reason': 'data_unchanged'})
                return result
            
            if dry_run:
                msg = "Dry run - would process data"
                logger.info(msg)
                result['status'] = 'dry_run'
                result['message'] = msg
                tracker.complete_run(run_id, {'status': 'dry_run'})
                return result
            
            # Step 2: Run data pipeline
            logger.info("Starting data pipeline...")
            pipeline_result = run_data_pipeline(logger)
            result['pipeline_result'] = pipeline_result
            
            # Step 3: Verify outputs
            logger.info("Verifying outputs...")
            verify_result = verify_outputs(logger)
            result['verification'] = verify_result
            
            if not verify_result['success']:
                raise RuntimeError(f"Output verification failed: {verify_result}")
            
            # Success
            result['status'] = 'success'
            result['finished_at'] = datetime.now().isoformat()
            
            tracker.complete_run(run_id, {
                'status': 'success',
                'data_hash': change_check['current_hash'],
                'files_created': len(DATA_CONFIG['output_files'])
            })
            
            logger.info("Data refresh completed successfully!")
            
            # Send success alert
            send_pipeline_alert(
                'data_refresh',
                'success',
                f"Data refresh completed. Hash: {change_check['current_hash'][:8]}",
                severity='info'
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Data refresh failed: {error_msg}")
            
            result['status'] = 'failed'
            result['error'] = error_msg
            
            tracker.fail_run(run_id, error_msg)
            
            # Send failure alert
            send_pipeline_alert(
                'data_refresh',
                'failed',
                f"Data refresh failed: {error_msg}",
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
        description="Refresh processed data from raw sources"
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force refresh even if data unchanged'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Check for changes without processing'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging("data_refresh", level=level)
    
    try:
        result = refresh_data(
            force=args.force,
            dry_run=args.dry_run,
            logger=logger
        )
        
        print(f"\n{'='*60}")
        print(f"Data Refresh Result: {result['status'].upper()}")
        print(f"{'='*60}")
        
        if result['status'] == 'success':
            print(f"  Data hash: {result['change_check']['current_hash'][:16]}...")
            if 'pipeline_result' in result:
                print(f"  Duration: {result['pipeline_result']['duration_seconds']:.1f}s")
        elif result.get('message'):
            print(f"  Message: {result['message']}")
        
        sys.exit(0 if result['status'] in ('success', 'skipped', 'dry_run') else 1)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
