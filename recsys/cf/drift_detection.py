"""
Drift Detection for CF Recommendations.

This module provides functions to detect:
- Rating distribution drift
- Popularity shift
- User behavior changes
- BERT embedding drift

Usage:
    >>> from recsys.cf.drift_detection import (
    ...     detect_rating_drift,
    ...     detect_popularity_shift,
    ...     detect_embedding_drift
    ... )
    >>> result = detect_rating_drift(historical_ratings, new_ratings)
    >>> if result['drift_detected']:
    ...     print("Retrain needed!")
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Rating Distribution Drift
# ============================================================================

def detect_rating_drift(
    historical_ratings: np.ndarray,
    new_ratings: np.ndarray,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Detect drift in rating distribution using Kolmogorov-Smirnov test.
    
    Args:
        historical_ratings: Array of historical ratings (train data)
        new_ratings: Array of new ratings (recent data)
        threshold: p-value threshold (default 0.05)
    
    Returns:
        dict: {
            'drift_detected': bool,
            'p_value': float,
            'statistic': float,
            'historical_mean': float,
            'new_mean': float,
            'recommendation': str
        }
    
    Example:
        >>> hist = np.array([5, 5, 4, 5, 3, 5, 5])
        >>> new = np.array([3, 4, 3, 2, 4, 3])
        >>> result = detect_rating_drift(hist, new)
        >>> print(result['drift_detected'])
    """
    if len(historical_ratings) == 0 or len(new_ratings) == 0:
        return {
            'drift_detected': False,
            'p_value': 1.0,
            'statistic': 0.0,
            'recommendation': 'Insufficient data'
        }
    
    # Kolmogorov-Smirnov test
    statistic, p_value = stats.ks_2samp(historical_ratings, new_ratings)
    
    drift_detected = p_value < threshold
    
    return {
        'drift_detected': drift_detected,
        'p_value': float(p_value),
        'statistic': float(statistic),
        'historical_mean': float(np.mean(historical_ratings)),
        'historical_std': float(np.std(historical_ratings)),
        'new_mean': float(np.mean(new_ratings)),
        'new_std': float(np.std(new_ratings)),
        'recommendation': 'Retrain model' if drift_detected else 'No action needed'
    }


def detect_rating_distribution_drift(
    historical_data: pd.DataFrame,
    new_data: pd.DataFrame,
    rating_col: str = 'rating',
    threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Detect drift in rating distribution from DataFrames.
    
    Args:
        historical_data: DataFrame with historical ratings
        new_data: DataFrame with new ratings
        rating_col: Name of rating column
        threshold: p-value threshold
    
    Returns:
        Drift detection result dict
    """
    hist_ratings = historical_data[rating_col].values
    new_ratings = new_data[rating_col].values
    
    return detect_rating_drift(hist_ratings, new_ratings, threshold)


# ============================================================================
# Popularity Shift Detection
# ============================================================================

def detect_popularity_shift(
    old_popularity: np.ndarray,
    new_popularity: np.ndarray,
    threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Detect shift in item popularity ranking using Spearman correlation.
    
    Args:
        old_popularity: Array of item popularities (historical)
        new_popularity: Array of item popularities (recent)
        threshold: Correlation threshold (default 0.8)
    
    Returns:
        dict: {
            'shift_detected': bool,
            'correlation': float,
            'p_value': float,
            'recommendation': str
        }
    """
    if len(old_popularity) != len(new_popularity):
        return {
            'shift_detected': True,
            'correlation': 0.0,
            'p_value': 0.0,
            'recommendation': 'Item sets differ - cannot compare'
        }
    
    if len(old_popularity) == 0:
        return {
            'shift_detected': False,
            'correlation': 1.0,
            'recommendation': 'No items to compare'
        }
    
    # Spearman rank correlation
    correlation, p_value = stats.spearmanr(old_popularity, new_popularity)
    
    shift_detected = correlation < threshold
    
    return {
        'shift_detected': shift_detected,
        'correlation': float(correlation),
        'p_value': float(p_value),
        'recommendation': 'Retrain with updated popularity' if shift_detected else 'No action'
    }


def compute_item_popularity(
    interactions: pd.DataFrame,
    item_col: str = 'product_id'
) -> Dict[int, int]:
    """
    Compute item popularity from interactions.
    
    Args:
        interactions: DataFrame with interactions
        item_col: Column name for item ID
    
    Returns:
        Dict mapping item_id to interaction count
    """
    return interactions[item_col].value_counts().to_dict()


# ============================================================================
# User Behavior Drift
# ============================================================================

def detect_user_activity_drift(
    historical_data: pd.DataFrame,
    new_data: pd.DataFrame,
    user_col: str = 'user_id',
    threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Detect changes in user activity patterns.
    
    Args:
        historical_data: Historical interactions
        new_data: Recent interactions
        user_col: User column name
        threshold: Threshold for significant change (20%)
    
    Returns:
        Drift detection result
    """
    # Interactions per user
    hist_per_user = historical_data.groupby(user_col).size()
    new_per_user = new_data.groupby(user_col).size()
    
    hist_mean = hist_per_user.mean()
    new_mean = new_per_user.mean()
    
    # Relative change
    if hist_mean > 0:
        relative_change = abs(new_mean - hist_mean) / hist_mean
    else:
        relative_change = 0.0 if new_mean == 0 else 1.0
    
    drift_detected = relative_change > threshold
    
    # New vs returning users
    hist_users = set(historical_data[user_col].unique())
    new_users = set(new_data[user_col].unique())
    
    returning_users = len(hist_users & new_users)
    brand_new_users = len(new_users - hist_users)
    
    return {
        'drift_detected': drift_detected,
        'historical_mean_interactions': float(hist_mean),
        'new_mean_interactions': float(new_mean),
        'relative_change': float(relative_change),
        'returning_users': returning_users,
        'brand_new_users': brand_new_users,
        'new_user_ratio': brand_new_users / max(1, len(new_users)),
        'recommendation': 'Consider retraining' if drift_detected else 'No action'
    }


# ============================================================================
# BERT Embedding Drift
# ============================================================================

def detect_embedding_drift(
    old_embeddings: np.ndarray,
    new_embeddings: np.ndarray,
    product_ids: Optional[List[int]] = None,
    threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Detect drift in BERT embeddings (e.g., after re-training PhoBERT).
    
    Args:
        old_embeddings: np.array (N, D) - old embeddings
        new_embeddings: np.array (N, D) - new embeddings
        product_ids: List of product IDs
        threshold: Cosine similarity threshold
    
    Returns:
        dict: {
            'drift_detected': bool,
            'mean_similarity': float,
            'num_drifted': int,
            'drifted_products': list
        }
    """
    if old_embeddings.shape != new_embeddings.shape:
        return {
            'drift_detected': True,
            'mean_similarity': 0.0,
            'recommendation': 'Embedding dimensions differ'
        }
    
    if len(old_embeddings) == 0:
        return {
            'drift_detected': False,
            'mean_similarity': 1.0,
            'recommendation': 'No embeddings to compare'
        }
    
    # Normalize embeddings
    old_norm = old_embeddings / (np.linalg.norm(old_embeddings, axis=1, keepdims=True) + 1e-8)
    new_norm = new_embeddings / (np.linalg.norm(new_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarities (self-similarity)
    similarities = np.sum(old_norm * new_norm, axis=1)
    
    # Statistics
    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    min_sim = float(np.min(similarities))
    
    # Identify drifted items
    drifted_mask = similarities < threshold
    drifted_indices = np.where(drifted_mask)[0]
    
    drifted_products = []
    if product_ids is not None:
        drifted_products = [product_ids[i] for i in drifted_indices[:10]]
    
    drift_detected = mean_sim < threshold
    
    return {
        'drift_detected': drift_detected,
        'mean_similarity': mean_sim,
        'std_similarity': std_sim,
        'min_similarity': min_sim,
        'num_drifted': int(np.sum(drifted_mask)),
        'drifted_products': drifted_products,
        'recommendation': 'Regenerate embeddings' if drift_detected else 'No action'
    }


def check_embedding_freshness(
    embeddings_path: str,
    max_age_days: int = 30
) -> Dict[str, Any]:
    """
    Check if BERT embeddings are stale.
    
    Args:
        embeddings_path: Path to embeddings file
        max_age_days: Maximum acceptable age in days
    
    Returns:
        dict with freshness info
    """
    import torch
    
    path = Path(embeddings_path)
    if not path.exists():
        return {
            'exists': False,
            'is_stale': True,
            'recommendation': 'Generate embeddings'
        }
    
    # Check file modification time
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age_days = (datetime.now() - mtime).days
    
    # Try to load metadata
    try:
        data = torch.load(embeddings_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict) and 'created_at' in data:
            created_at = datetime.fromisoformat(data['created_at'])
            age_days = (datetime.now() - created_at).days
    except Exception:
        pass
    
    is_stale = age_days > max_age_days
    
    return {
        'exists': True,
        'age_days': age_days,
        'is_stale': is_stale,
        'max_age_days': max_age_days,
        'recommendation': f'Regenerate embeddings (age: {age_days} days)' if is_stale else 'Embeddings are fresh'
    }


# ============================================================================
# Comprehensive Drift Check
# ============================================================================

def run_drift_detection(
    historical_data: pd.DataFrame,
    new_data: pd.DataFrame,
    embeddings_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive drift detection.
    
    Args:
        historical_data: Historical interactions DataFrame
        new_data: Recent interactions DataFrame
        embeddings_path: Path to BERT embeddings (optional)
    
    Returns:
        Comprehensive drift report
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }
    
    # Rating drift
    try:
        rating_drift = detect_rating_distribution_drift(
            historical_data, new_data, 'rating'
        )
        report['checks']['rating_distribution'] = rating_drift
    except Exception as e:
        report['checks']['rating_distribution'] = {'error': str(e)}
    
    # Popularity shift
    try:
        old_pop = compute_item_popularity(historical_data)
        new_pop = compute_item_popularity(new_data)
        
        # Align items
        common_items = sorted(set(old_pop.keys()) & set(new_pop.keys()))
        if common_items:
            old_arr = np.array([old_pop[i] for i in common_items])
            new_arr = np.array([new_pop[i] for i in common_items])
            pop_shift = detect_popularity_shift(old_arr, new_arr)
            report['checks']['popularity_shift'] = pop_shift
    except Exception as e:
        report['checks']['popularity_shift'] = {'error': str(e)}
    
    # User activity drift
    try:
        activity_drift = detect_user_activity_drift(historical_data, new_data)
        report['checks']['user_activity'] = activity_drift
    except Exception as e:
        report['checks']['user_activity'] = {'error': str(e)}
    
    # Embedding freshness
    if embeddings_path:
        try:
            freshness = check_embedding_freshness(embeddings_path)
            report['checks']['embedding_freshness'] = freshness
        except Exception as e:
            report['checks']['embedding_freshness'] = {'error': str(e)}
    
    # Overall decision
    any_drift = any(
        check.get('drift_detected', False) or check.get('is_stale', False)
        for check in report['checks'].values()
        if isinstance(check, dict) and 'error' not in check
    )
    
    report['overall'] = {
        'drift_detected': any_drift,
        'recommendation': 'Consider retraining' if any_drift else 'No action needed'
    }
    
    return report


# ============================================================================
# Retrain Decision
# ============================================================================

def should_retrain(
    historical_data: Optional[pd.DataFrame] = None,
    new_data: Optional[pd.DataFrame] = None,
    data_age_days: Optional[int] = None,
    recent_ctr: Optional[float] = None,
    baseline_ctr: Optional[float] = None,
    embeddings_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Determine if model should be retrained.
    
    Args:
        historical_data: Historical interactions
        new_data: Recent interactions
        data_age_days: Age of training data in days
        recent_ctr: Recent click-through rate (if available)
        baseline_ctr: Baseline CTR for comparison
        embeddings_path: Path to embeddings
    
    Returns:
        dict: {
            'should_retrain': bool,
            'reasons': list[str]
        }
    """
    reasons = []
    
    # Check data drift
    if historical_data is not None and new_data is not None:
        drift_result = detect_rating_distribution_drift(historical_data, new_data)
        if drift_result['drift_detected']:
            reasons.append(f"Rating distribution drift (p={drift_result['p_value']:.4f})")
    
    # Check online performance (if CTR available)
    if recent_ctr is not None and baseline_ctr is not None:
        if recent_ctr < baseline_ctr * 0.9:  # >10% drop
            drop_pct = (1 - recent_ctr / baseline_ctr) * 100
            reasons.append(f"CTR degraded {drop_pct:.1f}%")
    
    # Check data freshness
    if data_age_days is not None and data_age_days > 30:
        reasons.append(f"Training data is {data_age_days} days old")
    
    # Check embedding freshness
    if embeddings_path:
        freshness = check_embedding_freshness(embeddings_path)
        if freshness.get('is_stale', False):
            reasons.append(f"BERT embeddings are {freshness['age_days']} days old")
    
    return {
        'should_retrain': len(reasons) > 0,
        'reasons': reasons,
        'timestamp': datetime.now().isoformat()
    }
