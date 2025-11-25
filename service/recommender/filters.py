"""
Attribute Filtering Module.

This module provides functions for filtering recommendations
based on product attributes (brand, skin_type, price range, etc.).

Also includes attribute boosting for hybrid reranking.

Example:
    >>> from service.recommender.filters import apply_filters, boost_by_attributes
    >>> filtered = apply_filters(recommendations, {'brand': 'Innisfree'})
    >>> boosted = boost_by_attributes(filtered, boost_config, metadata)
"""

from typing import Dict, List, Optional, Any, Set, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Filter Functions
# ============================================================================

def apply_filters(
    recommendations: List[Dict[str, Any]],
    filter_params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Filter recommendations based on attribute criteria.
    
    Args:
        recommendations: List of recommendation dicts
        filter_params: Filter criteria, e.g.:
            {
                'brand': 'Innisfree',
                'skin_type': ['oily', 'acne'],
                'price_min': 100000,
                'price_max': 500000,
                'min_rating': 4.0
            }
    
    Returns:
        Filtered recommendations with updated ranks
    """
    if not recommendations or not filter_params:
        return recommendations
    
    filtered = []
    
    for rec in recommendations:
        passes_all = True
        
        for key, value in filter_params.items():
            # Handle special filter keys
            if key == 'price_min':
                price = rec.get('price', 0)
                if price < value:
                    passes_all = False
                    break
                continue
            
            if key == 'price_max':
                price = rec.get('price', float('inf'))
                if price > value:
                    passes_all = False
                    break
                continue
            
            if key == 'min_rating':
                rating = rec.get('avg_star', 0)
                if rating < value:
                    passes_all = False
                    break
                continue
            
            # Standard attribute matching
            rec_value = rec.get(key)
            
            if rec_value is None:
                # Missing attribute - skip filter or fail?
                # For now, skip (don't filter out)
                continue
            
            # Handle list-type values (e.g., skin_type)
            if isinstance(value, list):
                # rec_value should contain at least one of the values
                if isinstance(rec_value, list):
                    if not any(v in rec_value for v in value):
                        passes_all = False
                        break
                elif isinstance(rec_value, str):
                    if not any(v in rec_value for v in value):
                        passes_all = False
                        break
                else:
                    if rec_value not in value:
                        passes_all = False
                        break
            else:
                # Exact match
                if rec_value != value:
                    passes_all = False
                    break
        
        if passes_all:
            filtered.append(rec)
    
    # Update ranks
    for i, rec in enumerate(filtered):
        rec['rank'] = i + 1
    
    return filtered


def get_valid_item_indices(
    filter_params: Dict[str, Any],
    item_metadata: pd.DataFrame,
    item_to_idx: Dict[str, int]
) -> Set[int]:
    """
    Get set of valid item indices based on filters.
    
    For pre-filtering before CF scoring.
    
    Args:
        filter_params: Filter criteria
        item_metadata: Product metadata DataFrame
        item_to_idx: product_id -> index mapping
    
    Returns:
        Set of valid item indices
    """
    if not filter_params or item_metadata is None:
        return set(int(v) for v in item_to_idx.values())
    
    mask = pd.Series([True] * len(item_metadata))
    
    for key, value in filter_params.items():
        if key not in item_metadata.columns:
            continue
        
        if key == 'price_min':
            if 'price' in item_metadata.columns:
                mask &= item_metadata['price'] >= value
            continue
        
        if key == 'price_max':
            if 'price' in item_metadata.columns:
                mask &= item_metadata['price'] <= value
            continue
        
        if key == 'min_rating':
            if 'avg_star' in item_metadata.columns:
                mask &= item_metadata['avg_star'] >= value
            continue
        
        col = item_metadata[key]
        
        if isinstance(value, list):
            # Any match
            mask &= col.apply(
                lambda x: any(v in str(x) for v in value) if pd.notna(x) else False
            )
        else:
            mask &= col == value
    
    valid_pids = set(item_metadata[mask]['product_id'].values)
    
    valid_indices = set()
    for pid in valid_pids:
        idx = item_to_idx.get(str(pid))
        if idx is not None:
            valid_indices.add(int(idx))
    
    return valid_indices


def filter_by_category(
    recommendations: List[Dict[str, Any]],
    category: str,
    category_column: str = 'category'
) -> List[Dict[str, Any]]:
    """
    Filter recommendations by product category.
    
    Args:
        recommendations: List of recommendations
        category: Target category
        category_column: Column name for category
    
    Returns:
        Filtered recommendations
    """
    return apply_filters(recommendations, {category_column: category})


def filter_by_brand(
    recommendations: List[Dict[str, Any]],
    brand: str
) -> List[Dict[str, Any]]:
    """
    Filter recommendations by brand.
    
    Args:
        recommendations: List of recommendations
        brand: Target brand name
    
    Returns:
        Filtered recommendations
    """
    return apply_filters(recommendations, {'brand': brand})


def filter_by_skin_type(
    recommendations: List[Dict[str, Any]],
    skin_types: List[str]
) -> List[Dict[str, Any]]:
    """
    Filter recommendations by skin type compatibility.
    
    Args:
        recommendations: List of recommendations
        skin_types: List of skin types (e.g., ['oily', 'acne'])
    
    Returns:
        Filtered recommendations
    """
    return apply_filters(recommendations, {'skin_type_standardized': skin_types})


def filter_by_price_range(
    recommendations: List[Dict[str, Any]],
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Filter recommendations by price range.
    
    Args:
        recommendations: List of recommendations
        min_price: Minimum price (inclusive)
        max_price: Maximum price (inclusive)
    
    Returns:
        Filtered recommendations
    """
    filters = {}
    if min_price is not None:
        filters['price_min'] = min_price
    if max_price is not None:
        filters['price_max'] = max_price
    
    return apply_filters(recommendations, filters)


def exclude_products(
    recommendations: List[Dict[str, Any]],
    exclude_ids: Set[int]
) -> List[Dict[str, Any]]:
    """
    Exclude specific products from recommendations.
    
    Args:
        recommendations: List of recommendations
        exclude_ids: Set of product IDs to exclude
    
    Returns:
        Filtered recommendations with updated ranks
    """
    if not exclude_ids:
        return recommendations
    
    filtered = [
        rec for rec in recommendations
        if rec.get('product_id') not in exclude_ids
    ]
    
    for i, rec in enumerate(filtered):
        rec['rank'] = i + 1
    
    return filtered


# ============================================================================
# Attribute Boosting Functions
# ============================================================================

def boost_by_attributes(
    recommendations: List[Dict[str, Any]],
    boost_config: Dict[str, Dict[Any, float]],
    metadata: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Boost scores for products matching desired attributes.
    
    Args:
        recommendations: List of recommendation dicts
        boost_config: Dict of attribute -> {value: boost_factor}
            e.g., {'brand': {'Innisfree': 1.2, 'Cetaphil': 1.1}}
        metadata: Optional DataFrame with product attributes
            (if not provided, uses attributes from recommendations)
    
    Returns:
        Recommendations with boosted scores (sorted by final_score)
    
    Example:
        >>> boost_config = {
        ...     'brand': {'Innisfree': 1.2, 'Cetaphil': 1.1},
        ...     'skin_type_standardized': {'oily': 1.15, 'acne': 1.1}
        ... }
        >>> boosted = boost_by_attributes(recs, boost_config, metadata)
    """
    if not recommendations or not boost_config:
        return recommendations
    
    for rec in recommendations:
        product_id = rec['product_id']
        
        # Get product attributes
        product_attrs = {}
        
        if metadata is not None:
            product_row = metadata[metadata['product_id'] == product_id]
            if not product_row.empty:
                product_attrs = product_row.iloc[0].to_dict()
        
        # Merge with attributes already in recommendation
        for key in boost_config.keys():
            if key not in product_attrs and key in rec:
                product_attrs[key] = rec[key]
        
        # Calculate boost factor
        boost_factor = 1.0
        boost_details = {}
        
        for attr, boost_values in boost_config.items():
            attr_value = product_attrs.get(attr)
            
            if attr_value is None:
                continue
            
            # Handle list-type attributes (e.g., skin_type_standardized)
            if isinstance(attr_value, (list, np.ndarray)):
                for val in attr_value:
                    if val in boost_values:
                        boost_factor *= boost_values[val]
                        boost_details[f"{attr}:{val}"] = boost_values[val]
            elif isinstance(attr_value, str):
                # Check if string contains any boost value
                for val, boost in boost_values.items():
                    if val in attr_value:
                        boost_factor *= boost
                        boost_details[f"{attr}:{val}"] = boost
                        break
            else:
                # Direct match
                if attr_value in boost_values:
                    boost_factor *= boost_values[attr_value]
                    boost_details[f"{attr}:{attr_value}"] = boost_values[attr_value]
        
        # Apply boost
        score_key = 'final_score' if 'final_score' in rec else 'score'
        original_score = rec.get(score_key, 0.0)
        rec['final_score'] = original_score * boost_factor
        rec['boost_factor'] = boost_factor
        rec['boost_details'] = boost_details
    
    # Re-sort by boosted score
    recommendations.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    
    # Update ranks
    for i, rec in enumerate(recommendations):
        rec['rank'] = i + 1
    
    return recommendations


def boost_by_user_preferences(
    recommendations: List[Dict[str, Any]],
    user_preferences: Dict[str, List[Any]],
    metadata: Optional[pd.DataFrame] = None,
    match_boost: float = 1.2,
    partial_match_boost: float = 1.1
) -> List[Dict[str, Any]]:
    """
    Boost products matching user's inferred preferences.
    
    Args:
        recommendations: List of recommendations
        user_preferences: Dict of attribute -> preferred_values
            e.g., {'brand': ['Innisfree', 'Cosrx'], 'skin_type': ['oily']}
        metadata: Product metadata DataFrame
        match_boost: Boost for exact match
        partial_match_boost: Boost for partial match
    
    Returns:
        Boosted recommendations
    """
    if not recommendations or not user_preferences:
        return recommendations
    
    # Convert preferences to boost config
    boost_config = {}
    for attr, values in user_preferences.items():
        boost_config[attr] = {}
        for i, val in enumerate(values):
            # First preference gets higher boost
            boost = match_boost if i == 0 else partial_match_boost
            boost_config[attr][val] = boost
    
    return boost_by_attributes(recommendations, boost_config, metadata)


def infer_user_preferences(
    user_history: List[int],
    metadata: pd.DataFrame,
    attributes: List[str] = ['brand', 'skin_type_standardized'],
    top_n: int = 3
) -> Dict[str, List[Any]]:
    """
    Infer user preferences from their interaction history.
    
    Args:
        user_history: List of product IDs user interacted with
        metadata: Product metadata DataFrame
        attributes: List of attribute names to analyze
        top_n: Number of top values per attribute
    
    Returns:
        Dict of attribute -> list of top preferred values
    """
    if not user_history or metadata is None:
        return {}
    
    preferences = {}
    
    # Get products from history
    history_df = metadata[metadata['product_id'].isin(user_history)]
    
    if history_df.empty:
        return {}
    
    for attr in attributes:
        if attr not in history_df.columns:
            continue
        
        # Count attribute values
        value_counts = {}
        
        for _, row in history_df.iterrows():
            val = row[attr]
            
            if val is None:
                continue
            
            # Handle pandas NA/NaN
            try:
                if pd.isna(val):
                    continue
            except (TypeError, ValueError):
                # val might be a list/array which can't be checked with pd.isna
                pass
            
            # Handle list-type values (including numpy arrays)
            if isinstance(val, (list, np.ndarray)):
                for v in val:
                    if v is not None:
                        value_counts[v] = value_counts.get(v, 0) + 1
            else:
                value_counts[val] = value_counts.get(val, 0) + 1
        
        # Get top N values
        if value_counts:
            sorted_values = sorted(
                value_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            preferences[attr] = [v for v, _ in sorted_values]
    
    return preferences


# ============================================================================
# Advanced Filtering Functions
# ============================================================================

def filter_and_boost(
    recommendations: List[Dict[str, Any]],
    filter_params: Optional[Dict[str, Any]] = None,
    boost_config: Optional[Dict[str, Dict[Any, float]]] = None,
    metadata: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Apply both filtering and boosting in one pass.
    
    Args:
        recommendations: List of recommendations
        filter_params: Filter criteria (hard filters)
        boost_config: Boost configuration (soft preferences)
        metadata: Product metadata
    
    Returns:
        Filtered and boosted recommendations
    """
    result = recommendations
    
    # Apply hard filters first
    if filter_params:
        result = apply_filters(result, filter_params)
    
    # Then apply soft boosts
    if boost_config:
        result = boost_by_attributes(result, boost_config, metadata)
    
    return result
