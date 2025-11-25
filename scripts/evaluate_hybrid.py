#!/usr/bin/env python
"""
Evaluate Hybrid Reranking System.

This script compares pure CF recommendations vs hybrid reranked recommendations
using multiple metrics: Recall@K, NDCG@K, Diversity, Semantic Alignment.

Usage:
    python scripts/evaluate_hybrid.py
    python scripts/evaluate_hybrid.py --num-users 500 --topk 10 20
    python scripts/evaluate_hybrid.py --cf-only  # Pure CF baseline
    python scripts/evaluate_hybrid.py --output reports/hybrid_eval.json
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Metrics Functions
# ============================================================================

def compute_recall_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int
) -> float:
    """
    Compute Recall@K.
    
    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of relevant (ground truth) item IDs
        k: Cutoff
    
    Returns:
        Recall@K score
    """
    if not relevant:
        return 0.0
    
    recommended_k = set(recommended[:k])
    hits = len(recommended_k & relevant)
    
    return hits / len(relevant)


def compute_ndcg_at_k(
    recommended: List[int],
    relevant: Set[int],
    k: int
) -> float:
    """
    Compute NDCG@K.
    
    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of relevant item IDs
        k: Cutoff
    
    Returns:
        NDCG@K score
    """
    if not relevant:
        return 0.0
    
    recommended_k = recommended[:k]
    
    # DCG
    dcg = 0.0
    for i, item_id in enumerate(recommended_k):
        if item_id in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0
    
    # IDCG (all relevant items at top positions)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_diversity_bert(
    recommended: List[int],
    phobert_loader: 'PhoBERTEmbeddingLoader'
) -> float:
    """
    Compute BERT-based intra-list diversity.
    
    Diversity = 1 - avg_pairwise_similarity
    
    Args:
        recommended: List of recommended item IDs
        phobert_loader: PhoBERTEmbeddingLoader instance
    
    Returns:
        Diversity score in [0, 1]
    """
    if len(recommended) < 2:
        return 1.0
    
    if phobert_loader is None or not phobert_loader.is_loaded():
        return 0.5  # Default diversity when no embeddings
    
    # Compute pairwise similarities
    similarities = []
    
    for i in range(len(recommended)):
        for j in range(i + 1, len(recommended)):
            emb_i = phobert_loader.get_embedding_normalized(recommended[i])
            emb_j = phobert_loader.get_embedding_normalized(recommended[j])
            
            if emb_i is not None and emb_j is not None:
                sim = float(np.dot(emb_i, emb_j))
                similarities.append(sim)
    
    if not similarities:
        return 1.0
    
    avg_similarity = np.mean(similarities)
    diversity = 1.0 - avg_similarity
    
    return diversity


def compute_semantic_alignment(
    recommended: List[int],
    user_history: List[int],
    phobert_loader: 'PhoBERTEmbeddingLoader'
) -> float:
    """
    Compute semantic alignment between recommendations and user profile.
    
    Args:
        recommended: List of recommended item IDs
        user_history: User's interaction history
        phobert_loader: PhoBERTEmbeddingLoader instance
    
    Returns:
        Semantic alignment score in [0, 1]
    """
    if not recommended or not user_history:
        return 0.0
    
    if phobert_loader is None or not phobert_loader.is_loaded():
        return 0.0
    
    # Compute user profile
    user_profile = phobert_loader.compute_user_profile(user_history)
    if user_profile is None or len(user_profile) == 0:
        return 0.0
    
    # Normalize user profile
    user_profile_norm = user_profile / (np.linalg.norm(user_profile) + 1e-9)
    
    # Compute alignment with each recommendation
    alignments = []
    for pid in recommended:
        emb = phobert_loader.get_embedding_normalized(pid)
        if emb is not None and len(emb) > 0:
            sim = float(np.dot(user_profile_norm, emb))
            alignments.append(max(0, sim))  # Clip negatives
    
    if not alignments:
        return 0.0
    
    return np.mean(alignments)


def compute_category_coverage(
    recommended: List[int],
    metadata: pd.DataFrame,
    column: str = 'brand'
) -> float:
    """
    Compute category coverage (unique categories / total items).
    
    Args:
        recommended: List of recommended item IDs
        metadata: Product metadata DataFrame
        column: Column to compute coverage on
    
    Returns:
        Coverage ratio in [0, 1]
    """
    if not recommended or metadata is None:
        return 0.0
    
    categories = []
    for pid in recommended:
        row = metadata[metadata['product_id'] == pid]
        if not row.empty and column in row.columns:
            val = row[column].iloc[0]
            if pd.notna(val):
                categories.append(val)
    
    if not categories:
        return 0.0
    
    return len(set(categories)) / len(categories)


# ============================================================================
# Evaluation Class
# ============================================================================

class HybridEvaluator:
    """Evaluator for hybrid reranking system."""
    
    def __init__(
        self,
        cf_recommender: 'CFRecommender',
        test_data: pd.DataFrame,
        phobert_loader: 'PhoBERTEmbeddingLoader',
        metadata: Optional[pd.DataFrame] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            cf_recommender: CFRecommender instance
            test_data: DataFrame with test interactions
            phobert_loader: PhoBERTEmbeddingLoader instance
            metadata: Product metadata DataFrame
        """
        self.recommender = cf_recommender
        self.test_data = test_data
        self.phobert_loader = phobert_loader
        self.metadata = metadata
        
        # Build ground truth: {user_id: set of relevant item_ids}
        self.ground_truth = self._build_ground_truth()
        
        # Get trainable users from test data
        self.test_users = list(self.ground_truth.keys())
        
        logger.info(
            f"Evaluator initialized: {len(self.test_users)} test users, "
            f"avg relevant items: {np.mean([len(v) for v in self.ground_truth.values()]):.2f}"
        )
    
    def _build_ground_truth(self) -> Dict[int, Set[int]]:
        """Build ground truth from test data."""
        ground_truth = defaultdict(set)
        
        # Test data should have positive interactions (rating >= 4)
        for _, row in self.test_data.iterrows():
            user_id = int(row['user_id'])
            product_id = int(row['product_id'])
            ground_truth[user_id].add(product_id)
        
        return dict(ground_truth)
    
    def evaluate_user(
        self,
        user_id: int,
        k_values: List[int],
        use_rerank: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate recommendations for a single user.
        
        Args:
            user_id: User ID to evaluate
            k_values: List of K values for metrics
            use_rerank: Whether to use hybrid reranking
        
        Returns:
            Dict with metrics
        """
        relevant = self.ground_truth.get(user_id, set())
        if not relevant:
            return {}
        
        # Get recommendations
        max_k = max(k_values)
        result = self.recommender.recommend(
            user_id=user_id,
            topk=max_k,
            exclude_seen=True,
            rerank=use_rerank
        )
        
        if not result.recommendations:
            return {}
        
        recommended = [rec['product_id'] for rec in result.recommendations]
        
        # Get user history for semantic alignment
        user_history = list(self.recommender.loader.get_user_history(user_id))
        
        metrics = {
            'user_id': user_id,
            'is_fallback': result.is_fallback,
            'latency_ms': result.latency_ms,
            'num_recommendations': len(recommended),
        }
        
        # Compute metrics for each K
        for k in k_values:
            metrics[f'recall@{k}'] = compute_recall_at_k(recommended, relevant, k)
            metrics[f'ndcg@{k}'] = compute_ndcg_at_k(recommended, relevant, k)
        
        # Diversity (using max K)
        metrics['diversity'] = compute_diversity_bert(
            recommended[:max_k],
            self.phobert_loader
        )
        
        # Semantic alignment
        metrics['semantic_alignment'] = compute_semantic_alignment(
            recommended[:max_k],
            user_history,
            self.phobert_loader
        )
        
        # Category coverage
        if self.metadata is not None:
            metrics['brand_coverage'] = compute_category_coverage(
                recommended[:max_k],
                self.metadata,
                'brand'
            )
        
        return metrics
    
    def evaluate(
        self,
        num_users: Optional[int] = None,
        k_values: List[int] = [5, 10, 20],
        use_rerank: bool = True,
        sample_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Evaluate over multiple users.
        
        Args:
            num_users: Number of users to evaluate (None = all)
            k_values: List of K values for metrics
            use_rerank: Whether to use hybrid reranking
            sample_seed: Random seed for user sampling
        
        Returns:
            Dict with aggregated metrics
        """
        # Sample users if needed
        if num_users is not None and num_users < len(self.test_users):
            np.random.seed(sample_seed)
            users_to_eval = np.random.choice(
                self.test_users, 
                size=num_users, 
                replace=False
            ).tolist()
        else:
            users_to_eval = self.test_users
        
        logger.info(f"Evaluating {len(users_to_eval)} users with rerank={use_rerank}")
        
        all_metrics = []
        start_time = time.time()
        
        for i, user_id in enumerate(users_to_eval):
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(users_to_eval)}")
            
            try:
                user_metrics = self.evaluate_user(user_id, k_values, use_rerank)
                if user_metrics:
                    all_metrics.append(user_metrics)
            except Exception as e:
                logger.warning(f"Failed to evaluate user {user_id}: {e}")
        
        total_time = time.time() - start_time
        
        if not all_metrics:
            return {'error': 'No metrics computed'}
        
        # Aggregate metrics
        aggregated = {
            'config': {
                'num_users_evaluated': len(all_metrics),
                'k_values': k_values,
                'use_rerank': use_rerank,
                'total_time_seconds': total_time,
                'avg_time_per_user_ms': (total_time * 1000) / len(all_metrics),
            },
            'metrics': {},
        }
        
        # Compute mean for each metric
        metric_keys = [k for k in all_metrics[0].keys() if k != 'user_id']
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m and m[key] is not None]
            if values:
                if isinstance(values[0], bool):
                    aggregated['metrics'][key] = sum(values) / len(values)  # % true
                else:
                    aggregated['metrics'][f'{key}_mean'] = float(np.mean(values))
                    aggregated['metrics'][f'{key}_std'] = float(np.std(values))
        
        return aggregated


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_cf_vs_hybrid(
    cf_recommender: 'CFRecommender',
    test_data: pd.DataFrame,
    phobert_loader: 'PhoBERTEmbeddingLoader',
    metadata: Optional[pd.DataFrame],
    num_users: int = 200,
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, Any]:
    """
    Compare pure CF vs hybrid reranking.
    
    Args:
        cf_recommender: CFRecommender instance
        test_data: Test interactions DataFrame
        phobert_loader: PhoBERTEmbeddingLoader instance
        metadata: Product metadata DataFrame
        num_users: Number of users to evaluate
        k_values: K values for metrics
    
    Returns:
        Comparison results
    """
    evaluator = HybridEvaluator(
        cf_recommender=cf_recommender,
        test_data=test_data,
        phobert_loader=phobert_loader,
        metadata=metadata
    )
    
    # Evaluate CF only
    logger.info("Evaluating pure CF (rerank=False)...")
    cf_results = evaluator.evaluate(
        num_users=num_users,
        k_values=k_values,
        use_rerank=False
    )
    
    # Evaluate Hybrid
    logger.info("Evaluating hybrid (rerank=True)...")
    hybrid_results = evaluator.evaluate(
        num_users=num_users,
        k_values=k_values,
        use_rerank=True
    )
    
    # Compute improvements
    improvements = {}
    cf_metrics = cf_results.get('metrics', {})
    hybrid_metrics = hybrid_results.get('metrics', {})
    
    for key in cf_metrics:
        if key.endswith('_mean') and key in hybrid_metrics:
            cf_val = cf_metrics[key]
            hybrid_val = hybrid_metrics[key]
            if cf_val > 0:
                improvement = (hybrid_val - cf_val) / cf_val * 100
                improvements[key.replace('_mean', '_improvement_%')] = round(improvement, 2)
    
    return {
        'cf_only': cf_results,
        'hybrid': hybrid_results,
        'improvements': improvements,
        'summary': {
            'diversity_improvement': improvements.get('diversity_improvement_%', 0),
            'recall@10_improvement': improvements.get('recall@10_improvement_%', 0),
            'ndcg@10_improvement': improvements.get('ndcg@10_improvement_%', 0),
            'semantic_alignment_improvement': improvements.get('semantic_alignment_improvement_%', 0),
        }
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid Reranking')
    parser.add_argument('--num-users', type=int, default=200,
                        help='Number of users to evaluate')
    parser.add_argument('--topk', type=int, nargs='+', default=[5, 10, 20],
                        help='K values for evaluation')
    parser.add_argument('--cf-only', action='store_true',
                        help='Evaluate only pure CF (no hybrid)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import modules
    try:
        from service.recommender.recommender import CFRecommender
        from service.recommender.phobert_loader import get_phobert_loader
        from service.recommender.loader import get_loader
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        logger.info("Make sure you're running from project root")
        return 1
    
    # Load data
    logger.info("Loading test data...")
    test_data_path = PROJECT_ROOT / 'data' / 'processed' / 'test_interactions.parquet'
    
    if not test_data_path.exists():
        # Try to extract from interactions.parquet
        interactions_path = PROJECT_ROOT / 'data' / 'processed' / 'interactions.parquet'
        if interactions_path.exists():
            df = pd.read_parquet(interactions_path)
            if 'split' in df.columns:
                test_data = df[df['split'] == 'test']
            else:
                logger.error("No 'split' column in interactions.parquet")
                return 1
        else:
            logger.error(f"Test data not found: {test_data_path}")
            return 1
    else:
        test_data = pd.read_parquet(test_data_path)
    
    # Filter to positive interactions only
    if 'rating' in test_data.columns:
        test_data = test_data[test_data['rating'] >= 4]
    
    logger.info(f"Test data: {len(test_data)} positive interactions")
    
    # Load item metadata
    metadata = None
    metadata_path = PROJECT_ROOT / 'data' / 'processed' / 'product_attributes_enriched.parquet'
    if metadata_path.exists():
        metadata = pd.read_parquet(metadata_path)
        logger.info(f"Loaded metadata: {len(metadata)} products")
    else:
        # Try merged_product_data.csv
        alt_path = PROJECT_ROOT / 'data' / 'processed' / 'merged_product_data.csv'
        if alt_path.exists():
            metadata = pd.read_csv(alt_path)
            logger.info(f"Loaded metadata: {len(metadata)} products")
    
    # Initialize components
    logger.info("Initializing recommender components...")
    
    try:
        recommender = CFRecommender(
            auto_load=True,
            enable_reranking=True,
            rerank_config_path=str(PROJECT_ROOT / 'service' / 'config' / 'rerank_config.yaml')
        )
    except Exception as e:
        logger.error(f"Failed to initialize CFRecommender: {e}")
        logger.info("Creating recommender without reranking for evaluation...")
        recommender = CFRecommender(auto_load=True, enable_reranking=False)
    
    try:
        phobert_loader = get_phobert_loader()
        if not phobert_loader.is_loaded():
            logger.warning("PhoBERT embeddings not loaded - diversity metrics will be limited")
    except Exception as e:
        logger.warning(f"PhoBERT loader failed: {e}")
        phobert_loader = None
    
    # Run evaluation
    if args.cf_only:
        # Pure CF evaluation
        logger.info("Evaluating pure CF...")
        evaluator = HybridEvaluator(
            cf_recommender=recommender,
            test_data=test_data,
            phobert_loader=phobert_loader,
            metadata=metadata
        )
        results = evaluator.evaluate(
            num_users=args.num_users,
            k_values=args.topk,
            use_rerank=False
        )
    else:
        # Compare CF vs Hybrid
        results = compare_cf_vs_hybrid(
            cf_recommender=recommender,
            test_data=test_data,
            phobert_loader=phobert_loader,
            metadata=metadata,
            num_users=args.num_users,
            k_values=args.topk
        )
    
    # Add metadata
    results['timestamp'] = datetime.now().isoformat()
    results['config'] = {
        'num_users': args.num_users,
        'k_values': args.topk,
        'model_id': recommender.model_id,
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("HYBRID RERANKING EVALUATION RESULTS")
    print("=" * 60)
    
    if 'improvements' in results:
        print("\n[COMPARISON] Pure CF vs Hybrid\n")
        
        cf_metrics = results['cf_only']['metrics']
        hybrid_metrics = results['hybrid']['metrics']
        
        print(f"{'Metric':<25} {'CF Only':>12} {'Hybrid':>12} {'Change':>12}")
        print("-" * 61)
        
        for key in sorted(cf_metrics.keys()):
            if key.endswith('_mean'):
                cf_val = cf_metrics[key]
                hybrid_val = hybrid_metrics.get(key, 0)
                change = hybrid_val - cf_val
                change_pct = (change / cf_val * 100) if cf_val != 0 else 0
                
                metric_name = key.replace('_mean', '')
                sign = '+' if change >= 0 else ''
                
                print(f"{metric_name:<25} {cf_val:>12.4f} {hybrid_val:>12.4f} {sign}{change_pct:>10.1f}%")
        
        print("\n[KEY IMPROVEMENTS]:")
        summary = results['summary']
        print(f"  - Diversity: {summary['diversity_improvement']:+.1f}%")
        print(f"  - Recall@10: {summary['recall@10_improvement']:+.1f}%")
        print(f"  - NDCG@10: {summary['ndcg@10_improvement']:+.1f}%")
        print(f"  - Semantic Alignment: {summary['semantic_alignment_improvement']:+.1f}%")
    else:
        # Single evaluation results
        metrics = results.get('metrics', {})
        print("\n[METRICS]:\n")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}")
    
    print("\n" + "=" * 60)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
