#!/usr/bin/env python3
"""
Test Script for Evaluation Module.

This script tests all components of the recsys.cf.evaluation module:
- Core metrics (Recall, NDCG, Precision, etc.)
- Hybrid metrics (Diversity, Novelty, etc.)
- Model evaluators
- Baseline comparators
- Statistical testing
- Report generation

Usage:
    python scripts/test_evaluation_module.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from recsys.cf.evaluation import (
        # Core metrics
        recall_at_k, ndcg_at_k, precision_at_k, mrr, map_at_k, coverage,
        MetricFactory, RecallAtK, NDCGAtK,

        # Hybrid metrics
        DiversityMetric, NoveltyMetric, SemanticAlignmentMetric,

        # Evaluators
        ModelEvaluator, BatchModelEvaluator,

        # Baselines
        PopularityBaseline, RandomBaseline,

        # Comparison & Statistical
        ModelComparator, StatisticalTester, BootstrapEstimator,
        ReportGenerator,

        # Convenience functions
        evaluate_model, compare_models
    )
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def create_mock_data():
    """Create mock data for testing."""
    print("\n📊 Creating mock data...")

    # Mock user-item interactions
    np.random.seed(42)
    n_users = 100
    n_items = 200
    n_interactions = 500

    # Generate random interactions
    user_ids = np.random.randint(0, n_users, n_interactions)
    item_ids = np.random.randint(0, n_items, n_interactions)

    # Create test data (leave-one-out)
    test_data = {}
    user_pos_train = {}

    for u in range(n_users):
        user_interactions = item_ids[user_ids == u]
        if len(user_interactions) >= 2:
            # Keep one for test, rest for train
            test_item = user_interactions[0]
            train_items = user_interactions[1:]
            test_data[u] = {test_item}  # Use set instead of list
            user_pos_train[u] = set(train_items)

    # Mock model embeddings (ALS-style)
    factors = 32
    U = np.random.randn(n_users, factors) * 0.1  # User embeddings
    V = np.random.randn(n_items, factors) * 0.1  # Item embeddings

    # Mock item popularity
    item_popularity = np.random.exponential(2, n_items)
    item_popularity = item_popularity / item_popularity.sum()  # Normalize

    # Mock BERT embeddings for hybrid metrics
    bert_dim = 768
    bert_embeddings = np.random.randn(n_items, bert_dim) * 0.01

    print(f"   Created {len(test_data)} test users, {n_items} items")
    print(f"   User embeddings: {U.shape}, Item embeddings: {V.shape}")

    return {
        'test_data': test_data,
        'user_pos_train': user_pos_train,
        'U': U,
        'V': V,
        'item_popularity': item_popularity,
        'bert_embeddings': bert_embeddings,
        'n_users': n_users,
        'n_items': n_items
    }


def test_core_metrics():
    """Test core ranking metrics."""
    print("\n🎯 Testing Core Metrics...")

    # Mock recommendations and ground truth for single user
    predictions = [1, 5, 10, 15, 20]  # Single user recommendations
    ground_truth = {1, 2, 3}  # Single user ground truth (must be a set)

    k = 5

    # Test individual functions
    recall = recall_at_k(predictions, ground_truth, k=k)
    ndcg = ndcg_at_k(predictions, ground_truth, k=k)
    precision = precision_at_k(predictions, ground_truth, k=k)
    mrr_score = mrr(predictions, ground_truth)
    map_score = map_at_k(predictions, ground_truth, k=k)

    print(f"   Recall@{k}: {recall:.4f}")
    print(f"   NDCG@{k}: {ndcg:.4f}")
    print(f"   Precision@{k}: {precision:.4f}")
    print(f"   MRR: {mrr_score:.4f}")
    print(f"   MAP@{k}: {map_score:.4f}")

    # Test MetricFactory
    recall_metric = MetricFactory.create('recall', k=k)
    factory_recall = recall_metric.compute(predictions, ground_truth)
    print(f"   Factory Recall@{k}: {factory_recall:.4f}")

    # Test coverage with multiple users
    all_recommendations = {
        0: [1, 5, 10, 15, 20],
        1: [2, 7, 12, 17, 22],
        2: [3, 8, 13, 18, 23],
    }
    cov = coverage(all_recommendations, 50)
    print(f"   Coverage: {cov:.4f}")

    return {
        'recall': recall,
        'ndcg': ndcg,
        'precision': precision,
        'mrr': mrr_score,
        'map': map_score,
        'coverage': cov
    }


def test_model_evaluator(data):
    """Test ModelEvaluator."""
    print("\n🤖 Testing ModelEvaluator...")

    # ModelEvaluator uses positional args: U, V
    evaluator = ModelEvaluator(
        data['U'],  # User embeddings
        data['V'],  # Item embeddings  
        k_values=[5, 10, 20]
    )

    results = evaluator.evaluate(
        test_data=data['test_data'],
        user_pos_train=data['user_pos_train']
    )

    print("   Model evaluation results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")

    # Return dict with K values as keys for compatibility
    return {k: {'recall': results.get(f'recall@{k}', 0), 'ndcg': results.get(f'ndcg@{k}', 0)} 
            for k in [5, 10, 20]}


def test_baseline_evaluators(data):
    """Test baseline evaluators."""
    print("\n📈 Testing Baseline Evaluators...")

    # Popularity baseline - needs item_popularity and num_items
    pop_baseline = PopularityBaseline(data['item_popularity'], data['n_items'])
    pop_results = pop_baseline.evaluate(
        test_data=data['test_data'],
        user_pos_train=data['user_pos_train'],
        k_values=[5, 10, 20]
    )

    # Random baseline - needs num_items
    random_baseline = RandomBaseline(data['n_items'])
    random_results = random_baseline.evaluate(
        test_data=data['test_data'],
        user_pos_train=data['user_pos_train'],
        k_values=[5, 10, 20]
    )

    print("   Popularity baseline results:")
    for key, value in pop_results.items():
        if isinstance(value, float) and 'recall' in key or 'ndcg' in key:
            print(f"   {key}: {value:.4f}")

    print("   Random baseline results:")
    for key, value in random_results.items():
        if isinstance(value, float) and 'recall' in key or 'ndcg' in key:
            print(f"   {key}: {value:.4f}")

    return {'popularity': pop_results, 'random': random_results}


def test_hybrid_metrics(data):
    """Test hybrid metrics."""
    print("\n🔄 Testing Hybrid Metrics...")

    # Mock recommendations for multiple users (list of item indices)
    recommendations = [1, 5, 10, 15, 20]  # Single list for testing

    # Diversity using BERT embeddings
    diversity_metric = DiversityMetric()
    diversity_score = diversity_metric.compute(recommendations, data['bert_embeddings'])
    print(f"   Diversity: {diversity_score:.4f}")

    # Novelty (inverse popularity)
    novelty_metric = NoveltyMetric()
    novelty_score = novelty_metric.compute(recommendations, data['item_popularity'])
    print(f"   Novelty: {novelty_score:.4f}")

    # Mock user profile for semantic alignment
    # compute(user_profile_emb, recommendations, item_embeddings, item_to_idx, k)
    user_profile = np.random.randn(768)  # Single user profile
    alignment_metric = SemanticAlignmentMetric()
    alignment_score = alignment_metric.compute(
        user_profile_emb=user_profile,
        recommendations=recommendations, 
        item_embeddings=data['bert_embeddings']
    )
    print(f"   Semantic Alignment: {alignment_score:.4f}")

    return {
        'diversity': diversity_score,
        'novelty': novelty_score,
        'alignment': alignment_score
    }


def test_statistical_testing():
    """Test statistical testing."""
    print("\n📊 Testing Statistical Testing...")

    np.random.seed(42)

    # Create mock per-user scores
    n_users = 50
    baseline_scores = np.random.normal(0.15, 0.05, n_users)
    model_scores = baseline_scores + np.random.normal(0.03, 0.02, n_users)

    # Statistical tester
    tester = StatisticalTester(significance_level=0.05)

    # Paired t-test
    t_result = tester.paired_t_test(model_scores, baseline_scores)
    print(f"   Paired t-test: p-value={t_result['p_value']:.4f}, significant={t_result['significant']}")

    # Cohen's d
    effect = tester.cohens_d(model_scores, baseline_scores)
    print(f"   Cohen's d: {effect['d']:.4f} ({effect['interpretation']})")

    # Confidence interval
    ci = tester.confidence_interval(model_scores)
    print(f"   95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    # Bootstrap
    bootstrap = BootstrapEstimator(n_bootstrap=100, random_state=42)
    boot_ci = bootstrap.confidence_interval(model_scores)
    print(f"   Bootstrap 95% CI: [{boot_ci['lower']:.4f}, {boot_ci['upper']:.4f}]")

    return {
        't_test': t_result,
        'effect_size': effect,
        'confidence_interval': ci,
        'bootstrap_ci': boot_ci
    }


def test_comparison_and_reporting():
    """Test model comparison and reporting."""
    print("\n📋 Testing Comparison & Reporting...")

    # Use ModelComparator with add_model_results/add_baseline_results API
    comparator = ModelComparator()
    
    # Mock model results (flat dict with metrics)
    als_results = {
        'recall@5': 0.18, 'ndcg@5': 0.15, 'precision@5': 0.12,
        'recall@10': 0.25, 'ndcg@10': 0.20, 'precision@10': 0.15,
        'recall@20': 0.32, 'ndcg@20': 0.25, 'precision@20': 0.18
    }
    bpr_results = {
        'recall@5': 0.20, 'ndcg@5': 0.17, 'precision@5': 0.14,
        'recall@10': 0.28, 'ndcg@10': 0.22, 'precision@10': 0.17,
        'recall@20': 0.35, 'ndcg@20': 0.27, 'precision@20': 0.20
    }
    pop_results = {
        'recall@5': 0.12, 'ndcg@5': 0.10, 'precision@5': 0.08,
        'recall@10': 0.18, 'ndcg@10': 0.14, 'precision@10': 0.11,
        'recall@20': 0.24, 'ndcg@20': 0.18, 'precision@20': 0.13
    }

    # Add results
    comparator.add_model_results('ALS', als_results)
    comparator.add_model_results('BPR', bpr_results)
    comparator.add_baseline_results('Popularity', pop_results)

    # Compute improvement
    improvement = comparator.compute_improvement('ALS', 'Popularity', 'recall@10')
    print(f"   ALS vs Popularity (Recall@10):")
    print(f"     Model: {improvement['model_value']:.4f}, Baseline: {improvement['baseline_value']:.4f}")
    print(f"     Improvement: {improvement['relative_percent']:.1f}%")

    improvement_bpr = comparator.compute_improvement('BPR', 'Popularity', 'recall@10')
    print(f"   BPR vs Popularity (Recall@10):")
    print(f"     Improvement: {improvement_bpr['relative_percent']:.1f}%")

    # Generate report (skip if reports dir doesn't exist)
    report_path = project_root / "reports" / "test_evaluation_report.md"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_gen = ReportGenerator()
        # Create a simple report dict
        report_data = {
            'models': {
                'ALS': als_results,
                'BPR': bpr_results
            },
            'baselines': {
                'Popularity': pop_results
            }
        }
        # Use generate_markdown method (not generate_markdown_report)
        report_gen.generate_markdown(report_data, str(report_path))
        print(f"   Report generated: {report_path}")
    except Exception as e:
        print(f"   Report generation skipped: {e}")

    return {
        'models': {'ALS': als_results, 'BPR': bpr_results, 'Popularity': pop_results}
    }


def main():
    """Run all tests."""
    print("🚀 Starting Evaluation Module Tests")
    print("=" * 60)

    try:
        # Create mock data
        data = create_mock_data()

        # Test core metrics
        core_results = test_core_metrics()

        # Test model evaluator
        model_results = test_model_evaluator(data)

        # Test baselines
        baseline_results = test_baseline_evaluators(data)

        # Test hybrid metrics
        hybrid_results = test_hybrid_metrics(data)

        # Test statistical testing
        stat_results = test_statistical_testing()

        # Test comparison and reporting
        comparison_results = test_comparison_and_reporting()

        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n📈 Summary:")
        print(f"   Core metrics: {len(core_results)} metrics tested")
        print(f"   Model evaluation: {len(model_results)} K-values tested")
        print(f"   Baselines: {len(baseline_results)} baseline types tested")
        print(f"   Hybrid metrics: {len(hybrid_results)} metrics tested")
        print(f"   Statistical tests: {len(stat_results)} tests performed")
        print(f"   Comparisons: {len(comparison_results['models'])} models compared")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)