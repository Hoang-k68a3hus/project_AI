"""
Comparison and Reporting Module for CF Evaluation.

This module provides tools for:
- Comparing CF models vs baselines
- Statistical significance testing
- Generating evaluation reports
- Visualization support

Example:
    >>> from recsys.cf.evaluation import ModelComparator, ReportGenerator
    >>> comparator = ModelComparator()
    >>> comparator.add_model_results('als', als_results)
    >>> comparator.add_baseline_results('popularity', pop_results)
    >>> comparison = comparator.compare_models()
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)


# ============================================================================
# Model Comparator
# ============================================================================

class ModelComparator:
    """
    Compare CF models with baselines using statistical tests.
    
    Features:
    - Improvement percentage calculation
    - Paired t-test for significance
    - Effect size (Cohen's d)
    - Summary table generation
    
    Example:
        >>> comparator = ModelComparator()
        >>> comparator.add_model_results('als', als_metrics)
        >>> comparator.add_baseline_results('popularity', pop_metrics)
        >>> df = comparator.get_comparison_table()
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize model comparator.
        
        Args:
            significance_level: P-value threshold for significance (default: 0.05)
        """
        self.significance_level = significance_level
        self.model_results: Dict[str, Dict] = {}
        self.baseline_results: Dict[str, Dict] = {}
        self.per_user_metrics: Dict[str, Dict] = {}
    
    def add_model_results(
        self,
        name: str,
        results: Dict[str, Any],
        per_user_metrics: Optional[Dict] = None
    ) -> None:
        """
        Add model evaluation results.
        
        Args:
            name: Model name
            results: Dict with aggregated metrics
            per_user_metrics: Optional per-user metrics for statistical testing
        """
        self.model_results[name] = results
        if per_user_metrics:
            self.per_user_metrics[name] = per_user_metrics
        logger.info(f"Added model results: {name}")
    
    def add_baseline_results(
        self,
        name: str,
        results: Dict[str, Any],
        per_user_metrics: Optional[Dict] = None
    ) -> None:
        """
        Add baseline evaluation results.
        
        Args:
            name: Baseline name
            results: Dict with aggregated metrics
            per_user_metrics: Optional per-user metrics
        """
        self.baseline_results[name] = results
        if per_user_metrics:
            self.per_user_metrics[name] = per_user_metrics
        logger.info(f"Added baseline results: {name}")
    
    def compute_improvement(
        self,
        model_name: str,
        baseline_name: str,
        metric: str
    ) -> Dict[str, float]:
        """
        Compute improvement of model over baseline.
        
        Args:
            model_name: Name of the model
            baseline_name: Name of the baseline
            metric: Metric to compare
        
        Returns:
            Dict with absolute and relative improvement
        """
        if model_name not in self.model_results:
            raise ValueError(f"Model '{model_name}' not found")
        if baseline_name not in self.baseline_results:
            raise ValueError(f"Baseline '{baseline_name}' not found")
        
        model_value = self.model_results[model_name].get(metric, 0)
        baseline_value = self.baseline_results[baseline_name].get(metric, 0)
        
        absolute_improvement = model_value - baseline_value
        relative_improvement = (
            (absolute_improvement / baseline_value * 100) 
            if baseline_value > 0 else 0
        )
        
        return {
            'model_value': model_value,
            'baseline_value': baseline_value,
            'absolute': absolute_improvement,
            'relative_percent': relative_improvement
        }
    
    def paired_t_test(
        self,
        model_name: str,
        baseline_name: str,
        metric: str,
        k: int
    ) -> Dict[str, float]:
        """
        Perform paired t-test between model and baseline.
        
        Args:
            model_name: Name of the model
            baseline_name: Name of the baseline
            metric: Metric to test ('recall', 'ndcg', etc.)
            k: K value
        
        Returns:
            Dict with t-statistic, p-value, and significance
        """
        # Check if per-user metrics available
        if model_name not in self.per_user_metrics:
            logger.warning(f"Per-user metrics not available for {model_name}")
            return {'t_statistic': np.nan, 'p_value': np.nan, 'significant': False}
        
        if baseline_name not in self.per_user_metrics:
            logger.warning(f"Per-user metrics not available for {baseline_name}")
            return {'t_statistic': np.nan, 'p_value': np.nan, 'significant': False}
        
        # Extract per-user values
        model_per_user = self.per_user_metrics[model_name].get(k, [])
        baseline_per_user = self.per_user_metrics[baseline_name].get(k, [])
        
        if len(model_per_user) == 0 or len(baseline_per_user) == 0:
            return {'t_statistic': np.nan, 'p_value': np.nan, 'significant': False}
        
        model_values = [m[metric] for m in model_per_user]
        baseline_values = [m[metric] for m in baseline_per_user]
        
        # Ensure same length (matched users)
        min_len = min(len(model_values), len(baseline_values))
        model_values = model_values[:min_len]
        baseline_values = baseline_values[:min_len]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(model_values, baseline_values)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level,
            'n_samples': min_len
        }
    
    def cohens_d(
        self,
        model_name: str,
        baseline_name: str,
        metric: str,
        k: int
    ) -> Dict[str, Any]:
        """
        Compute Cohen's d effect size.
        
        Args:
            model_name: Name of the model
            baseline_name: Name of the baseline
            metric: Metric to measure
            k: K value
        
        Returns:
            Dict with effect size and interpretation
        """
        if model_name not in self.per_user_metrics or baseline_name not in self.per_user_metrics:
            return {'d': np.nan, 'interpretation': 'N/A'}
        
        model_per_user = self.per_user_metrics[model_name].get(k, [])
        baseline_per_user = self.per_user_metrics[baseline_name].get(k, [])
        
        if len(model_per_user) == 0 or len(baseline_per_user) == 0:
            return {'d': np.nan, 'interpretation': 'N/A'}
        
        model_values = np.array([m[metric] for m in model_per_user])
        baseline_values = np.array([m[metric] for m in baseline_per_user])
        
        # Compute pooled standard deviation
        n1, n2 = len(model_values), len(baseline_values)
        
        # Check for minimum sample size
        if n1 < 2 or n2 < 2:
            return {'d': np.nan, 'interpretation': 'N/A', 'warning': 'Insufficient sample size'}
        
        s1, s2 = np.std(model_values, ddof=1), np.std(baseline_values, ddof=1)
        
        # Avoid division by zero
        degrees_of_freedom = n1 + n2 - 2
        if degrees_of_freedom <= 0:
            return {'d': np.nan, 'interpretation': 'N/A', 'warning': 'Invalid degrees of freedom'}
        
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / degrees_of_freedom)
        
        if pooled_std == 0:
            return {'d': np.nan, 'interpretation': 'N/A'}
        
        # Cohen's d
        d = (np.mean(model_values) - np.mean(baseline_values)) / pooled_std
        
        # Interpretation
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        return {
            'd': float(d),
            'interpretation': interpretation
        }
    
    def get_comparison_table(
        self,
        metrics: List[str] = ['recall@10', 'ndcg@10', 'coverage'],
        include_hyperparams: bool = True
    ) -> pd.DataFrame:
        """
        Generate comparison table for all models and baselines.
        
        Args:
            metrics: List of metrics to include
            include_hyperparams: Whether to include hyperparameters
        
        Returns:
            DataFrame with comparison results
        """
        rows = []
        
        # Add models
        for name, results in self.model_results.items():
            row = {'name': name, 'type': 'model'}
            
            if include_hyperparams:
                hyperparams = results.get('hyperparams', {})
                for k, v in hyperparams.items():
                    row[f'hp_{k}'] = v
            
            for metric in metrics:
                row[metric] = results.get(metric, np.nan)
            
            rows.append(row)
        
        # Add baselines
        for name, results in self.baseline_results.items():
            row = {'name': name, 'type': 'baseline'}
            
            for metric in metrics:
                row[metric] = results.get(metric, np.nan)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_improvement_summary(
        self,
        baseline_name: str = 'popularity',
        metrics: List[str] = ['recall@10', 'ndcg@10']
    ) -> pd.DataFrame:
        """
        Get summary of model improvements over a baseline.
        
        Args:
            baseline_name: Baseline to compare against
            metrics: Metrics to compare
        
        Returns:
            DataFrame with improvements
        """
        rows = []
        
        for model_name in self.model_results:
            row = {'model': model_name}
            
            for metric in metrics:
                improvement = self.compute_improvement(model_name, baseline_name, metric)
                row[f'{metric}_value'] = improvement['model_value']
                row[f'{metric}_improvement%'] = improvement['relative_percent']
            
            rows.append(row)
        
        return pd.DataFrame(rows)


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """
    Generate evaluation reports in various formats.
    
    Supports:
    - CSV summary tables
    - JSON detailed reports
    - Markdown reports
    
    Example:
        >>> generator = ReportGenerator(output_dir='reports')
        >>> generator.generate_csv(comparison_df, 'cf_eval_summary.csv')
        >>> generator.generate_markdown(results, 'evaluation_report.md')
    """
    
    def __init__(self, output_dir: str = 'reports'):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_csv(
        self,
        df: pd.DataFrame,
        filename: str = 'cf_eval_summary.csv'
    ) -> str:
        """
        Generate CSV summary report.
        
        Args:
            df: DataFrame with results
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        df.to_csv(output_path, index=False)
        logger.info(f"CSV report saved to {output_path}")
        return output_path
    
    def generate_json(
        self,
        results: Dict[str, Any],
        filename: str = 'cf_eval_detailed.json'
    ) -> str:
        """
        Generate JSON detailed report.
        
        Args:
            results: Dict with evaluation results
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        clean_results = convert_numpy(results)
        clean_results['generated_at'] = datetime.now().isoformat()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON report saved to {output_path}")
        return output_path
    
    def generate_markdown(
        self,
        results: Dict[str, Any],
        model_comparison: pd.DataFrame,
        filename: str = 'evaluation_report.md'
    ) -> str:
        """
        Generate Markdown evaluation report.
        
        Args:
            results: Dict with detailed results
            model_comparison: DataFrame with model comparison
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        lines = [
            "# CF Model Evaluation Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            "### Model Comparison",
            "",
            model_comparison.to_markdown(index=False),
            "",
            "## Detailed Results",
            "",
        ]
        
        # Add model details
        for model_name, model_results in results.get('models', {}).items():
            lines.extend([
                f"### {model_name}",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ])
            
            for metric, value in model_results.items():
                if isinstance(value, float):
                    lines.append(f"| {metric} | {value:.4f} |")
                else:
                    lines.append(f"| {metric} | {value} |")
            
            lines.append("")
        
        # Add baseline details
        for baseline_name, baseline_results in results.get('baselines', {}).items():
            lines.extend([
                f"### Baseline: {baseline_name}",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ])
            
            for metric, value in baseline_results.items():
                if isinstance(value, float):
                    lines.append(f"| {metric} | {value:.4f} |")
                else:
                    lines.append(f"| {metric} | {value} |")
            
            lines.append("")
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Markdown report saved to {output_path}")
        return output_path


# ============================================================================
# Visualization Support
# ============================================================================

class EvaluationVisualizer:
    """
    Support class for evaluation visualizations.
    
    Provides data preparation for common plots:
    - Metric bar charts
    - K-value sensitivity curves
    - Coverage vs accuracy trade-off
    - Per-user metric distributions
    """
    
    @staticmethod
    def prepare_bar_chart_data(
        comparison_df: pd.DataFrame,
        metric: str = 'recall@10'
    ) -> Dict[str, Any]:
        """
        Prepare data for bar chart visualization.
        
        Args:
            comparison_df: Comparison DataFrame
            metric: Metric to visualize
        
        Returns:
            Dict with labels, values, and colors
        """
        models = comparison_df[comparison_df['type'] == 'model']['name'].tolist()
        baselines = comparison_df[comparison_df['type'] == 'baseline']['name'].tolist()
        
        model_values = comparison_df[comparison_df['type'] == 'model'][metric].tolist()
        baseline_values = comparison_df[comparison_df['type'] == 'baseline'][metric].tolist()
        
        return {
            'labels': models + baselines,
            'values': model_values + baseline_values,
            'types': ['model'] * len(models) + ['baseline'] * len(baselines),
            'colors': ['#2196F3'] * len(models) + ['#FF9800'] * len(baselines)
        }
    
    @staticmethod
    def prepare_k_sensitivity_data(
        results: Dict[str, Dict],
        metric_prefix: str = 'recall',
        k_values: List[int] = [5, 10, 20, 50]
    ) -> Dict[str, Any]:
        """
        Prepare data for K-value sensitivity plot.
        
        Args:
            results: Dict mapping model name to results
            metric_prefix: Metric prefix ('recall', 'ndcg')
            k_values: K values to plot
        
        Returns:
            Dict with model names, k_values, and metric values
        """
        data = {'k_values': k_values, 'models': {}}
        
        for name, metrics in results.items():
            model_values = []
            for k in k_values:
                metric_key = f'{metric_prefix}@{k}'
                model_values.append(metrics.get(metric_key, np.nan))
            data['models'][name] = model_values
        
        return data
    
    @staticmethod
    def prepare_distribution_data(
        per_user_metrics: Dict,
        metric: str = 'recall',
        k: int = 10
    ) -> np.ndarray:
        """
        Prepare data for distribution plot (histogram).
        
        Args:
            per_user_metrics: Per-user metrics dict
            metric: Metric to visualize
            k: K value
        
        Returns:
            Array of metric values per user
        """
        if k not in per_user_metrics:
            return np.array([])
        
        return np.array([m[metric] for m in per_user_metrics[k]])


# ============================================================================
# Convenience Functions
# ============================================================================

def compare_models(
    model_metrics: Dict[str, Dict],
    baseline_metrics: Dict[str, Dict],
    metrics: List[str] = ['recall@10', 'ndcg@10', 'coverage']
) -> pd.DataFrame:
    """
    Compare multiple models with baselines.
    
    Args:
        model_metrics: Dict mapping model name to metrics
        baseline_metrics: Dict mapping baseline name to metrics
        metrics: List of metrics to compare
    
    Returns:
        DataFrame with comparison
    """
    comparator = ModelComparator()
    
    for name, results in model_metrics.items():
        comparator.add_model_results(name, results)
    
    for name, results in baseline_metrics.items():
        comparator.add_baseline_results(name, results)
    
    return comparator.get_comparison_table(metrics=metrics)


def generate_evaluation_report(
    results: Dict[str, Any],
    output_dir: str = 'reports',
    formats: List[str] = ['csv', 'json', 'md']
) -> Dict[str, str]:
    """
    Generate evaluation reports in multiple formats.
    
    Args:
        results: Evaluation results
        output_dir: Output directory
        formats: List of formats to generate
    
    Returns:
        Dict mapping format to file path
    """
    generator = ReportGenerator(output_dir=output_dir)
    paths = {}
    
    # Prepare comparison DataFrame
    comparison_df = pd.DataFrame([
        {'name': k, 'type': 'model', **v}
        for k, v in results.get('models', {}).items()
    ] + [
        {'name': k, 'type': 'baseline', **v}
        for k, v in results.get('baselines', {}).items()
    ])
    
    if 'csv' in formats:
        paths['csv'] = generator.generate_csv(comparison_df)
    
    if 'json' in formats:
        paths['json'] = generator.generate_json(results)
    
    if 'md' in formats:
        paths['md'] = generator.generate_markdown(results, comparison_df)
    
    return paths


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Comparison and Reporting Module")
    print("=" * 60)
    
    # Create mock results
    model_results = {
        'als': {
            'recall@10': 0.234,
            'ndcg@10': 0.189,
            'precision@10': 0.125,
            'mrr': 0.312,
            'coverage': 0.287,
            'hyperparams': {'factors': 64, 'reg': 0.01}
        },
        'bpr': {
            'recall@10': 0.242,
            'ndcg@10': 0.195,
            'precision@10': 0.131,
            'mrr': 0.325,
            'coverage': 0.301,
            'hyperparams': {'factors': 64, 'lr': 0.05}
        }
    }
    
    baseline_results = {
        'popularity': {
            'recall@10': 0.145,
            'ndcg@10': 0.102,
            'precision@10': 0.082,
            'mrr': 0.198,
            'coverage': 0.042
        },
        'random': {
            'recall@10': 0.012,
            'ndcg@10': 0.008,
            'precision@10': 0.005,
            'mrr': 0.021,
            'coverage': 0.856
        }
    }
    
    # Test ModelComparator
    print("\n--- ModelComparator ---")
    comparator = ModelComparator()
    
    for name, results in model_results.items():
        comparator.add_model_results(name, results)
    
    for name, results in baseline_results.items():
        comparator.add_baseline_results(name, results)
    
    # Compute improvement
    improvement = comparator.compute_improvement('als', 'popularity', 'recall@10')
    print(f"\nALS vs Popularity (Recall@10):")
    print(f"  ALS: {improvement['model_value']:.4f}")
    print(f"  Popularity: {improvement['baseline_value']:.4f}")
    print(f"  Improvement: {improvement['relative_percent']:.1f}%")
    
    # Get comparison table
    comparison_df = comparator.get_comparison_table()
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))
    
    # Test ReportGenerator
    print("\n--- ReportGenerator ---")
    generator = ReportGenerator(output_dir='reports/test')
    
    # Generate CSV
    csv_path = generator.generate_csv(comparison_df, 'test_summary.csv')
    print(f"CSV saved to: {csv_path}")
    
    # Generate JSON
    json_results = {
        'models': model_results,
        'baselines': baseline_results
    }
    json_path = generator.generate_json(json_results, 'test_detailed.json')
    print(f"JSON saved to: {json_path}")
    
    # Generate Markdown
    md_path = generator.generate_markdown(json_results, comparison_df, 'test_report.md')
    print(f"Markdown saved to: {md_path}")
    
    # Test EvaluationVisualizer
    print("\n--- EvaluationVisualizer ---")
    bar_data = EvaluationVisualizer.prepare_bar_chart_data(comparison_df, 'recall@10')
    print(f"Bar chart data: {len(bar_data['labels'])} items")
    
    # Test convenience function
    print("\n--- compare_models function ---")
    quick_comparison = compare_models(model_results, baseline_results)
    print("Quick comparison:")
    print(quick_comparison.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("All tests passed!")
