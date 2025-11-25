"""
Statistical Testing Module for CF Evaluation.

This module provides statistical significance testing utilities:
- Paired t-tests
- Wilcoxon signed-rank tests
- Effect size calculations (Cohen's d)
- Multiple comparison corrections (Bonferroni, Holm-Bonferroni)

Example:
    >>> from recsys.cf.evaluation.statistical_tests import StatisticalTester
    >>> tester = StatisticalTester()
    >>> result = tester.paired_t_test(model_scores, baseline_scores)
    >>> print(f"p-value: {result['p_value']:.4f}, significant: {result['significant']}")
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Statistical Tester
# ============================================================================

class StatisticalTester:
    """
    Statistical significance testing for model comparisons.
    
    Provides:
    - Paired t-tests
    - Wilcoxon signed-rank tests (non-parametric)
    - Effect size (Cohen's d, Glass's delta)
    - Confidence intervals
    - Multiple comparison corrections
    
    Example:
        >>> tester = StatisticalTester(significance_level=0.05)
        >>> result = tester.paired_t_test(model_scores, baseline_scores)
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical tester.
        
        Args:
            significance_level: P-value threshold for significance (default: 0.05)
        """
        self.significance_level = significance_level
    
    def paired_t_test(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """
        Perform paired t-test.
        
        Tests the null hypothesis that the mean difference between paired samples is zero.
        
        Args:
            sample1: First sample (e.g., model per-user metrics)
            sample2: Second sample (e.g., baseline per-user metrics)
            alternative: 'two-sided', 'greater', or 'less'
        
        Returns:
            Dict with t-statistic, p-value, and significance
        """
        sample1 = np.asarray(sample1)
        sample2 = np.asarray(sample2)
        
        if len(sample1) != len(sample2):
            raise ValueError("Samples must have same length for paired test")
        
        if len(sample1) < 3:
            logger.warning("Sample size too small for reliable t-test")
            return {
                't_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'n_samples': len(sample1),
                'warning': 'Sample size too small'
            }
        
        t_stat, p_value = stats.ttest_rel(sample1, sample2, alternative=alternative)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level,
            'n_samples': len(sample1),
            'mean_diff': float(np.mean(sample1 - sample2)),
            'std_diff': float(np.std(sample1 - sample2, ddof=1))
        }
    
    def wilcoxon_test(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
        
        More robust when data is not normally distributed.
        
        Args:
            sample1: First sample
            sample2: Second sample
            alternative: 'two-sided', 'greater', or 'less'
        
        Returns:
            Dict with statistic, p-value, and significance
        """
        sample1 = np.asarray(sample1)
        sample2 = np.asarray(sample2)
        
        if len(sample1) != len(sample2):
            raise ValueError("Samples must have same length")
        
        # Remove zero differences (Wilcoxon requirement)
        diff = sample1 - sample2
        non_zero_mask = diff != 0
        
        if np.sum(non_zero_mask) < 10:
            logger.warning("Too few non-zero differences for Wilcoxon test")
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'n_samples': len(sample1),
                'n_non_zero': int(np.sum(non_zero_mask)),
                'warning': 'Too few non-zero differences'
            }
        
        stat, p_value = stats.wilcoxon(
            sample1[non_zero_mask], 
            sample2[non_zero_mask],
            alternative=alternative
        )
        
        return {
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < self.significance_level,
            'n_samples': len(sample1),
            'n_non_zero': int(np.sum(non_zero_mask))
        }
    
    def cohens_d(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        paired: bool = True
    ) -> Dict[str, Any]:
        """
        Compute Cohen's d effect size.
        
        Interpretation:
        - d < 0.2: negligible
        - 0.2 <= d < 0.5: small
        - 0.5 <= d < 0.8: medium
        - d >= 0.8: large
        
        Args:
            sample1: First sample
            sample2: Second sample
            paired: Whether samples are paired
        
        Returns:
            Dict with effect size and interpretation
        """
        sample1 = np.asarray(sample1)
        sample2 = np.asarray(sample2)
        
        mean_diff = np.mean(sample1) - np.mean(sample2)
        
        if paired:
            # For paired samples, use std of differences
            std = np.std(sample1 - sample2, ddof=1)
        else:
            # Pooled standard deviation
            n1, n2 = len(sample1), len(sample2)
            s1, s2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
            std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        if std == 0:
            d = 0.0 if mean_diff == 0 else np.inf
        else:
            d = mean_diff / std
        
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
            'interpretation': interpretation,
            'mean_diff': float(mean_diff),
            'std': float(std)
        }
    
    def glass_delta(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute Glass's delta effect size.
        
        Uses standard deviation of control group (sample2) as denominator.
        Useful when sample variances are unequal.
        
        Args:
            sample1: Treatment sample
            sample2: Control sample (std used as denominator)
        
        Returns:
            Dict with effect size and interpretation
        """
        sample1 = np.asarray(sample1)
        sample2 = np.asarray(sample2)
        
        mean_diff = np.mean(sample1) - np.mean(sample2)
        std_control = np.std(sample2, ddof=1)
        
        if std_control == 0:
            delta = 0.0 if mean_diff == 0 else np.inf
        else:
            delta = mean_diff / std_control
        
        # Same interpretation as Cohen's d
        abs_delta = abs(delta)
        if abs_delta < 0.2:
            interpretation = 'negligible'
        elif abs_delta < 0.5:
            interpretation = 'small'
        elif abs_delta < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        return {
            'delta': float(delta),
            'interpretation': interpretation,
            'mean_diff': float(mean_diff),
            'std_control': float(std_control)
        }
    
    def confidence_interval(
        self,
        sample: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute confidence interval for sample mean.
        
        Args:
            sample: Data sample
            confidence: Confidence level (default: 0.95)
        
        Returns:
            Dict with mean, lower, upper bounds
        """
        sample = np.asarray(sample)
        n = len(sample)
        mean = np.mean(sample)
        std_err = np.std(sample, ddof=1) / np.sqrt(n)
        
        # t-critical value
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        
        margin = t_crit * std_err
        
        return {
            'mean': float(mean),
            'lower': float(mean - margin),
            'upper': float(mean + margin),
            'confidence': confidence,
            'std_err': float(std_err)
        }
    
    def bonferroni_correction(
        self,
        p_values: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Most conservative correction method.
        
        Args:
            p_values: List of raw p-values
        
        Returns:
            List of dicts with corrected p-values and significance
        """
        n_tests = len(p_values)
        corrected_alpha = self.significance_level / n_tests
        
        results = []
        for i, p in enumerate(p_values):
            corrected_p = min(p * n_tests, 1.0)
            results.append({
                'original_p': p,
                'corrected_p': corrected_p,
                'significant': p < corrected_alpha,
                'corrected_alpha': corrected_alpha
            })
        
        return results
    
    def holm_bonferroni_correction(
        self,
        p_values: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Apply Holm-Bonferroni (step-down) correction.
        
        Less conservative than Bonferroni, more powerful.
        
        Args:
            p_values: List of raw p-values
        
        Returns:
            List of dicts with corrected p-values and significance
        """
        n_tests = len(p_values)
        
        # Sort p-values with original indices
        sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
        
        results = [None] * n_tests
        rejected_so_far = True
        
        for rank, (orig_idx, p) in enumerate(sorted_pairs):
            adjusted_alpha = self.significance_level / (n_tests - rank)
            
            if rejected_so_far and p < adjusted_alpha:
                significant = True
            else:
                significant = False
                rejected_so_far = False
            
            # Adjusted p-value
            corrected_p = min(p * (n_tests - rank), 1.0)
            
            results[orig_idx] = {
                'original_p': p,
                'corrected_p': corrected_p,
                'significant': significant,
                'rank': rank + 1,
                'adjusted_alpha': adjusted_alpha
            }
        
        return results
    
    def normality_test(
        self,
        sample: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test if sample is normally distributed (Shapiro-Wilk test).
        
        Args:
            sample: Data sample
        
        Returns:
            Dict with test statistic, p-value, and normality conclusion
        """
        sample = np.asarray(sample)
        
        if len(sample) < 3:
            return {
                'statistic': np.nan,
                'p_value': np.nan,
                'is_normal': None,
                'warning': 'Sample too small'
            }
        
        if len(sample) > 5000:
            # Use subset for large samples
            sample = np.random.choice(sample, 5000, replace=False)
        
        stat, p_value = stats.shapiro(sample)
        
        return {
            'statistic': float(stat),
            'p_value': float(p_value),
            'is_normal': p_value >= self.significance_level,
            'n_samples': len(sample)
        }
    
    def compare_multiple_models(
        self,
        model_scores: Dict[str, np.ndarray],
        baseline_name: str,
        metric_name: str = 'metric'
    ) -> Dict[str, Dict]:
        """
        Compare multiple models against a baseline with proper corrections.
        
        Args:
            model_scores: Dict mapping model name to per-user scores
            baseline_name: Name of baseline model (must be in model_scores)
            metric_name: Name of metric for reporting
        
        Returns:
            Dict with pairwise comparisons and corrected significance
        """
        if baseline_name not in model_scores:
            raise ValueError(f"Baseline '{baseline_name}' not in model_scores")
        
        baseline_scores = model_scores[baseline_name]
        other_models = [k for k in model_scores if k != baseline_name]
        
        # Pairwise comparisons
        comparisons = {}
        p_values = []
        
        for model_name in other_models:
            scores = model_scores[model_name]
            
            # Paired t-test
            t_result = self.paired_t_test(scores, baseline_scores, alternative='greater')
            
            # Effect size
            effect = self.cohens_d(scores, baseline_scores, paired=True)
            
            comparisons[model_name] = {
                'mean_model': float(np.mean(scores)),
                'mean_baseline': float(np.mean(baseline_scores)),
                't_test': t_result,
                'effect_size': effect
            }
            
            p_values.append(t_result['p_value'])
        
        # Apply correction
        if len(p_values) > 1:
            corrected = self.holm_bonferroni_correction(p_values)
            for i, model_name in enumerate(other_models):
                comparisons[model_name]['corrected'] = corrected[i]
        
        return {
            'baseline': baseline_name,
            'metric': metric_name,
            'n_comparisons': len(other_models),
            'comparisons': comparisons
        }


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

class BootstrapEstimator:
    """
    Bootstrap resampling for confidence intervals and hypothesis testing.
    
    Example:
        >>> estimator = BootstrapEstimator(n_bootstrap=1000)
        >>> ci = estimator.confidence_interval(scores)
    """
    
    def __init__(self, n_bootstrap: int = 1000, random_state: Optional[int] = None):
        """
        Initialize bootstrap estimator.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_state)
    
    def confidence_interval(
        self,
        sample: np.ndarray,
        statistic_fn=np.mean,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval.
        
        Args:
            sample: Data sample
            statistic_fn: Function to compute statistic
            confidence: Confidence level
        
        Returns:
            Dict with observed, lower, upper bounds
        """
        sample = np.asarray(sample)
        n = len(sample)
        
        # Observed statistic
        observed = statistic_fn(sample)
        
        # Bootstrap samples
        bootstrap_stats = []
        for _ in range(self.n_bootstrap):
            bootstrap_sample = self.rng.choice(sample, size=n, replace=True)
            bootstrap_stats.append(statistic_fn(bootstrap_sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Percentile method
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
        
        return {
            'observed': float(observed),
            'lower': float(lower),
            'upper': float(upper),
            'confidence': confidence,
            'bootstrap_std': float(np.std(bootstrap_stats))
        }
    
    def permutation_test(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        statistic_fn=lambda x, y: np.mean(x) - np.mean(y),
        n_permutations: int = 10000
    ) -> Dict[str, Any]:
        """
        Perform permutation test for difference between groups.
        
        Args:
            sample1: First sample
            sample2: Second sample
            statistic_fn: Function to compute test statistic
            n_permutations: Number of permutations
        
        Returns:
            Dict with observed statistic, p-value
        """
        sample1 = np.asarray(sample1)
        sample2 = np.asarray(sample2)
        
        # Observed statistic
        observed = statistic_fn(sample1, sample2)
        
        # Combined samples
        combined = np.concatenate([sample1, sample2])
        n1 = len(sample1)
        
        # Permutation distribution
        perm_stats = []
        for _ in range(n_permutations):
            self.rng.shuffle(combined)
            perm_sample1 = combined[:n1]
            perm_sample2 = combined[n1:]
            perm_stats.append(statistic_fn(perm_sample1, perm_sample2))
        
        perm_stats = np.array(perm_stats)
        
        # P-value (two-sided)
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
        
        return {
            'observed': float(observed),
            'p_value': float(p_value),
            'n_permutations': n_permutations
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def paired_t_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Perform paired t-test.
    
    Args:
        sample1: First sample
        sample2: Second sample
        significance_level: P-value threshold
    
    Returns:
        Test results
    """
    tester = StatisticalTester(significance_level)
    return tester.paired_t_test(sample1, sample2)


def cohens_d(
    sample1: np.ndarray,
    sample2: np.ndarray,
    paired: bool = True
) -> Dict[str, Any]:
    """
    Compute Cohen's d effect size.
    
    Args:
        sample1: First sample
        sample2: Second sample
        paired: Whether samples are paired
    
    Returns:
        Effect size results
    """
    tester = StatisticalTester()
    return tester.cohens_d(sample1, sample2, paired)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Statistical Tests Module")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create mock data
    n_users = 100
    
    # Model performs slightly better than baseline
    baseline_scores = np.random.normal(0.15, 0.1, n_users)
    baseline_scores = np.clip(baseline_scores, 0, 1)
    
    model_a_scores = baseline_scores + np.random.normal(0.05, 0.03, n_users)
    model_a_scores = np.clip(model_a_scores, 0, 1)
    
    model_b_scores = baseline_scores + np.random.normal(0.08, 0.04, n_users)
    model_b_scores = np.clip(model_b_scores, 0, 1)
    
    print(f"Baseline mean: {np.mean(baseline_scores):.4f}")
    print(f"Model A mean: {np.mean(model_a_scores):.4f}")
    print(f"Model B mean: {np.mean(model_b_scores):.4f}")
    
    # Test StatisticalTester
    print("\n--- Statistical Tester ---")
    tester = StatisticalTester(significance_level=0.05)
    
    # Paired t-test
    t_result = tester.paired_t_test(model_a_scores, baseline_scores)
    print(f"\nPaired t-test (Model A vs Baseline):")
    print(f"  t-statistic: {t_result['t_statistic']:.4f}")
    print(f"  p-value: {t_result['p_value']:.4f}")
    print(f"  Significant: {t_result['significant']}")
    
    # Cohen's d
    effect = tester.cohens_d(model_a_scores, baseline_scores)
    print(f"\nCohen's d: {effect['d']:.4f} ({effect['interpretation']})")
    
    # Wilcoxon test
    wilcox_result = tester.wilcoxon_test(model_a_scores, baseline_scores)
    print(f"\nWilcoxon test p-value: {wilcox_result['p_value']:.4f}")
    
    # Confidence interval
    ci = tester.confidence_interval(model_a_scores)
    print(f"\n95% CI for Model A: [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    
    # Normality test
    normality = tester.normality_test(model_a_scores)
    print(f"\nNormality test p-value: {normality['p_value']:.4f} (is_normal: {normality['is_normal']})")
    
    # Multiple comparisons
    print("\n--- Multiple Comparisons ---")
    model_scores = {
        'baseline': baseline_scores,
        'model_a': model_a_scores,
        'model_b': model_b_scores
    }
    
    comparison = tester.compare_multiple_models(model_scores, 'baseline', 'recall@10')
    print(f"Baseline: {comparison['baseline']}")
    print(f"Number of comparisons: {comparison['n_comparisons']}")
    
    for model, result in comparison['comparisons'].items():
        print(f"\n{model}:")
        print(f"  Mean: {result['mean_model']:.4f} vs {result['mean_baseline']:.4f}")
        print(f"  p-value: {result['t_test']['p_value']:.4f}")
        print(f"  Effect: {result['effect_size']['d']:.4f} ({result['effect_size']['interpretation']})")
        if 'corrected' in result:
            print(f"  Corrected significant: {result['corrected']['significant']}")
    
    # Test Bootstrap
    print("\n--- Bootstrap Estimator ---")
    bootstrap = BootstrapEstimator(n_bootstrap=1000, random_state=42)
    
    boot_ci = bootstrap.confidence_interval(model_a_scores, np.mean, confidence=0.95)
    print(f"Bootstrap 95% CI: [{boot_ci['lower']:.4f}, {boot_ci['upper']:.4f}]")
    
    perm_result = bootstrap.permutation_test(model_a_scores, baseline_scores)
    print(f"Permutation test p-value: {perm_result['p_value']:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
