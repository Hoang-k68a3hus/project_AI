#!/usr/bin/env python
"""
Test Script for Task 08: Hybrid Reranking.

Tests:
1. HybridReranker initialization and config loading
2. Signal computation (CF, content, popularity, quality)
3. Global normalization
4. Diversity penalty
5. Attribute boosting
6. Integration with CFRecommender
7. Cold-start reranking
8. Config update

Usage:
    python scripts/test_hybrid_reranking.py
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a test."""
    name: str
    passed: bool
    message: str
    duration_ms: float = 0


class HybridRerankerTests:
    """Test suite for hybrid reranking system."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.reranker = None
        self.recommender = None
        self.phobert_loader = None
    
    def run_test(self, name: str, test_func):
        """Run a single test and record result."""
        import time
        start = time.perf_counter()
        
        try:
            test_func()
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(name=name, passed=True, message="OK", duration_ms=duration)
            logger.info(f"✅ {name}: PASSED ({duration:.1f}ms)")
        except AssertionError as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(name=name, passed=False, message=str(e), duration_ms=duration)
            logger.error(f"❌ {name}: FAILED - {e}")
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = TestResult(name=name, passed=False, message=f"Error: {e}", duration_ms=duration)
            logger.error(f"❌ {name}: ERROR - {e}")
        
        self.results.append(result)
    
    def test_imports(self):
        """Test that all modules can be imported."""
        from service.recommender.rerank import (
            HybridReranker, 
            get_reranker,
            RerankerConfig,
            RerankedResult
        )
        from service.recommender.filters import (
            boost_by_attributes,
            boost_by_user_preferences,
            infer_user_preferences,
            filter_and_boost
        )
        
        assert HybridReranker is not None
        assert get_reranker is not None
        assert RerankerConfig is not None
        assert RerankedResult is not None
        assert boost_by_attributes is not None
    
    def test_reranker_config(self):
        """Test RerankerConfig initialization."""
        from service.recommender.rerank import RerankerConfig
        
        config = RerankerConfig()
        
        # Check default weights
        assert 'cf' in config.weights_trainable
        assert 'content' in config.weights_trainable
        assert 'popularity' in config.weights_trainable
        assert 'quality' in config.weights_trainable
        
        # Check weights sum approximately to 1
        sum_trainable = sum(config.weights_trainable.values())
        assert abs(sum_trainable - 1.0) < 0.01, f"Trainable weights sum to {sum_trainable}"
        
        sum_cold_start = sum(config.weights_cold_start.values())
        assert abs(sum_cold_start - 1.0) < 0.01, f"Cold-start weights sum to {sum_cold_start}"
        
        # Check diversity settings
        assert config.diversity_enabled is True
        assert 0 <= config.diversity_penalty <= 1
        assert 0 <= config.diversity_threshold <= 1
    
    def test_reranker_initialization(self):
        """Test HybridReranker initialization."""
        from service.recommender.rerank import HybridReranker, get_reranker
        
        # Test singleton
        reranker1 = get_reranker()
        reranker2 = get_reranker()
        
        assert reranker1 is reranker2, "get_reranker should return singleton"
        
        # Store for later tests
        self.reranker = reranker1
        
        # Check config loaded
        assert self.reranker.config is not None
        assert hasattr(self.reranker.config, 'weights_trainable')
    
    def test_global_normalization(self):
        """Test global normalization with fixed ranges."""
        from service.recommender.rerank import HybridReranker, RerankerConfig
        
        config = RerankerConfig(
            cf_score_min=0.0,
            cf_score_max=1.5,
            popularity_p01=0.0,
            popularity_p99=6.0
        )
        
        reranker = HybridReranker(config=config)
        
        # Test CF normalization
        cf_scores = {1: 0.0, 2: 0.75, 3: 1.5, 4: 2.0}  # 4 is above max
        normalized = reranker._normalize_global(cf_scores, 'cf')
        
        assert abs(normalized[1] - 0.0) < 0.01, f"CF 0.0 should normalize to 0: {normalized[1]}"
        assert abs(normalized[2] - 0.5) < 0.01, f"CF 0.75 should normalize to 0.5: {normalized[2]}"
        assert abs(normalized[3] - 1.0) < 0.01, f"CF 1.5 should normalize to 1.0: {normalized[3]}"
        assert normalized[4] <= 1.0, f"CF 2.0 should be clipped to 1.0: {normalized[4]}"
        
        # Test popularity normalization
        pop_scores = {1: 0.0, 2: 3.0, 3: 6.0, 4: 10.0}  # 4 is above p99
        normalized_pop = reranker._normalize_global(pop_scores, 'popularity')
        
        assert abs(normalized_pop[1] - 0.0) < 0.01
        assert abs(normalized_pop[2] - 0.5) < 0.01
        assert abs(normalized_pop[3] - 1.0) < 0.01
        assert normalized_pop[4] <= 1.0
    
    def test_signal_computation(self):
        """Test signal computation for candidates."""
        if self.reranker is None:
            from service.recommender.rerank import get_reranker
            self.reranker = get_reranker()
        
        # Mock candidates
        candidate_ids = [1, 2, 3]
        cf_scores = {1: 0.8, 2: 0.6, 3: 0.4}
        user_history = []  # Empty for this test
        
        signals = self.reranker._compute_signals(candidate_ids, cf_scores, user_history)
        
        assert 'cf' in signals
        assert 'content' in signals
        assert 'popularity' in signals
        assert 'quality' in signals
        
        # Check CF scores are passed through
        assert signals['cf'][1] == 0.8
        assert signals['cf'][2] == 0.6
        assert signals['cf'][3] == 0.4
    
    def test_score_combination(self):
        """Test weighted score combination."""
        from service.recommender.rerank import HybridReranker, RerankerConfig
        
        config = RerankerConfig()
        reranker = HybridReranker(config=config)
        
        # Mock normalized signals
        normalized = {
            'cf': {1: 0.8, 2: 0.6},
            'content': {1: 0.5, 2: 0.9},
            'popularity': {1: 0.7, 2: 0.3},
            'quality': {1: 0.6, 2: 0.8}
        }
        
        weights = {'cf': 0.3, 'content': 0.4, 'popularity': 0.2, 'quality': 0.1}
        
        combined = reranker._combine_scores(normalized, weights)
        
        # Manual calculation for product 1
        expected_1 = 0.3 * 0.8 + 0.4 * 0.5 + 0.2 * 0.7 + 0.1 * 0.6
        assert abs(combined[1] - expected_1) < 0.01, f"Expected {expected_1}, got {combined[1]}"
        
        # Manual calculation for product 2
        expected_2 = 0.3 * 0.6 + 0.4 * 0.9 + 0.2 * 0.3 + 0.1 * 0.8
        assert abs(combined[2] - expected_2) < 0.01, f"Expected {expected_2}, got {combined[2]}"
    
    def test_rerank_method(self):
        """Test main rerank method."""
        from service.recommender.rerank import HybridReranker, RerankerConfig
        
        config = RerankerConfig(diversity_enabled=False)  # Disable for simpler test
        reranker = HybridReranker(config=config)
        
        # Mock recommendations
        recommendations = [
            {'product_id': 1, 'score': 0.9, 'rank': 1},
            {'product_id': 2, 'score': 0.7, 'rank': 2},
            {'product_id': 3, 'score': 0.5, 'rank': 3},
        ]
        
        result = reranker.rerank(
            cf_recommendations=recommendations,
            user_id=12345,
            user_history=[],
            topk=3
        )
        
        assert result is not None
        assert result.recommendations is not None
        assert len(result.recommendations) == 3
        assert result.num_candidates == 3
        assert result.latency_ms > 0
        
        # Check recommendations have final_score
        for rec in result.recommendations:
            assert 'final_score' in rec
            assert 'cf_score' in rec
            assert 'signals' in rec
    
    def test_attribute_boosting(self):
        """Test attribute boosting function."""
        from service.recommender.filters import boost_by_attributes
        import pandas as pd
        
        recommendations = [
            {'product_id': 1, 'score': 0.5, 'final_score': 0.5, 'brand': 'Innisfree'},
            {'product_id': 2, 'score': 0.6, 'final_score': 0.6, 'brand': 'Cetaphil'},
            {'product_id': 3, 'score': 0.7, 'final_score': 0.7, 'brand': 'Other'},
        ]
        
        boost_config = {
            'brand': {'Innisfree': 1.5, 'Cetaphil': 1.2}
        }
        
        boosted = boost_by_attributes(recommendations, boost_config)
        
        # Check boost factors applied
        for rec in boosted:
            if rec['product_id'] == 1:
                assert rec['boost_factor'] == 1.5
                assert rec['final_score'] == 0.5 * 1.5
            elif rec['product_id'] == 2:
                assert rec['boost_factor'] == 1.2
                assert rec['final_score'] == 0.6 * 1.2
            elif rec['product_id'] == 3:
                assert rec['boost_factor'] == 1.0
        
        # Innisfree should be first after boosting (0.5*1.5=0.75 > 0.6*1.2=0.72)
        assert boosted[0]['product_id'] == 1
    
    def test_infer_user_preferences(self):
        """Test user preference inference."""
        from service.recommender.filters import infer_user_preferences
        import pandas as pd
        
        metadata = pd.DataFrame({
            'product_id': [1, 2, 3, 4],
            'brand': ['Innisfree', 'Innisfree', 'Cetaphil', 'Other'],
            'skin_type_standardized': [['oily'], ['oily', 'acne'], ['dry'], ['normal']]
        })
        
        # User bought Innisfree products twice and Cetaphil once
        user_history = [1, 2, 3]
        
        prefs = infer_user_preferences(user_history, metadata)
        
        assert 'brand' in prefs
        assert prefs['brand'][0] == 'Innisfree'  # Most frequent
        
        assert 'skin_type_standardized' in prefs
        assert 'oily' in prefs['skin_type_standardized']
    
    def test_config_yaml_exists(self):
        """Test that config YAML file exists."""
        config_path = PROJECT_ROOT / 'service' / 'config' / 'rerank_config.yaml'
        
        assert config_path.exists(), f"Config file not found: {config_path}"
        
        # Try to load it
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        assert 'reranking' in config
        assert 'weights_trainable' in config['reranking']
        assert 'weights_cold_start' in config['reranking']
        assert 'diversity' in config['reranking']
    
    def test_config_update(self):
        """Test dynamic config update."""
        from service.recommender.rerank import HybridReranker, RerankerConfig
        
        reranker = HybridReranker()
        
        # Update weights
        new_weights = {'cf': 0.5, 'content': 0.3, 'popularity': 0.1, 'quality': 0.1}
        reranker.update_config(weights_trainable=new_weights)
        
        assert reranker.config.weights_trainable == new_weights
        
        # Update diversity
        reranker.update_config(diversity_enabled=False, diversity_penalty=0.2)
        
        assert reranker.config.diversity_enabled is False
        assert reranker.config.diversity_penalty == 0.2
    
    def test_cold_start_reranking(self):
        """Test cold-start reranking uses different weights."""
        from service.recommender.rerank import HybridReranker, RerankerConfig
        
        config = RerankerConfig(
            weights_trainable={'cf': 0.5, 'content': 0.3, 'popularity': 0.1, 'quality': 0.1},
            weights_cold_start={'content': 0.6, 'popularity': 0.3, 'quality': 0.1},
            diversity_enabled=False  # Disable for simpler test
        )
        
        reranker = HybridReranker(config=config)
        
        recommendations = [
            {'product_id': 1, 'score': 0.8},
            {'product_id': 2, 'score': 0.6},
        ]
        
        # Cold-start rerank with empty history (not None)
        result = reranker.rerank_cold_start(
            recommendations=recommendations,
            user_history=[],  # Empty list, not None
            topk=2
        )
        
        assert result is not None
        assert result.weights_used == config.weights_cold_start
        assert len(result.recommendations) == 2
    
    def test_filter_and_boost(self):
        """Test combined filter and boost."""
        from service.recommender.filters import filter_and_boost
        
        recommendations = [
            {'product_id': 1, 'score': 0.9, 'brand': 'Innisfree', 'price': 200000},
            {'product_id': 2, 'score': 0.8, 'brand': 'Cetaphil', 'price': 500000},
            {'product_id': 3, 'score': 0.7, 'brand': 'Innisfree', 'price': 600000},
        ]
        
        # Filter by price and boost by brand
        result = filter_and_boost(
            recommendations=recommendations,
            filter_params={'price_max': 550000},
            boost_config={'brand': {'Innisfree': 1.3}}
        )
        
        # Product 3 should be filtered out (price > 500000)
        assert len(result) == 2
        
        # Product 1 should be boosted
        assert result[0]['boost_factor'] == 1.3
    
    def run_all(self) -> bool:
        """Run all tests and return success status."""
        print("\n" + "=" * 60)
        print("TASK 08: HYBRID RERANKING TESTS")
        print("=" * 60 + "\n")
        
        tests = [
            ("Import modules", self.test_imports),
            ("RerankerConfig defaults", self.test_reranker_config),
            ("HybridReranker initialization", self.test_reranker_initialization),
            ("Global normalization", self.test_global_normalization),
            ("Signal computation", self.test_signal_computation),
            ("Score combination", self.test_score_combination),
            ("Rerank method", self.test_rerank_method),
            ("Attribute boosting", self.test_attribute_boosting),
            ("Infer user preferences", self.test_infer_user_preferences),
            ("Config YAML exists", self.test_config_yaml_exists),
            ("Config update", self.test_config_update),
            ("Cold-start reranking", self.test_cold_start_reranking),
            ("Filter and boost", self.test_filter_and_boost),
        ]
        
        for name, test_func in tests:
            self.run_test(name, test_func)
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 60)
        print(f"RESULTS: {passed}/{total} tests passed")
        print("=" * 60)
        
        if passed < total:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
        
        return passed == total


def main():
    """Main entry point."""
    tester = HybridRerankerTests()
    success = tester.run_all()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
