"""
Deployment Optimization Report Generator.

This script analyzes and generates a comprehensive report on
deployment optimization, including cache analysis, latency profiling,
and recommendations for cold-start path optimization.

Usage:
    python scripts/deployment_optimizer.py --analyze
    python scripts/deployment_optimizer.py --profile
    python scripts/deployment_optimizer.py --report
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LatencyProfile:
    """Latency profile for a component."""
    component: str
    mean_ms: float
    median_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    samples: int


@dataclass
class OptimizationReport:
    """Comprehensive optimization report."""
    timestamp: str
    summary: Dict[str, Any]
    cache_analysis: Dict[str, Any]
    latency_profiles: List[LatencyProfile]
    recommendations: List[str]
    cold_start_analysis: Dict[str, Any]
    targets: Dict[str, Any]


# ============================================================================
# Latency Profiler
# ============================================================================

class LatencyProfiler:
    """Profile latency of different recommendation components."""
    
    def __init__(self):
        self.samples: Dict[str, List[float]] = {}
    
    def profile_component(
        self,
        name: str,
        func,
        iterations: int = 100,
        warmup: int = 10,
        **kwargs
    ) -> LatencyProfile:
        """
        Profile a component's latency.
        
        Args:
            name: Component name
            func: Function to profile
            iterations: Number of iterations
            warmup: Warmup iterations
            **kwargs: Arguments for func
        
        Returns:
            LatencyProfile with statistics
        """
        import numpy as np
        
        # Warmup
        for _ in range(warmup):
            try:
                func(**kwargs)
            except Exception:
                pass
        
        # Profile
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                func(**kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
            except Exception as e:
                logger.warning(f"Error in {name}: {e}")
        
        if not latencies:
            return LatencyProfile(
                component=name,
                mean_ms=0, median_ms=0, p90_ms=0, p99_ms=0,
                min_ms=0, max_ms=0, samples=0
            )
        
        latencies_arr = np.array(latencies)
        
        return LatencyProfile(
            component=name,
            mean_ms=float(np.mean(latencies_arr)),
            median_ms=float(np.median(latencies_arr)),
            p90_ms=float(np.percentile(latencies_arr, 90)),
            p99_ms=float(np.percentile(latencies_arr, 99)),
            min_ms=float(np.min(latencies_arr)),
            max_ms=float(np.max(latencies_arr)),
            samples=len(latencies)
        )
    
    def profile_recommendation_path(self, num_samples: int = 50) -> Dict[str, LatencyProfile]:
        """Profile the full recommendation path."""
        try:
            from service.recommender import CFRecommender
            from service.recommender.loader import get_loader
            from service.recommender.phobert_loader import get_phobert_loader
            from service.recommender.fallback import FallbackRecommender
            from service.recommender.cache import get_cache_manager
            import json
            import random
            
            logger.info("Loading components for profiling...")
            
            # Initialize components
            recommender = CFRecommender(auto_load=True)
            loader = get_loader()
            phobert = get_phobert_loader()
            cache = get_cache_manager()
            
            # Get sample users
            with open('data/processed/user_item_mappings.json', 'r') as f:
                mappings = json.load(f)
            
            all_users = [int(uid) for uid in mappings['user_to_idx'].keys()]
            trainable_users = list(loader.trainable_user_set or set())
            coldstart_users = [u for u in all_users[:10000] if u not in (loader.trainable_user_set or set())]
            
            profiles = {}
            
            # Profile CF path
            if trainable_users:
                sample_cf_users = random.sample(trainable_users, min(num_samples, len(trainable_users)))
                
                def cf_recommend():
                    uid = random.choice(sample_cf_users)
                    return recommender.recommend(uid, topk=10, rerank=False)
                
                profiles['cf_path'] = self.profile_component(
                    'cf_path',
                    cf_recommend,
                    iterations=num_samples
                )
                logger.info(f"CF Path: mean={profiles['cf_path'].mean_ms:.1f}ms")
            
            # Profile cold-start path
            if coldstart_users:
                sample_cs_users = random.sample(coldstart_users, min(num_samples, len(coldstart_users)))
                
                def coldstart_recommend():
                    uid = random.choice(sample_cs_users)
                    return recommender.recommend(uid, topk=10, rerank=False)
                
                profiles['coldstart_path'] = self.profile_component(
                    'coldstart_path',
                    coldstart_recommend,
                    iterations=num_samples
                )
                logger.info(f"Cold-Start Path: mean={profiles['coldstart_path'].mean_ms:.1f}ms")
            
            # Profile with reranking
            def cf_rerank_recommend():
                uid = random.choice(sample_cf_users)
                return recommender.recommend(uid, topk=10, rerank=True)
            
            if trainable_users:
                profiles['cf_path_rerank'] = self.profile_component(
                    'cf_path_rerank',
                    cf_rerank_recommend,
                    iterations=num_samples
                )
                logger.info(f"CF Path (Rerank): mean={profiles['cf_path_rerank'].mean_ms:.1f}ms")
            
            # Profile PhoBERT similarity
            if phobert.is_loaded():
                sample_items = list(phobert.product_id_to_idx.keys())[:100]
                
                def phobert_similar():
                    pid = random.choice(sample_items)
                    return phobert.find_similar_items(pid, topk=50)
                
                profiles['phobert_similarity'] = self.profile_component(
                    'phobert_similarity',
                    phobert_similar,
                    iterations=num_samples
                )
                logger.info(f"PhoBERT Similarity: mean={profiles['phobert_similarity'].mean_ms:.1f}ms")
            
            # Profile user profile computation
            def compute_profile():
                history = random.sample(sample_items, min(3, len(sample_items)))
                return phobert.compute_user_profile(history)
            
            if phobert.is_loaded():
                profiles['user_profile_compute'] = self.profile_component(
                    'user_profile_compute',
                    compute_profile,
                    iterations=num_samples
                )
                logger.info(f"User Profile Compute: mean={profiles['user_profile_compute'].mean_ms:.1f}ms")
            
            return profiles
            
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
            import traceback
            traceback.print_exc()
            return {}


# ============================================================================
# Optimization Analyzer
# ============================================================================

class OptimizationAnalyzer:
    """Analyze and generate optimization recommendations."""
    
    def __init__(self):
        self.profiler = LatencyProfiler()
    
    def analyze_cache_effectiveness(self) -> Dict[str, Any]:
        """Analyze cache hit rates and effectiveness."""
        try:
            from service.recommender.cache import get_cache_manager
            
            cache = get_cache_manager()
            stats = cache.get_stats()
            
            analysis = {
                'warmed_up': stats.get('warmed_up', False),
                'caches': {}
            }
            
            for cache_name, cache_stats in stats.get('caches', {}).items():
                hit_rate = cache_stats.get('hit_rate', 0)
                analysis['caches'][cache_name] = {
                    'size': cache_stats.get('size', 0),
                    'max_size': cache_stats.get('max_size', 0),
                    'hit_rate': hit_rate,
                    'hits': cache_stats.get('hits', 0),
                    'misses': cache_stats.get('misses', 0),
                    'status': 'good' if hit_rate > 0.7 else 'needs_improvement' if hit_rate > 0.3 else 'poor'
                }
            
            # Check precomputed data
            precomputed = stats.get('precomputed', {})
            analysis['precomputed'] = {
                'popular_items': precomputed.get('popular_items', 0),
                'popular_similarities': precomputed.get('popular_similarities', 0),
                'status': 'ready' if precomputed.get('popular_items', 0) > 0 else 'not_ready'
            }
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Cache analysis failed: {e}")
            return {'error': str(e)}
    
    def analyze_cold_start_path(self) -> Dict[str, Any]:
        """Analyze cold-start path optimizations."""
        try:
            from service.recommender.loader import get_loader
            from service.recommender.phobert_loader import get_phobert_loader
            
            loader = get_loader()
            phobert = get_phobert_loader()
            
            # Calculate traffic split
            total_users = loader.mappings['metadata']['num_users'] if loader.mappings else 0
            trainable_users = len(loader.trainable_user_set or set())
            coldstart_users = total_users - trainable_users
            
            coldstart_pct = coldstart_users / max(total_users, 1) * 100
            
            analysis = {
                'total_users': total_users,
                'trainable_users': trainable_users,
                'coldstart_users': coldstart_users,
                'coldstart_percentage': coldstart_pct,
                'traffic_warning': coldstart_pct > 90,
                'phobert_loaded': phobert.is_loaded() if phobert else False,
                'phobert_items': phobert.num_products if phobert and phobert.is_loaded() else 0,
                'similarity_precomputed': False
            }
            
            # Check if item-item similarity is precomputed
            if phobert and phobert._similarity_computed:
                analysis['similarity_precomputed'] = True
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Cold-start analysis failed: {e}")
            return {'error': str(e)}
    
    def generate_recommendations(
        self,
        cache_analysis: Dict[str, Any],
        cold_start_analysis: Dict[str, Any],
        latency_profiles: Dict[str, LatencyProfile]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Cache recommendations
        if not cache_analysis.get('warmed_up', False):
            recommendations.append(
                "⚠️ CRITICAL: Cache not warmed up. Call /cache_warmup endpoint or ensure "
                "warmup runs during service startup."
            )
        
        for cache_name, stats in cache_analysis.get('caches', {}).items():
            if stats.get('status') == 'poor':
                recommendations.append(
                    f"🔴 {cache_name} cache has low hit rate ({stats['hit_rate']*100:.1f}%). "
                    f"Consider increasing cache size or adjusting TTL."
                )
            elif stats.get('status') == 'needs_improvement':
                recommendations.append(
                    f"🟡 {cache_name} cache hit rate ({stats['hit_rate']*100:.1f}%) could be improved."
                )
        
        # Cold-start recommendations
        cs = cold_start_analysis
        if cs.get('traffic_warning'):
            recommendations.append(
                f"⚠️ Cold-start path handles {cs['coldstart_percentage']:.1f}% of traffic. "
                "Ensure PhoBERT embeddings are loaded and popular items are pre-computed."
            )
        
        if not cs.get('phobert_loaded'):
            recommendations.append(
                "🔴 CRITICAL: PhoBERT embeddings not loaded. Cold-start recommendations "
                "will fall back to popularity-only, reducing quality."
            )
        
        if cs.get('phobert_loaded') and not cs.get('similarity_precomputed'):
            recommendations.append(
                "🟡 Item-item similarity matrix not pre-computed. Consider calling "
                "phobert_loader.precompute_item_similarity() for faster similar item lookups."
            )
        
        # Latency recommendations
        target_latency = 200  # ms
        
        for name, profile in latency_profiles.items():
            if profile.p99_ms > target_latency:
                recommendations.append(
                    f"🔴 {name} P99 latency ({profile.p99_ms:.1f}ms) exceeds {target_latency}ms target. "
                    f"Mean: {profile.mean_ms:.1f}ms, Max: {profile.max_ms:.1f}ms"
                )
            elif profile.p90_ms > target_latency * 0.8:
                recommendations.append(
                    f"🟡 {name} P90 latency ({profile.p90_ms:.1f}ms) approaching {target_latency}ms target."
                )
        
        # Specific optimizations
        if 'coldstart_path' in latency_profiles:
            cs_profile = latency_profiles['coldstart_path']
            if cs_profile.mean_ms > 150:
                recommendations.append(
                    "💡 Cold-start path is slow. Consider:\n"
                    "   - Pre-computing user profiles for frequent users\n"
                    "   - Caching fallback results\n"
                    "   - Using approximate nearest neighbor search for PhoBERT similarity"
                )
        
        if 'phobert_similarity' in latency_profiles:
            phobert_profile = latency_profiles['phobert_similarity']
            if phobert_profile.mean_ms > 50:
                recommendations.append(
                    "💡 PhoBERT similarity computation is slow. Consider:\n"
                    "   - Pre-computing V @ V.T matrix\n"
                    "   - Using FAISS for approximate nearest neighbor\n"
                    "   - Reducing embedding dimension with PCA"
                )
        
        if not recommendations:
            recommendations.append("✅ All optimizations look good! Latency targets are met.")
        
        return recommendations
    
    def generate_report(self, output_path: Optional[str] = None) -> OptimizationReport:
        """Generate comprehensive optimization report."""
        logger.info("Generating optimization report...")
        
        # Profile components
        logger.info("Profiling recommendation components...")
        latency_profiles = self.profiler.profile_recommendation_path(num_samples=30)
        
        # Analyze cache
        logger.info("Analyzing cache effectiveness...")
        cache_analysis = self.analyze_cache_effectiveness()
        
        # Analyze cold-start path
        logger.info("Analyzing cold-start path...")
        cold_start_analysis = self.analyze_cold_start_path()
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            cache_analysis,
            cold_start_analysis,
            latency_profiles
        )
        
        # Create report
        report = OptimizationReport(
            timestamp=datetime.now().isoformat(),
            summary={
                'status': 'needs_attention' if any('🔴' in r for r in recommendations) else 'good',
                'num_recommendations': len(recommendations),
                'critical_issues': sum(1 for r in recommendations if '🔴' in r or 'CRITICAL' in r)
            },
            cache_analysis=cache_analysis,
            latency_profiles=[asdict(p) for p in latency_profiles.values()],
            recommendations=recommendations,
            cold_start_analysis=cold_start_analysis,
            targets={
                'latency_p99_ms': 200,
                'cache_hit_rate': 0.7,
                'compliance_target_pct': 95
            }
        )
        
        # Save report
        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            logger.info(f"Report saved to {output}")
        
        return report


# ============================================================================
# CLI
# ============================================================================

def print_report(report: OptimizationReport) -> None:
    """Print formatted report to console."""
    print("\n" + "="*70)
    print("📊 DEPLOYMENT OPTIMIZATION REPORT")
    print("="*70)
    print(f"Generated: {report.timestamp}")
    print(f"Status: {report.summary['status'].upper()}")
    print(f"Critical Issues: {report.summary['critical_issues']}")
    
    print("\n" + "-"*70)
    print("⏱️ LATENCY PROFILES")
    print("-"*70)
    
    for profile in report.latency_profiles:
        p = profile if isinstance(profile, dict) else asdict(profile)
        status = "✓" if p['p99_ms'] < 200 else "✗"
        print(f"\n{status} {p['component']}:")
        print(f"   Mean: {p['mean_ms']:.1f}ms | Median: {p['median_ms']:.1f}ms")
        print(f"   P90: {p['p90_ms']:.1f}ms | P99: {p['p99_ms']:.1f}ms")
        print(f"   Range: [{p['min_ms']:.1f}ms, {p['max_ms']:.1f}ms]")
    
    print("\n" + "-"*70)
    print("💾 CACHE ANALYSIS")
    print("-"*70)
    
    ca = report.cache_analysis
    print(f"Warmed Up: {'Yes ✓' if ca.get('warmed_up') else 'No ✗'}")
    
    for cache_name, stats in ca.get('caches', {}).items():
        hit_rate = stats.get('hit_rate', 0) * 100
        status = "✓" if stats.get('status') == 'good' else "⚠" if stats.get('status') == 'needs_improvement' else "✗"
        print(f"{status} {cache_name}: {stats.get('size', 0)}/{stats.get('max_size', 0)} entries, {hit_rate:.1f}% hit rate")
    
    print("\n" + "-"*70)
    print("🥶 COLD-START PATH ANALYSIS")
    print("-"*70)
    
    csa = report.cold_start_analysis
    print(f"Cold-Start Users: {csa.get('coldstart_users', 0):,} ({csa.get('coldstart_percentage', 0):.1f}% of traffic)")
    print(f"PhoBERT Loaded: {'Yes ✓' if csa.get('phobert_loaded') else 'No ✗'}")
    print(f"Similarity Pre-computed: {'Yes ✓' if csa.get('similarity_precomputed') else 'No ⚠'}")
    
    print("\n" + "-"*70)
    print("💡 RECOMMENDATIONS")
    print("-"*70)
    
    for rec in report.recommendations:
        print(f"\n{rec}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Deployment optimization analyzer")
    parser.add_argument("--analyze", action="store_true", help="Run full analysis")
    parser.add_argument("--profile", action="store_true", help="Run latency profiling only")
    parser.add_argument("--cache", action="store_true", help="Analyze cache only")
    parser.add_argument("--output", default="reports/optimization_report.json", help="Output file")
    parser.add_argument("--samples", type=int, default=30, help="Number of profiling samples")
    
    args = parser.parse_args()
    
    analyzer = OptimizationAnalyzer()
    
    if args.profile:
        logger.info("Running latency profiling...")
        profiles = analyzer.profiler.profile_recommendation_path(num_samples=args.samples)
        
        print("\n📊 LATENCY PROFILES")
        for name, profile in profiles.items():
            print(f"\n{name}:")
            print(f"   Mean: {profile.mean_ms:.1f}ms, P99: {profile.p99_ms:.1f}ms")
    
    elif args.cache:
        logger.info("Analyzing cache...")
        cache_analysis = analyzer.analyze_cache_effectiveness()
        print(json.dumps(cache_analysis, indent=2))
    
    else:
        # Full analysis
        report = analyzer.generate_report(output_path=args.output)
        print_report(report)


if __name__ == "__main__":
    main()
