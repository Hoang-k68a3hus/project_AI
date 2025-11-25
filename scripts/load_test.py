"""
Load Test Script for Recommendation API.

This script performs load testing on the recommendation API to ensure
latency requirements (<200ms) are met, especially for cold-start path.

Usage:
    # Start server first:
    uvicorn service.api:app --host 0.0.0.0 --port 8000 --workers 4

    # Run load test:
    python scripts/load_test.py --host localhost --port 8000

Requirements:
    pip install aiohttp tqdm
"""

import asyncio
import aiohttp
import time
import argparse
import json
import statistics
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RequestResult:
    """Result of a single request."""
    user_id: int
    latency_ms: float
    status_code: int
    is_fallback: bool
    fallback_method: Optional[str]
    num_recommendations: int
    success: bool
    error: Optional[str] = None


@dataclass
class LoadTestResult:
    """Aggregate load test results."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # Latency statistics (ms)
    mean_latency: float
    median_latency: float
    p90_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    
    # Cold-start vs CF breakdown
    cf_requests: int
    cf_mean_latency: float
    coldstart_requests: int
    coldstart_mean_latency: float
    
    # Throughput
    requests_per_second: float
    total_duration_s: float
    
    # Target compliance
    under_100ms_pct: float
    under_200ms_pct: float
    under_500ms_pct: float
    
    # Details
    errors: List[str] = field(default_factory=list)


# ============================================================================
# Load Test Client
# ============================================================================

class LoadTestClient:
    """Async HTTP client for load testing."""
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        max_connections: int = 100
    ):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.connector = aiohttp.TCPConnector(limit=max_connections)
    
    async def recommend(
        self,
        session: aiohttp.ClientSession,
        user_id: int,
        topk: int = 10,
        rerank: bool = False
    ) -> RequestResult:
        """Make a single recommendation request."""
        url = f"{self.base_url}/recommend"
        payload = {
            "user_id": user_id,
            "topk": topk,
            "exclude_seen": True,
            "rerank": rerank
        }
        
        start_time = time.perf_counter()
        
        try:
            async with session.post(url, json=payload) as response:
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    return RequestResult(
                        user_id=user_id,
                        latency_ms=latency_ms,
                        status_code=response.status,
                        is_fallback=data.get('is_fallback', False),
                        fallback_method=data.get('fallback_method'),
                        num_recommendations=data.get('count', 0),
                        success=True
                    )
                else:
                    error_text = await response.text()
                    return RequestResult(
                        user_id=user_id,
                        latency_ms=latency_ms,
                        status_code=response.status,
                        is_fallback=False,
                        fallback_method=None,
                        num_recommendations=0,
                        success=False,
                        error=f"HTTP {response.status}: {error_text[:200]}"
                    )
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RequestResult(
                user_id=user_id,
                latency_ms=latency_ms,
                status_code=0,
                is_fallback=False,
                fallback_method=None,
                num_recommendations=0,
                success=False,
                error="Request timeout"
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RequestResult(
                user_id=user_id,
                latency_ms=latency_ms,
                status_code=0,
                is_fallback=False,
                fallback_method=None,
                num_recommendations=0,
                success=False,
                error=str(e)
            )


# ============================================================================
# Load Test Runner
# ============================================================================

class LoadTestRunner:
    """Run load tests with configurable parameters."""
    
    def __init__(
        self,
        base_url: str,
        user_ids: List[int],
        concurrency: int = 10,
        total_requests: int = 100,
        warmup_requests: int = 10,
        timeout: float = 10.0
    ):
        self.client = LoadTestClient(base_url, timeout=timeout)
        self.user_ids = user_ids
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.warmup_requests = warmup_requests
    
    async def run_warmup(self, session: aiohttp.ClientSession) -> None:
        """Run warmup requests to prime caches."""
        print(f"\n🔥 Running {self.warmup_requests} warmup requests...")
        
        warmup_users = random.sample(self.user_ids, min(self.warmup_requests, len(self.user_ids)))
        
        tasks = [
            self.client.recommend(session, uid)
            for uid in warmup_users
        ]
        
        await asyncio.gather(*tasks)
        print("✓ Warmup complete")
    
    async def run_batch(
        self,
        session: aiohttp.ClientSession,
        user_ids_batch: List[int],
        rerank: bool = False
    ) -> List[RequestResult]:
        """Run a batch of concurrent requests."""
        tasks = [
            self.client.recommend(session, uid, rerank=rerank)
            for uid in user_ids_batch
        ]
        return await asyncio.gather(*tasks)
    
    async def run(self, rerank: bool = False) -> LoadTestResult:
        """Execute the full load test."""
        results: List[RequestResult] = []
        
        async with aiohttp.ClientSession(
            timeout=self.client.timeout,
            connector=self.client.connector
        ) as session:
            # Warmup
            await self.run_warmup(session)
            
            # Main test
            print(f"\n📊 Running {self.total_requests} requests with concurrency {self.concurrency}...")
            
            start_time = time.perf_counter()
            
            # Generate request sequence (sample with replacement)
            request_users = [
                random.choice(self.user_ids) 
                for _ in range(self.total_requests)
            ]
            
            # Process in batches
            for i in range(0, self.total_requests, self.concurrency):
                batch = request_users[i:i + self.concurrency]
                batch_results = await self.run_batch(session, batch, rerank=rerank)
                results.extend(batch_results)
                
                # Progress
                completed = min(i + self.concurrency, self.total_requests)
                pct = completed / self.total_requests * 100
                print(f"\r  Progress: {completed}/{self.total_requests} ({pct:.1f}%)", end="")
            
            total_duration = time.perf_counter() - start_time
            print("\n✓ Load test complete")
        
        return self._compute_stats(results, total_duration)
    
    def _compute_stats(
        self,
        results: List[RequestResult],
        total_duration: float
    ) -> LoadTestResult:
        """Compute aggregate statistics from results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # Extract latencies
        latencies = [r.latency_ms for r in successful]
        cf_results = [r for r in successful if not r.is_fallback]
        coldstart_results = [r for r in successful if r.is_fallback]
        
        cf_latencies = [r.latency_ms for r in cf_results]
        coldstart_latencies = [r.latency_ms for r in coldstart_results]
        
        # Compute percentiles
        if latencies:
            latencies_sorted = sorted(latencies)
            
            def percentile(data, pct):
                k = (len(data) - 1) * pct / 100
                f = int(k)
                c = f + 1 if f + 1 < len(data) else f
                return data[f] + (k - f) * (data[c] - data[f]) if c != f else data[f]
            
            return LoadTestResult(
                total_requests=len(results),
                successful_requests=len(successful),
                failed_requests=len(failed),
                
                mean_latency=statistics.mean(latencies),
                median_latency=statistics.median(latencies),
                p90_latency=percentile(latencies_sorted, 90),
                p95_latency=percentile(latencies_sorted, 95),
                p99_latency=percentile(latencies_sorted, 99),
                min_latency=min(latencies),
                max_latency=max(latencies),
                
                cf_requests=len(cf_results),
                cf_mean_latency=statistics.mean(cf_latencies) if cf_latencies else 0,
                coldstart_requests=len(coldstart_results),
                coldstart_mean_latency=statistics.mean(coldstart_latencies) if coldstart_latencies else 0,
                
                requests_per_second=len(results) / total_duration,
                total_duration_s=total_duration,
                
                under_100ms_pct=sum(1 for l in latencies if l < 100) / len(latencies) * 100,
                under_200ms_pct=sum(1 for l in latencies if l < 200) / len(latencies) * 100,
                under_500ms_pct=sum(1 for l in latencies if l < 500) / len(latencies) * 100,
                
                errors=[r.error for r in failed if r.error]
            )
        else:
            return LoadTestResult(
                total_requests=len(results),
                successful_requests=0,
                failed_requests=len(failed),
                mean_latency=0,
                median_latency=0,
                p90_latency=0,
                p95_latency=0,
                p99_latency=0,
                min_latency=0,
                max_latency=0,
                cf_requests=0,
                cf_mean_latency=0,
                coldstart_requests=0,
                coldstart_mean_latency=0,
                requests_per_second=0,
                total_duration_s=total_duration,
                under_100ms_pct=0,
                under_200ms_pct=0,
                under_500ms_pct=0,
                errors=[r.error for r in failed if r.error]
            )


# ============================================================================
# User ID Loading
# ============================================================================

def load_user_ids(data_dir: str = "data/processed") -> Dict[str, List[int]]:
    """Load trainable and cold-start user IDs from mappings."""
    data_path = Path(data_dir)
    
    # Load mappings
    mappings_path = data_path / "user_item_mappings.json"
    trainable_path = data_path / "trainable_user_mapping.json"
    
    with open(mappings_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    all_user_ids = [int(uid) for uid in mappings['user_to_idx'].keys()]
    
    # Load trainable users
    trainable_u_idx = set()
    if trainable_path.exists():
        with open(trainable_path, 'r', encoding='utf-8') as f:
            trainable_data = json.load(f)
        trainable_u_idx = set(int(k) for k in trainable_data.get('u_idx_to_u_idx_cf', {}).keys())
    
    # Convert to user_ids
    idx_to_user = {int(k): int(v) for k, v in mappings['idx_to_user'].items()}
    
    trainable_user_ids = []
    coldstart_user_ids = []
    
    for u_idx, user_id in idx_to_user.items():
        if u_idx in trainable_u_idx:
            trainable_user_ids.append(user_id)
        else:
            coldstart_user_ids.append(user_id)
    
    print(f"📊 User breakdown:")
    print(f"   - Total users: {len(all_user_ids):,}")
    print(f"   - Trainable (CF): {len(trainable_user_ids):,} ({len(trainable_user_ids)/len(all_user_ids)*100:.1f}%)")
    print(f"   - Cold-start: {len(coldstart_user_ids):,} ({len(coldstart_user_ids)/len(all_user_ids)*100:.1f}%)")
    
    return {
        'all': all_user_ids,
        'trainable': trainable_user_ids,
        'coldstart': coldstart_user_ids
    }


# ============================================================================
# Report Generation
# ============================================================================

def print_results(result: LoadTestResult, test_name: str = "Load Test") -> None:
    """Print formatted test results."""
    print(f"\n{'='*60}")
    print(f"📊 {test_name} Results")
    print(f"{'='*60}")
    
    print(f"\n📈 Request Summary:")
    print(f"   Total:      {result.total_requests:,}")
    print(f"   Successful: {result.successful_requests:,} ({result.successful_requests/result.total_requests*100:.1f}%)")
    print(f"   Failed:     {result.failed_requests:,}")
    print(f"   Duration:   {result.total_duration_s:.2f}s")
    print(f"   Throughput: {result.requests_per_second:.1f} req/s")
    
    print(f"\n⏱️ Latency (ms):")
    print(f"   Mean:   {result.mean_latency:.1f}")
    print(f"   Median: {result.median_latency:.1f}")
    print(f"   P90:    {result.p90_latency:.1f}")
    print(f"   P95:    {result.p95_latency:.1f}")
    print(f"   P99:    {result.p99_latency:.1f}")
    print(f"   Min:    {result.min_latency:.1f}")
    print(f"   Max:    {result.max_latency:.1f}")
    
    print(f"\n🎯 Path Breakdown:")
    print(f"   CF Path:         {result.cf_requests:,} requests, mean {result.cf_mean_latency:.1f}ms")
    print(f"   Cold-Start Path: {result.coldstart_requests:,} requests, mean {result.coldstart_mean_latency:.1f}ms")
    
    print(f"\n✅ Target Compliance:")
    
    # Check targets
    target_200 = result.under_200ms_pct >= 95
    target_icon_200 = "✓" if target_200 else "✗"
    print(f"   <100ms: {result.under_100ms_pct:.1f}%")
    print(f"   <200ms: {result.under_200ms_pct:.1f}% {target_icon_200} (target: 95%)")
    print(f"   <500ms: {result.under_500ms_pct:.1f}%")
    
    if result.coldstart_mean_latency > 200:
        print(f"\n⚠️  WARNING: Cold-start latency ({result.coldstart_mean_latency:.1f}ms) exceeds 200ms target!")
        print(f"   Consider enabling warm-up cache strategies.")
    
    if result.errors:
        print(f"\n❌ Errors ({len(result.errors)}):")
        for error in result.errors[:5]:
            print(f"   - {error}")
        if len(result.errors) > 5:
            print(f"   ... and {len(result.errors) - 5} more")


def save_results(
    results: Dict[str, LoadTestResult],
    output_path: str = "reports/load_test_results.json"
) -> None:
    """Save results to JSON file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    for name, result in results.items():
        data['tests'][name] = {
            'total_requests': result.total_requests,
            'successful_requests': result.successful_requests,
            'failed_requests': result.failed_requests,
            'latency_ms': {
                'mean': result.mean_latency,
                'median': result.median_latency,
                'p90': result.p90_latency,
                'p95': result.p95_latency,
                'p99': result.p99_latency,
                'min': result.min_latency,
                'max': result.max_latency
            },
            'path_breakdown': {
                'cf_requests': result.cf_requests,
                'cf_mean_latency_ms': result.cf_mean_latency,
                'coldstart_requests': result.coldstart_requests,
                'coldstart_mean_latency_ms': result.coldstart_mean_latency
            },
            'throughput': {
                'requests_per_second': result.requests_per_second,
                'total_duration_s': result.total_duration_s
            },
            'compliance': {
                'under_100ms_pct': result.under_100ms_pct,
                'under_200ms_pct': result.under_200ms_pct,
                'under_500ms_pct': result.under_500ms_pct
            }
        }
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Results saved to {output}")


# ============================================================================
# Main
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Load test for recommendation API")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--total", type=int, default=500, help="Total requests")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent requests")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup requests")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout (s)")
    parser.add_argument("--test-type", choices=['all', 'cf', 'coldstart', 'mixed'], 
                        default='mixed', help="Which users to test")
    parser.add_argument("--rerank", action="store_true", help="Enable hybrid reranking")
    parser.add_argument("--output", default="reports/load_test_results.json", help="Output file")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    print(f"\n🚀 Load Test Configuration:")
    print(f"   Base URL:    {base_url}")
    print(f"   Total:       {args.total} requests")
    print(f"   Concurrency: {args.concurrency}")
    print(f"   Warmup:      {args.warmup} requests")
    print(f"   Test Type:   {args.test_type}")
    print(f"   Reranking:   {'enabled' if args.rerank else 'disabled'}")
    
    # Load user IDs
    user_ids = load_user_ids()
    
    # Select user set based on test type
    test_users = {
        'all': user_ids['all'],
        'cf': user_ids['trainable'],
        'coldstart': user_ids['coldstart'],
        'mixed': user_ids['all']  # Natural distribution
    }
    
    results = {}
    
    # Run tests based on type
    if args.test_type == 'all':
        # Run all three tests
        for test_name, test_user_ids in [
            ('Mixed (Natural)', user_ids['all']),
            ('CF Only', user_ids['trainable']),
            ('Cold-Start Only', user_ids['coldstart'][:10000])  # Sample coldstart
        ]:
            if not test_user_ids:
                print(f"\n⚠️ Skipping {test_name}: no users")
                continue
            
            runner = LoadTestRunner(
                base_url=base_url,
                user_ids=test_user_ids,
                concurrency=args.concurrency,
                total_requests=args.total,
                warmup_requests=args.warmup,
                timeout=args.timeout
            )
            
            result = await runner.run(rerank=args.rerank)
            results[test_name] = result
            print_results(result, test_name)
    else:
        # Run single test
        test_user_ids = test_users[args.test_type]
        
        if args.test_type == 'coldstart':
            # Sample cold-start users
            test_user_ids = test_user_ids[:10000]
        
        runner = LoadTestRunner(
            base_url=base_url,
            user_ids=test_user_ids,
            concurrency=args.concurrency,
            total_requests=args.total,
            warmup_requests=args.warmup,
            timeout=args.timeout
        )
        
        result = await runner.run(rerank=args.rerank)
        results[args.test_type] = result
        print_results(result, f"{args.test_type.title()} Test")
    
    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
