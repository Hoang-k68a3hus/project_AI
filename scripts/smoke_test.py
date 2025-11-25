"""
Smoke Test Script for Vietnamese Cosmetics Recommendation System.

Validates basic API functionality after deployment.

Usage:
    python scripts/smoke_test.py --host localhost --port 8000
    python scripts/smoke_test.py --host staging-server --port 8000 --verbose
    
Examples:
    # Quick check
    python scripts/smoke_test.py
    
    # Full verbose output
    python scripts/smoke_test.py --verbose
    
    # Against staging
    python scripts/smoke_test.py --host staging.example.com --port 8000
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)


@dataclass
class TestResult:
    """Result of a single smoke test."""
    name: str
    passed: bool
    duration_ms: float
    status_code: Optional[int] = None
    error: Optional[str] = None
    response: Optional[Dict] = None


class SmokeTestRunner:
    """Runs smoke tests against the recommendation API."""
    
    def __init__(self, base_url: str, verbose: bool = False, timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.timeout = timeout
        self.results: List[TestResult] = []
        
    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict] = None
    ) -> Tuple[Optional[requests.Response], float, Optional[str]]:
        """Make HTTP request and return response, duration, and error."""
        url = f"{self.base_url}{endpoint}"
        start = time.time()
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=self.timeout)
            elif method.upper() == "POST":
                response = requests.post(
                    url, 
                    json=payload, 
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )
            else:
                return None, 0, f"Unsupported method: {method}"
            
            duration_ms = (time.time() - start) * 1000
            return response, duration_ms, None
            
        except requests.exceptions.Timeout:
            duration_ms = (time.time() - start) * 1000
            return None, duration_ms, "Request timed out"
        except requests.exceptions.ConnectionError:
            duration_ms = (time.time() - start) * 1000
            return None, duration_ms, "Connection refused"
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return None, duration_ms, str(e)
    
    def _run_test(
        self,
        name: str,
        method: str,
        endpoint: str,
        payload: Optional[Dict] = None,
        expected_status: int = 200,
        validate_response: Optional[callable] = None
    ) -> TestResult:
        """Run a single test."""
        response, duration_ms, error = self._request(method, endpoint, payload)
        
        if error:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration_ms,
                error=error
            )
        
        if response.status_code != expected_status:
            return TestResult(
                name=name,
                passed=False,
                duration_ms=duration_ms,
                status_code=response.status_code,
                error=f"Expected {expected_status}, got {response.status_code}"
            )
        
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            response_data = {"raw": response.text[:200]}
        
        # Custom validation
        if validate_response:
            try:
                validation_error = validate_response(response_data)
                if validation_error:
                    return TestResult(
                        name=name,
                        passed=False,
                        duration_ms=duration_ms,
                        status_code=response.status_code,
                        error=validation_error,
                        response=response_data
                    )
            except Exception as e:
                return TestResult(
                    name=name,
                    passed=False,
                    duration_ms=duration_ms,
                    status_code=response.status_code,
                    error=f"Validation error: {e}",
                    response=response_data
                )
        
        return TestResult(
            name=name,
            passed=True,
            duration_ms=duration_ms,
            status_code=response.status_code,
            response=response_data
        )
    
    def run_all_tests(self) -> bool:
        """Run all smoke tests and return overall success."""
        print(f"\n{'='*60}")
        print(f"SMOKE TESTS - {self.base_url}")
        print(f"{'='*60}\n")
        
        # Define tests
        tests = [
            # Basic connectivity
            {
                "name": "1. Health Check",
                "method": "GET",
                "endpoint": "/health",
                "validate": lambda r: None if r.get("status") in ["healthy", "ok", True] or "healthy" in str(r).lower() else "Health check failed"
            },
            
            # Model information
            {
                "name": "2. Model Info",
                "method": "GET",
                "endpoint": "/model_info",
                "validate": lambda r: None  # Just check it responds
            },
            
            # Cache status
            {
                "name": "3. Cache Stats",
                "method": "GET",
                "endpoint": "/cache_stats",
                "validate": lambda r: None  # Just check it responds
            },
            
            # Service stats
            {
                "name": "4. Service Stats",
                "method": "GET",
                "endpoint": "/stats",
                "validate": lambda r: None  # Just check it responds
            },
            
            # Recommendation for likely trainable user (low user_id)
            {
                "name": "5. Recommend (trainable user)",
                "method": "POST",
                "endpoint": "/recommend",
                "payload": {"user_id": 1, "topk": 10},
                "validate": lambda r: None if "recommendations" in r or "items" in r or isinstance(r, list) else "No recommendations in response"
            },
            
            # Recommendation for cold-start user (high user_id)
            {
                "name": "6. Recommend (cold-start user)",
                "method": "POST",
                "endpoint": "/recommend",
                "payload": {"user_id": 999999999, "topk": 10},
                "validate": lambda r: None if "recommendations" in r or "items" in r or "fallback" in str(r).lower() or isinstance(r, list) else "No recommendations for cold-start"
            },
            
            # Batch recommendation
            {
                "name": "7. Batch Recommend",
                "method": "POST",
                "endpoint": "/batch_recommend",
                "payload": {"user_ids": [1, 2, 3], "topk": 5},
                "validate": lambda r: None if "results" in r or isinstance(r, list) or isinstance(r, dict) else "No batch results"
            },
            
            # Similar items
            {
                "name": "8. Similar Items",
                "method": "POST",
                "endpoint": "/similar_items",
                "payload": {"item_id": 1, "topk": 5},
                "validate": lambda r: None  # May not exist, so just check it responds
            },
        ]
        
        # Run tests
        for test_config in tests:
            result = self._run_test(
                name=test_config["name"],
                method=test_config["method"],
                endpoint=test_config["endpoint"],
                payload=test_config.get("payload"),
                validate_response=test_config.get("validate")
            )
            self.results.append(result)
            self._print_result(result)
        
        # Print summary
        return self._print_summary()
    
    def _print_result(self, result: TestResult) -> None:
        """Print a single test result."""
        status = "✓ PASS" if result.passed else "✗ FAIL"
        latency = f"{result.duration_ms:.1f}ms"
        
        if result.passed:
            print(f"{status} | {result.name} | {latency}")
        else:
            print(f"{status} | {result.name} | {latency}")
            print(f"       Error: {result.error}")
        
        if self.verbose and result.response:
            print(f"       Response: {json.dumps(result.response, indent=2)[:500]}")
    
    def _print_summary(self) -> bool:
        """Print test summary and return success status."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        avg_latency = sum(r.duration_ms for r in self.results) / total if total > 0 else 0
        max_latency = max((r.duration_ms for r in self.results), default=0)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
        print(f"Avg Latency: {avg_latency:.1f}ms | Max Latency: {max_latency:.1f}ms")
        
        if failed > 0:
            print(f"\n⚠️  {failed} test(s) failed!")
            print("Failed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.error}")
        else:
            print(f"\n✅ All tests passed!")
        
        print(f"{'='*60}\n")
        
        return failed == 0


def run_latency_check(base_url: str, target_ms: float = 200) -> bool:
    """Run latency-focused tests to verify P99 < target."""
    print(f"\n{'='*60}")
    print(f"LATENCY CHECK - Target: P99 < {target_ms}ms")
    print(f"{'='*60}\n")
    
    latencies = []
    endpoint = "/recommend"
    
    # Run 20 requests to get latency distribution
    for i in range(20):
        try:
            start = time.time()
            response = requests.post(
                f"{base_url}{endpoint}",
                json={"user_id": i + 1, "topk": 10},
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            duration_ms = (time.time() - start) * 1000
            
            if response.status_code == 200:
                latencies.append(duration_ms)
                print(f"  Request {i+1}: {duration_ms:.1f}ms")
            else:
                print(f"  Request {i+1}: Failed (status {response.status_code})")
        except Exception as e:
            print(f"  Request {i+1}: Error - {e}")
    
    if len(latencies) < 10:
        print(f"\n⚠️  Not enough successful requests for latency analysis")
        return False
    
    # Calculate percentiles
    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[len(sorted_latencies) // 2]
    p95_idx = int(len(sorted_latencies) * 0.95)
    p99_idx = int(len(sorted_latencies) * 0.99)
    p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
    p99 = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
    
    print(f"\n{'='*60}")
    print(f"LATENCY RESULTS")
    print(f"{'='*60}")
    print(f"P50: {p50:.1f}ms")
    print(f"P95: {p95:.1f}ms")
    print(f"P99: {p99:.1f}ms")
    print(f"Max: {max(latencies):.1f}ms")
    
    if p99 < target_ms:
        print(f"\n✅ P99 latency ({p99:.1f}ms) is below target ({target_ms}ms)")
        return True
    else:
        print(f"\n⚠️  P99 latency ({p99:.1f}ms) exceeds target ({target_ms}ms)")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smoke tests for Vietnamese Cosmetics Recommendation API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --host staging-server --port 8000
    python scripts/smoke_test.py --verbose --latency-check
        """
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="API host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port (default: 8000)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed response data"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)"
    )
    parser.add_argument(
        "--latency-check",
        action="store_true",
        help="Run additional latency verification"
    )
    parser.add_argument(
        "--latency-target",
        type=float,
        default=200,
        help="Target P99 latency in ms (default: 200)"
    )
    
    args = parser.parse_args()
    
    # Construct base URL
    base_url = f"http://{args.host}:{args.port}"
    
    print(f"\n🚀 Starting Smoke Tests")
    print(f"   Target: {base_url}")
    print(f"   Timeout: {args.timeout}s")
    
    # Run smoke tests
    runner = SmokeTestRunner(
        base_url=base_url,
        verbose=args.verbose,
        timeout=args.timeout
    )
    
    smoke_passed = runner.run_all_tests()
    
    # Optional latency check
    latency_passed = True
    if args.latency_check:
        latency_passed = run_latency_check(base_url, args.latency_target)
    
    # Exit code
    if smoke_passed and latency_passed:
        print("✅ All checks passed - Ready for production!")
        sys.exit(0)
    else:
        print("⚠️  Some checks failed - Review before proceeding")
        sys.exit(1)


if __name__ == "__main__":
    main()
