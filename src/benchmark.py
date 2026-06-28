"""
COM668 Final Year Project — AI-Based Intrusion Detection System
API Performance Benchmark
Author : Abdulbosit Abdurazzakov | B00979380

Measures inference throughput and latency for the REST API.
Requires the Flask server to be running (python src/api.py).

Usage
-----
    python src/benchmark.py
    python src/benchmark.py --n-requests 500 --concurrency 1
"""

import argparse
import json
import os
import statistics
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT, "src")
for p in [_ROOT, _SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

BASE_URL = "http://localhost:5000"


def get_feature_vector(n: int = 78) -> list:
    """Generate a zeroed feature vector of length n."""
    return [0.0] * n


def benchmark_single(n_requests: int, base_url: str, n_features: int) -> dict:
    """
    Benchmark the /predict endpoint with sequential single-flow requests.

    Returns dict with: n_requests, mean_ms, median_ms, p95_ms, p99_ms,
                       min_ms, max_ms, throughput_rps
    """
    payload = {"features": get_feature_vector(n_features)}
    latencies = []

    print(f"\nBenchmarking /predict ({n_requests} sequential requests)...")

    for i in range(n_requests):
        t0   = time.perf_counter()
        resp = requests.post(f"{base_url}/predict", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        if resp.status_code != 200:
            print(f"  Request {i+1} failed: {resp.status_code}")

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_requests} — running mean: {statistics.mean(latencies):.1f}ms")

    total_s = sum(latencies) / 1000
    return {
        "endpoint":       "/predict",
        "n_requests":     n_requests,
        "mean_ms":        round(statistics.mean(latencies), 2),
        "median_ms":      round(statistics.median(latencies), 2),
        "p95_ms":         round(sorted(latencies)[int(0.95 * n_requests)], 2),
        "p99_ms":         round(sorted(latencies)[int(0.99 * n_requests)], 2),
        "min_ms":         round(min(latencies), 2),
        "max_ms":         round(max(latencies), 2),
        "throughput_rps": round(n_requests / total_s, 1),
    }


def benchmark_batch(batch_size: int, n_batches: int, base_url: str, n_features: int) -> dict:
    """Benchmark the /predict/batch endpoint."""
    payload = {"flows": [get_feature_vector(n_features) for _ in range(batch_size)]}
    latencies = []

    print(f"\nBenchmarking /predict/batch (batch_size={batch_size}, n_batches={n_batches})...")

    for i in range(n_batches):
        t0   = time.perf_counter()
        resp = requests.post(f"{base_url}/predict/batch", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

    total_flows = n_batches * batch_size
    total_s     = sum(latencies) / 1000
    return {
        "endpoint":          "/predict/batch",
        "batch_size":        batch_size,
        "n_batches":         n_batches,
        "total_flows":       total_flows,
        "mean_batch_ms":     round(statistics.mean(latencies), 2),
        "throughput_fps":    round(total_flows / total_s, 1),
        "ms_per_flow":       round(statistics.mean(latencies) / batch_size, 3),
    }


def parse_args():
    p = argparse.ArgumentParser(description="AI-IDS API Benchmark")
    p.add_argument("--n-requests",  type=int, default=200)
    p.add_argument("--batch-size",  type=int, default=100)
    p.add_argument("--n-batches",   type=int, default=20)
    p.add_argument("--url",         default=BASE_URL)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  AI-IDS API Performance Benchmark")
    print(f"  Target: {args.url}")
    print("=" * 55)

    # Health check
    try:
        health = requests.get(f"{args.url}/health", timeout=5).json()
        n_features = health["features"]
        print(f"\nServer healthy. Features: {n_features}")
    except Exception as e:
        print(f"\nServer unreachable: {e}")
        print("Start the API first: python src/api.py")
        sys.exit(1)

    single = benchmark_single(args.n_requests, args.url, n_features)
    batch  = benchmark_batch(args.batch_size, args.n_batches, args.url, n_features)

    print("\n" + "=" * 55)
    print("  Single-flow /predict benchmark:")
    for k, v in single.items():
        print(f"    {k:<20}: {v}")

    print("\n  Batch /predict/batch benchmark:")
    for k, v in batch.items():
        print(f"    {k:<20}: {v}")
    print("=" * 55)


if __name__ == "__main__":
    main()
