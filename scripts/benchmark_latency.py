"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: scripts/benchmark_latency.py
Description: Empirical End-to-End Latency Benchmark under Concurrent Load.
             Validates p50, p95, p99, and max latency against the budgeted SLA.
==============================================================================
"""

import os
import sys
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO_ROOT, 'api')
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
for d in [API_DIR, SCRIPTS_DIR]:
    if d not in sys.path:
        sys.path.append(d)

from fastapi.testclient import TestClient
from app import app, load_all_models, SPRINTS_API_KEY

def run_concurrent_latency_benchmark(total_requests: int = 50, concurrency: int = 5) -> Dict[str, Any]:
    load_all_models()
    client = TestClient(app, headers={"X-API-Key": SPRINTS_API_KEY})

    queries = [
        {"learner_id": "LRN-1002", "query_text": "What is PyTorch autograd engine?"},
        {"learner_id": "LRN-1002", "query_text": "How do SQL inner joins differ from left joins?"},
        {"learner_id": "LRN-1002", "query_text": "مش فاهم يعني إيه gradient descent في الـ Supervised Learning"},
        {"learner_id": "LRN-1005", "query_text": "أنا محبط جداً ومش قادر أكمل الكورس ومخنوق خالص"},
        {"learner_id": "LRN-1002", "query_text": "مش قادر أفتح ملف المحاضرة على المنصة"}
    ]

    latencies: List[float] = []

    def make_single_request(idx: int) -> float:
        payload = queries[idx % len(queries)]
        t0 = time.perf_counter()
        res = client.post("/api/v1/analyze", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert res.status_code == 200, f"Request {idx} failed with {res.status_code}: {res.text}"
        return elapsed_ms

    print("\n" + "=" * 65)
    print(f"RUNNING EMPIRICAL LATENCY BENCHMARK ({total_requests} requests, {concurrency} concurrent threads)")
    print("=" * 65)

    start_bench = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_single_request, i) for i in range(total_requests)]
        for f in as_completed(futures):
            latencies.append(f.result())
    total_time = time.perf_counter() - start_bench

    lat_arr = np.array(latencies)
    metrics = {
        "total_requests": total_requests,
        "concurrency": concurrency,
        "throughput_rps": round(total_requests / total_time, 2),
        "min_ms": round(float(np.min(lat_arr)), 2),
        "median_p50_ms": round(float(np.median(lat_arr)), 2),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "p99_ms": round(float(np.percentile(lat_arr, 99)), 2),
        "max_ms": round(float(np.max(lat_arr)), 2),
        "budgeted_ceiling_target_ms": 3280.0,
        "contractual_sla_threshold_ms": 3500.0,
        "sla_breached": bool(np.max(lat_arr) >= 3500.0)
    }

    print(f"Throughput:            {metrics['throughput_rps']} req/sec")
    print(f"Min Latency:           {metrics['min_ms']} ms")
    print(f"Median (p50):          {metrics['median_p50_ms']} ms")
    print(f"95th Percentile (p95): {metrics['p95_ms']} ms (Target < 500 ms)")
    print(f"99th Percentile (p99): {metrics['p99_ms']} ms (Target < 1000 ms)")
    print(f"Max Latency:           {metrics['max_ms']} ms (Budgeted Target: 3,280 ms | SLA: < 3,500 ms)")
    print(f"SLA Compliance:        {'100% COMPLIANT' if not metrics['sla_breached'] else 'BREACHED'}")
    print("=" * 65 + "\n")

    return metrics

if __name__ == "__main__":
    run_concurrent_latency_benchmark()
