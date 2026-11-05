"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: scripts/benchmark_live_webhook.py
Description: Empirical End-to-End Network & Concurrency Benchmark Suite.
             Benchmarks real HTTP/TCP network requests across concurrent threads:
             - Evaluates throughput (req/sec)
             - Measures Round-Trip Time (RTT): Min, Median (p50), p95, p99, Max
             - Validates strict compliance with the sub-3,500ms SLA budget
==============================================================================
"""

import os
import sys
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(REPO_ROOT, ".env")

# Load environment
if os.path.exists(ENV_FILE):
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

FASTAPI_URL = os.environ.get("FASTAPI_BASE_URL", "http://localhost:8000")
N8N_URL = os.environ.get("N8N_BASE_URL", "http://localhost:5678")
API_KEY = os.environ.get("SPRINTS_API_KEY", "")

TEST_QUERIES = [
    {"learner_id": "LRN-1002", "query_text": "What are the core principles of relational SQL databases?", "expected": "AUTOMATED_RAG_PIPELINE"},
    {"learner_id": "LRN-1019", "query_text": "أنا مش فاهم PyTorch خالص ومحبط جداً ومش قادر أكمل وعايز مساعدة ضروري", "expected": "HUMAN_OPERATOR_ESCALATION"},
    {"learner_id": "LRN-1005", "query_text": "I am completely failing this module, cannot keep up, and want to drop out urgently", "expected": "HUMAN_OPERATOR_ESCALATION"},
    {"learner_id": "LRN-1002", "query_text": "مش قادر أفتح ملف المحاضرة، ممكن تساعدني؟", "expected": "AUTOMATED_RAG_PIPELINE"},
    {"learner_id": "LRN-1005", "query_text": "الحقوني، أنا هسيب الكورس ومخنوق جدا ومش عارف أكمل", "expected": "HUMAN_OPERATOR_ESCALATION"}
]

def send_request(idx: int, target_url: str, headers: dict) -> dict:
    payload = TEST_QUERIES[idx % len(TEST_QUERIES)].copy()
    payload["correlation_id"] = f"net-bench-{int(time.time()*1000)}-{idx}"
    payload["idempotency_key"] = f"idemp-net-bench-{idx}-{time.time()}"

    t0 = time.time()
    try:
        resp = requests.post(target_url, json=payload, headers=headers, timeout=5.0)
        lat = (time.time() - t0) * 1000.0
        return {
            "index": idx,
            "status_code": resp.status_code,
            "latency_ms": lat,
            "success": (resp.status_code == 200),
            "error": None if resp.status_code == 200 else resp.text
        }
    except Exception as e:
        lat = (time.time() - t0) * 1000.0
        return {
            "index": idx,
            "status_code": 0,
            "latency_ms": lat,
            "success": False,
            "error": str(e)
        }

def run_network_benchmark(total_requests: int = 50, concurrency: int = 5):
    print("=" * 70)
    print("SPRINTS E2E NETWORK CONCURRENCY & SLA LATENCY BENCHMARK")
    print("=" * 70)

    # 1. Determine active endpoint
    webhook_target = f"{N8N_URL}/webhook/learner-query"
    api_target = f"{FASTAPI_URL}/api/v1/analyze"

    n8n_live = False
    try:
        p = requests.get(N8N_URL, timeout=1.5)
        n8n_live = (p.status_code in [200, 302, 401])
    except Exception:
        pass

    target_url = webhook_target if n8n_live else api_target
    target_name = "n8n Webhook (/webhook/learner-query)" if n8n_live else "FastAPI Backend Network Socket (/api/v1/analyze)"
    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}

    print(f"Target Endpoint:  {target_name}")
    print(f"Target URL:       {target_url}")
    print(f"Total Requests:   {total_requests}")
    print(f"Worker Threads:   {concurrency}")
    print(f"Target SLA:       < 3,500.0 ms")
    print("-" * 70)

    t_start = time.time()
    latencies = []
    successes = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i, target_url, headers) for i in range(total_requests)]
        for f in as_completed(futures):
            res = f.result()
            latencies.append(res["latency_ms"])
            if res["success"]:
                successes += 1

    total_time = time.time() - t_start
    throughput = len(latencies) / total_time if total_time > 0 else 0

    sorted_lats = sorted(latencies)
    min_lat = sorted_lats[0]
    p50 = statistics.median(sorted_lats)
    p95 = sorted_lats[int(len(sorted_lats) * 0.95)]
    p99 = sorted_lats[int(len(sorted_lats) * 0.99)]
    max_lat = sorted_lats[-1]
    sla_breaches = sum(1 for l in latencies if l >= 3500.0)

    print(f"Completed:        {successes}/{total_requests} requests successful ({successes/total_requests*100:.1f}%)")
    print(f"Total Wall Time:  {total_time:.2f} seconds")
    print(f"Throughput:       {throughput:.2f} requests/sec")
    print(f"Min Latency:      {min_lat:.2f} ms")
    print(f"p50 (Median):     {p50:.2f} ms")
    print(f"p95 Latency:      {p95:.2f} ms")
    print(f"p99 Latency:      {p99:.2f} ms")
    print(f"Max Latency:      {max_lat:.2f} ms")
    print(f"SLA Compliance:   {((len(latencies) - sla_breaches) / len(latencies) * 100):.1f}% (< 3500ms)")
    print("=" * 70)

    assert sla_breaches == 0, f"SLA breached on {sla_breaches} requests!"
    assert successes == total_requests, f"Only {successes}/{total_requests} succeeded!"

    return {
        "target": target_name,
        "total_requests": total_requests,
        "concurrency": concurrency,
        "throughput_req_sec": round(throughput, 2),
        "min_latency_ms": round(min_lat, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "max_latency_ms": round(max_lat, 2),
        "sla_compliance_pct": 100.0
    }

if __name__ == "__main__":
    run_network_benchmark()
