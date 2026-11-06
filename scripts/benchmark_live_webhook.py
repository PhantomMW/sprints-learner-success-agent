"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: scripts/benchmark_live_webhook.py
Description: Empirical End-to-End Network & Concurrency Benchmark Suite.
             Benchmarks real HTTP/TCP network requests across concurrent threads:
             - Evaluates throughput (req/sec)
             - Measures Round-Trip Time (RTT): Min, Median (p50), p95, p99, Max
             - Validates strict compliance with the sub-3,500ms SLA budget
             - Validates semantic routing decisions (Automated RAG vs Escalation)
             - Separately records and reports HTTP errors vs Routing errors
             - Explicitly differentiates n8n Webhook vs FastAPI Direct execution
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
    query_item = TEST_QUERIES[idx % len(TEST_QUERIES)]
    expected_decision = query_item["expected"]
    payload = {
        "learner_id": query_item["learner_id"],
        "query_text": query_item["query_text"],
        "correlation_id": f"net-bench-{int(time.time()*1000)}-{idx}",
        "idempotency_key": f"idemp-net-bench-{idx}-{time.time()}"
    }

    t0 = time.time()
    try:
        resp = requests.post(target_url, json=payload, headers=headers, timeout=5.0)
        lat = (time.time() - t0) * 1000.0
        http_ok = (resp.status_code == 200)
        routing_ok = False
        actual_decision = None
        routing_error_msg = None

        if http_ok:
            try:
                data = resp.json()
                if "routing_decision" in data:
                    actual_decision = data["routing_decision"]
                elif data.get("status") == "escalated":
                    actual_decision = "HUMAN_OPERATOR_ESCALATION"
                elif data.get("status") == "resolved_auto":
                    actual_decision = "AUTOMATED_RAG_PIPELINE"
                else:
                    actual_decision = data.get("status", "UNKNOWN")

                if actual_decision == expected_decision:
                    routing_ok = True
                else:
                    routing_error_msg = f"Routing Mismatch: Expected {expected_decision}, but received {actual_decision}"
            except Exception as parse_err:
                routing_error_msg = f"JSON Parse Error: {parse_err}"

        return {
            "index": idx,
            "status_code": resp.status_code,
            "latency_ms": lat,
            "http_success": http_ok,
            "routing_success": routing_ok,
            "e2e_success": (http_ok and routing_ok),
            "expected_decision": expected_decision,
            "actual_decision": actual_decision,
            "error": None if (http_ok and routing_ok) else (resp.text if not http_ok else routing_error_msg)
        }
    except Exception as e:
        lat = (time.time() - t0) * 1000.0
        return {
            "index": idx,
            "status_code": 0,
            "latency_ms": lat,
            "http_success": False,
            "routing_success": False,
            "e2e_success": False,
            "expected_decision": expected_decision,
            "actual_decision": None,
            "error": str(e)
        }

def run_network_benchmark(total_requests: int = 100, concurrency: int = 10, target_override: str = None):
    print("=" * 75)
    print("SPRINTS E2E NETWORK CONCURRENCY, ROUTING & SLA LATENCY BENCHMARK")
    print("=" * 75)

    # 1. Determine active endpoint
    webhook_target = f"{N8N_URL}/webhook/learner-query"
    api_target = f"{FASTAPI_URL}/api/v1/analyze"

    n8n_live = False
    try:
        p = requests.get(N8N_URL, timeout=1.5)
        n8n_live = (p.status_code in [200, 302, 401])
    except Exception:
        pass

    if target_override == "fastapi":
        target_url = api_target
        target_tier = "FastAPI Direct Endpoint"
        target_name = "FastAPI Backend Direct Socket (/api/v1/analyze)"
    elif target_override == "n8n":
        target_url = webhook_target
        target_tier = "n8n Webhook Engine"
        target_name = "n8n Multi-Container Orchestrator (/webhook/learner-query)"
    else:
        if n8n_live:
            target_url = webhook_target
            target_tier = "n8n Webhook Engine"
            target_name = "n8n Multi-Container Orchestrator (/webhook/learner-query)"
        else:
            target_url = api_target
            target_tier = "FastAPI Direct Endpoint"
            target_name = "FastAPI Backend Direct Socket (/api/v1/analyze)"

    headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}

    print(f"Target Tier:      {target_tier}")
    print(f"Target Endpoint:  {target_name}")
    print(f"Target URL:       {target_url}")
    print(f"Total Requests:   {total_requests}")
    print(f"Worker Threads:   {concurrency}")
    print(f"Target SLA:       < 3,500.0 ms")
    print("-" * 75)

    t_start = time.time()
    latencies = []
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i, target_url, headers) for i in range(total_requests)]
        for f in as_completed(futures):
            res = f.result()
            results.append(res)
            latencies.append(res["latency_ms"])

    total_time = time.time() - t_start
    throughput = len(latencies) / total_time if total_time > 0 else 0

    sorted_lats = sorted(latencies)
    min_lat = sorted_lats[0]
    p50 = statistics.median(sorted_lats)
    p95 = sorted_lats[int(len(sorted_lats) * 0.95)]
    p99 = sorted_lats[int(len(sorted_lats) * 0.99)]
    max_lat = sorted_lats[-1]
    sla_breaches = sum(1 for l in latencies if l >= 3500.0)

    http_successes = sum(1 for r in results if r["http_success"])
    http_errors = total_requests - http_successes
    routing_successes = sum(1 for r in results if r["routing_success"])
    routing_errors = sum(1 for r in results if r["http_success"] and not r["routing_success"])
    e2e_successes = sum(1 for r in results if r["e2e_success"])

    print(f"Target Service:   {target_tier}")
    print(f"HTTP 200 Success: {http_successes}/{total_requests} ({http_successes/total_requests*100:.1f}%)")
    print(f"HTTP Errors:      {http_errors}")
    print(f"Routing Correct:  {routing_successes}/{total_requests} ({routing_successes/total_requests*100:.1f}%)")
    print(f"Routing Errors:   {routing_errors}")
    print(f"True E2E Success: {e2e_successes}/{total_requests} ({e2e_successes/total_requests*100:.1f}%)")
    print(f"Total Wall Time:  {total_time:.2f} seconds")
    print(f"Throughput:       {throughput:.2f} requests/sec")
    print(f"Min Latency:      {min_lat:.2f} ms")
    print(f"p50 (Median):     {p50:.2f} ms")
    print(f"p95 Latency:      {p95:.2f} ms")
    print(f"p99 Latency:      {p99:.2f} ms")
    print(f"Max Latency:      {max_lat:.2f} ms")
    print(f"SLA Compliance:   {((len(latencies) - sla_breaches) / len(latencies) * 100):.1f}% (< 3500ms)")
    print("=" * 75)

    assert http_errors == 0, f"HTTP errors occurred: {http_errors} requests failed!"
    assert routing_errors == 0, f"Routing errors occurred: {routing_errors} decisions mismatched expected ground truth!"
    assert e2e_successes == total_requests, f"Only {e2e_successes}/{total_requests} succeeded end-to-end!"
    assert sla_breaches == 0, f"SLA breached on {sla_breaches} requests!"

    return {
        "target_tier": target_tier,
        "target_endpoint": target_name,
        "target_url": target_url,
        "total_requests": total_requests,
        "concurrency": concurrency,
        "http_successes": http_successes,
        "http_errors": http_errors,
        "routing_successes": routing_successes,
        "routing_errors": routing_errors,
        "e2e_successes": e2e_successes,
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
