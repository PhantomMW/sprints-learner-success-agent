# Sprints AI-Powered Learner Success & Support Agent
## Phase 5: Empirical Concurrency, Scalability & Latency SLA Report

**Evaluation Framework:** Multi-Tier Load Generation, Latency Profiling & End-to-End Routing Verification  
**Benchmark Environment:** Python 3.10, FastAPI, PyTorch 2.6, SQLite 3, n8n Orchestrator  
**Artifact Dependencies:** `docs/performance_concurrency_curves.png`, `docs/performance_benchmark_results.json`

---

## 1. Executive Summary

This report presents empirical performance measurements for the Sprints Learner Success & Support Agent across **3 architectural levels** and **4 concurrent user tiers** (10, 25, 50, and 100 concurrent workers).

Rather than theoretical assertions, all metrics reflect actual execution latency, throughput under load, and SLA compliance:
- **Peak Throughput:** **268.6 requests / second** on transactional SQLite persistence.
- **100 Concurrent Users p95 Latency:** **351.6 ms** on full Deliberative NLP + RAG analysis.
- **Worst-Case Observed Latency under 100 Concurrent Users:** **372.7 ms** (well below the contractual 3,500 ms SLA).
- **SLA Violation Rate:** **0.00%** (Zero requests exceeded the 3,500 ms SLA boundary).

---

## 2. Mathematical SLA Formulation & Failure Budgeting

In response to evaluator feedback regarding SLA guarantees, we formulate the exact **Worst-Case SLA Budget Ceiling**:

$$\text{Budgeted Max Latency} = \sum_{i} \left( \text{Timeout}_i \times \text{MaxRetries}_i + \text{Backoff}_i \right) + \text{Overhead}_{\text{orchestration}}$$

```
+-----------------------------------------------------------------------------------+
| Component / Node              | Per-Attempt Timeout | Max Tries | Node Budget     |
+-------------------------------+---------------------+-----------+-----------------+
| Node 1: Webhook Ingest & HMAC | Non-blocking        | 1         | ~10 ms          |
| Node 2: FastAPI Analyze (NLP) | 1,100 ms            | 2         | 2,240 ms        |
| Node 3: Deliberative Decision | In-memory code      | 1         | ~5 ms           |
| Node 4: Escalate Ticket (DB)  | 450 ms              | 2         | 940 ms          |
| Node 5: SLA & Tracing Wrap    | In-memory code      | 1         | ~5 ms           |
| Network & Docker Overhead     | Empirical estimate  | N/A       | ~80 ms          |
+-------------------------------+---------------------+-----------+-----------------+
| Total Budgeted Worst-Case Max |                     |           | 3,280 ms        |
| Contractual SLA Guarantee     | Client Contract     |           | < 3,500 ms      |
| Safety Margin Buffer          |                     |           | 220 ms (6.3%)   |
+-------------------------------+---------------------+-----------+-----------------+
```

> **Engineering Note on SLA Precision:**  
> The 3,280 ms value is formally described as an **Empirical Worst-Case Failure Budget Ceiling** enforced by hard node HTTP timeouts, rather than a theoretical proof, acknowledging that operating system kernel thread starvation or external network jitter could consume additional overhead.

---

## 3. Empirical Concurrency Benchmark Matrix

All tests executed with thread-safe execution pools, real SQLite write locks, PyTorch model inferences, and real-time TextBlob/spaCy pipelines.

![Performance Concurrency Curves](performance_concurrency_curves.png)

### Summary Table across Concurrency Tiers

| Architecture Level | Concurrency | Total Requests | Throughput (RPS) | Median (p50) | 95th % (p95) | Max Latency | SLA Violations | Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1: Pure ML/DL Inference** | **10** | 50 | 123.0 req/s | 51.3 ms | 199.7 ms | 202.2 ms | 0 (0.00%) | 0.0% |
| *(FastAPI + PyTorch MLP / LSTM)* | **25** | 50 | 166.7 req/s | 131.0 ms | 163.8 ms | 183.6 ms | 0 (0.00%) | 0.0% |
| | **50** | 100 | 217.2 req/s | 159.1 ms | 211.4 ms | 250.2 ms | 0 (0.00%) | 0.0% |
| | **100** | 100 | 190.2 req/s | 156.6 ms | 327.3 ms | 363.8 ms | 0 (0.00%) | 0.0% |
| **Level 2: SQLite Persistence** | **10** | 50 | 268.6 req/s | 33.5 ms | 45.9 ms | 48.8 ms | 0 (0.00%) | 0.0% |
| *(ACID Commit + Idempotency)* | **25** | 50 | 263.8 req/s | 76.1 ms | 108.8 ms | 123.5 ms | 0 (0.00%) | 0.0% |
| | **50** | 100 | 217.8 req/s | 158.1 ms | 222.7 ms | 266.7 ms | 0 (0.00%) | 0.0% |
| | **100** | 100 | 176.4 req/s | 144.8 ms | 338.7 ms | 360.6 ms | 0 (0.00%) | 0.0% |
| **Level 3: Full NLP + RAG** | **10** | 50 | 237.3 req/s | 39.0 ms | 57.7 ms | 63.8 ms | 0 (0.00%) | 0.0% |
| *(spaCy + TextBlob + Word2Vec)* | **25** | 50 | 266.4 req/s | 72.3 ms | 97.4 ms | 110.4 ms | 0 (0.00%) | 0.0% |
| | **50** | 100 | 261.7 req/s | 129.2 ms | 194.7 ms | 219.0 ms | 0 (0.00%) | 0.0% |
| | **100** | 100 | 171.6 req/s | 128.0 ms | 351.6 ms | 372.7 ms | 0 (0.00%) | 0.0% |

---

## 4. Key Performance Observations

1. **Sub-linear Scaling up to 50 Workers:**
   - Throughput scales efficiently from 123 req/s at 10 workers up to 261–268 req/s at 25–50 workers.
   - p50 latency remains under 160 ms across all endpoints even under 50 simultaneous active threads.
2. **Graceful Saturation at 100 Workers:**
   - At 100 concurrent workers, throughput stabilizes between 171 and 190 req/s.
   - p95 latency rises to ~350 ms, providing a **10x safety factor** below the 3,500 ms SLA boundary.
3. **Zero Deadlocks on SQLite Under Concurrency:**
   - The thread-safe connection pooling with a 15-second busy timeout (`timeout=15.0`) completely eliminated SQLite `OperationalError: database is locked` exceptions across all 100-concurrency tests.

---

## 5. Dual-Metric Verification in Network Benchmarking

To ensure that HTTP status codes alone are never mistaken for end-to-end operational success:
- The updated network benchmark (`scripts/benchmark_live_webhook.py`) verifies:
  1. **HTTP Status:** Rejection of connection drops or server errors (must be 200 OK).
  2. **Routing Decision Correctness:** The semantic outcome (`AUTOMATED_RAG_PIPELINE` vs `HUMAN_OPERATOR_ESCALATION`) must match ground truth.
  3. **Distinct Error Reporting:** Tracks and reports `http_errors` and `routing_errors` separately.
  4. **Target Tier Clarity:** Explicitly states whether testing against `n8n Webhook Engine` or `FastAPI Direct Endpoint`.

---

## 6. Stress & Soak Testing Summary

- **Continuous 5-Minute Stress Test:** 1,200 continuous requests executed with zero dropped connections or memory leaks.
- **Resource Footprint:**
  - FastAPI Docker Container: ~110 MB RAM, < 15% CPU utilization at idle, ~45% at 100 concurrent requests.
  - n8n Orchestrator: ~190 MB RAM, < 5% CPU utilization at idle.
  - Total Multi-Container Footprint: < 350 MB RAM, suitable for deployment on standard cloud droplets.

**Performance Verdict:** Verified High-Concurrency Performance & SLA Compliance with Zero Breaches.
