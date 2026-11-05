"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: tests/test_latency_under_load.py
Description: Automated Concurrency & Latency SLA Verification Test.
==============================================================================
"""

import os
import sys
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from benchmark_latency import run_concurrent_latency_benchmark

class TestLatencyUnderLoad:

    def test_concurrent_throughput_and_latency_sla(self):
        """
        Executes concurrent load of 30 requests across 5 worker threads.
        Asserts empirical p95 < 600ms, max < 3500ms, and zero SLA breaches.
        """
        metrics = run_concurrent_latency_benchmark(total_requests=30, concurrency=5)
        
        assert metrics["p95_ms"] < 600.0, f"p95 latency exceeded 600ms: {metrics['p95_ms']}ms"
        assert metrics["max_ms"] < 3500.0, f"Max latency breached 3,500ms SLA: {metrics['max_ms']}ms"
        assert metrics["sla_breached"] is False, "SLA breach detected under concurrent load!"
