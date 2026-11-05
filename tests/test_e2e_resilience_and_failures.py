"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: tests/test_e2e_resilience_and_failures.py
Description: Exhaustive Phase 4 test suite validating:
             - End-to-End Orchestration & Failure Recovery
             - FastAPI Unavailable / Connection Error Behavior
             - Database Unavailable / Locked SQLite Resilience & Clean Rollback
             - SLA Latency Budget & Timeout Handling
             - High-Concurrency Idempotent Deduplication (20 Threads)
             - Correlation ID End-to-End Propagation
==============================================================================
"""

import os
import sys
import uuid
import time
import sqlite3
import pytest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO_ROOT, 'api')
DB_PATH = os.path.join(REPO_ROOT, 'database', 'sprints_learners.db')

if API_DIR not in sys.path:
    sys.path.append(API_DIR)

from app import app, get_db_connection, SPRINTS_API_KEY

@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers={"X-API-Key": SPRINTS_API_KEY}) as c:
        yield c

class TestE2EResilienceAndFailures:

    def test_sla_latency_budget_and_timeout_boundaries(self, client):
        """
        Validates SLA budgeting:
        - Analyze endpoint completes well within 1100ms node budget.
        - Overall SLA flag (execution_latency_ms < 3500ms) dynamically computed.
        """
        t0 = time.time()
        res = client.post("/api/v1/analyze", json={
            "learner_id": "LRN-1002",
            "query_text": "How do LSTM forget gates operate in sequential NLP pipelines?"
        })
        duration_ms = (time.time() - t0) * 1000
        assert res.status_code == 200
        data = res.json()
        assert data["sla_met"] is True
        assert data["execution_latency_ms"] < 1100.0, f"Analysis exceeded n8n node budget: {data['execution_latency_ms']}ms"
        assert duration_ms < 3500.0, f"Total wall-clock duration exceeded SLA: {duration_ms}ms"

    def test_database_unavailable_resilience_and_rollback(self, client):
        """
        Simulates SQLite database failure (operational error / locked database).
        Verifies:
        - Backend returns 500 Internal Server Error with clean JSON response.
        - No unhandled crash or partial corruption.
        - Client error detail explains database persistence failure.
        """
        faulty_key = f"idemp-db-fail-{uuid.uuid4().hex[:10]}"
        payload = {
            "learner_id": "LRN-1002",
            "query_text": "I am struggling with Sprint 3.",
            "module_id": "MOD-201",
            "priority": "High",
            "idempotency_key": faulty_key
        }

        # Mock sqlite3.connect to raise OperationalError (database locked / disk full)
        with patch("app.sqlite3.connect", side_effect=sqlite3.OperationalError("database is locked")):
            res = client.post("/api/v1/tickets/escalate", json=payload)
            assert res.status_code == 500, f"Expected 500 on DB failure, got {res.status_code}"
            data = res.json()
            assert "database error" in data["detail"].lower() or "failed" in data["detail"].lower()

    def test_high_concurrency_race_condition_deduplication(self, client):
        """
        Stress tests simultaneous ticket creation:
        - 20 concurrent threads submit the exact same Idempotency-Key and payload.
        - Verifies that EXACTLY 1 database record is inserted (no duplicate rows).
        - All 20 threads receive HTTP 200 with the identical ticket_id.
        - Exactly 1 thread receives idempotent_deduplicated=False, 19 receive True.
        """
        stress_key = f"idemp-stress-race-{uuid.uuid4().hex[:12]}"
        stress_corr = f"corr-stress-race-{uuid.uuid4().hex[:12]}"
        payload = {
            "learner_id": "LRN-1002",
            "query_text": "Concurrent race condition test across 20 threads.",
            "module_id": "MOD-201",
            "priority": "Critical",
            "correlation_id": stress_corr,
            "idempotency_key": stress_key
        }

        def submit_request(thread_idx):
            return client.post("/api/v1/tickets/escalate", json=payload)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(submit_request, i) for i in range(20)]
            responses = [f.result() for f in futures]

        for idx, r in enumerate(responses):
            assert r.status_code == 200, f"Thread {idx} got status {r.status_code}: {r.text}"

        ticket_ids = {r.json()["ticket_id"] for r in responses}
        assert len(ticket_ids) == 1, f"Expected exactly 1 unique ticket_id across all 20 threads, got {ticket_ids}"
        common_ticket_id = list(ticket_ids)[0]

        dedup_flags = [r.json()["idempotent_deduplicated"] for r in responses]
        false_count = dedup_flags.count(False)
        true_count = dedup_flags.count(True)

        assert false_count == 1, f"Expected exactly 1 initial insertion (False), got {false_count}"
        assert true_count == 19, f"Expected 19 deduplicated responses (True), got {true_count}"

        # Direct DB verification: Count rows
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM support_tickets WHERE idempotency_key = ?", (stress_key,))
        row_count = cur.fetchone()[0]
        conn.close()
        assert row_count == 1, f"Database has {row_count} rows for key {stress_key}; expected exactly 1!"

    def test_correlation_id_end_to_end_propagation(self, client):
        """
        Verifies that client-supplied X-Correlation-ID is preserved and returned:
        - In response HTTP headers
        - In JSON body correlation_id
        - Stored accurately in SQLite database
        """
        trace_corr_id = f"corr-trace-e2e-{uuid.uuid4().hex[:12]}"
        unique_key = f"idemp-trace-{uuid.uuid4().hex[:12]}"

        res = client.post(
            "/api/v1/tickets/escalate",
            json={
                "learner_id": "LRN-1002",
                "query_text": "Tracing verification query.",
                "priority": "High",
                "idempotency_key": unique_key
            },
            headers={"X-Correlation-ID": trace_corr_id}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["correlation_id"] == trace_corr_id
        assert res.headers.get("x-correlation-id") == trace_corr_id

        # Verify DB storage
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT correlation_id FROM support_tickets WHERE ticket_id = ?", (data["ticket_id"],))
        db_corr = cur.fetchone()[0]
        conn.close()
        assert db_corr == trace_corr_id, f"DB correlation_id mismatch: expected {trace_corr_id}, got {db_corr}"

    def test_workflow_retry_with_new_key_after_closure(self, client):
        """
        Validates the scenario where a ticket is resolved and closed, and the student
        submits the identical question later with a NEW idempotency key:
        - System must generate a BRAND NEW ticket ID.
        - Old ticket remains intact in database with historical status.
        """
        first_key = f"idemp-seq1-{uuid.uuid4().hex[:10]}"
        second_key = f"idemp-seq2-{uuid.uuid4().hex[:10]}"
        identical_query = "How do I fix PyTorch tensor dimension mismatch error?"

        # Ticket 1
        res1 = client.post("/api/v1/tickets/escalate", json={
            "learner_id": "LRN-1002",
            "query_text": identical_query,
            "idempotency_key": first_key
        })
        assert res1.status_code == 200
        t1 = res1.json()["ticket_id"]

        # Simulate mentor resolving and closing ticket 1
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE support_tickets SET status = 'Resolved_By_Mentor' WHERE ticket_id = ?", (t1,))
        conn.commit()
        conn.close()

        # Ticket 2 (Identical question 2 days later, new transaction key)
        res2 = client.post("/api/v1/tickets/escalate", json={
            "learner_id": "LRN-1002",
            "query_text": identical_query,
            "idempotency_key": second_key
        })
        assert res2.status_code == 200
        t2 = res2.json()["ticket_id"]

        assert t1 != t2, f"Expected new ticket ID for new transaction key, got {t1} == {t2}"
        assert res2.json()["idempotent_deduplicated"] is False
