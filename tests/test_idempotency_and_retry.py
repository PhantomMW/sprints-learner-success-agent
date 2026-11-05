"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: tests/test_idempotency_and_retry.py
Description: Exhaustive verification suite for:
             - True Idempotency-Key handling & DB unique constraint
             - Network failure & retry simulation (same ticket ID returned)
             - Payload conflict detection (HTTP 409 Conflict on payload mismatch with same key)
             - Database Migration Fail-Fast & post-migration verification (RuntimeError on failure)
             - Collision-free UUID ticket ID generation
             - Dynamic SLA calculation across all endpoints
             - Same learner with different queries (distinct tickets created)
             - Same query resubmitted after resolution with new key (new ticket created)
             - High-concurrency race condition testing (Thread pool simultaneous requests)
             - End-to-End Correlation ID tracing across API & SQLite DB
==============================================================================
"""

import os
import sys
import uuid
import sqlite3
import pytest
from concurrent.futures import ThreadPoolExecutor
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO_ROOT, 'api')
DB_PATH = os.path.join(REPO_ROOT, 'database', 'sprints_learners.db')

if API_DIR not in sys.path:
    sys.path.append(API_DIR)

from app import app, get_db_connection, ensure_db_schema, SPRINTS_API_KEY

@pytest.fixture(scope="module")
def client():
    with TestClient(app, headers={"X-API-Key": SPRINTS_API_KEY}) as c:
        yield c

class TestIdempotencyAndRetry:

    def test_idempotent_retry_returns_same_ticket_id(self, client):
        """
        Simulates:
        1. Client sends escalation request with unique Idempotency-Key.
        2. Ticket is committed to SQLite.
        3. Response is dropped / connection interrupted.
        4. Client/n8n retries the identical request with the same Idempotency-Key.
        5. API must recognize the key and return the EXACT SAME ticket_id with idempotent_deduplicated=True.
        """
        unique_key = f"idemp-retry-test-{uuid.uuid4().hex[:10]}"
        corr_id = f"corr-retry-test-{uuid.uuid4().hex[:10]}"

        payload = {
            "learner_id": "LRN-1002",
            "query_text": "I am struggling and need mentor advice on Sprint 2.",
            "module_id": "MOD-102",
            "sentiment_polarity": -0.55,
            "sentiment_subjectivity": 0.75,
            "priority": "High",
            "status": "Pending_Human_Escalation",
            "correlation_id": corr_id,
            "idempotency_key": unique_key
        }

        # First request (Initial submission)
        res1 = client.post("/api/v1/tickets/escalate", json=payload)
        assert res1.status_code == 200, f"Expected 200, got {res1.status_code}: {res1.text}"
        data1 = res1.json()
        assert data1["status"] == "persisted"
        assert data1["idempotent_deduplicated"] is False
        assert data1["idempotency_key"] == unique_key
        assert data1["correlation_id"] == corr_id
        ticket_id = data1["ticket_id"]

        # Second request (Retry simulation with exact same Idempotency-Key)
        res2 = client.post("/api/v1/tickets/escalate", json=payload)
        assert res2.status_code == 200, f"Expected 200 on retry, got {res2.status_code}: {res2.text}"
        data2 = res2.json()
        assert data2["ticket_id"] == ticket_id, f"Expected same ticket {ticket_id}, got {data2['ticket_id']}"
        assert data2["idempotent_deduplicated"] is True, "Expected idempotent_deduplicated to be True on retry"
        assert data2["idempotency_key"] == unique_key

        # Third request (Using HTTP Header Idempotency-Key)
        payload_no_body_key = payload.copy()
        payload_no_body_key.pop("idempotency_key")
        res3 = client.post(
            "/api/v1/tickets/escalate",
            json=payload_no_body_key,
            headers={"Idempotency-Key": unique_key, "X-Correlation-ID": corr_id}
        )
        assert res3.status_code == 200
        data3 = res3.json()
        assert data3["ticket_id"] == ticket_id
        assert data3["idempotent_deduplicated"] is True

        # Database verification: Exactly 1 row persisted
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM support_tickets WHERE idempotency_key = ?", (unique_key,))
        count = cur.fetchone()[0]
        conn.close()
        assert count == 1, f"Expected exactly 1 DB row for key {unique_key}, found {count}"

    def test_idempotency_payload_conflict_returns_409(self, client):
        """
        P0 Security & Integrity Check:
        If a client sends an existing Idempotency-Key but with a DIFFERENT request payload
        (e.g. different query_text or learner_id), the API must reject it with HTTP 409 Conflict.
        """
        shared_key = f"idemp-conflict-test-{uuid.uuid4().hex[:10]}"
        
        payload_original = {
            "learner_id": "LRN-1002",
            "query_text": "Original query: I need help with PyTorch tensors.",
            "module_id": "MOD-201",
            "priority": "High",
            "idempotency_key": shared_key
        }

        # Step 1: Submit original request -> 200 OK
        res1 = client.post("/api/v1/tickets/escalate", json=payload_original)
        assert res1.status_code == 200
        assert res1.json()["idempotent_deduplicated"] is False

        # Step 2: Submit conflicting payload with the SAME key
        payload_conflicting = {
            "learner_id": "LRN-1002",
            "query_text": "Mismatched query: How to use SQL GROUP BY clauses?",
            "module_id": "MOD-101",
            "priority": "Medium",
            "idempotency_key": shared_key
        }
        res2 = client.post("/api/v1/tickets/escalate", json=payload_conflicting)
        assert res2.status_code == 409, f"Expected 409 Conflict, got {res2.status_code}: {res2.text}"
        error_detail = res2.json().get("detail", "")
        assert "Idempotency key collision" in error_detail or "different request payload" in error_detail

    def test_idempotency_payload_conflict_on_attribute_changes(self, client):
        """
        P0 Security Check:
        Verifies that payload hash covers ALL significant ticket attributes.
        If a client sends an existing Idempotency-Key with identical text/learner
        but altered priority, module, or sentiment, it MUST return HTTP 409 Conflict.
        """
        shared_key = f"idemp-attr-conflict-{uuid.uuid4().hex[:10]}"
        base_payload = {
            "learner_id": "LRN-1002",
            "query_text": "I am struggling with PyTorch neural network autograd.",
            "module_id": "MOD-201",
            "priority": "High",
            "sentiment_polarity": -0.65,
            "sentiment_subjectivity": 0.85,
            "idempotency_key": shared_key
        }

        # Step 1: Initial submission -> 200 OK
        res1 = client.post("/api/v1/tickets/escalate", json=base_payload)
        assert res1.status_code == 200

        # Step 2: Same key, but altered priority (High -> Critical)
        p_priority = base_payload.copy()
        p_priority["priority"] = "Critical"
        res2 = client.post("/api/v1/tickets/escalate", json=p_priority)
        assert res2.status_code == 409, f"Expected 409 on priority alteration, got {res2.status_code}"

        # Step 3: Same key, but altered sentiment polarity (-0.65 -> 0.10)
        p_sentiment = base_payload.copy()
        p_sentiment["sentiment_polarity"] = 0.10
        res3 = client.post("/api/v1/tickets/escalate", json=p_sentiment)
        assert res3.status_code == 409, f"Expected 409 on sentiment alteration, got {res3.status_code}"

        # Step 4: Same key, but altered module (MOD-201 -> MOD-101)
        p_module = base_payload.copy()
        p_module["module_id"] = "MOD-101"
        res4 = client.post("/api/v1/tickets/escalate", json=p_module)
        assert res4.status_code == 409, f"Expected 409 on module alteration, got {res4.status_code}"

    def test_database_migration_fail_fast_and_verification(self):
        """
        P0 Database Migration & Resilience Check:
        1. Calling ensure_db_schema on a database missing support_tickets table strictly raises RuntimeError.
        2. Calling ensure_db_schema on a legacy database table automatically applies migrations
           and verifies column and index presence.
        """
        # Test 1: Missing table raises RuntimeError
        in_memory_conn = sqlite3.connect(":memory:")
        with pytest.raises(RuntimeError) as exc_info:
            ensure_db_schema(in_memory_conn)
        assert "support_tickets" in str(exc_info.value)
        in_memory_conn.close()

        # Test 2: Migration of legacy table adds missing columns & indexes
        test_conn = sqlite3.connect(":memory:")
        test_cur = test_conn.cursor()
        test_cur.execute("""
            CREATE TABLE support_tickets (
                ticket_id VARCHAR(32) PRIMARY KEY,
                learner_id VARCHAR(32),
                query_text TEXT,
                status VARCHAR(32)
            );
        """)
        test_conn.commit()

        # Run migration
        ensure_db_schema(test_conn)

        # Verify columns exist
        test_cur.execute("PRAGMA table_info(support_tickets);")
        cols = {c[1] for c in test_cur.fetchall()}
        assert "correlation_id" in cols
        assert "idempotency_key" in cols
        assert "payload_hash" in cols

        # Verify indexes exist
        test_cur.execute("PRAGMA index_list(support_tickets);")
        indexes = {idx[1] for idx in test_cur.fetchall()}
        assert "idx_tickets_idempotency" in indexes
        assert "idx_tickets_correlation" in indexes
        test_conn.close()

    def test_collision_free_uuid_ticket_ids(self, client):
        """
        P1 Ticket ID Collision Check:
        Verifies that ticket IDs generated by escalate endpoint use collision-free UUIDs
        (e.g., TCK-ESC-...).
        """
        ticket_ids = set()
        for i in range(5):
            res = client.post("/api/v1/tickets/escalate", json={
                "learner_id": "LRN-1002",
                "query_text": f"Distinct inquiry #{i} on advanced neural network backprop.",
                "module_id": "MOD-201",
                "priority": "High"
            })
            assert res.status_code == 200
            t_id = res.json()["ticket_id"]
            assert t_id.startswith("TCK-ESC-")
            assert len(t_id) >= 16  # "TCK-ESC-" + 12 hex chars
            assert t_id not in ticket_ids
            ticket_ids.add(t_id)

    def test_dynamic_sla_met_across_all_endpoints(self, client):
        """
        P1 SLA Monitoring Check:
        Verifies that execution_latency_ms is computed and sla_met is a dynamic boolean
        (execution_latency_ms < 3500.0) across all endpoints.
        """
        endpoints = [
            ("/api/v1/analyze", {"learner_id": "LRN-1001", "query_text": "What is an LSTM forget gate?"}),
            ("/api/v1/predict_risk", {
                "learner_id": "LRN-1001", "week_1_quiz": 85.0, "week_2_quiz": 90.0,
                "week_3_quiz": 80.0, "week_4_quiz": 75.0, "week_5_quiz": 85.0
            }),
            ("/api/v1/predict_risk_early", {
                "learner_id": "LRN-1001", "week_1_quiz": 80.0, "week_2_quiz": 75.0, "week_3_quiz": 70.0
            }),
            ("/api/v1/predict_sequence", {
                "learner_id": "LRN-1001", "quiz_scores": [80.0, 85.0, 90.0, 75.0, 80.0]
            })
        ]

        for path, body in endpoints:
            res = client.post(path, json=body)
            assert res.status_code == 200, f"Endpoint {path} failed: {res.text}"
            data = res.json()
            assert "execution_latency_ms" in data
            assert "sla_met" in data
            assert isinstance(data["sla_met"], bool)
            expected_sla = data["execution_latency_ms"] < 3500.0
            assert data["sla_met"] == expected_sla

    def test_different_queries_create_distinct_tickets(self, client):
        """
        Same student asking two different questions with two different Idempotency-Keys
        must result in two distinct ticket IDs.
        """
        key_a = f"idemp-diff-a-{uuid.uuid4().hex[:10]}"
        key_b = f"idemp-diff-b-{uuid.uuid4().hex[:10]}"

        res_a = client.post("/api/v1/tickets/escalate", json={
            "learner_id": "LRN-1002",
            "query_text": "Question A about PyTorch autograd tensors.",
            "module_id": "MOD-201",
            "priority": "High",
            "idempotency_key": key_a
        })
        res_b = client.post("/api/v1/tickets/escalate", json={
            "learner_id": "LRN-1002",
            "query_text": "Question B about SQL recursive common table expressions.",
            "module_id": "MOD-101",
            "priority": "High",
            "idempotency_key": key_b
        })

        assert res_a.status_code == 200
        assert res_b.status_code == 200
        data_a = res_a.json()
        data_b = res_b.json()
        assert data_a["ticket_id"] != data_b["ticket_id"]
        assert data_a["idempotent_deduplicated"] is False
        assert data_b["idempotent_deduplicated"] is False

    def test_reopen_resolved_ticket_creates_new_ticket(self, client):
        """
        If a student asks the same question after their previous ticket was resolved,
        using a new Idempotency-Key must create a brand-new ticket.
        """
        key_old = f"idemp-resolved-{uuid.uuid4().hex[:10]}"
        query_text = "Repeated question: How to install PyTorch on Mac M1?"

        # Initial ticket
        res1 = client.post("/api/v1/tickets/escalate", json={
            "learner_id": "LRN-1003",
            "query_text": query_text,
            "module_id": "MOD-201",
            "priority": "Medium",
            "idempotency_key": key_old
        })
        ticket_id_1 = res1.json()["ticket_id"]

        # Admin resolves ticket in DB
        conn = get_db_connection()
        conn.execute("UPDATE support_tickets SET status = 'Resolved_By_Agent' WHERE ticket_id = ?", (ticket_id_1,))
        conn.commit()
        conn.close()

        # Student submits same question later with a NEW Idempotency-Key
        key_new = f"idemp-new-{uuid.uuid4().hex[:10]}"
        res2 = client.post("/api/v1/tickets/escalate", json={
            "learner_id": "LRN-1003",
            "query_text": query_text,
            "module_id": "MOD-201",
            "priority": "Medium",
            "idempotency_key": key_new
        })
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["ticket_id"] != ticket_id_1, "Expected new ticket ID for new request after resolution"
        assert data2["idempotent_deduplicated"] is False

    def test_concurrent_idempotent_requests_threadpool(self, client):
        """
        Stress & concurrency test:
        10 threads simultaneously submit escalation requests with the SAME Idempotency-Key.
        All 10 must succeed with HTTP 200, return the identical ticket_id,
        and exactly 1 ticket must exist in SQLite.
        """
        shared_key = f"idemp-concurrent-{uuid.uuid4().hex[:10]}"
        shared_corr = f"corr-concurrent-{uuid.uuid4().hex[:10]}"

        def send_escalation():
            return client.post("/api/v1/tickets/escalate", json={
                "learner_id": "LRN-1004",
                "query_text": "Concurrent stress query testing SQLite UNIQUE lock handling.",
                "module_id": "MOD-101",
                "priority": "Critical",
                "correlation_id": shared_corr,
                "idempotency_key": shared_key
            })

        # Run 10 requests concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(send_escalation) for _ in range(10)]
            responses = [f.result() for f in futures]

        for r in responses:
            assert r.status_code == 200, f"Concurrent request failed: {r.text}"

        ticket_ids = {r.json()["ticket_id"] for r in responses}
        assert len(ticket_ids) == 1, f"Expected exactly 1 distinct ticket ID, got {ticket_ids}"

        # DB assertion
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM support_tickets WHERE idempotency_key = ?", (shared_key,))
        count = cur.fetchone()[0]
        conn.close()
        assert count == 1, f"Expected 1 record in SQLite, found {count}"

    def test_correlation_id_propagated_to_all_endpoints_and_db(self, client):
        """
        Verifies end-to-end correlation_id tracking across all endpoints:
        - /api/v1/analyze
        - /api/v1/tickets/escalate (and persisted in SQLite)
        - /api/v1/predict_risk
        - /api/v1/predict_risk_early
        - /api/v1/predict_sequence
        """
        test_corr = f"corr-e2e-trace-{uuid.uuid4().hex[:8]}"

        # 1. Analyze endpoint
        res_an = client.post("/api/v1/analyze", json={
            "learner_id": "LRN-1001",
            "query_text": "How do I calculate cross entropy loss in PyTorch?",
            "correlation_id": test_corr
        })
        assert res_an.status_code == 200
        assert res_an.json()["correlation_id"] == test_corr

        # 2. Escalate endpoint & DB persistence
        res_esc = client.post("/api/v1/tickets/escalate", json={
            "learner_id": "LRN-1001",
            "query_text": "I am struggling with module MOD-201.",
            "module_id": "MOD-201",
            "priority": "High",
            "correlation_id": test_corr,
            "idempotency_key": f"idemp-trace-{uuid.uuid4().hex[:8]}"
        })
        assert res_esc.status_code == 200
        assert res_esc.json()["correlation_id"] == test_corr
        esc_ticket_id = res_esc.json()["ticket_id"]

        # Check DB column correlation_id
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT correlation_id FROM support_tickets WHERE ticket_id = ?", (esc_ticket_id,))
        row = cur.fetchone()
        conn.close()
        assert row is not None
        assert row["correlation_id"] == test_corr, f"Expected DB correlation_id {test_corr}, got {row['correlation_id']}"

        # 3. Predict risk endpoint
        res_risk = client.post("/api/v1/predict_risk", json={
            "learner_id": "LRN-1001",
            "week_1_quiz": 80.0, "week_2_quiz": 85.0, "week_3_quiz": 75.0,
            "week_4_quiz": 70.0, "week_5_quiz": 90.0,
            "correlation_id": test_corr
        })
        assert res_risk.status_code == 200, f"Predict risk failed: {res_risk.text}"
        assert res_risk.json()["correlation_id"] == test_corr

        # 4. Predict risk early endpoint
        res_early = client.post("/api/v1/predict_risk_early", json={
            "learner_id": "LRN-1001",
            "week_1_quiz": 70.0, "week_2_quiz": 65.0, "week_3_quiz": 60.0,
            "correlation_id": test_corr
        })
        assert res_early.status_code == 200, f"Predict early risk failed: {res_early.text}"
        assert res_early.json()["correlation_id"] == test_corr

        # 5. Predict sequence endpoint
        res_seq = client.post("/api/v1/predict_sequence", json={
            "learner_id": "LRN-1001",
            "quiz_scores": [85.0, 80.0, 75.0, 70.0, 60.0],
            "correlation_id": test_corr
        })
        assert res_seq.status_code == 200, f"Predict sequence failed: {res_seq.text}"
        assert res_seq.json()["correlation_id"] == test_corr

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
