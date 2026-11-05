"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: tests/test_security_and_auth.py
Description: Production-Grade Security, API Authentication, Authorization,
             and Sliding-Window Rate Limiting Verification Suite.
==============================================================================
"""

import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO_ROOT, 'api')
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
for d in [API_DIR, SCRIPTS_DIR]:
    if d not in sys.path:
        sys.path.append(d)

from app import app, load_all_models, SPRINTS_API_KEY, check_rate_limit, _rate_limit_history, RATE_LIMIT_PER_MINUTE

@pytest.fixture(scope="module", autouse=True)
def init_app():
    load_all_models()

@pytest.fixture
def unauthenticated_client():
    """Client with zero credentials to test authentication boundaries."""
    return TestClient(app)

@pytest.fixture
def authenticated_client():
    """Client authenticated via standard X-API-Key header."""
    return TestClient(app, headers={"X-API-Key": SPRINTS_API_KEY})

class TestSecurityAndAuthentication:

    def test_public_endpoints_accessible_without_auth(self, unauthenticated_client):
        """
        Confirms that healthcheck and interactive demo root are public for evaluators
        and infrastructure monitoring, while sensitive computational endpoints are locked.
        """
        # Healthcheck must be publicly reachable
        res_health = unauthenticated_client.get("/api/v1/health")
        assert res_health.status_code == 200, f"Public healthcheck failed: {res_health.text}"
        data = res_health.json()
        assert data.get("status") == "healthy"

        # Interactive dashboard root must be publicly viewable
        res_root = unauthenticated_client.get("/")
        assert res_root.status_code == 200, f"Public dashboard failed: {res_root.text}"
        assert "Sprints AI" in res_root.text

    def test_unauthenticated_requests_fail_with_401(self, unauthenticated_client):
        """
        Verifies that all 5 predictive and write endpoints return HTTP 401 Unauthorized
        when called without authentication credentials.
        """
        protected_endpoints = [
            ("/api/v1/analyze", {"learner_id": "LRN-1002", "query_text": "What is autograd?"}),
            ("/api/v1/tickets/escalate", {
                "learner_id": "LRN-1005", "query_text": "Failing class",
                "priority": "Critical", "status": "Pending_Human_Escalation"
            }),
            ("/api/v1/predict_risk", {
                "learner_id": "LRN-1002", "week_1_quiz": 80.0, "week_2_quiz": 75.0,
                "week_3_quiz": 70.0, "week_4_quiz": 65.0, "week_5_quiz": 60.0
            }),
            ("/api/v1/predict_risk_early", {
                "learner_id": "LRN-1002", "week_1_quiz": 80.0, "week_2_quiz": 75.0, "week_3_quiz": 70.0
            }),
            ("/api/v1/predict_sequence", {
                "learner_id": "LRN-1002", "quiz_scores": [80.0, 75.0, 70.0, 65.0, 60.0]
            })
        ]

        for path, body in protected_endpoints:
            res = unauthenticated_client.post(path, json=body)
            assert res.status_code == 401, f"Expected 401 Unauthorized on {path}, got {res.status_code}: {res.text}"
            detail = res.json().get("detail", "")
            assert "Unauthorized" in detail or "Missing or invalid API Key" in detail
            assert "WWW-Authenticate" in res.headers

    def test_invalid_api_key_fails_with_401(self, unauthenticated_client):
        """
        Verifies that sending an incorrect or forged API key strictly fails with HTTP 401.
        """
        attacker_headers = {"X-API-Key": "malicious-attacker-forged-key-999"}
        res = unauthenticated_client.post(
            "/api/v1/analyze",
            json={"learner_id": "LRN-1002", "query_text": "Test query"},
            headers=attacker_headers
        )
        assert res.status_code == 401
        assert "Unauthorized" in res.json().get("detail", "")

    def test_valid_x_api_key_header_succeeds(self, authenticated_client):
        """
        Verifies that standard X-API-Key header permits authorized execution.
        """
        res = authenticated_client.post(
            "/api/v1/analyze",
            json={"learner_id": "LRN-1002", "query_text": "How do PyTorch tensors work?"}
        )
        assert res.status_code == 200, f"Authorized call failed: {res.text}"
        assert res.json().get("routing_decision") == "AUTOMATED_RAG_PIPELINE"

    def test_bearer_token_authorization_header_succeeds(self, unauthenticated_client):
        """
        Verifies that Authorization: Bearer <token> is supported as an industry-standard alternative.
        """
        bearer_headers = {"Authorization": f"Bearer {SPRINTS_API_KEY}"}
        res = unauthenticated_client.post(
            "/api/v1/analyze",
            json={"learner_id": "LRN-1002", "query_text": "What is SQL schema normalization?"},
            headers=bearer_headers
        )
        assert res.status_code == 200
        assert res.json().get("routing_decision") == "AUTOMATED_RAG_PIPELINE"

    def test_student_data_and_tickets_protected_against_unauthorized_probes(self, unauthenticated_client):
        """
        Verifies that unauthorized clients cannot extract predictions or create tickets
        for arbitrary learner IDs.
        """
        # Attempt to escalate ticket without credentials
        ticket_payload = {
            "learner_id": "LRN-1005",
            "query_text": "Unauthorized ticket creation probe",
            "priority": "Critical",
            "status": "Pending_Human_Escalation"
        }
        res = unauthenticated_client.post("/api/v1/tickets/escalate", json=ticket_payload)
        assert res.status_code == 401, "Data write boundary breached without authentication!"

    def test_rate_limiting_sliding_window_logic(self):
        """
        Tests the core in-memory sliding window rate limiter function.
        Simulates exceeding the limit and verifying pruning of expired entries.
        """
        test_ip = "192.168.1.200"
        limit = 5

        # Clear any prior history for test IP
        _rate_limit_history[test_ip] = []

        # Send exactly 5 requests -> all must be allowed
        for i in range(limit):
            allowed = check_rate_limit(test_ip, limit=limit)
            assert allowed is True, f"Request {i+1} should be allowed"

        # 6th request within same window must be rejected
        blocked = check_rate_limit(test_ip, limit=limit)
        assert blocked is False, "6th request should exceed the limit and be blocked!"

        # Simulate time passing by modifying timestamps to 65 seconds ago
        _rate_limit_history[test_ip] = [t - 65.0 for t in _rate_limit_history[test_ip]]

        # Now a new request must be permitted
        allowed_after_decay = check_rate_limit(test_ip, limit=limit)
        assert allowed_after_decay is True, "Request after sliding window expiry must be permitted"

    def test_rate_limiting_http_429_response(self, unauthenticated_client):
        """
        Verifies that when a client IP exceeds rate limit on an endpoint,
        FastAPI returns HTTP 429 Too Many Requests with a Retry-After header.
        """
        flood_ip = "10.0.0.88"
        headers = {"X-Forwarded-For": flood_ip}
        _rate_limit_history[flood_ip] = []

        # Flood RATE_LIMIT_PER_MINUTE requests
        for _ in range(RATE_LIMIT_PER_MINUTE):
            check_rate_limit(flood_ip, limit=RATE_LIMIT_PER_MINUTE)

        # The 121st request via HTTP must return 429
        res = unauthenticated_client.get("/api/v1/health", headers=headers)
        assert res.status_code == 429, f"Expected 429 Too Many Requests, got {res.status_code}"
        assert "Retry-After" in res.headers
        assert "Rate limit exceeded" in res.json().get("detail", "")
