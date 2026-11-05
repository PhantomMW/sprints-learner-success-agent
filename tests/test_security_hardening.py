"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: tests/test_security_hardening.py
Description: Exhaustive Automated Test Suite for Phase 1: Security Hardening.
             Validates Enterprise-Grade OWASP Boundaries:
             1. Webhook HMAC-SHA256 signature verification & tamper detection.
             2. Replay attack rejection & sliding timestamp drift protection.
             3. HTTP 413 Payload Too Large enforcement on oversized requests.
             4. Session lifecycle management (Login, Cookie auth, Logout revocation).
             5. Protection against XSS script tags and SQL injection vectors.
             6. Strict CORS origin access control.
==============================================================================
"""

import os
import sys
import time
import json
import hmac
import hashlib
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO_ROOT, 'api')
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
for d in [API_DIR, SCRIPTS_DIR]:
    if d not in sys.path:
        sys.path.append(d)

from app import (
    app,
    load_all_models,
    SPRINTS_API_KEY,
    WEBHOOK_SECRET,
    create_session_token,
    verify_session_token,
    revoke_session_token,
    _seen_webhook_signatures
)

@pytest.fixture(scope="module", autouse=True)
def init_models():
    load_all_models()

@pytest.fixture
def client():
    return TestClient(app)

def generate_webhook_signature(body: bytes, timestamp: str, secret: str = WEBHOOK_SECRET) -> str:
    """Computes canonical HMAC-SHA256 signature over f'{timestamp}.{body}'."""
    signed_payload = f"{timestamp}.".encode('utf-8') + body
    return "sha256=" + hmac.new(secret.encode('utf-8'), signed_payload, hashlib.sha256).hexdigest()

class TestSecurityHardening:

    # ==========================================================================
    # 1. n8n WEBHOOK HMAC-SHA256 SIGNATURE VERIFICATION
    # ==========================================================================

    def test_webhook_missing_signature_returns_401(self, client):
        """Webhooks missing X-Signature-SHA256 must be rejected with HTTP 401."""
        payload = {"learner_id": "LRN-1002", "query_text": "What is backpropagation in PyTorch?"}
        res = client.post("/api/v1/webhook/learner-query", json=payload)
        assert res.status_code == 401
        assert "Missing signature header" in res.json().get("detail", "")

    def test_webhook_missing_timestamp_returns_401(self, client):
        """Webhooks with signature but missing X-Timestamp must be rejected with HTTP 401."""
        body = json.dumps({"learner_id": "LRN-1002", "query_text": "What is backpropagation?"}).encode('utf-8')
        sig = generate_webhook_signature(body, str(time.time()))
        res = client.post(
            "/api/v1/webhook/learner-query",
            content=body,
            headers={"Content-Type": "application/json", "X-Signature-SHA256": sig}
        )
        assert res.status_code == 401
        assert "Missing timestamp header" in res.json().get("detail", "")

    def test_webhook_tampered_or_invalid_signature_returns_401(self, client):
        """Tampered payload or forged signature must fail with HTTP 401."""
        body = json.dumps({"learner_id": "LRN-1002", "query_text": "Tampered payload"}).encode('utf-8')
        ts = str(time.time())
        fake_sig = "sha256=" + "a" * 64
        res = client.post(
            "/api/v1/webhook/learner-query",
            content=body,
            headers={"Content-Type": "application/json", "X-Signature-SHA256": fake_sig, "X-Timestamp": ts}
        )
        assert res.status_code == 401
        assert "Invalid HMAC-SHA256 signature" in res.json().get("detail", "")

    def test_webhook_valid_hmac_signature_succeeds(self, client):
        """Legitimate webhook request with authentic HMAC signature must succeed with 200."""
        payload = {"learner_id": "LRN-1002", "query_text": "Can you explain how optimizer step works?"}
        body = json.dumps(payload).encode('utf-8')
        ts = str(time.time())
        sig = generate_webhook_signature(body, ts)

        res = client.post(
            "/api/v1/webhook/learner-query",
            content=body,
            headers={"Content-Type": "application/json", "X-Signature-SHA256": sig, "X-Timestamp": ts}
        )
        assert res.status_code == 200
        data = res.json()
        assert data.get("webhook_authenticated") is True
        assert data.get("signature_verified") is True
        assert data.get("routing_decision") == "AUTOMATED_RAG_PIPELINE"

    # ==========================================================================
    # 2. REPLAY ATTACK DEFENSE & TIMESTAMP DRIFT
    # ==========================================================================

    def test_webhook_replay_attack_rejected(self, client):
        """Replaying the exact same signed event twice must be blocked with HTTP 401."""
        payload = {"learner_id": "LRN-1002", "query_text": "Replay test question"}
        body = json.dumps(payload).encode('utf-8')
        ts = str(time.time() + 0.123)
        sig = generate_webhook_signature(body, ts)
        headers = {"Content-Type": "application/json", "X-Signature-SHA256": sig, "X-Timestamp": ts}

        # First request succeeds
        res1 = client.post("/api/v1/webhook/learner-query", content=body, headers=headers)
        assert res1.status_code == 200

        # Immediate second identical replay must fail
        res2 = client.post("/api/v1/webhook/learner-query", content=body, headers=headers)
        assert res2.status_code == 401
        assert "Replay attack detected" in res2.json().get("detail", "")

    def test_webhook_stale_expired_timestamp_rejected(self, client):
        """Requests with timestamps drifting > 300 seconds must fail with HTTP 401."""
        body = json.dumps({"learner_id": "LRN-1002", "query_text": "Old timestamp query"}).encode('utf-8')
        stale_ts = str(time.time() - 400.0) # 400 seconds in the past (> 300s tolerance)
        sig = generate_webhook_signature(body, stale_ts)

        res = client.post(
            "/api/v1/webhook/learner-query",
            content=body,
            headers={"Content-Type": "application/json", "X-Signature-SHA256": sig, "X-Timestamp": stale_ts}
        )
        assert res.status_code == 401
        assert "clock drift" in res.json().get("detail", "")

    # ==========================================================================
    # 3. HTTP 413 PAYLOAD TOO LARGE DEFENSE
    # ==========================================================================

    def test_oversized_payload_returns_http_413(self, client):
        """Payloads exceeding MAX_PAYLOAD_SIZE (64KB) must be rejected with HTTP 413."""
        giant_query = "x" * (70 * 1024) # 70 KB > 64 KB limit
        res = client.post(
            "/api/v1/analyze",
            json={"learner_id": "LRN-1002", "query_text": giant_query},
            headers={"X-API-Key": SPRINTS_API_KEY}
        )
        assert res.status_code == 413
        assert "Payload Too Large" in res.json().get("detail", "")

    # ==========================================================================
    # 4. DASHBOARD SESSION MANAGEMENT LIFECYCLE
    # ==========================================================================

    def test_session_login_with_valid_key_sets_httponly_cookie(self, client):
        """POST /api/v1/auth/login exchanges API Key for secure session token & cookie."""
        res = client.post("/api/v1/auth/login", json={"api_key": SPRINTS_API_KEY})
        assert res.status_code == 200
        data = res.json()
        assert data.get("status") == "authenticated"
        assert "session_token" in data
        assert "sprints_session_id" in res.cookies

    def test_session_login_with_invalid_key_fails(self, client):
        """POST /api/v1/auth/login with wrong key fails with 401."""
        res = client.post("/api/v1/auth/login", json={"api_key": "wrong-secret-key-xyz"})
        assert res.status_code == 401

    def test_session_cookie_authenticates_operational_endpoints(self, client):
        """Caller possessing valid session cookie can query endpoints without sending raw API key."""
        # 1. Login to obtain session cookie
        login_res = client.post("/api/v1/auth/login", json={"api_key": SPRINTS_API_KEY})
        assert login_res.status_code == 200

        # 2. Invoke /api/v1/analyze relying purely on session cookie (no X-API-Key header!)
        res = client.post(
            "/api/v1/analyze",
            json={"learner_id": "LRN-1002", "query_text": "Explain PyTorch tensors briefly."},
            cookies=login_res.cookies
        )
        assert res.status_code == 200
        assert res.json().get("routing_decision") == "AUTOMATED_RAG_PIPELINE"

    def test_session_logout_revokes_token_and_invalidates_access(self, client):
        """POST /api/v1/auth/logout revokes session token; subsequent calls fail with 401."""
        # 1. Login
        login_res = client.post("/api/v1/auth/login", json={"api_key": SPRINTS_API_KEY})
        cookies = login_res.cookies

        # Verify active session
        sess_check = client.get("/api/v1/auth/session", cookies=cookies)
        assert sess_check.status_code == 200

        # 2. Logout
        logout_res = client.post("/api/v1/auth/logout", cookies=cookies)
        assert logout_res.status_code == 200

        # 3. Call with old cookie must fail with 401
        res = client.post(
            "/api/v1/analyze",
            json={"learner_id": "LRN-1002", "query_text": "Explain PyTorch tensors briefly."},
            cookies=cookies
        )
        assert res.status_code == 401

    # ==========================================================================
    # 5. INPUT SANITIZATION & INJECTION ATTACK VECTORS
    # ==========================================================================

    def test_xss_script_injection_rejected(self, client):
        """Payloads containing <script> or javascript: tags must be rejected with HTTP 422."""
        xss_payload = "<script>alert('pwned')</script> I am confused about tensors"
        res = client.post(
            "/api/v1/analyze",
            json={"learner_id": "LRN-1002", "query_text": xss_payload},
            headers={"X-API-Key": SPRINTS_API_KEY}
        )
        assert res.status_code == 422
        assert "XSS vector" in str(res.json())

    def test_sql_injection_learner_id_rejected(self, client):
        """Learner IDs containing SQL injection syntax must be rejected with HTTP 422."""
        sqli_id = "LRN-1001'--"
        res = client.post(
            "/api/v1/analyze",
            json={"learner_id": sqli_id, "query_text": "Routine explanation request"},
            headers={"X-API-Key": SPRINTS_API_KEY}
        )
        assert res.status_code == 422
        assert "SQL injection" in str(res.json())

    # ==========================================================================
    # 6. KEY ROTATION & OUTDATED CREDENTIAL INVALIDATION
    # ==========================================================================

    def test_outdated_or_rotated_key_fails(self, client):
        """An old, rotated key (e.g. sprints-capstone-secret-api-key-2026) must fail with HTTP 401."""
        res = client.post(
            "/api/v1/analyze",
            json={"learner_id": "LRN-1002", "query_text": "What is cross entropy?"},
            headers={"X-API-Key": "sprints-capstone-secret-api-key-2026"}
        )
        assert res.status_code == 401
