"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: tests/test_dashboard_e2e.py
Description: End-to-End Automated Test Suite for the Interactive Live Dashboard.
             Validates:
             1. Clean HTML Rendering & UTF-8 Encoding.
             2. Mobile Viewport & RTL Localization.
             3. In-Page Dynamic Alerts & Absence of Stale States.
             4. Active Server Health Polling Integration.
             5. Real-Time API Key Authentication Boundary.
==============================================================================
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO_ROOT, 'api')
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
for d in [API_DIR, SCRIPTS_DIR]:
    if d not in sys.path:
        sys.path.append(d)

from app import app, load_all_models, SPRINTS_API_KEY

@pytest.fixture(scope="module", autouse=True)
def setup_models():
    load_all_models()

@pytest.fixture
def client():
    return TestClient(app)

class TestDashboardEndToEnd:

    def test_dashboard_html_rendering_and_metadata(self, client):
        """
        Verifies that GET / returns the live dashboard with valid HTML5 semantics,
        mobile responsiveness meta viewport, Arabic RTL direction, and required DOM controls.
        """
        res = client.get("/")
        assert res.status_code == 200, f"Dashboard route failed: {res.status_code}"
        assert "text/html" in res.headers.get("content-type", "")

        html = res.text
        # Mobile responsiveness check
        assert '<meta name="viewport" content="width=device-width, initial-scale=1.0">' in html
        # Direction and localization check
        assert 'dir="rtl"' in html
        assert 'lang="ar"' in html

        # Key DOM components check
        assert 'id="globalAlertBanner"' in html, "Missing global alert banner for error handling!"
        assert 'id="liveStatusBadge"' in html, "Missing dynamic live server status badge!"
        assert 'id="apiKeyInput"' in html, "Missing API key input field in navbar!"
        assert 'id="nlpQueryText"' in html, "Missing NLP query input!"
        assert 'id="w1"' in html and 'id="w5"' in html, "Missing PyTorch risk quiz sliders!"
        assert 'id="l1"' in html and 'id="l5"' in html, "Missing LSTM sequence sliders!"

    def test_health_probe_polled_by_dashboard(self, client):
        """
        Verifies that the endpoint polled by checkServerHealth() returns healthy
        and lists loaded models, allowing the dynamic live badge to turn green.
        """
        res = client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data.get("status") == "healthy"
        assert len(data.get("models_loaded", [])) >= 5
        assert data.get("database_connected") is True

    def test_dashboard_does_not_leak_api_key_in_html(self, client):
        """
        P0 Security Invariant: GET / must NEVER leak SPRINTS_API_KEY or session tokens
        in the response HTML or source code.
        Users must provide their key via header X-API-Key or client-side input.
        """
        res = client.get("/")
        assert res.status_code == 200
        assert "{{SESSION_API_KEY}}" not in res.text, "Unreplaced template placeholder in HTML!"
        assert SPRINTS_API_KEY not in res.text, "CRITICAL SECURITY BREACH: SPRINTS_API_KEY was leaked in GET / HTML!"

    def test_dashboard_api_key_auth_boundary(self, client):
        """
        Verifies that requests sent from the dashboard with the default key succeed,
        while requests with corrupted keys fail with HTTP 401 Unauthorized.
        """
        # 1. Default dashboard key succeeds
        res_ok = client.post(
            "/api/v1/analyze",
            json={"learner_id": "LRN-1002", "query_text": "مش فاهم يعني إيه autograd في PyTorch"},
            headers={"X-API-Key": SPRINTS_API_KEY}
        )
        assert res_ok.status_code == 200
        assert res_ok.json().get("routing_decision") == "AUTOMATED_RAG_PIPELINE"

        # 2. Corrupted key receives 401 (triggers in-page red alert banner)
        res_unauth = client.post(
            "/api/v1/analyze",
            json={"learner_id": "LRN-1002", "query_text": "مش فاهم يعني إيه autograd في PyTorch"},
            headers={"X-API-Key": "wrong-corrupted-key-123"}
        )
        assert res_unauth.status_code == 401
        assert "Unauthorized" in res_unauth.json().get("detail", "")

    def test_client_validation_and_rejection_of_malformed_inputs(self, client):
        """
        Verifies that malformed inputs that bypass frontend checks are strictly
        rejected by FastAPI with HTTP 422, enabling the dashboard to display the error.
        """
        # Blank query rejection
        res_blank = client.post(
            "/api/v1/analyze",
            json={"learner_id": "LRN-1002", "query_text": "   "},
            headers={"X-API-Key": SPRINTS_API_KEY}
        )
        assert res_blank.status_code == 422

        # Score > 100 rejection
        res_score_overflow = client.post(
            "/api/v1/predict_risk",
            json={"learner_id": "LRN-1002", "week_1_quiz": 150.0},
            headers={"X-API-Key": SPRINTS_API_KEY}
        )
        assert res_score_overflow.status_code == 422
