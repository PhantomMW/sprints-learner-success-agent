"""
Test script verifying the FastAPI backend and simulated n8n workflow execution.
Meets Acceptance Criteria: Latency < 3.5 seconds, Deliberative Agent routing accuracy.
"""

import time
import json
from fastapi.testclient import TestClient
import sys
import os

project_root = r'C:\Users\mw851\.gemini\antigravity\scratch\sprints_capstone'
sys.path.append(os.path.join(project_root, 'api'))

from app import app, load_all_models

load_all_models()
client = TestClient(app)

print("\n" + "="*60)
print("TESTING SPRINTS FASTAPI BACKEND & N8N AGENT ROUTING")
print("="*60)

# 1. Test Healthcheck
resp = client.get("/api/v1/health")
print("Healthcheck:", resp.status_code, resp.json())
assert resp.status_code == 200

# 2. Test Neutral Curriculum Query (RAG Routing)
neutral_payload = {
    "learner_id": "LRN-1006",
    "query_text": "Can someone explain the forget gate in LSTM? The lecture from last Tuesday was cut off."
}
t0 = time.time()
resp_neutral = client.post("/api/v1/analyze", json=neutral_payload)
lat_neutral = (time.time() - t0) * 1000
print(f"\n[Test 1] Neutral Curriculum Query ({lat_neutral:.2f}ms):")
print(json.dumps(resp_neutral.json(), indent=2))
assert resp_neutral.status_code == 200
assert resp_neutral.json()["routing_decision"] == "AUTOMATED_RAG_PIPELINE"
assert resp_neutral.json()["execution_latency_ms"] < 3500.0

# 3. Test Negative / Frustrated Query (Deliberative Human Escalation)
negative_payload = {
    "learner_id": "LRN-1019",
    "query_text": "This is the worst experience, none of the code from the notebook works. Thinking of quitting."
}
t0 = time.time()
resp_neg = client.post("/api/v1/analyze", json=negative_payload)
lat_neg = (time.time() - t0) * 1000
print(f"\n[Test 2] Frustrated Query Escalation ({lat_neg:.2f}ms):")
print(json.dumps(resp_neg.json(), indent=2))
assert resp_neg.status_code == 200
assert resp_neg.json()["routing_decision"] == "HUMAN_OPERATOR_ESCALATION"
assert resp_neg.json()["priority"] == "Critical"
assert resp_neg.json()["execution_latency_ms"] < 3500.0

# 4. Test PyTorch Supervised Risk Classifier (Sprint 2)
risk_payload_high = {
    "learner_id": "LRN-1005",
    "week_1_quiz": 60.0,
    "week_2_quiz": 55.0,
    "week_3_quiz": 45.0,
    "week_4_quiz": 40.0,
    "week_5_quiz": 35.0
}
resp_risk = client.post("/api/v1/predict_risk", json=risk_payload_high)
print("\n[Test 3] PyTorch Supervised Risk Prediction (High Risk Profile):")
print(json.dumps(resp_risk.json(), indent=2))
assert resp_risk.status_code == 200
assert resp_risk.json()["risk_level"] == "High Risk"

# 5. Test PyTorch Sequence LSTM Predictor (Sprint 3)
seq_payload = {
    "learner_id": "LRN-1001",
    "quiz_scores": [85.0, 80.0, 75.0, 70.0, 60.0]
}
resp_seq = client.post("/api/v1/predict_sequence", json=seq_payload)
print("\n[Test 4] PyTorch LSTM Sequence Predictor (Capstone Forecast):")
print(json.dumps(resp_seq.json(), indent=2))
assert resp_seq.status_code == 200

print("\n" + "="*60)
print("ALL API ENDPOINTS & DELIBERATIVE ROUTING: 100% SUCCESSFUL!")
print(f"Max Latency Observed: {max(lat_neutral, lat_neg):.2f}ms (SLA < 3500ms STRICTLY MET)")
print("="*60)
