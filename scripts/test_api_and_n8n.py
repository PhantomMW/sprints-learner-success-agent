"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: scripts/test_api_and_n8n.py
Description: Automated test script verifying the FastAPI backend, n8n simulated routing,
             semantic vector RAG retrieval, persistent SQLite escalation logging,
             and strict Pydantic input validation.
Meets Acceptance Criteria: Latency < 3.5 seconds, Deliberative Agent routing accuracy.
==============================================================================
"""

import os
import sys
import time
import json
import sqlite3
from fastapi.testclient import TestClient

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
api_dir = os.path.join(project_root, 'api')
if api_dir not in sys.path:
    sys.path.append(api_dir)

from app import app, load_all_models

load_all_models()
client = TestClient(app)

print("\n" + "="*65)
print("TESTING SPRINTS FASTAPI BACKEND & N8N AGENT ROUTING")
print("="*65)

# 1. Test Healthcheck
resp = client.get("/api/v1/health")
print("1. Healthcheck:", resp.status_code, resp.json())
assert resp.status_code == 200
assert resp.json()["database_connected"] is True

# 2. Test Neutral Curriculum Query (Semantic Vector RAG Routing)
neutral_payload = {
    "learner_id": "LRN-1006",
    "query_text": "Can someone explain the forget gate in LSTM? The lecture from last Tuesday was cut off."
}
t0 = time.time()
resp_neutral = client.post("/api/v1/analyze", json=neutral_payload)
lat_neutral = (time.time() - t0) * 1000
print(f"\n2. Neutral Curriculum Query (Semantic Vector RAG) ({lat_neutral:.2f}ms):")
data_neutral = resp_neutral.json()
print("   Routing Decision:", data_neutral["routing_decision"])
print("   Retrieved Chunk:", data_neutral["retrieved_chunk"]["chunk_id"] if data_neutral.get("retrieved_chunk") else None)
print("   Similarity Score:", data_neutral["retrieved_chunk"].get("similarity_score") if data_neutral.get("retrieved_chunk") else None)
assert resp_neutral.status_code == 200
assert data_neutral["routing_decision"] == "AUTOMATED_RAG_PIPELINE"
assert data_neutral["execution_latency_ms"] < 3500.0

# 3. Test Negative / Frustrated Query (Deliberative Human Escalation)
negative_payload = {
    "learner_id": "LRN-1019",
    "query_text": "This is the worst experience, none of the code from the notebook works. Thinking of quitting."
}
t0 = time.time()
resp_neg = client.post("/api/v1/analyze", json=negative_payload)
lat_neg = (time.time() - t0) * 1000
print(f"\n3. Frustrated Query Escalation ({lat_neg:.2f}ms):")
data_neg = resp_neg.json()
print("   Routing Decision:", data_neg["routing_decision"])
print("   Assigned Queue:", data_neg["assigned_queue"])
print("   Priority:", data_neg["priority"])
assert resp_neg.status_code == 200
assert data_neg["routing_decision"] == "HUMAN_OPERATOR_ESCALATION"
assert data_neg["priority"] == "Critical"
assert data_neg["execution_latency_ms"] < 3500.0

# 4. Test Persistent Escalation Ticket Logging to SQLite DB
esc_payload = {
    "learner_id": "LRN-1019",
    "query_text": "This is the worst experience, none of the code works. Thinking of quitting.",
    "module_id": "MOD-PYT-02",
    "sentiment_polarity": data_neg["sentiment"]["polarity"],
    "sentiment_subjectivity": data_neg["sentiment"]["subjectivity"],
    "priority": "Critical",
    "status": "Pending_Human_Escalation"
}
resp_esc = client.post("/api/v1/tickets/escalate", json=esc_payload)
print(f"\n4. Persistent Ticket Escalation Write to SQLite DB:")
data_esc = resp_esc.json()
print("   Status:", data_esc["status"])
print("   Ticket ID:", data_esc["ticket_id"])
print("   DB Status:", data_esc["db_record_status"])
assert resp_esc.status_code == 200
assert data_esc["status"] == "persisted"

# Verify in SQLite database directly
db_path = os.path.join(project_root, 'database', 'sprints_learners.db')
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute("SELECT ticket_id, learner_id, status, priority, escalation_flag FROM support_tickets WHERE ticket_id = ?", (data_esc["ticket_id"],))
db_row = cur.fetchone()
conn.close()
print("   Direct DB Verification:", db_row)
assert db_row is not None
assert db_row[4] == 1 # escalation_flag = 1

# 5. Test PyTorch Supervised Risk Classifier (Sprint 2)
risk_payload_high = {
    "learner_id": "LRN-1005",
    "week_1_quiz": 60.0,
    "week_2_quiz": 55.0,
    "week_3_quiz": 45.0,
    "week_4_quiz": 40.0,
    "week_5_quiz": 35.0
}
resp_risk = client.post("/api/v1/predict_risk", json=risk_payload_high)
print("\n5. PyTorch Supervised Risk Prediction (High Risk Profile):")
data_risk = resp_risk.json()
print("   Risk Level:", data_risk["risk_level"])
print("   Risk Probability:", data_risk["risk_probability"])
print("   Decision Threshold:", data_risk["decision_threshold"])
assert resp_risk.status_code == 200
assert data_risk["risk_level"] == "High Risk"

# 6. Test PyTorch Sequence LSTM Predictor (Sprint 3)
seq_payload = {
    "learner_id": "LRN-1001",
    "quiz_scores": [85.0, 80.0, 75.0, 70.0, 60.0]
}
resp_seq = client.post("/api/v1/predict_sequence", json=seq_payload)
print("\n6. PyTorch LSTM Sequence Predictor (Capstone Forecast):")
data_seq = resp_seq.json()
print("   Predicted Capstone Score:", data_seq["predicted_capstone_score"])
print("   Forecast Status:", data_seq["forecast_status"])
print("   Recurrent Mechanism:", data_seq["recurrent_mechanism"])
assert resp_seq.status_code == 200
assert data_seq["predicted_capstone_score"] > 60.0

# 7. Test Strict Validation Handling (Bad Requests)
resp_bad_query = client.post("/api/v1/analyze", json={"learner_id": "LRN-1001", "query_text": "   "})
print("\n7. Input Validation Checks:")
print("   Blank query rejection status code:", resp_bad_query.status_code)
assert resp_bad_query.status_code == 422

resp_bad_score = client.post("/api/v1/predict_risk", json={"learner_id": "LRN-1001", "week_1_quiz": 150.0, "week_2_quiz": 50.0, "week_3_quiz": 50.0})
print("   Score > 100 rejection status code:", resp_bad_score.status_code)
assert resp_bad_score.status_code == 422

print("\n" + "="*65)
print("ALL API ENDPOINTS & DELIBERATIVE ROUTING: 100% SUCCESSFUL!")
print(f"Max Latency Observed: {max(lat_neutral, lat_neg):.2f}ms (SLA < 3500ms STRICTLY MET)")
print("="*65)
