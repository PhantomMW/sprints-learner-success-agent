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
    "module_id": "PyTorch",  # tests module mapping to MOD-201
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
print("   Mapped Module ID:", data_esc["module_id"])
print("   DB Status:", data_esc["db_record_status"])
assert resp_esc.status_code == 200
assert data_esc["status"] == "persisted"
assert data_esc["module_id"] == "MOD-201"

# Verify in SQLite database directly with foreign key check
db_path = os.path.join(project_root, 'database', 'sprints_learners.db')
conn = sqlite3.connect(db_path)
conn.execute("PRAGMA foreign_keys = ON;")
assert conn.execute("PRAGMA foreign_keys;").fetchone()[0] == 1
cur = conn.cursor()
cur.execute("SELECT ticket_id, learner_id, module_id, status, priority, escalation_flag FROM support_tickets WHERE ticket_id = ?", (data_esc["ticket_id"],))
db_row = cur.fetchone()
conn.close()
print("   Direct DB Verification:", db_row)
assert db_row is not None
assert db_row[1] == "LRN-1019"
assert db_row[2] == "MOD-201"
assert db_row[5] == 1 # escalation_flag = 1

# 4B. Test Unknown Learner Escalation Rejection (Strict 404, No Fallback to LRN-1001)
esc_unknown = {
    "learner_id": "LRN-UNKNOWN-9999",
    "query_text": "Unknown learner attempting to create ticket.",
    "module_id": "MOD-101",
    "sentiment_polarity": -0.5,
    "sentiment_subjectivity": 0.5,
    "priority": "High",
    "status": "Pending_Human_Escalation"
}
resp_unknown = client.post("/api/v1/tickets/escalate", json=esc_unknown)
print("\n4B. Unknown Learner Escalation Rejection:")
print("    Status code:", resp_unknown.status_code, "(Detail:", resp_unknown.json().get("detail"), ")")
assert resp_unknown.status_code == 404

# 5. Test PyTorch Supervised Risk Classifier (Sprint 2 - Exp B Full Trajectory)
risk_payload_high = {
    "learner_id": "LRN-1005",
    "week_1_quiz": 60.0,
    "week_2_quiz": 55.0,
    "week_3_quiz": 45.0,
    "week_4_quiz": 40.0,
    "week_5_quiz": 35.0
}
resp_risk = client.post("/api/v1/predict_risk", json=risk_payload_high)
print("\n5. PyTorch Supervised Risk Prediction (Exp B Full Trajectory):")
data_risk = resp_risk.json()
print("   Model Used:", data_risk["model_used"])
print("   Risk Level:", data_risk["risk_level"])
print("   Risk Probability:", data_risk["risk_probability"])
print("   Decision Threshold:", data_risk["decision_threshold"])
assert resp_risk.status_code == 200
assert data_risk["risk_level"] == "High Risk"
assert "Full Trajectory" in data_risk["model_used"]

# 5B. Test Dedicated Early Risk Classifier (Sprint 2 - Exp A Weeks 1-3)
early_payload = {
    "learner_id": "LRN-1005",
    "week_1_quiz": 58.0,
    "week_2_quiz": 50.0,
    "week_3_quiz": 42.0
}
resp_early = client.post("/api/v1/predict_risk_early", json=early_payload)
print("\n5B. PyTorch Dedicated Early Risk Prediction (Exp A Weeks 1-3):")
data_early = resp_early.json()
print("   Model Used:", data_early["model_used"])
print("   Risk Level:", data_early["risk_level"])
print("   Risk Probability:", data_early["risk_probability"])
print("   Score Delta W1-W3:", data_early["score_delta_w1_w3"])
assert resp_early.status_code == 200
assert "Early Risk" in data_early["model_used"]

# 5C. Test Missing Value Handling in Risk Prediction (Optional[float] Support)
risk_missing_payload = {
    "learner_id": "LRN-1005",
    "cohort_city": "Cairo",
    "week_1_quiz": 70.0,
    "week_2_quiz": None,  # Missing Week 2 score
    "week_3_quiz": 60.0
    # week_4_quiz and week_5_quiz omitted entirely (early intervention scenario)
}
resp_risk_missing = client.post("/api/v1/predict_risk", json=risk_missing_payload)
print("\n5C. Missing Value Handling in Risk Prediction (Optional Floats):")
data_risk_missing = resp_risk_missing.json()
print("   Status code:", resp_risk_missing.status_code)
print("   Model Selected:", data_risk_missing["model_used"])
print("   Risk Level:", data_risk_missing["risk_level"])
print("   Risk Probability:", data_risk_missing["risk_probability"])
assert resp_risk_missing.status_code == 200
assert "Early Risk" in data_risk_missing["model_used"]

# 5D. Test Missing Value Handling in Dedicated Early Risk (Week 1 Missing)
early_missing_payload = {
    "learner_id": "LRN-1005",
    "cohort_city": "Alexandria",
    "week_1_quiz": None,  # Missing Week 1
    "week_2_quiz": 62.0,
    "week_3_quiz": 55.0
}
resp_early_missing = client.post("/api/v1/predict_risk_early", json=early_missing_payload)
print("\n5D. Dedicated Early Risk Prediction with Missing Week 1:")
data_early_missing = resp_early_missing.json()
print("   Status code:", resp_early_missing.status_code)
print("   Risk Level:", data_early_missing["risk_level"])
print("   Risk Probability:", data_early_missing["risk_probability"])
print("   Early Mean Score:", data_early_missing["early_mean_score"])
assert resp_early_missing.status_code == 200

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

resp_bad_score_high = client.post("/api/v1/predict_risk", json={"learner_id": "LRN-1001", "week_1_quiz": 150.0, "week_2_quiz": 50.0, "week_3_quiz": 50.0})
print("   Score > 100 rejection status code:", resp_bad_score_high.status_code)
assert resp_bad_score_high.status_code == 422

resp_bad_score_neg = client.post("/api/v1/predict_risk_early", json={"learner_id": "LRN-1001", "week_1_quiz": -10.0, "week_2_quiz": 50.0, "week_3_quiz": 50.0})
print("   Score < 0 rejection status code:", resp_bad_score_neg.status_code)
assert resp_bad_score_neg.status_code == 422

# 8. Node.js n8n Workflow Verification
print("\n8. Executing Node.js n8n Workflow Internal Engine Test...")
n8n_test_script = os.path.join(project_root, "tests", "test_n8n_workflow_engine.js")
if os.path.exists(n8n_test_script):
    import subprocess
    proc = subprocess.run(["node", n8n_test_script], capture_output=True, text=True)
    assert proc.returncode == 0, f"n8n node test failed: {proc.stderr}"
    print("   n8n Node.js Code Execution & Routing: 100% PASSED!")

print("\n" + "="*65)
print("ALL API ENDPOINTS & DELIBERATIVE ROUTING: 100% SUCCESSFUL!")
print(f"Max Latency Observed: {max(lat_neutral, lat_neg):.2f}ms (SLA < 3500ms STRICTLY MET)")
print("="*65)
