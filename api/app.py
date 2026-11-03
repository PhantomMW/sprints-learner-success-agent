"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: api/app.py
Description: Production FastAPI service backing the n8n Workflow and LMS integrations.
             Exposes sub-second endpoints for:
             - Deliberative Agent Query Routing (Sentiment + NER + RAG)
             - PyTorch Supervised Risk Classification (Sprint 2)
             - PyTorch LSTM Sequence Trajectory Prediction (Sprint 3)
Guarantees < 3.5s latency SLA (typical execution: ~20-50ms).
==============================================================================
"""

import os
import sys
import time
import torch
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add scripts directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'scripts'))

from nlp_pipeline import (
    build_spacy_ner_pipeline,
    extract_entities,
    analyze_sentiment,
    deliberative_agent_router
)
from train_pytorch import (
    LearnerRiskClassifier,
    LearnerSequenceLSTM
)

app = FastAPI(
    title="Sprints Learner Success & Support Agent API",
    description="High-performance backend for the Sprints Deliberative Agent and n8n Workflow.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model caches
GLOBAL_MODELS: Dict[str, Any] = {}

# Sprints Curriculum Knowledge Base for RAG retrieval
CURRICULUM_KB = {
    "pytorch": (
        "PyTorch Tensors & Autograd: In PyTorch, tensors are multi-dimensional arrays optimized for GPU acceleration. "
        "To fix dimension mismatches during forward passes, verify tensor shapes using `tensor.shape` and utilize "
        "`view()`, `reshape()`, or `unsqueeze()` appropriately. For gradient computation, ensure `requires_grad=True`."
    ),
    "sql": (
        "SQL Window Functions & Aggregations: Advanced joins (INNER, LEFT, FULL) allow relational extraction. "
        "Use `ROW_NUMBER()`, `RANK()`, or `AVG() OVER (PARTITION BY ...)` to perform multi-row analytical calculations "
        "without collapsing data via standard GROUP BY."
    ),
    "pandas": (
        "Data Imputation with pandas: When handling missing quiz scores or sensor metrics, utilize forward-fill (`ffill()`) "
        "for sequential temporal time-series, or grouped median imputation (`df.groupby('cohort_city')['quiz'].transform('median')`) "
        "to avoid data leakage across cohorts."
    ),
    "lstm": (
        "LSTM Architecture & Forget Gate: The forget gate in an LSTM cell computes f_t = sigma(W_f * [h_{t-1}, x_t] + b_f). "
        "It outputs values between 0 (completely discard past cell state) and 1 (completely retain), effectively discarding "
        "early-week noise and mitigating vanishing gradients across 5-week trajectories."
    ),
    "gensim": (
        "Gensim Word2Vec: Implements Skip-gram and CBOW architectures. Skip-gram predicts surrounding context words given a target word, "
        "producing dense continuous vectors that capture semantic relationships like King - Man + Woman = Queen."
    ),
    "word2vec": (
        "Gensim Word2Vec: Implements Skip-gram and CBOW architectures. Skip-gram predicts surrounding context words given a target word, "
        "producing dense continuous vectors that capture semantic relationships like King - Man + Woman = Queen."
    ),
    "textblob": (
        "TextBlob Sentiment Analysis: Computes sentiment polarity in the range [-1.0, 1.0]. In the Sprints Deliberative Agent, "
        "negative polarity (< -0.15) triggers an automatic escalation rule to alert human mentors."
    ),
    "n8n": (
        "n8n Event-Driven Workflows: Operates via Webhook triggers. Incoming HTTP requests pass through TextBlob sentiment nodes, "
        "followed by an IF / Switch node that routes positive/neutral queries to RAG synthesis and negative queries to human escalations."
    ),
    "rag": (
        "Retrieval-Augmented Generation (RAG): Combines dense vector retrieval with LLM generation to ground responses "
        "in verified Sprints curriculum documentation, preventing hallucination."
    )
}

@app.on_event("startup")
def load_all_models():
    print("Pre-loading NLP pipelines and PyTorch checkpoints...")
    models_dir = os.path.join(project_root, 'models')
    
    # 1. Load spaCy NER
    GLOBAL_MODELS["nlp"] = build_spacy_ner_pipeline()
    
    # 2. Load PyTorch Risk Classifier
    risk_path = os.path.join(models_dir, 'risk_classifier.pth')
    if os.path.exists(risk_path):
        ckpt = torch.load(risk_path, weights_only=False)
        classifier = LearnerRiskClassifier(input_dim=len(ckpt['feature_cols']), hidden_dim=64)
        classifier.load_state_dict(ckpt['model_state_dict'])
        classifier.eval()
        GLOBAL_MODELS["risk_classifier"] = classifier
        GLOBAL_MODELS["risk_meta"] = ckpt
        print("Loaded PyTorch Risk Classifier.")
        
    # 3. Load PyTorch Sequence LSTM
    lstm_path = os.path.join(models_dir, 'sequence_predictor.pth')
    if os.path.exists(lstm_path):
        ckpt_lstm = torch.load(lstm_path, weights_only=False)
        lstm_model = LearnerSequenceLSTM(input_size=1, hidden_size=ckpt_lstm['hidden_size'], num_layers=ckpt_lstm['num_layers'])
        lstm_model.load_state_dict(ckpt_lstm['model_state_dict'])
        lstm_model.eval()
        GLOBAL_MODELS["sequence_lstm"] = lstm_model
        print("Loaded PyTorch Sequence LSTM.")

# Pydantic Schemas
class LearnerQueryRequest(BaseModel):
    learner_id: str = Field(..., example="LRN-1001")
    query_text: str = Field(..., example="I am struggling a bit with PyTorch tensors. The module on Oct 12 was too fast for me.")
    risk_level: Optional[str] = Field("Low", example="Low")

class RiskPredictionRequest(BaseModel):
    learner_id: str = Field(..., example="LRN-1005")
    week_1_quiz: float = Field(..., example=60.0)
    week_2_quiz: float = Field(..., example=55.0)
    week_3_quiz: float = Field(..., example=45.0)
    week_4_quiz: float = Field(..., example=40.0)
    week_5_quiz: float = Field(..., example=35.0)

class SequencePredictionRequest(BaseModel):
    learner_id: str = Field(..., example="LRN-1001")
    quiz_scores: List[float] = Field(..., example=[85.0, 80.0, 75.0, 70.0, 60.0])

@app.get("/api/v1/health")
def healthcheck():
    return {
        "status": "healthy",
        "service": "Sprints AI Learner Success Agent",
        "models_loaded": list(GLOBAL_MODELS.keys())
    }

@app.post("/api/v1/analyze")
def analyze_learner_query(req: LearnerQueryRequest):
    """
    Sub-second endpoint called directly by the n8n Webhook / HTTP node.
    Performs sentiment analysis, NER extraction, deliberative routing,
    and curriculum RAG retrieval.
    """
    start_time = time.time()
    
    # Deliberative agent routing
    routing_result = deliberative_agent_router(
        req.query_text,
        nlp_model=GLOBAL_MODELS.get("nlp"),
        risk_level=req.risk_level
    )
    
    retrieved_content = None
    ai_synthesized_response = None
    
    # If routed to automated RAG
    if routing_result["routing_decision"] == "AUTOMATED_RAG_PIPELINE":
        extracted_mods = [m.lower() for m in routing_result["extracted_modules"]]
        # Retrieve from curriculum KB
        matched_kb = []
        for mod in extracted_mods:
            for k, doc in CURRICULUM_KB.items():
                if k in mod:
                    matched_kb.append(doc)
                    
        if not matched_kb:
            # Fallback general query retrieval
            for k, doc in CURRICULUM_KB.items():
                if k in req.query_text.lower():
                    matched_kb.append(doc)
                    
        retrieved_content = " ".join(matched_kb) if matched_kb else "General Sprints AI Pioneers Bootcamp Knowledge Base."
        ai_synthesized_response = (
            f"Hello {req.learner_id}! Here is guidance based on your curriculum query: {retrieved_content} "
            f"If you need additional live code support, feel free to submit a code review ticket to your cohort TA."
        )
    else:
        # Escalated
        ai_synthesized_response = (
            f"Hello {req.learner_id}, we noticed that you are experiencing challenges with this module. "
            f"Your inquiry has been escalated directly to our Senior Teaching Assistant and Mentor desk "
            f"(Queue: {routing_result['assigned_queue']}). A mentor will contact you within 2 business hours."
        )
        
    execution_latency_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "learner_id": req.learner_id,
        "query_text": req.query_text,
        "routing_decision": routing_result["routing_decision"],
        "assigned_queue": routing_result["assigned_queue"],
        "priority": routing_result["priority"],
        "sentiment": routing_result["sentiment"],
        "extracted_modules": routing_result["extracted_modules"],
        "extracted_dates": routing_result["extracted_dates"],
        "retrieved_knowledge": retrieved_content,
        "agent_response": ai_synthesized_response,
        "execution_latency_ms": execution_latency_ms,
        "sla_met": execution_latency_ms < 3500.0
    }

@app.post("/api/v1/predict_risk")
def predict_learner_risk(req: RiskPredictionRequest):
    """
    Sprint 2: Evaluates a student's structured quiz profile using the PyTorch MLP.
    Returns High Risk vs Low Risk classification and risk probability.
    """
    if "risk_classifier" not in GLOBAL_MODELS:
        raise HTTPException(status_code=503, detail="Risk classifier model not loaded.")
        
    start_time = time.time()
    quizzes = [req.week_1_quiz, req.week_2_quiz, req.week_3_quiz, req.week_4_quiz, req.week_5_quiz]
    
    score_delta = quizzes[0] - quizzes[-1]
    quiz_mean = float(np.mean(quizzes))
    quiz_std = float(np.std(quizzes))
    quiz_min = float(np.min(quizzes))
    perf_ratio = quizzes[-1] / (quizzes[0] + 1e-5)
    missing_count = sum(1 for q in quizzes if q <= 0 or np.isnan(q))
    
    raw_feats = np.array([quizzes + [score_delta, quiz_mean, quiz_std, quiz_min, perf_ratio, missing_count]])
    
    ckpt = GLOBAL_MODELS["risk_meta"]
    mean = ckpt['scaler_mean']
    scale = ckpt['scaler_scale']
    scaled_feats = (raw_feats - mean) / scale
    
    model = GLOBAL_MODELS["risk_classifier"]
    with torch.no_grad():
        tensor_x = torch.tensor(scaled_feats, dtype=torch.float32)
        logits = model(tensor_x)
        prob = float(torch.sigmoid(logits).item())
        
    threshold = ckpt.get('decision_threshold', 0.42)
    is_high_risk = prob >= threshold
    
    return {
        "learner_id": req.learner_id,
        "risk_probability": round(prob, 4),
        "risk_level": "High Risk" if is_high_risk else "Low Risk",
        "flagged_for_intervention": is_high_risk,
        "score_delta_w1_w5": round(score_delta, 1),
        "execution_latency_ms": round((time.time() - start_time) * 1000, 2)
    }

@app.post("/api/v1/predict_sequence")
def predict_sequence_trajectory(req: SequencePredictionRequest):
    """
    Sprint 3: Feeds 5-week quiz vector into PyTorch LSTM to forecast capstone outcome.
    """
    if "sequence_lstm" not in GLOBAL_MODELS:
        raise HTTPException(status_code=503, detail="Sequence LSTM model not loaded.")
        
    if len(req.quiz_scores) != 5:
        raise HTTPException(status_code=400, detail="quiz_scores must contain exactly 5 scores.")
        
    start_time = time.time()
    norm_scores = np.array(req.quiz_scores, dtype=np.float32) / 100.0
    tensor_seq = torch.tensor(norm_scores).unsqueeze(0).unsqueeze(-1) # (1, 5, 1)
    
    model = GLOBAL_MODELS["sequence_lstm"]
    with torch.no_grad():
        pred_norm, _ = model(tensor_seq)
        predicted_capstone = float(pred_norm.item() * 100.0)
        
    return {
        "learner_id": req.learner_id,
        "input_trajectory": req.quiz_scores,
        "predicted_capstone_score": round(predicted_capstone, 2),
        "forecast_status": "Projected Pass" if predicted_capstone >= 60.0 else "Projected Fail (At Risk)",
        "execution_latency_ms": round((time.time() - start_time) * 1000, 2)
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
