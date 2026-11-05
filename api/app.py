"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: api/app.py
Description: Production FastAPI service backing the n8n Deliberative Agent Workflow.
             Exposes sub-second endpoints for:
             - Deliberative Agent Query Routing (Sentiment + NER + Semantic Vector RAG)
             - Persistent Escalation Ticket Logging into SQLite Relational DB
             - PyTorch Supervised Risk Classification (Sprint 2)
             - PyTorch LSTM Sequence Trajectory Forecasting (Sprint 3)
Guarantees < 3.5s latency SLA (typical execution: ~15-45ms).
==============================================================================
"""

import os
import sys
import time
import sqlite3
import torch
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import uuid
import hashlib
import json
from fastapi import FastAPI, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from gensim.models import Word2Vec

# Add scripts directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
scripts_dir = os.path.join(project_root, 'scripts')
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from nlp_pipeline import (
    build_spacy_ner_pipeline,
    extract_entities,
    analyze_sentiment,
    semantic_rag_retrieve,
    deliberative_agent_router,
    CURRICULUM_KNOWLEDGE_CHUNKS
)
from train_pytorch import (
    LearnerRiskClassifier,
    LearnerSequenceLSTM,
    extract_single_learner_features
)

app = FastAPI(
    title="Sprints Learner Success & Support Agent API",
    description="Production backend service for Sprints Deliberative Agent, PyTorch ML/DL, and n8n orchestration.",
    version="1.0.0"
)

# CORS Policy configuration
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
allowed_origins = [
    origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()
] or [
    "http://localhost:5678",
    "http://127.0.0.1:5678",
    "http://sprints-ai-api:8000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global model caches
GLOBAL_MODELS: Dict[str, Any] = {}
DB_PATH = os.path.join(project_root, 'database', 'sprints_learners.db')

import logging
from contextlib import contextmanager

logger = logging.getLogger("sprints_agent")

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    assert conn.execute("PRAGMA foreign_keys;").fetchone()[0] == 1, "Foreign key enforcement failed"
    return conn

@contextmanager
def get_db_session():
    """
    Robust SQLite context manager guaranteeing proper connection lifecycle management,
    foreign key enforcement, and explicit error logging.
    """
    conn = None
    try:
        conn = get_db_connection()
        yield conn
    except sqlite3.Error as e:
        logger.error(f"SQLite database error occurred: {e}", exc_info=True)
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception as ce:
                logger.warning(f"Error closing database connection: {ce}")

def lookup_learner_city(learner_id: str) -> Optional[str]:
    """Safely retrieves a learner's cohort city from the database using a managed session."""
    try:
        with get_db_session() as conn:
            cur = conn.cursor()
            cur.execute("SELECT cohort_city FROM learners WHERE learner_id = ?", (learner_id,))
            row = cur.fetchone()
            if row and row['cohort_city']:
                return str(row['cohort_city'])
    except Exception as e:
        logger.warning(f"Failed to lookup cohort_city for learner {learner_id}: {e}")
    return None

def resolve_module_id(mod_input: Optional[str], cur: sqlite3.Cursor) -> Optional[str]:
    """
    Maps student / NER module keywords to valid foreign keys in course_modules table.
    Returns None if unmapped to maintain referential integrity without foreign key violations.
    """
    if not mod_input or not str(mod_input).strip():
        return None
    mod_clean = str(mod_input).strip().lower()
    mapping = {
        'sql': 'MOD-101',
        'relational sql': 'MOD-101',
        'mod-101': 'MOD-101',
        'pandas': 'MOD-102',
        'pandas wrangling': 'MOD-102',
        'mod-102': 'MOD-102',
        'pytorch': 'MOD-201',
        'supervised ml': 'MOD-201',
        'mod-201': 'MOD-201',
        'mod-pyt-02': 'MOD-201',
        'lstm': 'MOD-301',
        'deep sequence': 'MOD-301',
        'mod-301': 'MOD-301',
        'nlp': 'MOD-401',
        'nlp & generative ai': 'MOD-401',
        'word2vec': 'MOD-401',
        'fasttext': 'MOD-401',
        'mod-401': 'MOD-401'
    }
    target_id = mapping.get(mod_clean, str(mod_input).strip().upper())
    cur.execute("SELECT module_id FROM course_modules WHERE module_id = ?", (target_id,))
    row = cur.fetchone()
    return row['module_id'] if row else None

def ensure_db_schema(conn: Optional[sqlite3.Connection] = None):
    """
    Executes SQLite database schema migrations and strictly verifies that
    all required columns and indexes exist.
    Fails fast with RuntimeError if migration or verification fails,
    preventing application startup in an inconsistent state.
    """
    def _migrate(connection: sqlite3.Connection):
        cur = connection.cursor()
        cur.execute("PRAGMA table_info(support_tickets);")
        existing_cols = [c[1] for c in cur.fetchall()]
        if not existing_cols:
            raise RuntimeError("Database migration error: 'support_tickets' table does not exist.")

        # 1. Apply required column migrations
        required_cols = {
            "correlation_id": "VARCHAR(64)",
            "idempotency_key": "VARCHAR(64)",
            "payload_hash": "VARCHAR(64)"
        }
        for col_name, col_type in required_cols.items():
            if col_name not in existing_cols:
                cur.execute(f"ALTER TABLE support_tickets ADD COLUMN {col_name} {col_type};")
                logger.info(f"Database migration: Added column '{col_name}' ({col_type}) to support_tickets.")

        # 2. Apply index migrations
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_tickets_idempotency ON support_tickets(idempotency_key);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_tickets_correlation ON support_tickets(correlation_id);")
        connection.commit()

        # 3. Post-migration verification assertion
        cur.execute("PRAGMA table_info(support_tickets);")
        verified_cols = {c[1] for c in cur.fetchall()}
        missing = [c for c in required_cols.keys() if c not in verified_cols]
        if missing:
            raise RuntimeError(f"Database schema verification failed: columns {missing} missing after migration.")

        cur.execute("PRAGMA index_list(support_tickets);")
        verified_indexes = {idx[1] for idx in cur.fetchall()}
        required_indexes = ["idx_tickets_idempotency", "idx_tickets_correlation"]
        missing_idx = [idx for idx in required_indexes if idx not in verified_indexes]
        if missing_idx:
            raise RuntimeError(f"Database schema verification failed: indexes {missing_idx} missing after migration.")

        logger.info("Database schema migration & verification passed 100%.")

    if conn is not None:
        _migrate(conn)
    else:
        with get_db_session() as session_conn:
            _migrate(session_conn)

@app.on_event("startup")
def load_all_models():
    ensure_db_schema()
    print("Pre-loading NLP pipelines, Word2Vec, and PyTorch checkpoints...")
    models_dir = os.path.join(project_root, 'models')
    
    # 1. Load spaCy NER
    GLOBAL_MODELS["nlp"] = build_spacy_ner_pipeline()
    print("Loaded spaCy NER Pipeline.")
    
    # 2. Load Word2Vec model
    w2v_path = os.path.join(models_dir, 'word2vec.model')
    if os.path.exists(w2v_path):
        GLOBAL_MODELS["word2vec"] = Word2Vec.load(w2v_path)
        print("Loaded Gensim Word2Vec Dense Embeddings.")
        
    # 3. Load PyTorch Risk Classifiers (Both Exp A and Exp B)
    risk_path = os.path.join(models_dir, 'risk_classifier.pth')
    if os.path.exists(risk_path):
        ckpt = torch.load(risk_path, weights_only=False)
        # Experiment B: Full Trajectory (Weeks 1-5)
        feature_cols = ckpt['feature_cols']
        classifier = LearnerRiskClassifier(input_dim=len(feature_cols), hidden_dim=64)
        classifier.load_state_dict(ckpt['model_state_dict'])
        classifier.eval()
        GLOBAL_MODELS["risk_classifier"] = classifier
        
        # Experiment A: Early Proactive Risk (Weeks 1-3)
        early_cols = ckpt.get('early_feature_cols')
        if early_cols and 'early_model_state_dict' in ckpt:
            early_classifier = LearnerRiskClassifier(input_dim=len(early_cols), hidden_dim=64)
            early_classifier.load_state_dict(ckpt['early_model_state_dict'])
            early_classifier.eval()
            GLOBAL_MODELS["early_risk_classifier"] = early_classifier
            
        GLOBAL_MODELS["risk_meta"] = ckpt
        print("Loaded PyTorch Risk Classifiers (Exp A Early & Exp B Full).")
        
    # 4. Load PyTorch Sequence LSTM
    lstm_path = os.path.join(models_dir, 'sequence_predictor.pth')
    if os.path.exists(lstm_path):
        ckpt_lstm = torch.load(lstm_path, weights_only=False)
        lstm_model = LearnerSequenceLSTM(input_size=1, hidden_size=ckpt_lstm['hidden_size'], num_layers=ckpt_lstm['num_layers'])
        lstm_model.load_state_dict(ckpt_lstm['model_state_dict'])
        lstm_model.eval()
        GLOBAL_MODELS["sequence_lstm"] = lstm_model
        GLOBAL_MODELS["lstm_meta"] = ckpt_lstm
        print("Loaded PyTorch Sequence LSTM (Zero Data Leakage).")

# Pydantic Schemas with Strict Validation
class LearnerQueryRequest(BaseModel):
    learner_id: str = Field(..., min_length=3, max_length=20, example="LRN-1001")
    query_text: str = Field(..., min_length=3, max_length=1500, example="I am struggling with PyTorch autograd tensors.")
    risk_level: Optional[str] = Field("Low", example="Low")
    correlation_id: Optional[str] = Field(None, example="corr-12345")
    idempotency_key: Optional[str] = Field(None, example="idemp-12345")

    @validator("query_text")
    def validate_query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("query_text cannot be blank or contain only whitespace.")
        return v.strip()

class TicketEscalationRequest(BaseModel):
    learner_id: str = Field(..., min_length=3, max_length=20, example="LRN-1005")
    query_text: str = Field(..., min_length=3, max_length=1500, example="I want to drop out, failing week 4 quiz.")
    module_id: Optional[str] = Field(None, example="MOD-PYT-02")
    sentiment_polarity: float = Field(-0.65, ge=-1.0, le=1.0, example=-0.65)
    sentiment_subjectivity: float = Field(0.70, ge=0.0, le=1.0, example=0.70)
    priority: str = Field("Critical", example="Critical")
    status: str = Field("Pending_Human_Escalation", example="Pending_Human_Escalation")
    correlation_id: Optional[str] = Field(None, example="corr-12345")
    idempotency_key: Optional[str] = Field(None, example="idemp-12345")

    @validator("priority")
    def validate_priority(cls, v):
        valid = ["Low", "Medium", "High", "Critical"]
        if v not in valid:
            raise ValueError(f"priority must be one of {valid}")
        return v

    @validator("status")
    def validate_status(cls, v):
        valid = ["Open", "Pending_Human_Escalation", "Resolved_By_Agent", "Resolved_By_Mentor"]
        if v not in valid:
            raise ValueError(f"status must be one of {valid}")
        return v

class RiskPredictionRequest(BaseModel):
    learner_id: str = Field(..., min_length=3, max_length=20, example="LRN-1005")
    correlation_id: Optional[str] = Field(None, example="corr-12345")
    cohort_city: Optional[str] = Field(None, example="Cairo")
    week_1_quiz: Optional[float] = Field(None, ge=0.0, le=100.0, example=65.0)
    week_2_quiz: Optional[float] = Field(None, ge=0.0, le=100.0, example=58.0)
    week_3_quiz: Optional[float] = Field(None, ge=0.0, le=100.0, example=48.0)
    week_4_quiz: Optional[float] = Field(None, ge=0.0, le=100.0, example=40.0)
    week_5_quiz: Optional[float] = Field(None, ge=0.0, le=100.0, example=35.0)

class EarlyRiskPredictionRequest(BaseModel):
    learner_id: str = Field(..., min_length=3, max_length=20, example="LRN-1005")
    correlation_id: Optional[str] = Field(None, example="corr-12345")
    cohort_city: Optional[str] = Field(None, example="Cairo")
    week_1_quiz: Optional[float] = Field(None, ge=0.0, le=100.0, example=65.0)
    week_2_quiz: Optional[float] = Field(None, ge=0.0, le=100.0, example=58.0)
    week_3_quiz: Optional[float] = Field(None, ge=0.0, le=100.0, example=48.0)

class SequencePredictionRequest(BaseModel):
    learner_id: str = Field(..., min_length=3, max_length=20, example="LRN-1001")
    correlation_id: Optional[str] = Field(None, example="corr-12345")
    quiz_scores: List[float] = Field(..., min_items=5, max_items=5, example=[85.0, 80.0, 75.0, 70.0, 60.0])

    @validator("quiz_scores")
    def validate_quiz_scores(cls, scores):
        for s in scores:
            if s < 0.0 or s > 100.0:
                raise ValueError("All quiz scores must be between 0.0 and 100.0.")
        return scores

# Endpoints
@app.get("/api/v1/health")
def healthcheck():
    return {
        "status": "healthy",
        "service": "Sprints AI Learner Success Agent",
        "models_loaded": list(GLOBAL_MODELS.keys()),
        "database_connected": os.path.exists(DB_PATH)
    }

@app.post("/api/v1/analyze")
def analyze_learner_query(
    req: LearnerQueryRequest,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
    idempotency_key_header: Optional[str] = Header(None, alias="Idempotency-Key")
):
    """
    Sub-second endpoint called directly by the n8n Webhook / HTTP node.
    Performs sentiment analysis, spaCy NER extraction, deliberative routing,
    and true Semantic Vector RAG via Word2Vec Cosine Similarity.
    """
    start_time = time.time()
    corr_id = req.correlation_id or x_correlation_id or f"corr-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
    idemp_key = req.idempotency_key or idempotency_key_header or f"idemp-{corr_id}"

    logger.info(f"[{corr_id}] Received query analysis for learner {req.learner_id} (idempotency_key: {idemp_key})")
    
    # Deliberative agent routing with Semantic Vector RAG
    routing_result = deliberative_agent_router(
        req.query_text,
        nlp_model=GLOBAL_MODELS.get("nlp"),
        w2v_model=GLOBAL_MODELS.get("word2vec"),
        risk_level=req.risk_level
    )
    
    execution_latency_ms = round((time.time() - start_time) * 1000, 2)
    
    return {
        "correlation_id": corr_id,
        "idempotency_key": idemp_key,
        "learner_id": req.learner_id,
        "query_text": req.query_text,
        "routing_decision": routing_result["routing_decision"],
        "assigned_queue": routing_result["assigned_queue"],
        "priority": routing_result["priority"],
        "sentiment": routing_result["sentiment"],
        "extracted_modules": routing_result["extracted_modules"],
        "extracted_dates": routing_result["extracted_dates"],
        "retrieved_chunk": routing_result["retrieved_chunk"],
        "retrieved_knowledge": routing_result["retrieved_chunk"]["text"] if routing_result.get("retrieved_chunk") else None,
        "agent_response": routing_result["agent_response"],
        "action_summary": routing_result["action_summary"],
        "execution_latency_ms": execution_latency_ms,
        "sla_met": execution_latency_ms < 3500.0
    }

def compute_payload_hash(
    learner_id: str,
    query_text: str,
    module_id: Optional[str] = None,
    priority: Optional[str] = None,
    status_val: Optional[str] = None,
    sentiment_polarity: Optional[float] = None,
    sentiment_subjectivity: Optional[float] = None
) -> str:
    """
    Computes deterministic SHA-256 fingerprint over all effective ticket payload attributes
    using canonical sorted JSON serialization.
    Ensures that priority, sentiment, and module changes under the same key are caught as conflicts.
    """
    canonical_payload = {
        "learner_id": str(learner_id).strip(),
        "query_text": str(query_text).strip(),
        "module_id": str(module_id).strip() if module_id else None,
        "priority": str(priority).strip() if priority else None,
        "status": str(status_val).strip() if status_val else None,
        "sentiment_polarity": round(float(sentiment_polarity), 4) if sentiment_polarity is not None else None,
        "sentiment_subjectivity": round(float(sentiment_subjectivity), 4) if sentiment_subjectivity is not None else None
    }
    canonical_json = json.dumps(canonical_payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

@app.post("/api/v1/tickets/escalate")
def escalate_ticket_to_db(
    req: TicketEscalationRequest,
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
    idempotency_key_header: Optional[str] = Header(None, alias="Idempotency-Key")
):
    """
    Persistent escalation endpoint: logs escalated tickets directly into
    the SQLite relational database (support_tickets table).
    Strictly verifies learner existence (returns 404 Not Found if missing).
    Resolves module keywords to valid foreign keys in course_modules table.
    Guarantees strict idempotency via Idempotency-Key, payload conflict detection,
    database UNIQUE constraint, and collision-free UUID ticket IDs.
    """
    start_time = time.time()
    corr_id = req.correlation_id or x_correlation_id or f"corr-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}"
    idemp_key = req.idempotency_key or idempotency_key_header
    ticket_id = f"TCK-ESC-{uuid.uuid4().hex[:12].upper()}"
    payload_hash = compute_payload_hash(
        learner_id=req.learner_id,
        query_text=req.query_text,
        module_id=req.module_id,
        priority=req.priority,
        status_val=req.status,
        sentiment_polarity=req.sentiment_polarity,
        sentiment_subjectivity=req.sentiment_subjectivity
    )
    
    resolved_mod = None
    try:
        with get_db_session() as conn:
            cur = conn.cursor()
            
            # Verify learner existence; strictly reject unknown IDs with 404
            cur.execute("SELECT learner_id FROM learners WHERE learner_id = ?", (req.learner_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Learner ID '{req.learner_id}' not found in database. Referential integrity prevented ticket escalation."
                )
                
            # Map module ID to valid course_modules foreign key or None
            resolved_mod = resolve_module_id(req.module_id, cur)

            # 1. Primary Idempotency Check: Dedicated Idempotency-Key
            if idemp_key:
                cur.execute("""
                    SELECT ticket_id, learner_id, module_id, query_text, status, priority, correlation_id, idempotency_key, payload_hash, created_at 
                    FROM support_tickets 
                    WHERE idempotency_key = ?
                    LIMIT 1
                """, (idemp_key,))
                existing_key_ticket = cur.fetchone()
                if existing_key_ticket:
                    # Idempotency Conflict Detection: Same key used with DIFFERENT payload!
                    existing_hash = existing_key_ticket['payload_hash'] or compute_payload_hash(
                        learner_id=existing_key_ticket['learner_id'],
                        query_text=existing_key_ticket['query_text'],
                        module_id=existing_key_ticket['module_id'],
                        priority=existing_key_ticket['priority'],
                        status_val=existing_key_ticket['status']
                    )
                    if existing_hash != payload_hash:
                        logger.warning(f"[{corr_id}] Idempotency key conflict: key '{idemp_key}' was previously used with a different payload!")
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=f"Idempotency key collision: provided key '{idemp_key}' was already used with a different request payload."
                        )

                    logger.info(f"[{corr_id}] Idempotency key match: Existing ticket {existing_key_ticket['ticket_id']} returned for key {idemp_key}.")
                    exec_lat = round((time.time() - start_time) * 1000, 2)
                    return {
                        "status": "persisted",
                        "ticket_id": existing_key_ticket['ticket_id'],
                        "learner_id": existing_key_ticket['learner_id'],
                        "module_id": existing_key_ticket['module_id'],
                        "db_record_status": existing_key_ticket['status'],
                        "priority": existing_key_ticket['priority'],
                        "correlation_id": existing_key_ticket['correlation_id'] or corr_id,
                        "idempotency_key": existing_key_ticket['idempotency_key'],
                        "payload_hash": payload_hash,
                        "persisted_to_db": True,
                        "idempotent_deduplicated": True,
                        "message": "Idempotent request recognized; existing ticket returned without duplicate creation.",
                        "execution_latency_ms": exec_lat,
                        "sla_met": exec_lat < 3500.0
                    }

            # 2. Secondary Deduplication Check: Same learner, same query text, pending status (only when NO explicit key is provided)
            else:
                cur.execute("""
                    SELECT ticket_id, status, priority, created_at, correlation_id, idempotency_key 
                    FROM support_tickets 
                    WHERE learner_id = ? AND query_text = ? AND status = 'Pending_Human_Escalation'
                    ORDER BY created_at DESC LIMIT 1
                """, (req.learner_id, req.query_text))
                existing_ticket = cur.fetchone()
                if existing_ticket:
                    logger.info(f"[{corr_id}] Pending ticket deduplication: Existing pending ticket {existing_ticket['ticket_id']} returned for learner {req.learner_id}.")
                    exec_lat = round((time.time() - start_time) * 1000, 2)
                    return {
                        "status": "persisted",
                        "ticket_id": existing_ticket['ticket_id'],
                        "learner_id": req.learner_id,
                        "module_id": resolved_mod,
                        "db_record_status": existing_ticket['status'],
                        "priority": existing_ticket['priority'],
                        "correlation_id": existing_ticket['correlation_id'] or corr_id,
                        "idempotency_key": existing_ticket['idempotency_key'],
                        "payload_hash": payload_hash,
                        "persisted_to_db": True,
                        "idempotent_deduplicated": True,
                        "message": "Duplicate pending request recognized; existing ticket returned.",
                        "execution_latency_ms": exec_lat,
                        "sla_met": exec_lat < 3500.0
                    }

            # 3. Insert fresh escalation ticket
            cur.execute("""
                INSERT INTO support_tickets (
                    ticket_id, learner_id, module_id, query_text,
                    sentiment_polarity, sentiment_subjectivity,
                    status, priority, escalation_flag, correlation_id, idempotency_key, payload_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                ticket_id,
                req.learner_id,
                resolved_mod,
                req.query_text,
                req.sentiment_polarity,
                req.sentiment_subjectivity,
                req.status,
                req.priority,
                corr_id,
                idemp_key,
                payload_hash
            ))
            conn.commit()
    except HTTPException:
        raise
    except sqlite3.IntegrityError as ie:
        # Handle concurrent race condition on idempotency_key UNIQUE constraint
        if idemp_key and "idempotency_key" in str(ie).lower():
            logger.info(f"[{corr_id}] Concurrent race caught by UNIQUE constraint for key {idemp_key}. Fetching winning record.")
            try:
                with get_db_session() as conn_dup:
                    cur_dup = conn_dup.cursor()
                    cur_dup.execute("""
                        SELECT ticket_id, learner_id, module_id, query_text, status, priority, correlation_id, idempotency_key, payload_hash 
                        FROM support_tickets WHERE idempotency_key = ?
                    """, (idemp_key,))
                    winning = cur_dup.fetchone()
                    if winning:
                        winning_hash = winning['payload_hash'] or compute_payload_hash(
                            learner_id=winning['learner_id'],
                            query_text=winning['query_text'],
                            module_id=winning['module_id'],
                            priority=winning['priority'],
                            status_val=winning['status']
                        )
                        if winning_hash != payload_hash:
                            raise HTTPException(
                                status_code=status.HTTP_409_CONFLICT,
                                detail=f"Idempotency key collision: provided key '{idemp_key}' was already used with a different request payload."
                            )
                        exec_lat = round((time.time() - start_time) * 1000, 2)
                        return {
                            "status": "persisted",
                            "ticket_id": winning['ticket_id'],
                            "learner_id": winning['learner_id'],
                            "module_id": winning['module_id'],
                            "db_record_status": winning['status'],
                            "priority": winning['priority'],
                            "correlation_id": winning['correlation_id'] or corr_id,
                            "idempotency_key": winning['idempotency_key'],
                            "payload_hash": payload_hash,
                            "persisted_to_db": True,
                            "idempotent_deduplicated": True,
                            "message": "Concurrent idempotent request safely deduplicated via database unique constraint.",
                            "execution_latency_ms": exec_lat,
                            "sla_met": exec_lat < 3500.0
                        }
            except HTTPException:
                raise
            except Exception as fe:
                logger.error(f"Error fetching concurrent winner: {fe}")
        logger.error(f"Integrity error during escalation write for learner {req.learner_id}: {ie}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database escalation write integrity error: {str(ie)}"
        )
    except Exception as e:
        logger.error(f"Escalation write failed for learner {req.learner_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database escalation write failed: {str(e)}"
        )
        
    execution_latency_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"[{corr_id}] Successfully persisted new escalation ticket {ticket_id} for learner {req.learner_id} (idempotency_key: {idemp_key}).")
    return {
        "status": "persisted",
        "ticket_id": ticket_id,
        "learner_id": req.learner_id,
        "module_id": resolved_mod,
        "db_record_status": req.status,
        "priority": req.priority,
        "correlation_id": corr_id,
        "idempotency_key": idemp_key,
        "payload_hash": payload_hash,
        "persisted_to_db": True,
        "idempotent_deduplicated": False,
        "message": "Escalation ticket persisted successfully into relational database with referential integrity verified.",
        "execution_latency_ms": execution_latency_ms,
        "sla_met": execution_latency_ms < 3500.0
    }

@app.post("/api/v1/predict_risk")
def predict_learner_risk(req: RiskPredictionRequest, x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID")):
    """
    Sprint 2: Evaluates a student's structured quiz profile using the PyTorch MLP.
    Supports both Experiment A (Early Risk W1-W3) and Experiment B (Full Trajectory W1-W5).
    Guarantees 100% preprocessing parity with train_pytorch.py:
    - Same training-set city/global median imputation
    - Causal forward fill (no backward fill leakage)
    - Sample standard deviation (ddof=1)
    """
    if "risk_classifier" not in GLOBAL_MODELS:
        raise HTTPException(status_code=503, detail="Risk classifier model not loaded.")
        
    start_time = time.time()
    ckpt = GLOBAL_MODELS["risk_meta"]
    city_medians = ckpt.get('city_medians', {})
    global_medians = ckpt.get('global_medians', {})
    
    # Resolve student's cohort_city (from payload or managed database lookup)
    city = req.cohort_city or lookup_learner_city(req.learner_id)

    # Check if this is an early intervention scenario (Weeks 1-3 only)
    is_early = (req.week_4_quiz is None and req.week_5_quiz is None)
    
    if is_early and "early_risk_classifier" in GLOBAL_MODELS:
        model = GLOBAL_MODELS["early_risk_classifier"]
        raw_feats = extract_single_learner_features(
            raw_quizzes=[req.week_1_quiz, req.week_2_quiz, req.week_3_quiz],
            cohort_city=city,
            city_medians=city_medians,
            global_medians=global_medians,
            mode="early"
        )
        mean = ckpt['early_scaler_mean']
        scale = ckpt['early_scaler_scale']
        scaled_feats = (raw_feats - mean) / scale
        threshold = float(ckpt.get('early_threshold', 0.35))
        model_used = "Experiment A: Proactive Early Risk (Weeks 1-3)"
        score_delta = float(raw_feats[0, 3])
    else:
        model = GLOBAL_MODELS["risk_classifier"]
        raw_feats = extract_single_learner_features(
            raw_quizzes=[req.week_1_quiz, req.week_2_quiz, req.week_3_quiz, req.week_4_quiz, req.week_5_quiz],
            cohort_city=city,
            city_medians=city_medians,
            global_medians=global_medians,
            mode="full"
        )
        mean = ckpt['scaler_mean']
        scale = ckpt['scaler_scale']
        scaled_feats = (raw_feats - mean) / scale
        threshold = float(ckpt.get('decision_threshold', 0.35))
        model_used = "Experiment B: Full Trajectory Risk (Weeks 1-5)"
        score_delta = float(raw_feats[0, 5])
        
    with torch.no_grad():
        tensor_x = torch.tensor(scaled_feats, dtype=torch.float32)
        logits = model(tensor_x)
        prob = float(torch.sigmoid(logits).item())
        
    is_high_risk = bool(prob >= threshold)
    
    return {
        "learner_id": req.learner_id,
        "cohort_city": city or "Unknown / Global Default",
        "model_used": model_used,
        "risk_probability": round(prob, 4),
        "risk_level": "High Risk" if is_high_risk else "Low Risk",
        "flagged_for_intervention": is_high_risk,
        "decision_threshold": round(threshold, 2),
        "score_delta": round(float(score_delta), 1),
        "correlation_id": req.correlation_id or x_correlation_id or f"corr-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}",
        "execution_latency_ms": round((time.time() - start_time) * 1000, 2),
        "sla_met": round((time.time() - start_time) * 1000, 2) < 3500.0
    }

@app.post("/api/v1/predict_risk_early")
def predict_learner_risk_early(req: EarlyRiskPredictionRequest, x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID")):
    """
    Sprint 2 (Experiment A): Dedicated Proactive Early Risk Classifier.
    Predicts dropout risk at Week 3 before learner attrition occurs.
    Test F1 = 0.9714 | Recall = 0.9855 | ROC-AUC = 0.9996
    Guarantees unified preprocessing parity (train city medians + sample std ddof=1).
    """
    if "early_risk_classifier" not in GLOBAL_MODELS:
        raise HTTPException(status_code=503, detail="Early risk classifier model not loaded.")
        
    start_time = time.time()
    ckpt = GLOBAL_MODELS["risk_meta"]
    model = GLOBAL_MODELS["early_risk_classifier"]
    city_medians = ckpt.get('city_medians', {})
    global_medians = ckpt.get('global_medians', {})
    
    city = req.cohort_city or lookup_learner_city(req.learner_id)
            
    raw_feats = extract_single_learner_features(
        raw_quizzes=[req.week_1_quiz, req.week_2_quiz, req.week_3_quiz],
        cohort_city=city,
        city_medians=city_medians,
        global_medians=global_medians,
        mode="early"
    )
    
    mean = ckpt['early_scaler_mean']
    scale = ckpt['early_scaler_scale']
    scaled_feats = (raw_feats - mean) / scale
    threshold = float(ckpt.get('early_threshold', 0.35))
    delta_w1_w3 = float(raw_feats[0, 3])
    early_mean = float(raw_feats[0, 4])
    
    with torch.no_grad():
        tensor_x = torch.tensor(scaled_feats, dtype=torch.float32)
        logits = model(tensor_x)
        prob = float(torch.sigmoid(logits).item())
        
    is_high_risk = bool(prob >= threshold)
    
    return {
        "learner_id": req.learner_id,
        "cohort_city": city or "Unknown / Global Default",
        "model_used": "Experiment A: Proactive Early Risk (Weeks 1-3)",
        "risk_probability": round(prob, 4),
        "risk_level": "High Risk" if is_high_risk else "Low Risk",
        "flagged_for_intervention": is_high_risk,
        "decision_threshold": round(threshold, 2),
        "score_delta_w1_w3": round(float(delta_w1_w3), 1),
        "early_mean_score": round(float(early_mean), 1),
        "correlation_id": req.correlation_id or x_correlation_id or f"corr-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}",
        "execution_latency_ms": round((time.time() - start_time) * 1000, 2),
        "sla_met": round((time.time() - start_time) * 1000, 2) < 3500.0
    }

@app.post("/api/v1/predict_sequence")
def predict_sequence_trajectory(req: SequencePredictionRequest, x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID")):
    """
    Sprint 3: Feeds 5-week quiz vector into PyTorch LSTM to forecast capstone outcome.
    Utilizes Forget Gate dynamics and the Constant Error Carousel (Cell State).
    """
    if "sequence_lstm" not in GLOBAL_MODELS:
        raise HTTPException(status_code=503, detail="Sequence LSTM model not loaded.")
        
    start_time = time.time()
    norm_scores = np.array(req.quiz_scores, dtype=np.float32) / 100.0
    tensor_seq = torch.tensor(norm_scores).unsqueeze(0).unsqueeze(-1) # Shape: (1, 5, 1)
    
    model = GLOBAL_MODELS["sequence_lstm"]
    with torch.no_grad():
        pred_norm, _ = model(tensor_seq)
        predicted_capstone = float(pred_norm.item() * 100.0)
        
    return {
        "learner_id": req.learner_id,
        "input_trajectory": req.quiz_scores,
        "predicted_capstone_score": round(predicted_capstone, 2),
        "forecast_status": "Projected Pass" if predicted_capstone >= 60.0 else "Projected Fail (At Risk)",
        "recurrent_mechanism": "LSTM Forget Gate & Cell State (Constant Error Carousel)",
        "correlation_id": req.correlation_id or x_correlation_id or f"corr-{int(time.time()*1000)}-{uuid.uuid4().hex[:6]}",
        "execution_latency_ms": round((time.time() - start_time) * 1000, 2),
        "sla_met": round((time.time() - start_time) * 1000, 2) < 3500.0
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
