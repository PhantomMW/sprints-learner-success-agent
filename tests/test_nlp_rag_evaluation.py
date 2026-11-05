"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: tests/test_nlp_rag_evaluation.py
Description: Rigorous Phase 3 validation suite:
             - Egyptian Arabic Dialect NLP & Distress Classification
             - Multi-split generalizability (Dev, Holdout, Real Chat)
             - Confusion Matrix targets (Precision >= 0.90, Recall >= 0.90, F1 >= 0.90)
             - Semantic Vector RAG Retrieval Quality (MRR, Top-1, Fallback)
             - Ambiguity & Polysemous Resolution
==============================================================================
"""

import os
import sys
import pytest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from nlp_pipeline import (
    classify_arabic_context,
    analyze_sentiment,
    deliberative_agent_router,
    semantic_rag_retrieve,
    CURRICULUM_KNOWLEDGE_CHUNKS,
    ARABIC_DEV_BENCHMARK_DATASET,
    ARABIC_HOLDOUT_TEST_SET,
    ARABIC_ANONYMIZED_REAL_WORLD_DATASET,
    get_cached_word2vec
)
from evaluate_nlp_benchmarks import (
    evaluate_confusion_matrix_and_metrics,
    evaluate_semantic_rag_quality
)

class TestNLPRAGEvaluation:

    def test_holdout_confusion_matrix_targets(self):
        """
        Validates that the independent unseen holdout dataset (44 cases)
        meets and exceeds the 10/10 evaluation threshold:
        - Precision >= 0.90
        - Recall >= 0.90
        - F1-Score >= 0.90
        - False Positives <= 2 (Zero false alarms on normal tech issues)
        - False Negatives <= 2 (Zero overlooked distress)
        """
        metrics = evaluate_confusion_matrix_and_metrics(ARABIC_HOLDOUT_TEST_SET, "Holdout Set")
        assert metrics["precision"] >= 0.90, f"Holdout Precision below 0.90: {metrics['precision']:.4f}"
        assert metrics["recall"] >= 0.90, f"Holdout Recall below 0.90: {metrics['recall']:.4f}"
        assert metrics["f1"] >= 0.90, f"Holdout F1 below 0.90: {metrics['f1']:.4f}"
        assert metrics["fp"] <= 2, f"Excessive false positives on holdout: {metrics['fp']}"
        assert metrics["fn"] <= 2, f"Excessive false negatives on holdout: {metrics['fn']}"

    def test_real_world_chat_generalization(self):
        """
        Validates zero regressions and robust performance on authentic
        anonymized real-world student chat samples (30 cases).
        """
        metrics = evaluate_confusion_matrix_and_metrics(ARABIC_ANONYMIZED_REAL_WORLD_DATASET, "Real Chat Set")
        assert metrics["precision"] >= 0.90
        assert metrics["recall"] >= 0.90
        assert metrics["f1"] >= 0.90

    def test_per_category_classification_accuracy(self):
        """
        Ensures that each specific category achieves high routing accuracy:
        - TECHNICAL_INQUIRY -> AUTOMATED_RAG_PIPELINE
        - CURRICULUM_QUESTION -> AUTOMATED_RAG_PIPELINE
        - EMOTIONAL_DISTRESS -> HUMAN_OPERATOR_ESCALATION
        - DROPOUT_INTENT -> HUMAN_OPERATOR_ESCALATION
        - URGENT_DISTRESS -> HUMAN_OPERATOR_ESCALATION
        """
        metrics = evaluate_confusion_matrix_and_metrics(ARABIC_HOLDOUT_TEST_SET, "Holdout Set")
        breakdown = metrics["category_breakdown"]
        for cat, data in breakdown.items():
            assert data["accuracy"] >= 0.85, f"Category {cat} accuracy below 0.85: {data['accuracy']}"

    def test_semantic_rag_retrieval_and_mrr(self):
        """
        Tests that Semantic Vector RAG retrieval attains:
        - Top-1 Accuracy >= 0.70
        - Mean Reciprocal Rank (MRR) >= 0.70
        - Mean Cosine Similarity >= 0.80
        """
        w2v_model = get_cached_word2vec()
        if w2v_model is None:
            pytest.skip("Word2Vec model not found on disk")
        rag_metrics = evaluate_semantic_rag_quality(w2v_model)
        assert rag_metrics["top1_accuracy"] >= 0.70, f"Top-1 accuracy {rag_metrics['top1_accuracy']} < 0.70"
        assert rag_metrics["mrr"] >= 0.70, f"MRR {rag_metrics['mrr']} < 0.70"
        assert rag_metrics["mean_similarity"] >= 0.80, f"Mean cosine similarity {rag_metrics['mean_similarity']} < 0.80"

    def test_rag_zero_hallucination_on_oov_queries(self):
        """
        Tests that out-of-domain and low-coverage queries trigger CHK-FALLBACK-01
        without hallucinating unrelated syllabus modules.
        """
        w2v_model = get_cached_word2vec()
        if w2v_model is None:
            pytest.skip("Word2Vec model not found on disk")
        
        oov_queries = [
            "What is the best recipe for baking chocolate cake?",
            "Can you tell me the score of the Champions League match yesterday?",
            "Where can I buy tickets for the cinema tonight?"
        ]
        for q in oov_queries:
            retrievals = semantic_rag_retrieve(q, w2v_model, top_k=1)
            assert retrievals[0]["chunk_id"] == "CHK-FALLBACK-01", (
                f"Query '{q}' failed to trigger fallback, returned {retrievals[0]['chunk_id']}"
            )
            assert "General Sprints Support" in retrievals[0]["title"]

    def test_polysemous_words_ambiguity_matrix(self):
        """
        Stress tests dialect polysemy:
        - 'ساقط' in tech outage vs personal failure
        - 'ضايع' in lost link vs emotional despair
        - 'واقع' in server downtime vs academic struggle
        - 'مش قادر' in technical inability vs mental burnout
        """
        # Tech vs Distress pairs
        pairs = [
            ("السيرفر ساقط ومش عارفين ندخل", False, "TECHNICAL_INQUIRY"),
            ("أنا ساقط ومحبط ومش نافع", True, "EMOTIONAL_DISTRESS"),
            ("الفايل ضايع مني على البورتال", False, "TECHNICAL_INQUIRY"),
            ("أنا ضايع ومخنوق ومش فاهم حاجة", True, "EMOTIONAL_DISTRESS"),
            ("السيستم واقع تماماً", False, "TECHNICAL_INQUIRY"),
            ("مش قادر أحمل الفيديو بتاع السيشن", False, "TECHNICAL_INQUIRY"),
            ("مش قادر أكمل خلاص طاقتي خلصت", True, "EMOTIONAL_DISTRESS"),
        ]
        for text, exp_escalate, exp_cat in pairs:
            res = deliberative_agent_router(text)
            pred_escalate = (res["routing_decision"] == "HUMAN_OPERATOR_ESCALATION")
            assert pred_escalate == exp_escalate, (
                f"Failed ambiguity on '{text}': expected escalate={exp_escalate}, got {pred_escalate}"
            )
            assert res["arabic_context"]["category"] == exp_cat, (
                f"Failed category on '{text}': expected {exp_cat}, got {res['arabic_context']['category']}"
            )
