"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: tests/test_arabic_nlp_metrics.py
Description: Rigorous automated evaluation suite for Egyptian Arabic dialect
             distress detection, technical task disambiguation, and rubric adherence.
Measures: Precision >= 0.90, Recall >= 0.90, F1-Score >= 0.90
Verifies: 6 Evaluator Scenarios + Contextual Ambiguity Cases ('ساقط', 'ضايع', 'مش قادر')
==============================================================================
"""

import os
import sys
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from nlp_pipeline import (
    classify_arabic_context,
    analyze_sentiment,
    deliberative_agent_router,
    evaluate_arabic_distress_benchmark
)

class TestArabicNLPMetrics:

    def test_arabic_distress_benchmark_metrics(self):
        """
        Evaluates the full 26-case gold-standard Arabic dialect benchmark.
        Strictly asserts Precision, Recall, and F1 all achieve >= 0.90.
        """
        metrics = evaluate_arabic_distress_benchmark()
        
        assert metrics["precision"] >= 0.90, f"Precision target (>= 0.90) breached: {metrics['precision']:.4f}"
        assert metrics["recall"] >= 0.90, f"Recall target (>= 0.90) breached: {metrics['recall']:.4f}"
        assert metrics["f1"] >= 0.90, f"F1-Score target (>= 0.90) breached: {metrics['f1']:.4f}"
        assert metrics["accuracy"] >= 0.90, f"Accuracy target (>= 0.90) breached: {metrics['accuracy']:.4f}"
        assert metrics["fp"] == 0, f"Expected 0 False Positives on technical queries, got {metrics['fp']}"
        assert metrics["fn"] == 0, f"Expected 0 False Negatives on distressed students, got {metrics['fn']}"

    @pytest.mark.parametrize("query,expected_decision,expected_priority,expected_cat", [
        # Scenario 1: Routine educational question (Gradient Descent explanation -> RAG)
        ("مش فاهم Gradient Descent، ممكن شرح؟", "AUTOMATED_RAG_PIPELINE", "Medium", "CURRICULUM_QUESTION"),
        # Scenario 2: Technical task inability (Cannot open file -> RAG)
        ("مش قادر أفتح الملف", "AUTOMATED_RAG_PIPELINE", "Low", "TECHNICAL_INQUIRY"),
        # Scenario 3: Clear emotional frustration (Frustrated & cannot continue -> Escalation)
        ("أنا محبط جدًا ومش قادر أكمل", "HUMAN_OPERATOR_ESCALATION", "Critical", "EMOTIONAL_DISTRESS"),
        # Scenario 4: Direct dropout intention (Wants to withdraw -> Escalation)
        ("أنا عايز أنسحب من الكورس", "HUMAN_OPERATOR_ESCALATION", "Critical", "DROPOUT_INTENT"),
        # Scenario 5: Academic delay (Cannot finish homework in time -> Escalation)
        ("مش لاحق أسلم التاسكات", "HUMAN_OPERATOR_ESCALATION", "High", "ACADEMIC_DELAY"),
        # Scenario 6: Urgent distress and emergency plea (Help me, leaving course -> Escalation)
        ("الحقوني، أنا هسيب الكورس", "HUMAN_OPERATOR_ESCALATION", "Critical", "DROPOUT_INTENT"),
    ])
    def test_six_evaluator_scenarios(self, query, expected_decision, expected_priority, expected_cat):
        """
        Validates all 6 core scenarios explicitly mandated by the evaluator.
        """
        res = deliberative_agent_router(query)
        assert res["routing_decision"] == expected_decision, (
            f"Query '{query}' expected '{expected_decision}', got '{res['routing_decision']}'"
        )
        assert res["priority"] == expected_priority, (
            f"Query '{query}' expected priority '{expected_priority}', got '{res['priority']}'"
        )
        assert res["arabic_context"]["category"] == expected_cat, (
            f"Query '{query}' expected category '{expected_cat}', got '{res['arabic_context']['category']}'"
        )

    def test_contextual_ambiguity_words(self):
        """
        Tests disambiguation of ambiguous polysemous words ('ساقط', 'ضايع', 'مش قادر')
        between technical tool issues and genuine student despair.
        """
        # Technical usage of 'ساقط' (Platform down) -> Automated RAG
        outage_res = deliberative_agent_router("السيستم ساقط ومش عارف أسجل الحضور")
        assert outage_res["routing_decision"] == "AUTOMATED_RAG_PIPELINE"
        assert outage_res["arabic_context"]["category"] == "TECHNICAL_INQUIRY"

        # Emotional usage of 'ساقط' (Student feeling failure) -> Human Escalation
        despair_res = deliberative_agent_router("أنا ساقط ومحبط ومش نافع في الكورس")
        assert despair_res["routing_decision"] == "HUMAN_OPERATOR_ESCALATION"
        assert despair_res["priority"] == "Critical"
        assert despair_res["arabic_context"]["category"] == "EMOTIONAL_DISTRESS"

        # Technical usage of 'ضايع' (File link missing) -> Automated RAG
        file_res = deliberative_agent_router("الملف ضايع مني على البورتال مش لاقيه")
        assert file_res["routing_decision"] == "AUTOMATED_RAG_PIPELINE"
        assert file_res["arabic_context"]["category"] == "TECHNICAL_INQUIRY"

        # Emotional usage of 'ضايع' (Student lost and burned out) -> Human Escalation
        lost_res = deliberative_agent_router("أنا ضايع ومحبط ومخنوق جدًا")
        assert lost_res["routing_decision"] == "HUMAN_OPERATOR_ESCALATION"
        assert lost_res["priority"] == "Critical"
        assert lost_res["arabic_context"]["category"] == "EMOTIONAL_DISTRESS"

        # Technical usage of 'مش قادر' (Cannot download video) -> Automated RAG
        download_res = deliberative_agent_router("مش قادر أحمل الفيديو بتاع السيشن")
        assert download_res["routing_decision"] == "AUTOMATED_RAG_PIPELINE"
        assert download_res["arabic_context"]["category"] == "TECHNICAL_INQUIRY"

        # Emotional usage of 'مش قادر' (Cannot continue bootcamp) -> Human Escalation
        dropout_res = deliberative_agent_router("مش قادر أكمل في البرنامج ومخنوق")
        assert dropout_res["routing_decision"] == "HUMAN_OPERATOR_ESCALATION"
        assert dropout_res["priority"] == "Critical"
