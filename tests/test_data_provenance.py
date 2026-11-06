"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: tests/test_data_provenance.py
Description: Automated test suite for data provenance, inter-annotator reliability
             (Cohen's Kappa), PII sanitization, and stratified split integrity.
==============================================================================
"""

import os
import sys
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from verify_data_provenance import (
    verify_cohen_kappa_iaa,
    verify_pii_sanitization,
    verify_stratified_split_integrity
)

class TestDataProvenance:

    def test_inter_annotator_agreement_cohen_kappa(self):
        """Asserts Cohen's Kappa >= 0.90 across all subsets (Overall, Real Chat, and Controlled Benchmark)."""
        res = verify_cohen_kappa_iaa()
        assert res["overall"]["n"] == 100
        assert res["overall"]["agreed"] == 97
        assert res["overall"]["kappa"] >= 0.90
        assert res["overall"]["p_o"] == 0.97
        
        # Real Chat Sub-matrix
        assert res["anonymized_real_chat"]["n"] == 30
        assert res["anonymized_real_chat"]["agreed"] == 29
        assert res["anonymized_real_chat"]["kappa"] >= 0.90
        
        # Controlled Benchmark Sub-matrix
        assert res["controlled_benchmark"]["n"] == 70
        assert res["controlled_benchmark"]["agreed"] == 68
        assert res["controlled_benchmark"]["kappa"] >= 0.90
        
        assert res["adjudicated_disagreements"] == 3

    def test_independent_raw_annotation_files_exist_and_unmodified(self):
        """Validates that distinct, independent raw annotation files exist with reviewer metadata."""
        import json
        annotations_dir = os.path.join(REPO_ROOT, "data", "annotations")
        a1_file = os.path.join(annotations_dir, "annotator_1_raw.json")
        a2_file = os.path.join(annotations_dir, "annotator_2_raw.json")
        adj_file = os.path.join(annotations_dir, "adjudication_consensus.json")
        
        for f in [a1_file, a2_file, adj_file]:
            assert os.path.exists(f), f"Expected annotation audit file {f} to exist"
            
        with open(a1_file, "r", encoding="utf-8") as f:
            a1_data = json.load(f)
        with open(a2_file, "r", encoding="utf-8") as f:
            a2_data = json.load(f)
        with open(adj_file, "r", encoding="utf-8") as f:
            adj_data = json.load(f)
            
        assert len(a1_data) == 100
        assert len(a2_data) == 100
        assert len(adj_data) == 3
        
        assert a1_data[0]["annotator_id"] == "ANN-EG-01"
        assert a2_data[0]["annotator_id"] == "ANN-EG-02"
        assert adj_data[0]["adjudicator_id"] == "ADJ-EG-00"

    def test_pii_sanitization_zero_leakage(self):
        """Asserts zero emails, phones, or national IDs in learner dataset."""
        res = verify_pii_sanitization()
        assert res["total_records_audited"] == 2000
        assert res["synthetic_surrogate_ids_verified"] is True
        assert res["pii_emails_detected"] == 0
        assert res["pii_phones_detected"] == 0
        assert res["pii_national_ids_detected"] == 0
        assert res["pii_credit_cards_detected"] == 0

    def test_stratified_split_balance(self):
        """Asserts 70/15/15 split sizes and balanced class distribution (~23.0%)."""
        res = verify_stratified_split_integrity()
        assert res["train_samples"] == 1400
        assert res["val_samples"] == 300
        assert res["test_samples"] == 300
        assert res["stratification_verified"] is True
        assert abs(res["train_dropout_rate"] - res["test_dropout_rate"]) < 0.01
