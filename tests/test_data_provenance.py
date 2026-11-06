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
        """Asserts Cohen's Kappa >= 0.90 across the 100 benchmark queries."""
        res = verify_cohen_kappa_iaa()
        assert res["total_queries"] == 100
        assert res["initial_agreed_queries"] == 97
        assert res["adjudicated_queries"] == 3
        assert res["cohens_kappa"] >= 0.90
        assert res["observed_agreement_po"] == 0.97

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
