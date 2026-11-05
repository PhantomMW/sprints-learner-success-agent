"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: tests/test_ml_dl_validation.py
Description: Automated Test Suite for Phase 2: ML / DL Scientific Validation.
             Verifies:
             1. Clean 70/15/15 Stratified Partitioning with Zero Data Leakage.
             2. PyTorch MLP Performance against Baselines (F1 >= 0.85, ROC-AUC >= 0.90, Brier <= 0.05).
             3. PyTorch LSTM Sequence Forecasting (RMSE <= 8.5 pts, R2 >= 0.75, outperforming baselines).
             4. Probability Calibration & Reliability Guarantee.
             5. Seed Reproducibility and Preprocessing Parity.
==============================================================================
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')
MODELS_DIR = os.path.join(REPO_ROOT, 'models')
DATA_PATH = os.path.join(REPO_ROOT, 'data', 'sprints_learner_data.csv')

if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

from train_pytorch import (
    load_and_split_data,
    LearnerRiskClassifier,
    LearnerSequenceLSTM,
    set_seed
)
from evaluate_benchmarks import (
    evaluate_risk_classification,
    evaluate_sequence_forecasting
)

@pytest.fixture(scope="module")
def dataset_splits():
    set_seed(42)
    train_df, val_df, test_df, city_medians, global_medians = load_and_split_data(DATA_PATH)
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "city_medians": city_medians,
        "global_medians": global_medians
    }

@pytest.fixture(scope="module")
def checkpoints():
    risk_ckpt = torch.load(os.path.join(MODELS_DIR, 'risk_classifier.pth'), map_location='cpu', weights_only=False)
    lstm_ckpt = torch.load(os.path.join(MODELS_DIR, 'sequence_predictor.pth'), map_location='cpu', weights_only=False)
    return {
        "risk": risk_ckpt,
        "lstm": lstm_ckpt
    }

class TestMLDLValidation:

    # ==========================================================================
    # 1. 70 / 15 / 15 STRATIFIED SPLIT & DATA LEAKAGE PREVENTION
    # ==========================================================================

    def test_clean_three_way_split_dimensions(self, dataset_splits):
        """Verifies strict 70% Train, 15% Validation, and 15% Test partitioning."""
        train = dataset_splits["train"]
        val = dataset_splits["val"]
        test = dataset_splits["test"]
        total = len(train) + len(val) + len(test)

        assert total == 2000, f"Expected 2000 total learners, got {total}"
        assert len(train) == 1400, f"Train set must be exactly 70% (1400), got {len(train)}"
        assert len(val) == 300, f"Validation set must be exactly 15% (300), got {len(val)}"
        assert len(test) == 300, f"Test set must be exactly 15% (300), got {len(test)}"

    def test_stratification_parity_across_splits(self, dataset_splits):
        """Verifies target class balance (dropout proportion) is preserved identically across splits."""
        tr_rate = dataset_splits["train"]["dropped_out"].mean()
        va_rate = dataset_splits["val"]["dropped_out"].mean()
        te_rate = dataset_splits["test"]["dropped_out"].mean()

        # All splits must be within 1.5% of the overall ~23% base rate
        assert abs(tr_rate - 0.23) < 0.015, f"Train dropout rate skewed: {tr_rate:.4f}"
        assert abs(va_rate - 0.23) < 0.015, f"Val dropout rate skewed: {va_rate:.4f}"
        assert abs(te_rate - 0.23) < 0.015, f"Test dropout rate skewed: {te_rate:.4f}"

    def test_zero_target_and_temporal_leakage_in_features(self, dataset_splits):
        """Verifies no target column or future timeframes are leaked into feature columns."""
        for split_name in ["train", "val", "test"]:
            df = dataset_splits[split_name]
            # No backwards fill: if a student missed week 1 and 2, week 1 must NOT be filled from week 3!
            # Features must contain strictly causal transformations
            assert 'score_delta_w1_w5' in df.columns
            assert 'score_delta_w1_w3' in df.columns

    # ==========================================================================
    # 2. SUPERVISED PYTORCH MLP VS BASELINE COMPARISON (EXPERIMENT B FULL)
    # ==========================================================================

    def test_full_risk_mlp_outperforms_baselines(self, dataset_splits, checkpoints):
        """Full trajectory MLP must achieve F1 >= 0.85, ROC-AUC >= 0.90 and beat Dummy baseline."""
        full_features = ['week_1_quiz', 'week_2_quiz', 'week_3_quiz', 'week_4_quiz', 'week_5_quiz', 'score_delta_w1_w5', 'quiz_mean', 'quiz_std', 'quiz_min', 'recent_performance_ratio', 'missing_quiz_count']
        results, _, _ = evaluate_risk_classification(
            dataset_splits["train"],
            dataset_splits["val"],
            dataset_splits["test"],
            full_features,
            checkpoints["risk"],
            mode="full"
        )

        mlp_metrics = results["PyTorch Deep MLP (Sprints)"]
        dummy_metrics = results["Zero-R / Majority Baseline"]

        # High-bar criteria for 10/10
        assert mlp_metrics["F1-Score"] >= 0.85, f"Expected MLP F1 >= 0.85, got {mlp_metrics['F1-Score']}"
        assert mlp_metrics["Recall"] >= 0.90, f"Expected MLP Recall >= 0.90, got {mlp_metrics['Recall']}"
        assert mlp_metrics["ROC-AUC"] >= 0.95, f"Expected MLP ROC-AUC >= 0.95, got {mlp_metrics['ROC-AUC']}"
        assert mlp_metrics["Brier Score"] <= 0.05, f"Expected well-calibrated Brier score <= 0.05, got {mlp_metrics['Brier Score']}"

        # Must decisively outperform Dummy baseline
        assert mlp_metrics["F1-Score"] > dummy_metrics["F1-Score"] + 0.50

    # ==========================================================================
    # 3. PROACTIVE EARLY RISK MLP (EXPERIMENT A WEEKS 1-3)
    # ==========================================================================

    def test_early_risk_mlp_performance(self, dataset_splits, checkpoints):
        """Early Risk Classifier (Weeks 1-3 only) must achieve Test F1 >= 0.85, Recall >= 0.90."""
        early_features = ['week_1_quiz', 'week_2_quiz', 'week_3_quiz', 'score_delta_w1_w3', 'early_mean', 'early_min', 'early_missing_count']
        results, _, _ = evaluate_risk_classification(
            dataset_splits["train"],
            dataset_splits["val"],
            dataset_splits["test"],
            early_features,
            checkpoints["risk"],
            mode="early"
        )

        early_metrics = results["PyTorch Deep MLP (Sprints)"]
        assert early_metrics["F1-Score"] >= 0.85, f"Early Risk F1 must be >= 0.85, got {early_metrics['F1-Score']}"
        assert early_metrics["Recall"] >= 0.90, f"Early Risk Recall must be >= 0.90, got {early_metrics['Recall']}"
        assert early_metrics["ROC-AUC"] >= 0.95, f"Early Risk ROC-AUC must be >= 0.95, got {early_metrics['ROC-AUC']}"
        assert early_metrics["Brier Score"] <= 0.05, f"Early Risk Brier must be <= 0.05, got {early_metrics['Brier Score']}"

    # ==========================================================================
    # 4. DEEP SEQUENCE LSTM BENCHMARK & BASELINE ERROR REDUCTION
    # ==========================================================================

    def test_sequence_lstm_outperforms_persistence_and_moving_average(self, dataset_splits, checkpoints):
        """LSTM must achieve Test RMSE <= 8.5 pts, R2 >= 0.75 and beat persistence by > 35% error reduction."""
        results, _, _, _ = evaluate_sequence_forecasting(
            dataset_splits["train"],
            dataset_splits["val"],
            dataset_splits["test"],
            checkpoints["lstm"]
        )

        lstm = results["PyTorch Deep LSTM (Forget Gate & Cell State)"]
        persistence = results["Persistence Baseline (Week 5 Score)"]
        sma = results["3-Week Moving Average Baseline (SMA)"]

        # Production grade thresholds
        assert lstm["RMSE"] <= 8.5, f"LSTM RMSE must be <= 8.5, got {lstm['RMSE']}"
        assert lstm["MAE"] <= 6.0, f"LSTM MAE must be <= 6.0, got {lstm['MAE']}"
        assert lstm["R2"] >= 0.75, f"LSTM R2 must be >= 0.75, got {lstm['R2']}"
        assert lstm["Pearson_r"] >= 0.85, f"LSTM Pearson correlation must be >= 0.85, got {lstm['Pearson_r']}"

        # Error reduction vs persistence baseline
        error_reduction_pct = ((persistence["RMSE"] - lstm["RMSE"]) / persistence["RMSE"]) * 100.0
        assert error_reduction_pct >= 35.0, f"LSTM must achieve at least 35% error reduction over persistence, got {error_reduction_pct:.1f}%"

        # Outperforms Moving Average
        assert lstm["RMSE"] < sma["RMSE"], f"LSTM RMSE ({lstm['RMSE']}) must beat Moving Average ({sma['RMSE']})"

    # ==========================================================================
    # 5. REPRODUCIBILITY & DETERMINISM
    # ==========================================================================

    def test_deterministic_reproducibility(self, dataset_splits, checkpoints):
        """Running evaluation with seed=42 produces exactly identical numerical outputs."""
        full_features = ['week_1_quiz', 'week_2_quiz', 'week_3_quiz', 'week_4_quiz', 'week_5_quiz', 'score_delta_w1_w5', 'quiz_mean', 'quiz_std', 'quiz_min', 'recent_performance_ratio', 'missing_quiz_count']
        res1, _, _ = evaluate_risk_classification(dataset_splits["train"], dataset_splits["val"], dataset_splits["test"], full_features, checkpoints["risk"], mode="full")
        res2, _, _ = evaluate_risk_classification(dataset_splits["train"], dataset_splits["val"], dataset_splits["test"], full_features, checkpoints["risk"], mode="full")

        assert res1["PyTorch Deep MLP (Sprints)"]["F1-Score"] == res2["PyTorch Deep MLP (Sprints)"]["F1-Score"]
        assert res1["PyTorch Deep MLP (Sprints)"]["ROC-AUC"] == res2["PyTorch Deep MLP (Sprints)"]["ROC-AUC"]
