import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch

# Ensure repo root is in python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.train_pytorch import extract_single_learner_features

class TestFeatureParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ckpt_path = os.path.join(REPO_ROOT, 'models', 'risk_classifier.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
        cls.ckpt = torch.load(ckpt_path, weights_only=False)
        cls.city_medians = cls.ckpt['city_medians']
        cls.global_medians = cls.ckpt['global_medians']
        cls.quiz_cols = ['week_1_quiz', 'week_2_quiz', 'week_3_quiz', 'week_4_quiz', 'week_5_quiz']
        cls.w13_cols = ['week_1_quiz', 'week_2_quiz', 'week_3_quiz']
        cls.early_feature_cols = [
            'week_1_quiz', 'week_2_quiz', 'week_3_quiz',
            'score_delta_w1_w3', 'early_mean', 'early_min', 'early_missing_count'
        ]
        cls.full_feature_cols = [
            'week_1_quiz', 'week_2_quiz', 'week_3_quiz', 'week_4_quiz', 'week_5_quiz',
            'score_delta_w1_w5', 'quiz_mean', 'quiz_std', 'quiz_min',
            'recent_performance_ratio', 'missing_quiz_count'
        ]

    def _compute_df_features(self, city, quizzes):
        """Replicates exact training pipeline DataFrame transformation from train_pytorch.py"""
        df_row = pd.DataFrame([{
            'cohort_city': city,
            'week_1_quiz': quizzes[0],
            'week_2_quiz': quizzes[1],
            'week_3_quiz': quizzes[2],
            'week_4_quiz': quizzes[3],
            'week_5_quiz': quizzes[4],
        }])

        # Missing indicator counts (computed before imputation)
        df_row['missing_quiz_count'] = df_row[self.quiz_cols].isna().sum(axis=1)
        df_row['early_missing_count'] = df_row[self.w13_cols].isna().sum(axis=1)

        # Impute using city medians or global medians
        for col in self.quiz_cols:
            city_val = self.city_medians.get(city, {}).get(col, self.global_medians.get(col, 50.0))
            df_row[col] = df_row[col].fillna(city_val)

        # Causal forward fill along temporal axis
        df_row[self.quiz_cols] = df_row[self.quiz_cols].ffill(axis=1).fillna(50.0)

        # Early feature engineering
        df_row['score_delta_w1_w3'] = df_row['week_1_quiz'] - df_row['week_3_quiz']
        df_row['early_mean'] = df_row[self.w13_cols].mean(axis=1)
        df_row['early_min'] = df_row[self.w13_cols].min(axis=1)

        # Full feature engineering
        df_row['score_delta_w1_w5'] = df_row['week_1_quiz'] - df_row['week_5_quiz']
        df_row['quiz_mean'] = df_row[self.quiz_cols].mean(axis=1)
        df_row['quiz_std'] = df_row[self.quiz_cols].std(axis=1).fillna(0.0)
        df_row['quiz_min'] = df_row[self.quiz_cols].min(axis=1)
        df_row['recent_performance_ratio'] = df_row['week_5_quiz'] / (df_row['week_1_quiz'] + 1e-5)

        early_feats = df_row[self.early_feature_cols].values.astype(np.float32)
        full_feats = df_row[self.full_feature_cols].values.astype(np.float32)
        return early_feats, full_feats

    def _assert_parity(self, city, quizzes, case_name):
        df_early, df_full = self._compute_df_features(city, quizzes)
        fn_early = extract_single_learner_features(
            quizzes[:3], cohort_city=city,
            city_medians=self.city_medians, global_medians=self.global_medians,
            mode='early'
        )
        fn_full = extract_single_learner_features(
            quizzes, cohort_city=city,
            city_medians=self.city_medians, global_medians=self.global_medians,
            mode='full'
        )

        np.testing.assert_allclose(
            fn_early, df_early, rtol=1e-5, atol=1e-5,
            err_msg=f"Early feature mismatch in test case: {case_name}"
        )
        np.testing.assert_allclose(
            fn_full, df_full, rtol=1e-5, atol=1e-5,
            err_msg=f"Full feature mismatch in test case: {case_name}"
        )

    def test_complete_data_no_missing(self):
        """Case 1: Complete data without missing values"""
        self._assert_parity('Cairo', [85.0, 80.0, 75.0, 70.0, 65.0], "Complete data")

    def test_missing_week_2(self):
        """Case 2: Missing Week 2 score (W1, W3-W5 present)"""
        self._assert_parity('Alexandria', [90.0, None, 80.0, 75.0, 70.0], "Missing Week 2")

    def test_missing_week_1(self):
        """Case 3: Missing Week 1 score (W2-W5 present)"""
        self._assert_parity('Giza', [None, 60.0, 55.0, 50.0, 45.0], "Missing Week 1")

    def test_missing_week_4_and_5(self):
        """Case 4: Missing Week 4 and Week 5 (typical dropout telemetry)"""
        self._assert_parity('Mansoura', [70.0, 65.0, 60.0, None, None], "Missing Week 4 and 5")

    def test_city_not_in_city_medians(self):
        """Case 5: Unknown city falling back to global training medians"""
        self._assert_parity('MarsaMatruh', [75.0, None, 65.0, 60.0, 55.0], "Unknown City Fallback")

    def test_all_quizzes_missing(self):
        """Case 6: Extreme boundary condition where all 5 quizzes are missing"""
        self._assert_parity('Aswan', [None, None, None, None, None], "All Quizzes Missing")

if __name__ == '__main__':
    unittest.main()
