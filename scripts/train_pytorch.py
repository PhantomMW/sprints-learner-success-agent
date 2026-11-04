"""
==============================================================================
SPRINTS AI-POWERED LEARNER SUCCESS & SUPPORT AGENT
Module: scripts/train_pytorch.py
Description: End-to-end PyTorch modeling pipeline for Sprint 2 and Sprint 3.
             - 100% Data-Leakage Free Preprocessing (Train-only imputation)
             - Rigorous 3-way Split: Train (70%) / Validation (15%) / Test (15%)
             - Dual Experiments in Sprint 2:
               * Experiment A: Proactive Early Risk Classifier (Weeks 1-3 only)
               * Experiment B: Full Trajectory Risk Classifier (Weeks 1-5)
             - Sprint 3 Deep Sequence LSTM Model:
               * Trained strictly on non-null capstone submissions (no target imputation)
               * Forget gate & constant error carousel cell state dynamics
               * Evaluated on held-out test set (RMSE, R^2)
==============================================================================
"""

import os
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    r2_score
)
from sklearn.decomposition import PCA

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==============================================================================
# 1. LEAK-FREE DATA PREPROCESSING & 3-WAY SPLIT
# ==============================================================================

def load_and_split_data(csv_path: str, random_state: int = 42):
    """
    Loads raw learner dataset and performs a strict 3-way split:
    Train (70%) / Validation (15%) / Test (15%) stratified by dropped_out.
    Imputation statistics are computed EXCLUSIVELY on the training set
    to prevent target and temporal data leakage.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at: {csv_path}")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded raw dataset from {csv_path}: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    quiz_cols = ['week_1_quiz', 'week_2_quiz', 'week_3_quiz', 'week_4_quiz', 'week_5_quiz']
    for col in quiz_cols + ['project_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df['dropped_out'] = df['dropped_out'].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}).fillna(0).astype(int)
    
    # Missing indicator features (known at runtime)
    df['missing_quiz_count'] = df[quiz_cols].isna().sum(axis=1)
    df['early_missing_count'] = df[['week_1_quiz', 'week_2_quiz', 'week_3_quiz']].isna().sum(axis=1)
    
    # 3-Way Stratified Split: Train 70%, Val 15%, Test 15%
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=random_state, stratify=df['dropped_out'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=random_state, stratify=temp_df['dropped_out'])
    
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    print(f"Dataset Split Sizes -> Train: {len(train_df)} (70%), Val: {len(val_df)} (15%), Test: {len(test_df)} (15%)")
    
    # Impute quiz scores using ONLY training set city medians (no target used!)
    city_medians_train = train_df.groupby('cohort_city')[quiz_cols].median()
    global_medians_train = train_df[quiz_cols].median()
    
    for split_df in [train_df, val_df, test_df]:
        for col in quiz_cols:
            city_col_map = city_medians_train[col].to_dict()
            imputed_city = split_df['cohort_city'].map(city_col_map).fillna(global_medians_train[col])
            split_df[col] = split_df[col].fillna(imputed_city)
            
        # Forward fill along temporal axis to carry forward demonstrated competence (strictly causal, no bfill future leakage)
        split_df[quiz_cols] = split_df[quiz_cols].ffill(axis=1).fillna(50.0)
        
        # Feature Engineering:
        # Full trajectory features
        split_df['score_delta_w1_w5'] = split_df['week_1_quiz'] - split_df['week_5_quiz']
        split_df['quiz_mean'] = split_df[quiz_cols].mean(axis=1)
        split_df['quiz_std'] = split_df[quiz_cols].std(axis=1).fillna(0.0)
        split_df['quiz_min'] = split_df[quiz_cols].min(axis=1)
        split_df['recent_performance_ratio'] = split_df['week_5_quiz'] / (split_df['week_1_quiz'] + 1e-5)
        
        # Early risk features (Weeks 1-3 only)
        w13_cols = ['week_1_quiz', 'week_2_quiz', 'week_3_quiz']
        split_df['score_delta_w1_w3'] = split_df['week_1_quiz'] - split_df['week_3_quiz']
        split_df['early_mean'] = split_df[w13_cols].mean(axis=1)
        split_df['early_min'] = split_df[w13_cols].min(axis=1)
        split_df['early_performance_ratio'] = split_df['week_3_quiz'] / (split_df['week_1_quiz'] + 1e-5)
        
    city_medians_dict = {
        city: {col: float(val) for col, val in row.items()}
        for city, row in city_medians_train.iterrows()
    }
    global_medians_dict = {col: float(val) for col, val in global_medians_train.items()}

    print("Leak-free preprocessing complete. Training statistics strictly isolated.")
    return train_df, val_df, test_df, city_medians_dict, global_medians_dict

def extract_single_learner_features(
    raw_quizzes: list,
    cohort_city: str = None,
    city_medians: dict = None,
    global_medians: dict = None,
    mode: str = "full"
) -> np.ndarray:
    """
    Unified feature extraction for inference (FastAPI & n8n) guaranteeing 100% mathematical
    and methodological parity with training:
    - Same training-set city/global median imputation
    - Causal forward fill (no backward fill)
    - Sample standard deviation (ddof=1) matching pandas
    - Matching feature engineering order and column keys
    """
    quiz_keys = ['week_1_quiz', 'week_2_quiz', 'week_3_quiz', 'week_4_quiz', 'week_5_quiz']
    city_medians = city_medians or {}
    global_medians = global_medians or {k: 75.0 for k in quiz_keys}
    
    if mode == "early":
        active_keys = quiz_keys[:3]
        quizzes_subset = [raw_quizzes[i] if i < len(raw_quizzes) else None for i in range(3)]
        early_missing_count = sum(1 for q in quizzes_subset if q is None or (isinstance(q, (int, float)) and np.isnan(q)))
        
        # 1. Imputation
        clean_quizzes = []
        for i in range(3):
            k = active_keys[i]
            val = quizzes_subset[i]
            if val is None or (isinstance(val, (int, float)) and np.isnan(val)):
                imputed = city_medians.get(cohort_city, {}).get(k, global_medians.get(k, 50.0))
                clean_quizzes.append(float(imputed))
            else:
                clean_quizzes.append(float(val))
                
        # 2. Sequential causal forward fill
        for i in range(1, len(clean_quizzes)):
            if clean_quizzes[i] is None or np.isnan(clean_quizzes[i]):
                clean_quizzes[i] = clean_quizzes[i-1]
        clean_quizzes = [50.0 if (q is None or np.isnan(q)) else q for q in clean_quizzes]
        
        delta_w1_w3 = clean_quizzes[0] - clean_quizzes[2]
        early_mean = float(np.mean(clean_quizzes))
        early_min = float(np.min(clean_quizzes))
        
        return np.array([[
            clean_quizzes[0],
            clean_quizzes[1],
            clean_quizzes[2],
            delta_w1_w3,
            early_mean,
            early_min,
            early_missing_count
        ]], dtype=np.float32)
        
    else:
        quizzes_5 = [raw_quizzes[i] if i < len(raw_quizzes) else None for i in range(5)]
        missing_quiz_count = sum(1 for q in quizzes_5 if q is None or (isinstance(q, (int, float)) and np.isnan(q)))
        
        # 1. Imputation
        clean_quizzes = []
        for i in range(5):
            k = quiz_keys[i]
            val = quizzes_5[i]
            if val is None or (isinstance(val, (int, float)) and np.isnan(val)):
                imputed = city_medians.get(cohort_city, {}).get(k, global_medians.get(k, 50.0))
                clean_quizzes.append(float(imputed))
            else:
                clean_quizzes.append(float(val))
                
        # 2. Sequential causal forward fill
        for i in range(1, len(clean_quizzes)):
            if clean_quizzes[i] is None or np.isnan(clean_quizzes[i]):
                clean_quizzes[i] = clean_quizzes[i-1]
        clean_quizzes = [50.0 if (q is None or np.isnan(q)) else q for q in clean_quizzes]
        
        score_delta_w1_w5 = clean_quizzes[0] - clean_quizzes[4]
        quiz_mean = float(np.mean(clean_quizzes))
        # Use ddof=1 to match pandas sample standard deviation exactly
        quiz_std = float(np.std(clean_quizzes, ddof=1)) if len(clean_quizzes) > 1 else 0.0
        quiz_min = float(np.min(clean_quizzes))
        recent_performance_ratio = float(clean_quizzes[4] / (clean_quizzes[0] + 1e-5))
        
        return np.array([[
            clean_quizzes[0],
            clean_quizzes[1],
            clean_quizzes[2],
            clean_quizzes[3],
            clean_quizzes[4],
            score_delta_w1_w5,
            quiz_mean,
            quiz_std,
            quiz_min,
            recent_performance_ratio,
            missing_quiz_count
        ]], dtype=np.float32)

# ==============================================================================
# 2. PYTORCH DATASETS
# ==============================================================================

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1) # (N, 5, 1)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# ==============================================================================
# 3. PYTORCH MODEL ARCHITECTURES
# ==============================================================================

class LearnerRiskClassifier(nn.Module):
    """
    Sprint 2: Supervised Feed-Forward Multi-Layer Perceptron (MLP).
    Uses BatchNorm to stabilize learning and Dropout(0.25) to prevent co-adaptation.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.25):
        super(LearnerRiskClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
        
    def extract_latent_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts penultimate 16-dimensional latent representations for clustering."""
        with torch.no_grad():
            h1 = torch.relu(self.net[1](self.net[0](x)))
            h2 = torch.relu(self.net[5](self.net[4](h1)))
            latent = self.net[8](h2)
            return latent

class LearnerSequenceLSTM(nn.Module):
    """
    Sprint 3: Deep Sequence LSTM Model.
    Processes weekly quiz trajectories to forecast capstone outcomes.
    The forget gate modulates memory retention while the additive cell state
    enables constant error propagation without vanishing gradients.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2, dropout_rate: float = 0.2):
        super(LearnerSequenceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x: torch.Tensor):
        lstm_out, (hn, cn) = self.lstm(x)
        # Use top-layer final hidden state h_5
        last_hidden = hn[-1]
        out = self.fc(last_hidden)
        return out, (hn, cn)

# ==============================================================================
# 4. SPRINT 2 TRAINING: EXPERIMENT A (EARLY) & EXPERIMENT B (FULL)
# ==============================================================================

def train_risk_model(train_df, val_df, test_df, feature_cols, exp_name, epochs=60, batch_size=32):
    """
    Trains a supervised risk model with early stopping on VALIDATION set,
    followed by a single uncorrupted evaluation on the TEST set.
    """
    print(f"\n--- Training {exp_name} ({len(feature_cols)} Features) ---")
    print(f"Features: {feature_cols}")
    
    assert len(train_df) > 0, "Train dataset must not be empty"
    assert len(val_df) > 0, "Validation dataset must not be empty"
    assert len(test_df) > 0, "Test dataset must not be empty"

    X_train = train_df[feature_cols].values
    y_train = train_df['dropped_out'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['dropped_out'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['dropped_out'].values
    
    assert len(y_val) > 0, "Validation targets must not be empty"
    assert len(y_test) > 0, "Test targets must not be empty"
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / (pos_count + 1e-5)
    
    train_loader = DataLoader(TabularDataset(X_train_scaled, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val_scaled, y_val), batch_size=batch_size, shuffle=False)
    
    model = LearnerRiskClassifier(input_dim=len(feature_cols), hidden_dim=64, dropout_rate=0.25)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight * 1.1]))
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    
    best_val_f1 = 0.0
    best_state = None
    best_threshold = 0.45
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
        # VALIDATION EVALUATION (Model selection on Validation ONLY)
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                probs = torch.sigmoid(model(batch_X)).squeeze().numpy()
                val_probs.extend(probs)
                val_targets.extend(batch_y.squeeze().numpy())
                
        val_probs = np.array(val_probs)
        val_targets = np.array(val_targets)
        
        # Calibrate threshold on VALIDATION set
        best_t, best_t_f1 = 0.5, 0.0
        for t in np.arange(0.35, 0.60, 0.05):
            preds_t = (val_probs >= t).astype(int)
            f1_t = f1_score(val_targets, preds_t, pos_label=1, zero_division=0)
            if f1_t > best_t_f1:
                best_t_f1 = f1_t
                best_t = t
                
        scheduler.step(best_t_f1)
        
        if best_t_f1 > best_val_f1:
            best_val_f1 = best_t_f1
            best_threshold = best_t
            best_state = copy.deepcopy(model.state_dict())
            
    # Load best validation model
    model.load_state_dict(best_state)
    model.eval()
    
    # FINAL INDEPENDENT TEST EVALUATION
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test_scaled, dtype=torch.float32))
        test_probs = torch.sigmoid(test_logits).squeeze().numpy()
        
    test_preds = (test_probs >= best_threshold).astype(int)
    test_f1 = f1_score(y_test, test_preds, pos_label=1)
    test_rec = recall_score(y_test, test_preds, pos_label=1)
    test_prec = precision_score(y_test, test_preds, pos_label=1)
    test_auc = roc_auc_score(y_test, test_probs)
    
    print(f"Results for {exp_name} on HELD-OUT TEST SET (N={len(y_test)}):")
    print(f"  - Validation Best F1: {best_val_f1:.4f} (Calibrated Threshold: {best_threshold:.2f})")
    print(f"  - Test F1 Score:       {test_f1:.4f} (Target >= 0.82: {'PASSED [OK]' if test_f1 >= 0.82 else 'FAILED'})")
    print(f"  - Test Recall:         {test_rec:.4f}")
    print(f"  - Test Precision:      {test_prec:.4f}")
    print(f"  - Test ROC-AUC:        {test_auc:.4f}")
    
    return {
        'model': model,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'threshold': best_threshold,
        'val_f1': best_val_f1,
        'test_f1': test_f1,
        'test_rec': test_rec,
        'test_prec': test_prec,
        'test_auc': test_auc,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'test_preds': test_preds,
        'test_probs': test_probs
    }

# ==============================================================================
# 5. SPRINT 3 TRAINING: LSTM SEQUENCE REGRESSION (NO TARGET IMPUTATION)
# ==============================================================================

def train_sequence_lstm(train_df, val_df, test_df, epochs=50, batch_size=32):
    """
    Trains PyTorch LSTM strictly on learners who submitted their capstone project.
    Eliminates target imputation leakage.
    Evaluates on independent validation set, tests on held-out test set.
    """
    print("\n" + "="*60)
    print("SPRINT 3: DEEP SEQUENCE LSTM MODEL TRAINING")
    print("="*60)
    
    seq_cols = ['week_1_quiz', 'week_2_quiz', 'week_3_quiz', 'week_4_quiz', 'week_5_quiz']
    
    # Filter strictly to actual submissions
    tr_clean = train_df[train_df['project_score'].notna()].copy()
    va_clean = val_df[val_df['project_score'].notna()].copy()
    te_clean = test_df[test_df['project_score'].notna()].copy()
    
    print(f"Learners with confirmed capstone project submissions:")
    print(f"  Train: {len(tr_clean)}, Val: {len(va_clean)}, Test: {len(te_clean)}")
    assert len(tr_clean) > 0, "Train clean subset must not be empty"
    assert len(va_clean) > 0, "Validation clean subset must not be empty"
    assert len(te_clean) > 0, "Test clean subset must not be empty"
    
    # Scale sequences and targets to [0, 1]
    X_train = tr_clean[seq_cols].values / 100.0
    y_train = tr_clean['project_score'].values / 100.0
    
    X_val = va_clean[seq_cols].values / 100.0
    y_val = va_clean['project_score'].values / 100.0
    
    X_test = te_clean[seq_cols].values / 100.0
    y_test = te_clean['project_score'].values / 100.0
    
    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    model = LearnerSequenceLSTM(input_size=1, hidden_size=32, num_layers=2, dropout_rate=0.2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    best_lstm_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch_seqs, batch_targets in train_loader:
            optimizer.zero_grad()
            preds, _ = model(batch_seqs)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_seqs, batch_targets in val_loader:
                preds, _ = model(batch_seqs)
                val_loss += criterion(preds, batch_targets).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_lstm_state = copy.deepcopy(model.state_dict())
            
    # Load best validation model
    model.load_state_dict(best_lstm_state)
    model.eval()
    
    # FINAL TEST EVALUATION
    with torch.no_grad():
        test_seq_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
        test_preds_norm, _ = model(test_seq_tensor)
        preds_100 = test_preds_norm.squeeze().numpy() * 100.0
        targets_100 = y_test * 100.0
        
    rmse = np.sqrt(mean_squared_error(targets_100, preds_100))
    r2 = r2_score(targets_100, preds_100)
    mae = np.mean(np.abs(targets_100 - preds_100))
    
    print("\n" + "-"*50)
    print(f"SEQUENCE LSTM RESULTS ON HELD-OUT TEST SET (N={len(y_test)}):")
    print(f"  - Test RMSE: {rmse:.2f} points (on 0-100 scale)")
    print(f"  - Test MAE:  {mae:.2f} points")
    print(f"  - Test R^2:  {r2:.4f}")
    print("-"*50)
    
    return {
        'model': model,
        'seq_cols': seq_cols,
        'val_mse': best_val_loss,
        'test_rmse': rmse,
        'test_r2': r2,
        'test_mae': mae,
        'X_test': X_test,
        'y_test_100': targets_100,
        'preds_100': preds_100
    }

# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Sprints Leak-Free PyTorch Models")
    default_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--base_dir', type=str, default=default_base, help="Project root directory")
    parser.add_argument('--epochs', type=int, default=60, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="DataLoader batch size")
    args = parser.parse_args()
    
    set_seed(42)
    csv_path = os.path.join(args.base_dir, 'data', 'sprints_learner_data.csv')
    models_dir = os.path.join(args.base_dir, 'models')
    docs_dir = os.path.join(args.base_dir, 'docs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    # 1. Split & Preprocess
    train_df, val_df, test_df, city_medians_dict, global_medians_dict = load_and_split_data(csv_path)
    
    # 2. Experiment A: Proactive Early Risk Classifier (Weeks 1 to 3)
    early_features = ['week_1_quiz', 'week_2_quiz', 'week_3_quiz', 'score_delta_w1_w3', 'early_mean', 'early_min', 'early_missing_count']
    exp_a = train_risk_model(train_df, val_df, test_df, early_features, "Experiment A: Proactive Early Risk (Weeks 1-3)", epochs=args.epochs, batch_size=args.batch_size)
    
    # 3. Experiment B: Full Trajectory Risk Classifier (Weeks 1 to 5)
    full_features = ['week_1_quiz', 'week_2_quiz', 'week_3_quiz', 'week_4_quiz', 'week_5_quiz', 'score_delta_w1_w5', 'quiz_mean', 'quiz_std', 'quiz_min', 'recent_performance_ratio', 'missing_quiz_count']
    exp_b = train_risk_model(train_df, val_df, test_df, full_features, "Experiment B: Full Trajectory Risk (Weeks 1-5)", epochs=args.epochs, batch_size=args.batch_size)
    
    # 4. Save Sprint 2 Checkpoint (Holding both models)
    risk_checkpoint_path = os.path.join(models_dir, 'risk_classifier.pth')
    torch.save({
        'model_state_dict': exp_b['model'].state_dict(),
        'feature_cols': full_features,
        'scaler_mean': exp_b['scaler'].mean_,
        'scaler_scale': exp_b['scaler'].scale_,
        'decision_threshold': exp_b['threshold'],
        'test_f1': exp_b['test_f1'],
        'test_recall': exp_b['test_rec'],
        'city_medians': city_medians_dict,
        'global_medians': global_medians_dict,
        'early_model_state_dict': exp_a['model'].state_dict(),
        'early_feature_cols': early_features,
        'early_scaler_mean': exp_a['scaler'].mean_,
        'early_scaler_scale': exp_a['scaler'].scale_,
        'early_threshold': exp_a['threshold'],
        'early_test_f1': exp_a['test_f1'],
        'early_test_recall': exp_a['test_rec']
    }, risk_checkpoint_path)
    print(f"\nSaved dual risk classifier checkpoint: {risk_checkpoint_path}")
    
    # 5. Sprint 3: LSTM Sequence Predictor
    lstm_results = train_sequence_lstm(train_df, val_df, test_df, epochs=50, batch_size=args.batch_size)
    lstm_checkpoint_path = os.path.join(models_dir, 'sequence_predictor.pth')
    torch.save({
        'model_state_dict': lstm_results['model'].state_dict(),
        'seq_cols': lstm_results['seq_cols'],
        'hidden_size': 32,
        'num_layers': 2,
        'test_rmse': lstm_results['test_rmse'],
        'test_r2': lstm_results['test_r2']
    }, lstm_checkpoint_path)
    print(f"Saved sequence LSTM checkpoint: {lstm_checkpoint_path}")
    
    # 6. Generate Updated Visualizations
    # A. Confusion Matrix for Full Model
    cm = confusion_matrix(exp_b['y_test'], exp_b['test_preds'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
    plt.title(f"PyTorch Risk Classifier (Test Set N={len(exp_b['y_test'])})\nTest F1={exp_b['test_f1']:.4f} | Recall={exp_b['test_rec']:.4f}", fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    cm_path = os.path.join(docs_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # B. Latent Space PCA Projection
    with torch.no_grad():
        latent_reps = exp_b['model'].extract_latent_features(torch.tensor(exp_b['X_test_scaled'], dtype=torch.float32)).numpy()
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_reps)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=exp_b['y_test'], cmap='coolwarm', alpha=0.8, edgecolors='k', s=45)
    plt.colorbar(scatter, label='Ground Truth Dropout Status (0: Retained, 1: High Risk)')
    plt.title('PyTorch Latent Space Distribution (Test Set PCA)\nSeparation of Behavioral Risk Clusters', fontsize=12, fontweight='bold')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    latent_path = os.path.join(docs_dir, 'latent_space_distribution.png')
    plt.savefig(latent_path, dpi=300)
    plt.close()
    
    # C. LSTM Trajectory Forecast Plot
    plt.figure(figsize=(9, 5))
    weeks = [1, 2, 3, 4, 5]
    sample_indices = [0, 4, 10, 18, 25]
    for idx in sample_indices:
        actual_trajectory = lstm_results['X_test'][idx] * 100.0
        pred_capstone = lstm_results['preds_100'][idx]
        actual_capstone = lstm_results['y_test_100'][idx]
        p = plt.plot(weeks, actual_trajectory, marker='o', label=f'Learner #{idx+1} Quizzes')
        color = p[0].get_color()
        plt.scatter(6, pred_capstone, color=color, marker='*', s=140, label=f'Pred: {pred_capstone:.1f}' if idx == 0 else "")
        plt.scatter(6, actual_capstone, color=color, marker='x', s=100, label=f'Actual: {actual_capstone:.1f}' if idx == 0 else "")
        
    plt.title(f"PyTorch LSTM 5-Week Sequence Modeling & Capstone Forecast\nTest RMSE: {lstm_results['test_rmse']:.2f} pts | R² = {lstm_results['test_r2']:.4f}", fontsize=12, fontweight='bold')
    plt.xlabel('Bootcamp Timeline')
    plt.ylabel('Assessment Score')
    plt.xticks([1, 2, 3, 4, 5, 6], ['W1 Quiz', 'W2 Quiz', 'W3 Quiz', 'W4 Quiz', 'W5 Quiz', 'Capstone'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    lstm_plot_path = os.path.join(docs_dir, 'lstm_sequence_predictions.png')
    plt.savefig(lstm_plot_path, dpi=300)
    plt.close()
    
    print(f"\nArtifacts generated & saved to {docs_dir}")
    print("="*60)
    print("LEAK-FREE PYTORCH MODELING COMPLETED WITH FULL VALIDATION!")
    print("="*60)

if __name__ == '__main__':
    main()
