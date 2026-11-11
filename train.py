"""
train.py
~~~~~~~~
Standalone training script for the Telco Customer Churn MLP.

Usage:
    python train.py

Outputs to /artifacts:
  - churn_mlp.keras           trained Keras model
  - scaler.pkl                 StandardScaler fitted on train split only
  - feature_columns.json       ordered feature list post one-hot encoding
  - metrics.json               test accuracy, precision, recall, F1, AUC, thresholds
  - history.csv                per-epoch train/val loss and accuracy
  - test_predictions.csv       y_true, y_prob, y_pred_default, y_pred_optimal
  - baseline_metrics.json      Logistic Regression metrics for benchmark
  - feature_importance.csv     permutation feature importance (ROC-AUC drop)
"""

import os
import json
import random
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "42"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from preprocessing import clean_dataframe, FEATURE_COLUMNS

# ---------------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"
DATA_PATH     = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
# ---------------------------------------------------------------------------


def load_and_prepare(data_path: str):
    print(f"[train.py] Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    # Drop ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Binary maps
    binary_map = {
        "gender":          {"Female": 0, "Male": 1},
        "Partner":         {"No": 0, "Yes": 1},
        "Dependents":      {"No": 0, "Yes": 1},
        "PhoneService":    {"No": 0, "Yes": 1},
        "PaperlessBilling":{"No": 0, "Yes": 1},
    }
    for col, mapping in binary_map.items():
        df[col] = df[col].map(mapping)

    # One-hot encode multi-level categoricals
    ohe_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Align to canonical feature order
    X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return X, y


def build_model(input_dim: int, lr: float = 0.001, dropout: float = 0.3,
                units: list = None) -> keras.Model:
    if units is None:
        units = [64, 32]
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="input_features"),
        layers.Dense(units[0], activation="relu", name="dense_1",
                     kernel_initializer="he_normal"),
        layers.Dropout(dropout, name="dropout_1", seed=SEED),
        layers.Dense(units[1], activation="relu", name="dense_2",
                     kernel_initializer="he_normal"),
        layers.Dropout(dropout, name="dropout_2", seed=SEED),
        layers.Dense(1, activation="sigmoid", name="output",
                     kernel_initializer="glorot_uniform"),
    ], name="Telco_Churn_MLP")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc", curve="ROC"),
        ],
    )
    return model


def compute_permutation_importance(model, X_test, y_test,
                                   feature_names, baseline_auc,
                                   n_repeats=5, seed=SEED):
    rng = np.random.default_rng(seed)
    means, stds = [], []
    for fi in range(X_test.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, fi] = rng.permutation(X_perm[:, fi])
            probs = model.predict(X_perm, verbose=0).ravel()
            drops.append(baseline_auc - roc_auc_score(y_test, probs))
        means.append(float(np.mean(drops)))
        stds.append(float(np.std(drops)))
    df_imp = pd.DataFrame({
        "feature":    feature_names,
        "importance": means,
        "std_dev":    stds,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp


def find_best_thresholds(y_true, y_prob):
    """Find best F1 threshold and minimum-cost threshold."""
    thresholds = np.linspace(0.05, 0.95, 181)
    COST_FN, COST_FP, COST_TP = 1000.0, 50.0, 50.0

    best_f1, best_f1_thresh = 0, 0.5
    min_cost, min_cost_thresh = float("inf"), 0.5

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        cm = confusion_matrix(y_true, preds).ravel()
        if len(cm) == 4:
            tn, fp, fn, tp = cm
            cost = (fn * COST_FN + fp * COST_FP + tp * COST_TP) / len(y_true)
        else:
            cost = float("inf")

        if f1 > best_f1:
            best_f1, best_f1_thresh = f1, float(round(t, 2))
        if cost < min_cost:
            min_cost, min_cost_thresh = cost, float(round(t, 2))

    return best_f1_thresh, min_cost_thresh


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load & prepare data
    # ------------------------------------------------------------------
    X, y = load_and_prepare(DATA_PATH)
    print(f"[train.py] Dataset shape: {X.shape}")

    # Save canonical feature list
    with open(os.path.join(ARTIFACTS_DIR, "feature_columns.json"), "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print("[train.py] Saved feature_columns.json")

    # ------------------------------------------------------------------
    # 2. Stratified split 70 / 15 / 15
    # ------------------------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(f"[train.py] Split — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Scale (fit only on train)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    print("[train.py] Saved scaler.pkl")

    # ------------------------------------------------------------------
    # 4. Class weights
    # ------------------------------------------------------------------
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weights = {0: float(cw[0]), 1: float(cw[1])}
    print(f"[train.py] Class weights: {class_weights}")

    # ------------------------------------------------------------------
    # 5. Train MLP
    # ------------------------------------------------------------------
    print("[train.py] Training MLP (up to 100 epochs)…")
    model = build_model(X_train_s.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-5, verbose=1,
        ),
    ]

    hist = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=100, batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks, verbose=1,
    )

    # Save model
    model.save(os.path.join(ARTIFACTS_DIR, "churn_mlp.keras"))
    print("[train.py] Saved churn_mlp.keras")

    # Save training history
    hist_df = pd.DataFrame(hist.history)
    hist_df["epoch"] = range(1, len(hist_df) + 1)
    hist_df.to_csv(os.path.join(ARTIFACTS_DIR, "history.csv"), index=False)
    print("[train.py] Saved history.csv")

    # ------------------------------------------------------------------
    # 6. Evaluate MLP on test set
    # ------------------------------------------------------------------
    y_prob = model.predict(X_test_s, verbose=0).ravel()
    y_pred_default = (y_prob >= 0.5).astype(int)

    best_f1_thresh, best_cost_thresh = find_best_thresholds(y_test, y_prob)
    y_pred_optimal = (y_prob >= best_cost_thresh).astype(int)

    test_metrics = {
        "accuracy":           float(accuracy_score(y_test, y_pred_default)),
        "precision":          float(precision_score(y_test, y_pred_default)),
        "recall":             float(recall_score(y_test, y_pred_default)),
        "f1_score":           float(f1_score(y_test, y_pred_default)),
        "roc_auc":            float(roc_auc_score(y_test, y_prob)),
        "pr_auc":             float(average_precision_score(y_test, y_prob)),
        "default_threshold":  0.5,
        "best_cost_threshold":best_cost_thresh,
        "best_f1_threshold":  best_f1_thresh,
        "best_epoch":         int(np.argmin(hist.history["val_loss"]) + 1),
    }

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"[train.py] Saved metrics.json — ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # Save test predictions
    pd.DataFrame({
        "y_true":           y_test.values,
        "y_prob":           y_prob,
        "y_pred_default":   y_pred_default,
        "y_pred_optimal":   y_pred_optimal,
    }).to_csv(os.path.join(ARTIFACTS_DIR, "test_predictions.csv"), index=False)
    print("[train.py] Saved test_predictions.csv")

    # ------------------------------------------------------------------
    # 7. Baseline — Logistic Regression
    # ------------------------------------------------------------------
    print("[train.py] Training Logistic Regression baseline…")
    lr_model = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)
    lr_model.fit(X_train_s, y_train)
    y_lr_prob = lr_model.predict_proba(X_test_s)[:, 1]
    y_lr_pred = (y_lr_prob >= 0.5).astype(int)

    baseline_metrics = {
        "accuracy":  float(accuracy_score(y_test, y_lr_pred)),
        "precision": float(precision_score(y_test, y_lr_pred)),
        "recall":    float(recall_score(y_test, y_lr_pred)),
        "f1_score":  float(f1_score(y_test, y_lr_pred)),
        "roc_auc":   float(roc_auc_score(y_test, y_lr_prob)),
        "pr_auc":    float(average_precision_score(y_test, y_lr_prob)),
    }

    with open(os.path.join(ARTIFACTS_DIR, "baseline_metrics.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    print("[train.py] Saved baseline_metrics.json")

    # ------------------------------------------------------------------
    # 8. Permutation feature importance
    # ------------------------------------------------------------------
    print("[train.py] Computing permutation feature importance (5 shuffles)…")
    base_auc = test_metrics["roc_auc"]
    df_imp = compute_permutation_importance(
        model, X_test_s, y_test.values, FEATURE_COLUMNS, base_auc
    )
    df_imp.to_csv(os.path.join(ARTIFACTS_DIR, "feature_importance.csv"), index=False)
    print("[train.py] Saved feature_importance.csv")

    # ------------------------------------------------------------------
    # 9. Sample batch CSV for demo
    # ------------------------------------------------------------------
    os.makedirs("sample_data", exist_ok=True)
    sample_csv = "sample_data/sample_batch.csv"
    if not os.path.exists(sample_csv):
        df_raw = pd.read_csv(DATA_PATH)
        sample = df_raw.sample(20, random_state=SEED)
        sample.to_csv(sample_csv, index=False)
        print("[train.py] Saved sample_data/sample_batch.csv")
    else:
        print("[train.py] sample_data/sample_batch.csv already exists, skipping.")

    # ------------------------------------------------------------------
    print("\n[train.py] ✅ All artifacts saved successfully!")
    print(f"  - ROC-AUC:        {test_metrics['roc_auc']:.4f}")
    print(f"  - Recall:         {test_metrics['recall']:.4f}")
    print(f"  - Best epoch:     {test_metrics['best_epoch']}")
    print(f"  - Cost threshold: {test_metrics['best_cost_threshold']}")


if __name__ == "__main__":
    main()

