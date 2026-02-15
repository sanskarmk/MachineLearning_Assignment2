"""
Offline Training Script for ML Assignment 2
=============================================
Phishing Website Detection - Binary Classification

This script:
  1. Loads training_data.csv
  2. Splits into 80% train / 20% test (stratified)
  3. Saves test_data.csv (for Streamlit upload)
  4. Trains all 6 classification models on the training split
  5. Evaluates each model on the test split
  6. Saves each trained model as model/<name>.pkl
  7. Saves a StandardScaler as model/scaler.pkl (for models that need scaling)
  8. Saves model_comparison_results.csv with all evaluation metrics

Usage:
    python train_all_models_offline.py --data training_data.csv
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def load_and_prepare(csv_path: str):
    """Load CSV, drop URL column, encode target, return X DataFrame and y array."""
    df = pd.read_csv(csv_path)
    if "url" in df.columns:
        df = df.drop(columns=["url"])
    X = df.drop(columns=["status"])
    y = df["status"]
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        le.fit(["legitimate", "phishing"])  # 0=legitimate, 1=phishing
        y = le.transform(y)
    return df, X, y


def compute_metrics(y_true, y_pred, y_prob):
    """Compute the 6 required evaluation metrics."""
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Train all 6 models offline")
    parser.add_argument("--data", default="training_data.csv", help="Path to full dataset CSV")
    parser.add_argument("--outdir", default="models", help="Directory to write *.pkl files")
    parser.add_argument("--test-csv", default="test_data.csv", help="Output path for test split CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    # ── 1. Load data ──
    full_df, X, y = load_and_prepare(args.data)
    print(f"Loaded {len(full_df)} rows, {X.shape[1]} features from {args.data}")

    # ── 2. Stratified train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # ── 3. Save test_data.csv (with original status labels for upload) ──
    # Reconstruct a DataFrame with the original 'status' column as text
    test_df = X_test.copy()
    test_df["status"] = np.where(y_test == 1, "phishing", "legitimate")
    test_df.to_csv(args.test_csv, index=False)
    print(f"Saved test split → {args.test_csv} ({len(test_df)} rows)")

    # ── 4. Fit a StandardScaler on training data (for models that need it) ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 5. Define all 6 models ──
    models = {
        "logistic_regression": {
            "clf": LogisticRegression(max_iter=2000, random_state=args.seed),
            "needs_scaling": True,
        },
        "decision_tree": {
            "clf": DecisionTreeClassifier(random_state=args.seed),
            "needs_scaling": False,
        },
        "knn": {
            "clf": KNeighborsClassifier(n_neighbors=5),
            "needs_scaling": True,
        },
        "naive_bayes": {
            "clf": GaussianNB(),
            "needs_scaling": False,
        },
        "random_forest": {
            "clf": RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1),
            "needs_scaling": False,
        },
        "xgboost": {
            "clf": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=args.seed,
                n_jobs=-1,
            ),
            "needs_scaling": False,
        },
    }

    display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "KNN",
        "naive_bayes": "Naive Bayes",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }

    # ── 6. Train, evaluate, and save each model ──
    os.makedirs(args.outdir, exist_ok=True)
    comparison_rows = []

    for name, spec in models.items():
        clf = spec["clf"]
        use_scaled = spec["needs_scaling"]

        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test

        print(f"\nTraining {display_names[name]}...")
        clf.fit(Xtr, y_train)

        y_pred = clf.predict(Xte)
        y_prob = clf.predict_proba(Xte)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = display_names[name]
        comparison_rows.append(metrics)

        # Save model
        pkl_path = os.path.join(args.outdir, f"{name}.pkl")
        joblib.dump(clf, pkl_path)
        print(f"  → Saved {pkl_path}")
        print(f"  Accuracy={metrics['Accuracy']}  AUC={metrics['AUC']}  "
              f"Precision={metrics['Precision']}  Recall={metrics['Recall']}  "
              f"F1={metrics['F1']}  MCC={metrics['MCC']}")

    # Save scaler
    scaler_path = os.path.join(args.outdir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\nSaved scaler → {scaler_path}")

    # ── 7. Save comparison CSV ──
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    comparison_df.to_csv("model_comparison_results.csv", index=False)
    print(f"\nSaved model_comparison_results.csv")
    print("\n" + "=" * 70)
    print(comparison_df.to_string(index=False))
    print("=" * 70)
    print("\nDone! All models trained and saved.")


if __name__ == "__main__":
    main()
