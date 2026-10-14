"""Train, evaluate, and save the synthetic-history RPS ANN."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from feature_engineering import MOVES, extract_features, feature_names, rotate_history

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "rps_synthetic.csv"
ARTIFACT_DIR = ROOT / "artifacts"


def train(data_path: Path = DEFAULT_DATA, seed: int = 42, epochs: int = 350) -> dict:
    df = pd.read_csv(data_path)
    names = feature_names()
    if df.empty:
        raise ValueError("Synthetic dataset is empty. Run generate_data.py first.")

    # Hold out complete synthetic sessions to reduce train/test leakage between
    # adjacent prefixes from the same generated sequence.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["session_id"]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Augment training episodes with all three cyclic relabellings of R/P/S.
    # These preserve the game's cyclic payoff structure and are not applied to test data.
    augmented_histories: list[list[int]] = []
    augmented_y: list[int] = []
    for row in train_df.itertuples(index=False):
        history = [int(x) for x in row.history.split()] if isinstance(row.history, str) and row.history.strip() else []
        target = int(row.target)
        for shift in range(3):
            augmented_histories.append(rotate_history(history, shift))
            augmented_y.append((target + shift) % 3)

    X_train = np.asarray([extract_features(h) for h in augmented_histories], dtype=np.float64)
    y_train = np.asarray(augmented_y, dtype=np.int64)
    X_test = test_df[names].to_numpy(dtype=np.float64)
    y_test = test_df["target"].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = MLPClassifier(
        hidden_layer_sizes=(48, 24),
        activation="relu",
        solver="adam",
        learning_rate_init=0.002,
        max_iter=epochs,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=18,
        random_state=seed,
    )
    model.fit(X_train_scaled, y_train)
    probabilities = model.predict_proba(X_test_scaled)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]

    majority = int(pd.Series(y_train).value_counts(normalize=True).max())
    baseline = float(pd.Series(y_train).value_counts(normalize=True).max())
    metrics = {
        "dataset_samples": int(len(df)),
        "synthetic_sessions": int(df["session_id"].nunique()),
        "train_samples_before_augmentation": int(len(train_df)),
        "train_samples_after_augmentation": int(len(X_train)),
        "test_samples": int(len(test_df)),
        "input_features": int(len(names)),
        "architecture": [len(names), 48, 24, 3],
        "activation": "ReLU hidden layers; softmax output probabilities",
        "epochs_requested": int(epochs),
        "iterations_run": int(model.n_iter_),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_log_loss": float(log_loss(y_test, probabilities, labels=model.classes_)),
        "majority_class_baseline": baseline,
        "labels": list(MOVES),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1, 2], target_names=list(MOVES), output_dict=True, zero_division=0
        ),
        "note": "Evaluation is on held-out synthetic sessions, not on real human users. Random behavior is inherently unpredictable.",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    dump({"scaler": scaler, "model": model, "feature_names": names, "moves": list(MOVES)}, ARTIFACT_DIR / "rps_ann.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ["dataset_samples", "train_samples_after_augmentation", "test_samples", "iterations_run", "test_accuracy", "test_log_loss", "majority_class_baseline"]}, indent=2))
    print(f"Saved model: {ARTIFACT_DIR / 'rps_ann.joblib'}")
    print(f"Saved metrics: {ARTIFACT_DIR / 'metrics.json'}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=350)
    args = parser.parse_args()
    train(args.data, args.seed, args.epochs)
