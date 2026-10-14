"""Flask API + animated frontend for the Rock-Paper-Scissors ANN tutorial."""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
from flask import Flask, jsonify, render_template, request
from joblib import load

from feature_engineering import MOVES, extract_features, feature_names, normalize_history

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "rps_ann.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
app = Flask(__name__)

BUNDLE = None
LOAD_ERROR = None
try:
    if MODEL_PATH.exists():
        BUNDLE = load(MODEL_PATH)
    else:
        LOAD_ERROR = "Trained model not found. Run: python generate_data.py && python train_model.py"
except Exception as exc:  # Explain setup issues through the health endpoint.
    LOAD_ERROR = f"Could not load model: {exc}"


def encode_history(values) -> list[int]:
    if not isinstance(values, list):
        raise ValueError("history must be a list of move names or integer move IDs")
    if len(values) > 1000:
        raise ValueError("history is too long (maximum 1,000 moves)")
    encoded = []
    for value in values:
        if isinstance(value, str):
            key = value.strip().lower()
            if key not in MOVES:
                raise ValueError(f"Invalid move '{value}'. Choose rock, paper, or scissors.")
            encoded.append(MOVES.index(key))
        else:
            encoded.append(int(value))
    return normalize_history(encoded)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def payoff(ai_move: int, player_move: int) -> int:
    if ai_move == player_move:
        return 0
    return 1 if (ai_move - player_move) % 3 == 1 else -1


def make_analysis(history: list[int]) -> dict:
    if BUNDLE is None:
        raise RuntimeError(LOAD_ERROR or "Model not loaded")
    scaler = BUNDLE["scaler"]
    model = BUNDLE["model"]
    names = BUNDLE.get("feature_names", feature_names())

    raw_features = np.asarray(extract_features(history), dtype=np.float64)
    scaled_features = scaler.transform(raw_features.reshape(1, -1))[0]

    # Explicitly run a forward pass so the UI can display actual neuron activations.
    # This mirrors the fitted sklearn MLP: affine transform -> ReLU hidden layers -> softmax.
    current = scaled_features
    layer_trace = [{
        "name": "Input",
        "kind": "input",
        "size": int(len(current)),
        "values": current.astype(float).tolist(),
    }]
    for idx, (weights, bias) in enumerate(zip(model.coefs_, model.intercepts_)):
        z = current @ weights + bias
        is_output = idx == len(model.coefs_) - 1
        current = softmax(z) if is_output else np.maximum(z, 0.0)
        layer_trace.append({
            "name": "Output / Softmax" if is_output else f"Hidden {idx + 1} / ReLU",
            "kind": "output" if is_output else "hidden",
            "size": int(len(current)),
            "values": current.astype(float).tolist(),
            "pre_activation": z.astype(float).tolist(),
        })

    # Consistency check against sklearn's public prediction path.
    probabilities_from_model = model.predict_proba(scaled_features.reshape(1, -1))[0]
    probabilities = {MOVES[int(class_id)]: float(probability)
                     for class_id, probability in zip(model.classes_, probabilities_from_model)}
    predicted_move = int(model.classes_[int(np.argmax(probabilities_from_model))])

    utilities = []
    for ai_move in range(3):
        expected = sum(
            probabilities[MOVES[player_move]] * payoff(ai_move, player_move)
            for player_move in range(3)
        )
        utilities.append({"move": MOVES[ai_move], "expected_payoff": float(expected)})
    best_value = max(item["expected_payoff"] for item in utilities)
    best_moves = [item["move"] for item in utilities if abs(item["expected_payoff"] - best_value) < 1e-9]
    selected_move = random.choice(best_moves)

    feature_detail = [
        {"name": name, "raw": float(raw), "scaled": float(scaled)}
        for name, raw, scaled in zip(names, raw_features, scaled_features)
    ]
    clipped_history = history[-20:]
    feature_history = history[-50:]
    frequency_counts = {move: feature_history.count(idx) for idx, move in enumerate(MOVES)}
    transition_counts = [[0 for _ in range(3)] for _ in range(3)]
    for before, after in zip(feature_history[:-1], feature_history[1:]):
        transition_counts[before][after] += 1

    try:
        model_prediction = model.predict_proba(scaled_features.reshape(1, -1))[0]
        forward_error = float(np.max(np.abs(model_prediction - current)))
    except Exception:
        forward_error = None

    return {
        "history": [MOVES[x] for x in history],
        "recent_history": [MOVES[x] for x in clipped_history],
        "history_length": len(history),
        "feature_window_length": len(feature_history),
        "feature_names": names,
        "features": feature_detail,
        "layers": layer_trace,
        "probabilities": probabilities,
        "predicted_player_move": MOVES[predicted_move],
        "utilities": utilities,
        "selected_ai_move": selected_move,
        "decision_reason": (
            f"{selected_move.title()} has the highest expected payoff ({best_value:+.3f}) "
            f"against the predicted move distribution."
        ),
        "frequency_counts": frequency_counts,
        "transition_counts": transition_counts,
        "forward_pass_max_abs_error": forward_error,
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": BUNDLE is not None, "error": LOAD_ERROR})


@app.get("/api/metrics")
def metrics():
    if METRICS_PATH.exists():
        return jsonify(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
    return jsonify({"error": "Metrics file not found. Train the model first."}), 404


@app.post("/api/analyze")
def analyze():
    try:
        payload = request.get_json(silent=True) or {}
        history = encode_history(payload.get("history", []))
        return jsonify(make_analysis(history))
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


if __name__ == "__main__":
    if BUNDLE is None:
        print(LOAD_ERROR)
    app.run(host="127.0.0.1", port=5000, debug=False)
