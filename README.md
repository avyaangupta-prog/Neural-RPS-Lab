# Neural RPS Lab — Rock, Paper, Scissors with an ANN

A beginner-friendly machine-learning project that trains an Artificial Neural Network (ANN) on **synthetic move histories** and lets you play Rock-Paper-Scissors in a browser. The UI animates the real inference trace from observed history and feature engineering through every network layer, softmax probabilities, and expected-payoff decision.

> **Important modeling choice:** the model predicts your *next* move from completed prior rounds. It does not use the move you are about to submit to predict that same move. That prevents data leakage. Because random moves are inherently unpredictable, the model is intended to learn recurring synthetic behaviors—not guarantee wins against people.

## What is included?

- `generate_data.py` — simulate 10 types of synthetic opponents and label histories with the next move.
- `feature_engineering.py` — build the 28-dimensional input vector used by training and live inference.
- `train_model.py` — group-aware train/test split, cyclic-label augmentation, scaling, ANN training, and evaluation.
- `app.py` — Flask routes plus an explicit forward pass that exposes the *actual trained model's* neuron activations.
- `templates/index.html`, `static/style.css`, `static/script.js` — responsive game UI and stage-by-stage inference animation.
- `data/rps_synthetic.csv` — generated labeled dataset (re-create it any time).
- `artifacts/rps_ann.joblib` — trained model and fitted `StandardScaler`.
- `artifacts/metrics.json` — evaluation metrics from the saved model.
- `ML_DEVELOPMENT_WALKTHROUGH.md` — detailed sequential learning guide.
- `test_pipeline.py` — sanity tests for feature extraction, model probabilities, and forward-pass parity.

## Run locally (macOS / Linux)

Run each command from the extracted `rock_paper_scissors_ann` folder.

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Generate synthetic data

```bash
python generate_data.py
```

Optional parameters:

```bash
python generate_data.py --sessions-per-strategy 250 --seed 123
```

### 4. Train and evaluate the ANN

```bash
python train_model.py
```

Optional shorter/faster experiment:

```bash
python train_model.py --epochs 100
```

The fitted model is saved in `artifacts/rps_ann.joblib` and metrics are saved in `artifacts/metrics.json`. Re-run training whenever you change feature extraction, the generated data, or the model architecture.

### 5. Run the web application

```bash
python app.py
```

Open **http://127.0.0.1:5000** in a browser. Choose Rock, Paper, or Scissors to trigger inference. Your current selection is locked while the model runs using only completed prior rounds; the AI action is revealed after the trace and the round is scored. Your move is then added to history for the next round.

### 6. Run sanity checks

```bash
python -m unittest test_pipeline.py
```

## Model at a glance

- **Input:** 28 features describing prior moves.
- **Hidden layers:** 48 and 24 neurons using ReLU.
- **Output:** 3 classes—rock, paper, scissors—with softmax probabilities.
- **Decision policy:** calculate expected payoff of each possible AI move against the predicted distribution and choose the maximum. A random tie-break is used when multiple actions have exactly equal expected payoff.
- **Training data:** simulated from random, biased, repeat-last, forward-cycle, backward-cycle, alternating-pair, Markov-transition, and mixed behavior policies.

The page's activation dots/bars are derived from the saved network weights and fitted scaler. The `forward pass check` compares the explicit layer-by-layer computation with scikit-learn's public `predict_proba` result.

## Current saved run

- Synthetic samples: **32,445** across **1,800** synthetic sessions.
- Training samples after cyclic-label augmentation: **78,213**.
- Held-out test samples: **6,374**, split by session so adjacent prefixes from one sequence do not leak across train/test.
- Test accuracy: **58.7%** on synthetic held-out sessions. This does not measure performance against arbitrary real humans or truly random choices.

See `artifacts/metrics.json` for the complete report. Metrics will change when you regenerate data, change the seed, alter features, or retrain the model.

## Troubleshooting

- **Model unavailable:** run `python generate_data.py`, then `python train_model.py`, then start `python app.py` again.
- **`ModuleNotFoundError`:** activate the virtual environment and install `requirements.txt`.
- **Frontend says it cannot contact Flask:** keep the terminal running and open `http://127.0.0.1:5000` rather than opening `index.html` directly.
- **You changed features:** regenerate the CSV and retrain. Training and inference must use the same feature extraction implementation.
