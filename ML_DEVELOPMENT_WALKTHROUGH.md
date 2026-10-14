# Machine Learning Development Walkthrough: Rock-Paper-Scissors ANN

This guide follows the development lifecycle from defining the prediction problem to generating data, training an ANN, validating the model, serving predictions, and watching the actual computation in a browser.

## 0. Define the problem correctly

Rock-Paper-Scissors has three classes:

| Encoded value | Move | Beats |
|---:|---|---|
| `0` | Rock | Scissors |
| `1` | Paper | Rock |
| `2` | Scissors | Paper |

A truly random player has no dependable next-move pattern. Therefore, a model cannot reliably predict every random move. For a meaningful supervised-learning exercise, this project generates synthetic players with a mixture of behavior policies: uniform random, move bias, repeating a move, cyclic patterns, alternating between two moves, and Markov-style transitions.

**Prediction target:** given a history of completed moves, predict the next move in that synthetic sequence. The history does not contain the target move. During gameplay, the AI's move for the current round is locked before you make your selection. After the round finishes, the new selection is added to history and the next inference predicts the *following* selection. This makes the demonstration free from current-round target leakage.

## 1. Set up the development environment

Use Python 3.11 or newer. A virtual environment keeps this project's dependencies separate from other Python projects.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main libraries are:

- **NumPy:** numerical arrays and matrix operations.
- **pandas:** loading and inspecting the generated tabular dataset.
- **scikit-learn:** feature scaling, MLP classifier, train/test split, and metrics.
- **joblib:** saving/loading the fitted scaler and trained ANN.
- **Flask:** serve the API and UI.

## 2. Generate a synthetic dataset

Run:

```bash
python generate_data.py
```

The script simulates 10 types of synthetic player behavior. Each episode has a hidden behavior policy and a generated move sequence. For every position, it creates a sample containing the prior prefix and the next move as the label.

Example concept:

```text
History so far: [rock, rock, paper, paper]
Target label:   scissors
```

This is an illustrative row, not a guarantee about what a real person will do.

The CSV `data/rps_synthetic.csv` includes `session_id`, `strategy`, `history`, 28 engineered feature columns, and `target`. The `strategy` column is metadata for inspection only; it is **not** fed into the ANN. The `history` column allows the training script to create symmetry-based augmentations without losing the original move sequence.

### Inspect it

```python
import pandas as pd

df = pd.read_csv("data/rps_synthetic.csv")
print(df.shape)
print(df.head())
print(df["target"].value_counts(normalize=True))
print(df["strategy"].value_counts())
```

Try changing the sample count or seed:

```bash
python generate_data.py --sessions-per-strategy 250 --seed 7
```

More rows do not automatically mean a better model. Diversity and realism of generated behaviors also matter.

## 3. Engineer the model inputs

Open `feature_engineering.py`. Every raw history is converted into **28 numeric features**:

| Feature group | Count | Meaning |
|---|---:|---|
| Recent five moves | 15 | Five positions × three one-hot flags per position; unused positions are zero padded. |
| Move frequencies | 3 | Proportion of rock, paper, and scissors in up to the last 50 moves. |
| Transition matrix | 9 | Smoothed chance of each following move conditioned on each previous move. |
| History length | 1 | History size scaled to the range 0–1. |
| **Total** | **28** | Fixed-size vector suitable for an ANN. |

Why not give the ANN a string such as `"rock, rock, paper"`? Neural networks operate on numerical tensors. Feature engineering maps the sequence to numerical signals while preserving several useful patterns.

Transition probabilities use Laplace smoothing. Each transition row starts with one count for each next-move class so that rows are valid probability distributions even when no transition has been observed.

### Try a feature example

```python
from feature_engineering import extract_features, feature_names

history = [0, 0, 1, 2, 1]  # rock, rock, paper, scissors, paper
values = extract_features(history)
for name, value in zip(feature_names(), values):
    print(f"{name:32} {value:.4f}")
```

Important discipline: feature generation in training and inference comes from the **same function**. Different transformations at prediction time can invalidate a trained model.

## 4. Create train and test sets without leaking sessions

Open `train_model.py`. The data is split using `GroupShuffleSplit`, grouped by `session_id`. Prefixes from the same synthetic sequence are correlated; if one prefix is in training and a nearby prefix from that episode is in testing, evaluation can be overly optimistic. Holding out complete sessions is more careful.

The test set is left unaugmented. Only training histories are transformed for data augmentation.

### Cyclic augmentation

Rock-Paper-Scissors has a cyclic structure: rock → paper → scissors → rock. Rotating all move labels by one or two places preserves the game's win/loss relationship. The training process creates three equivalent versions of a history and its target. This supplies additional examples without changing the underlying pattern.

This is **data augmentation**, not collection of new real-world observations. It cannot repair a poor synthetic behavior design by itself.

## 5. Scale the features

`StandardScaler` is fitted on the training features and then reused for both the test set and live predictions. Each feature is transformed using training-set means and standard deviations.

Why? One-hot indicators, frequencies, and transition probabilities may have different distributions. Scaling often helps gradient-based optimization behave more consistently.

Avoid fitting a separate scaler on the test set or on every user request: that would change the meaning of values during inference.

## 6. Build and train the ANN

The scikit-learn `MLPClassifier` is configured with this architecture:

```text
28 inputs → Dense(48, ReLU) → Dense(24, ReLU) → Dense(3, Softmax)
```

The model uses Adam optimization, a learning rate of `0.002`, early stopping, and a fixed seed for reproducibility.

The output is a probability distribution over the player's **next** rock, paper, or scissors move. Softmax converts the final class scores to three values that sum to 1.

Run training:

```bash
python train_model.py
```

The script saves:

- `artifacts/rps_ann.joblib` — fitted scaler and ANN weights.
- `artifacts/metrics.json` — sample sizes, architecture, accuracy, log loss, confusion matrix, and class report.

Model file and scaler are a pair: the ANN expects the same scaling transformation that was learned during training.

## 7. Evaluate the model instead of guessing from the UI

The current saved run produced:

| Metric | Result |
|---|---:|
| Dataset samples | 32,445 |
| Synthetic sessions | 1,800 |
| Training samples after augmentation | 78,213 |
| Held-out test samples | 6,374 |
| Test accuracy | 58.7% |
| Majority-class baseline after augmentation | 33.3% |

The model converged before the configured maximum of 350 iterations because early stopping was enabled. See `artifacts/metrics.json` for precision, recall, F1, class-level results, and log loss.

Interpretation matters:

- The test set contains held-out **synthetic sessions** produced by the same family of generators.
- Accuracy measures the fraction of next-move labels correctly classified.
- Log loss evaluates the probability assigned to the true class; lower is generally better.
- This does not mean the AI can predict 58.7% of all real users' moves.
- A real player who selects moves randomly offers no learnable next-move signal beyond chance-level uncertainty.

## 8. Turn predicted probabilities into an action

An ANN classifies the likely move; the game still needs a policy that picks the AI's move. Instead of blindly countering only the single most likely class, the application calculates each action's expected payoff against all three probabilities.

For each possible AI move:

```text
expected payoff = P(win) - P(loss)
```

A tie contributes zero. The policy selects the move with the highest expected payoff. If multiple moves are exactly tied, it uses a random tie-breaker. This distinction matters: a probability distribution can be spread across all moves, and the best expected response may differ from the counter of just the top-probability class.

## 9. Inspect the real forward pass in the backend

Open `app.py`. The browser sends the completed move history to `POST /api/analyze`. The route then:

1. Validates and encodes move labels.
2. Calls `extract_features` to produce the 28 raw features.
3. Uses the already-fitted `StandardScaler` to transform those features.
4. Uses the actual saved weights and biases (`model.coefs_` and `model.intercepts_`) to compute each layer's activations.
5. Applies ReLU after both hidden layers and softmax at the output.
6. Compares the manual output with `model.predict_proba` as a consistency check.
7. Calculates expected payoff for each AI action and returns a structured trace to the UI.

This lets you inspect neuron activations directly rather than displaying random or hand-written values. The model runs in the backend; JavaScript only visualizes the response.

### API response ideas

The response contains:

- `features`: every feature name, raw value, and scaled value.
- `layers`: actual activation values for input, both hidden layers, and softmax output.
- `probabilities`: estimated next-move class probabilities.
- `utilities`: expected payoff for rock, paper, and scissors as the AI move.
- `selected_ai_move`: chosen action.
- `forward_pass_max_abs_error`: maximum absolute difference between the explicit pass and scikit-learn's prediction.

## 10. Understand the UI animation

Open `templates/index.html`, `static/style.css`, and `static/script.js`.

- **Observed history:** shows previously completed inputs only.
- **Feature engineering:** breaks down the 15 + 3 + 9 + 1 features.
- **Input layer:** visualizes the 28 standardized input values.
- **Hidden layer 1:** visualizes 48 ReLU activations.
- **Hidden layer 2:** visualizes 24 ReLU activations.
- **Output probabilities:** visualizes the three softmax probabilities.
- **Decision policy:** compares expected payoff and highlights the selected move.
- **Feature inspector:** makes it possible to compare raw and standardized input values.
- **Forward-pass check:** reports whether the manually calculated output matches scikit-learn's output to numerical precision.

The seven stage cards light up in order and the moving packet indicates the active stage. All values originate from the backend trace for that inference request.

### Why the analysis runs after a round

At the start of a round, the ANN has already chosen the move using the history available at that time. You submit your move; the round is scored; your move is appended to history; then a new inference predicts your *next* move. The newly selected response is locked for the next round. That is why the input is not leaked into the prediction that was used to play the current round.

## 11. Run the full application

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000`. Do not double-click the HTML file; the UI uses Flask API routes.

Each round:

1. Choose Rock, Paper, or Scissors; the selected input becomes locked.
2. Watch the model process only completed prior rounds through features and neuron activations.
3. The ANN estimates the submitted move from prior behavior and selects a response using expected payoff.
4. The AI move is revealed, the round is scored, and the session record updates.
5. Your move is added to history and will inform the next round’s inference.

Use **Replay pipeline** to rerun inference against the same history without scoring a round, or **Reset match** to clear the session and start from an empty history.

## 12. Test the implementation

```bash
python -m unittest test_pipeline.py
```

The tests check the 28-feature contract, input validation, probability sums, legal output classes, and that the manual forward pass matches scikit-learn predictions.

## 13. Suggested student experiments

1. Change hidden layers from `(48, 24)` to `(64, 32)` and compare test accuracy, log loss, training time, and number of iterations.
2. Turn off augmentation and measure what changes.
3. Remove transition features and compare performance; explain why sequential behavior may become harder to recognize.
4. Train only with random synthetic players. Inspect how stable the output probabilities are and explain why exact prediction is not possible.
5. Increase or decrease the repeat probability in `generate_data.py` and measure generalization on held-out sessions.
6. Inspect confusion matrix errors: which move is commonly confused with which class?
7. Reduce history to the last two moves or expand the recent window. Update `feature_names`, feature extraction, and model training together.
8. Compare choosing the counter of the single most likely class against the implemented expected-payoff policy.

Whenever you change synthetic data, features, or model architecture, regenerate the dataset and retrain before testing the UI. Keep the test set separate from training and base conclusions on measured metrics.
