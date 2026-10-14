"""Feature extraction for the Rock-Paper-Scissors ANN demo.

Moves are encoded as 0=rock, 1=paper, 2=scissors. All input features are
calculated using historical moves only; the next move is the prediction target.
"""
from __future__ import annotations

from typing import Iterable, Sequence

MOVES = ("rock", "paper", "scissors")
RECENT_WINDOW = 5
MAX_HISTORY_FOR_FEATURES = 50


def normalize_history(history: Iterable[int]) -> list[int]:
    """Validate and normalize a list of encoded moves."""
    result = []
    for value in history:
        move = int(value)
        if move not in (0, 1, 2):
            raise ValueError(f"Invalid move id {move}; expected 0, 1, or 2")
        result.append(move)
    return result


def feature_names() -> list[str]:
    names: list[str] = []
    # Recent positions run from oldest to newest; zero vectors are left padding.
    for offset in range(RECENT_WINDOW, 0, -1):
        for move in MOVES:
            names.append(f"recent_{offset}_ago_{move}")
    names.extend([f"frequency_{move}" for move in MOVES])
    for previous in MOVES:
        for following in MOVES:
            names.append(f"transition_{previous}_to_{following}")
    names.append("history_length_scaled")
    return names


def extract_features(history: Sequence[int]) -> list[float]:
    """Convert the player's prior moves into a fixed 28-dimensional vector.

    Components:
      * 15 values: one-hot encodings of the most recent five moves
      * 3 values: empirical move frequencies over up to 50 moves
      * 9 values: smoothed conditional transition probabilities
      * 1 value: history length scaled to [0, 1]
    """
    moves = normalize_history(history)[-MAX_HISTORY_FOR_FEATURES:]
    features: list[float] = []

    recent = moves[-RECENT_WINDOW:]
    padded: list[int | None] = [None] * (RECENT_WINDOW - len(recent)) + recent
    for move_id in padded:
        for candidate in range(3):
            features.append(1.0 if move_id == candidate else 0.0)

    count = len(moves)
    for candidate in range(3):
        features.append(moves.count(candidate) / count if count else 0.0)

    # Laplace smoothing keeps rows defined when no transition has been observed.
    transition_counts = [[1.0, 1.0, 1.0] for _ in range(3)]
    for left, right in zip(moves[:-1], moves[1:]):
        transition_counts[left][right] += 1.0
    for previous in range(3):
        total = sum(transition_counts[previous])
        for following in range(3):
            features.append(transition_counts[previous][following] / total)

    features.append(min(len(normalize_history(history)), MAX_HISTORY_FOR_FEATURES) / MAX_HISTORY_FOR_FEATURES)
    return features


def rotate_move(move: int, shift: int) -> int:
    """Rotate labels cyclically; this preserves RPS win/loss structure."""
    return (int(move) + int(shift)) % 3


def rotate_history(history: Sequence[int], shift: int) -> list[int]:
    return [rotate_move(move, shift) for move in history]
