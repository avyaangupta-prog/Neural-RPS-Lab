"""Generate labelled synthetic player histories for supervised ANN training."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from feature_engineering import MOVES, extract_features, feature_names

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data" / "rps_synthetic.csv"


def weighted_choice(rng: random.Random, probabilities: list[float]) -> int:
    return rng.choices([0, 1, 2], weights=probabilities, k=1)[0]


def generate_sequence(strategy: str, length: int, rng: random.Random) -> list[int]:
    """Simulate one synthetic opponent episode with a hidden behavior policy."""
    seq: list[int] = []
    biased = {
        "biased_rock": [0.68, 0.20, 0.12],
        "biased_paper": [0.16, 0.68, 0.16],
        "biased_scissors": [0.12, 0.20, 0.68],
    }
    transition_matrix = None
    if strategy == "markov":
        # Each episode has its own stochastic transition policy.
        transition_matrix = [
            [rng.uniform(0.15, 0.70) for _ in range(3)] for _ in range(3)
        ]
        transition_matrix = [[v / sum(row) for v in row] for row in transition_matrix]

    alternate_pair = rng.sample([0, 1, 2], k=2)
    for index in range(length):
        previous = seq[-1] if seq else None
        if strategy == "random":
            move = rng.randrange(3)
        elif strategy in biased:
            move = weighted_choice(rng, biased[strategy])
        elif strategy == "repeat_last":
            if previous is None or rng.random() < 0.25:
                move = rng.randrange(3)
            else:
                move = previous
        elif strategy == "cycle_forward":
            if previous is None or rng.random() < 0.18:
                move = rng.randrange(3)
            else:
                move = (previous + 1) % 3
        elif strategy == "cycle_backward":
            if previous is None or rng.random() < 0.18:
                move = rng.randrange(3)
            else:
                move = (previous - 1) % 3
        elif strategy == "alternate_pair":
            if index == 0 or rng.random() < 0.12:
                move = rng.choice(alternate_pair)
            else:
                move = alternate_pair[1] if previous == alternate_pair[0] else alternate_pair[0]
        elif strategy == "markov":
            if previous is None:
                move = rng.randrange(3)
            else:
                assert transition_matrix is not None
                move = weighted_choice(rng, transition_matrix[previous])
        elif strategy == "win_stay_lose_shift":
            # Synthetic pattern proxy: mostly repeats, with occasional rotation.
            if previous is None or rng.random() < 0.22:
                move = rng.randrange(3)
            else:
                move = (previous + rng.choice([0, 1])) % 3
        else:
            raise ValueError(f"Unknown synthetic strategy: {strategy}")
        seq.append(move)
    return seq


def generate_dataset(output_path: Path = DEFAULT_OUT, sessions_per_strategy: int = 220, seed: int = 42) -> int:
    rng = random.Random(seed)
    strategies = [
        "random", "biased_rock", "biased_paper", "biased_scissors",
        "repeat_last", "cycle_forward", "cycle_backward", "alternate_pair",
        "markov", "win_stay_lose_shift",
    ]
    names = feature_names()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["session_id", "strategy", "history", *names, "target"],
        )
        writer.writeheader()
        session_number = 0
        for strategy in strategies:
            for _ in range(sessions_per_strategy):
                session_number += 1
                length = rng.randint(8, 28)
                seq = generate_sequence(strategy, length, rng)
                # One sample for each prefix; label is the next move after that prefix.
                for index, target in enumerate(seq):
                    history = seq[:index]
                    row = {
                        "session_id": f"session_{session_number:05d}",
                        "strategy": strategy,
                        "history": " ".join(map(str, history)),
                        "target": target,
                    }
                    row.update(dict(zip(names, extract_features(history))))
                    writer.writerow(row)
                    total_rows += 1
    print(f"Generated {total_rows:,} samples from {session_number:,} synthetic sessions.")
    print(f"Saved dataset: {output_path}")
    print(f"Input features: {len(names)}; target classes: {', '.join(MOVES)}")
    return total_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-per-strategy", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    generate_dataset(args.output, args.sessions_per_strategy, args.seed)
