"""Lightweight sanity checks; run with: python -m unittest test_pipeline.py"""
import unittest
from pathlib import Path

import numpy as np
from joblib import load

from feature_engineering import MOVES, extract_features, feature_names, normalize_history

ROOT = Path(__file__).resolve().parent


def softmax(values):
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / exp.sum()


class FeatureTests(unittest.TestCase):
    def test_feature_vector_has_28_finite_values(self):
        values = extract_features([0, 1, 2, 0, 0, 2])
        self.assertEqual(len(values), 28)
        self.assertEqual(len(feature_names()), 28)
        self.assertTrue(np.isfinite(values).all())

    def test_empty_history_is_valid(self):
        values = extract_features([])
        self.assertEqual(len(values), 28)
        self.assertEqual(sum(values[:15]), 0)

    def test_invalid_move_rejected(self):
        with self.assertRaises(ValueError):
            normalize_history([0, 1, 4])

    def test_cyclic_rotation_maps_move_labels(self):
        self.assertEqual([(move + 1) % 3 for move in [0, 1, 2]], [1, 2, 0])
        self.assertEqual(MOVES, ("rock", "paper", "scissors"))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifact = ROOT / "artifacts" / "rps_ann.joblib"
        if not artifact.exists():
            raise RuntimeError("Run generate_data.py and train_model.py before tests")
        cls.bundle = load(artifact)
        cls.scaler = cls.bundle["scaler"]
        cls.model = cls.bundle["model"]

    def test_probabilities_sum_to_one(self):
        x = self.scaler.transform(np.asarray([extract_features([0, 0, 1, 2, 2])]))
        p = self.model.predict_proba(x)[0]
        self.assertAlmostEqual(float(p.sum()), 1.0, places=7)
        self.assertEqual(len(p), 3)

    def test_manual_forward_pass_matches_sklearn(self):
        x = self.scaler.transform(np.asarray([extract_features([0, 1, 0, 2, 1, 1])]))[0]
        current = x
        for layer_idx, (weights, bias) in enumerate(zip(self.model.coefs_, self.model.intercepts_)):
            z = current @ weights + bias
            if layer_idx < len(self.model.coefs_) - 1:
                current = np.maximum(z, 0.0)
            else:
                current = softmax(z)
        public = self.model.predict_proba(x.reshape(1, -1))[0]
        self.assertTrue(np.allclose(current, public, atol=1e-8))

    def test_prediction_is_a_valid_move_class(self):
        x = self.scaler.transform(np.asarray([extract_features([2, 2, 0, 1, 0])]))
        prediction = int(self.model.predict(x)[0])
        self.assertIn(prediction, [0, 1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
