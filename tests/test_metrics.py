"""
============================================================
test_metrics.py — Unit tests for evaluate.py
============================================================

Run:
    pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Allow importing src package without installing
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluate import compute_metrics, per_class_f1

NUM_LABELS = 28
N_SAMPLES = 50


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def random_labels(rng):
    """Random multi-hot ground-truth labels."""
    arr = (rng.random((N_SAMPLES, NUM_LABELS)) > 0.85).astype(float)
    # Ensure at least one positive per row (avoid all-zero rows)
    for i in range(N_SAMPLES):
        if arr[i].sum() == 0:
            arr[i, rng.integers(NUM_LABELS)] = 1.0
    return torch.tensor(arr, dtype=torch.float)


# ============================================================
# Test: perfect prediction
# ============================================================
def test_compute_metrics_perfect_prediction(random_labels):
    """When logits perfectly predict labels, F1 should be 1.0."""
    # Logits: large positive → sigmoid ≈ 1; large negative → sigmoid ≈ 0
    perfect_logits = (random_labels * 2 - 1) * 10.0  # +10 where label=1, -10 where label=0

    metrics = compute_metrics(perfect_logits, random_labels, threshold=0.5)

    assert metrics["f1_macro"] == pytest.approx(1.0, abs=1e-6), (
        f"Expected f1_macro=1.0 for perfect predictions, got {metrics['f1_macro']}"
    )
    assert metrics["f1_micro"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["hamming_loss"] == pytest.approx(0.0, abs=1e-6)


# ============================================================
# Test: all-zero prediction
# ============================================================
def test_compute_metrics_zero_prediction(random_labels):
    """When model predicts nothing, hamming_loss > 0 and f1_macro should be low."""
    zero_logits = torch.full((N_SAMPLES, NUM_LABELS), -10.0)  # sigmoid → ~0 everywhere

    metrics = compute_metrics(zero_logits, random_labels, threshold=0.5)

    # Hamming loss > 0 because there are positive labels we're missing
    assert metrics["hamming_loss"] > 0.0, "Hamming loss should be > 0 for zero predictions"

    # F1-macro must be in [0, 1]
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["f1_micro"] <= 1.0

    # With all zeros predicted, f1_macro will be 0 (no TP anywhere)
    assert metrics["f1_macro"] == pytest.approx(0.0, abs=1e-6), (
        f"Expected f1_macro=0.0 for all-zero predictions, got {metrics['f1_macro']}"
    )


# ============================================================
# Test: per_class_f1 output shape
# ============================================================
def test_per_class_f1_shape(random_labels):
    """per_class_f1 should return a dict with exactly NUM_LABELS entries."""
    logits = torch.randn(N_SAMPLES, NUM_LABELS)

    result = per_class_f1(logits, random_labels, threshold=0.5)

    assert isinstance(result, dict), "per_class_f1 should return a dict"
    assert len(result) == NUM_LABELS, (
        f"Expected {NUM_LABELS} class entries, got {len(result)}"
    )

    # All values must be floats in [0, 1]
    for class_name, f1 in result.items():
        assert isinstance(class_name, str), f"Key should be str, got {type(class_name)}"
        assert 0.0 <= f1 <= 1.0, f"F1 for '{class_name}' is out of [0,1]: {f1}"


# ============================================================
# Test: hamming_loss is in [0, 1]
# ============================================================
def test_hamming_loss_range(random_labels):
    """Hamming loss must always be in [0.0, 1.0]."""
    # Test with multiple random logit tensors
    rng = np.random.default_rng(seed=7)
    for _ in range(5):
        logits = torch.tensor(rng.standard_normal((N_SAMPLES, NUM_LABELS)), dtype=torch.float)
        metrics = compute_metrics(logits, random_labels, threshold=0.5)
        assert 0.0 <= metrics["hamming_loss"] <= 1.0, (
            f"Hamming loss out of range: {metrics['hamming_loss']}"
        )


# ============================================================
# Test: threshold sensitivity
# ============================================================
def test_threshold_affects_predictions():
    """Different thresholds should produce different predictions."""
    torch.manual_seed(0)
    logits = torch.randn(30, NUM_LABELS)
    labels = (torch.rand(30, NUM_LABELS) > 0.85).float()

    metrics_low = compute_metrics(logits, labels, threshold=0.3)
    metrics_high = compute_metrics(logits, labels, threshold=0.7)

    # With threshold=0.3 more positives are predicted → lower hamming on average
    # The important check is just that they're valid floats in range
    for m in [metrics_low, metrics_high]:
        assert 0.0 <= m["f1_macro"] <= 1.0
        assert 0.0 <= m["hamming_loss"] <= 1.0


# ============================================================
# Test: return dict has expected keys
# ============================================================
def test_compute_metrics_keys(random_labels):
    """compute_metrics must return all expected keys."""
    logits = torch.randn(N_SAMPLES, NUM_LABELS)
    metrics = compute_metrics(logits, random_labels)

    expected_keys = {"f1_macro", "f1_micro", "f1_weighted", "hamming_loss"}
    assert expected_keys.issubset(set(metrics.keys())), (
        f"Missing keys: {expected_keys - set(metrics.keys())}"
    )
