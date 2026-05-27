"""
Tests cho data module.

Run:
    pytest tests/test_data.py -v
"""

import pytest
import torch

from src.data import NUM_LABELS, EMOTION_NAMES


def test_num_labels():
    """Verify số lượng labels."""
    assert NUM_LABELS == 28
    assert len(EMOTION_NAMES) == 28


def test_emotion_names_unique():
    """Verify không có duplicate emotion names."""
    assert len(set(EMOTION_NAMES)) == len(EMOTION_NAMES)


def test_neutral_in_labels():
    """Verify 'neutral' là 1 trong các labels."""
    assert "neutral" in EMOTION_NAMES


def test_compute_pos_weights_shape():
    """Verify pos_weights output shape (mock test)."""
    from src.data import compute_pos_weights

    # Mock dataset
    class MockDataset:
        def __init__(self, n=100):
            self.n = n
            self.items = [{"labels": [i % NUM_LABELS]} for i in range(n)]

        def __len__(self):
            return self.n

        def __iter__(self):
            return iter(self.items)

    ds = MockDataset(n=100)
    weights = compute_pos_weights(ds)
    assert weights.shape == (NUM_LABELS,)
    assert weights.dtype == torch.float
    assert (weights >= 1.0).all()
    assert (weights <= 50.0).all()
