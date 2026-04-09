"""Exercise 4 — Causal mask tests.

Run:  pytest tests/test_mask.py -v
"""

import torch

from model import create_causal_mask


class TestCausalMask:
    def test_shape(self):
        mask = create_causal_mask(8)
        assert mask.shape == (8, 8)

    def test_dtype_is_bool(self):
        mask = create_causal_mask(4)
        assert mask.dtype == torch.bool

    def test_diagonal_is_visible(self):
        """Every position must be able to attend to itself."""
        mask = create_causal_mask(6)
        for i in range(6):
            assert not mask[i, i], f"Position {i} should see itself"

    def test_lower_triangle_visible(self):
        """Each position sees all earlier positions."""
        mask = create_causal_mask(5)
        for i in range(5):
            for j in range(i + 1):
                assert not mask[i, j], f"Position {i} should see position {j}"

    def test_upper_triangle_masked(self):
        """Each position must NOT see future positions."""
        mask = create_causal_mask(5)
        for i in range(5):
            for j in range(i + 1, 5):
                assert mask[i, j], f"Position {i} should NOT see future position {j}"

    def test_single_token(self):
        mask = create_causal_mask(1)
        assert mask.shape == (1, 1)
        assert not mask[0, 0]
