"""Exercise 7 — Top-k / top-p sampling tests.

Run:  pytest tests/test_sampling.py -v
"""

import torch

from generate import top_k_top_p_filtering


class TestTopKTopPFiltering:
    def test_top_k_keeps_k_tokens(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        filtered = top_k_top_p_filtering(logits, top_k=3)

        finite_count = filtered.isfinite().sum().item()
        assert finite_count == 3, f"Expected 3 finite values, got {finite_count}"

        kept = set(filtered[filtered.isfinite()].tolist())
        assert kept == {5.0, 4.0, 3.0}

    def test_top_k_zero_is_noop(self):
        logits = torch.randn(1, 100)
        filtered = top_k_top_p_filtering(logits.clone(), top_k=0)
        assert torch.equal(logits, filtered)

    def test_top_p_filters_tail(self):
        logits = torch.tensor([[10.0, 1.0, 0.1, 0.01, 0.001]])
        filtered = top_k_top_p_filtering(logits, top_p=0.9)
        remaining = filtered.isfinite().sum().item()
        assert 1 <= remaining < logits.shape[-1], (
            f"top_p should remove low-probability tail, got {remaining} remaining"
        )

    def test_top_p_one_is_noop(self):
        logits = torch.randn(1, 50)
        filtered = top_k_top_p_filtering(logits.clone(), top_p=1.0)
        assert torch.equal(logits, filtered)

    def test_combined_top_k_top_p(self):
        logits = torch.randn(1, 100)
        filtered = top_k_top_p_filtering(logits, top_k=10, top_p=0.5)
        remaining = filtered.isfinite().sum().item()
        assert 1 <= remaining <= 10
