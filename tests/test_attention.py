"""Exercise 3 — Scaled dot-product attention tests.

Run:  pytest tests/test_attention.py -v
"""

import torch
import torch.nn.functional as F

from model import scaled_dot_product_attention, create_causal_mask


class TestScaledDotProductAttention:
    def test_output_shape(self):
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        v = torch.randn(2, 8, 16, 64)
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (2, 8, 16, 64)

    def test_against_pytorch_no_mask(self):
        """Compare with torch.nn.functional.scaled_dot_product_attention (no mask)."""
        torch.manual_seed(42)
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        v = torch.randn(2, 8, 16, 64)

        ours = scaled_dot_product_attention(q, k, v)
        ref = F.scaled_dot_product_attention(q, k, v)
        assert torch.allclose(ours, ref, atol=1e-5), (
            f"Max diff (no mask): {(ours - ref).abs().max().item()}"
        )

    def test_against_pytorch_causal(self):
        """Causal-masked attention should match PyTorch's is_causal=True."""
        torch.manual_seed(42)
        seq_len = 16
        q = torch.randn(2, 8, seq_len, 64)
        k = torch.randn(2, 8, seq_len, 64)
        v = torch.randn(2, 8, seq_len, 64)
        mask = create_causal_mask(seq_len)

        ours = scaled_dot_product_attention(q, k, v, mask)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert torch.allclose(ours, ref, atol=1e-5), (
            f"Max diff (causal): {(ours - ref).abs().max().item()}"
        )

    def test_different_qk_lengths(self):
        """Q and K can have different sequence lengths (KV cache decode scenario)."""
        q = torch.randn(1, 4, 1, 32)
        k = torch.randn(1, 4, 10, 32)
        v = torch.randn(1, 4, 10, 32)
        out = scaled_dot_product_attention(q, k, v)
        assert out.shape == (1, 4, 1, 32)
