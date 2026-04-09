"""Exercise 1 — RMSNorm tests.

Run:  pytest tests/test_rmsnorm.py -v
"""

import torch

from model import RMSNorm


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization_scale(self):
        """After RMSNorm (weight=1), the RMS of each vector should be ~1."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64) * 5
        out = norm(x)
        rms = (out.float() ** 2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), (
            f"RMS should be ~1.0, got min={rms.min():.4f} max={rms.max():.4f}"
        )

    def test_dtype_preservation(self):
        """Output dtype must match input dtype (even float16)."""
        norm = RMSNorm(dim=32).half()
        x = torch.randn(1, 5, 32, dtype=torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16

    def test_against_manual_computation(self):
        """Compare output with a direct manual calculation."""
        dim = 16
        norm = RMSNorm(dim=dim, eps=1e-6)
        x = torch.randn(1, 4, dim)

        x_f = x.float()
        rms = (x_f ** 2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        expected = (x_f / rms * norm.weight.float()).to(x.dtype)

        out = norm(x)
        assert torch.allclose(out, expected, atol=1e-5), (
            f"Max diff: {(out - expected).abs().max().item()}"
        )
