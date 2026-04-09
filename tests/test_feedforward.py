"""Exercise 5 — SwiGLU feed-forward tests.

Run:  pytest tests/test_feedforward.py -v
"""

import torch
import torch.nn.functional as F

from config import ModelConfig
from model import FeedForward


class TestFeedForward:
    def setup_method(self):
        self.config = ModelConfig.tiny()
        torch.manual_seed(42)
        self.ff = FeedForward(self.config)

    def test_output_shape(self):
        x = torch.randn(2, 10, self.config.dim)
        out = self.ff(x)
        assert out.shape == x.shape

    def test_against_manual_swiglu(self):
        """Verify: output = w2( silu(w1(x)) * w3(x) )."""
        x = torch.randn(1, 4, self.config.dim)

        gate = F.silu(self.ff.w1(x))
        value = self.ff.w3(x)
        expected = self.ff.w2(gate * value)

        out = self.ff(x)
        assert torch.allclose(out, expected, atol=1e-5), (
            f"Max diff: {(out - expected).abs().max().item()}"
        )

    def test_nonlinearity(self):
        """SwiGLU is nonlinear: f(a + b) != f(a) + f(b)."""
        a = torch.randn(1, 4, self.config.dim)
        b = torch.randn(1, 4, self.config.dim)
        assert not torch.allclose(self.ff(a + b), self.ff(a) + self.ff(b), atol=1e-3)
