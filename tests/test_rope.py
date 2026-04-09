"""Exercise 2 — Rotary Position Embeddings tests.

Run:  pytest tests/test_rope.py -v
"""

import torch

from rope import precompute_rope_frequencies, apply_rope


class TestPrecomputeFrequencies:
    def test_output_shape(self):
        freqs = precompute_rope_frequencies(head_dim=64, max_seq_len=128)
        assert freqs.shape == (128, 32)

    def test_is_complex(self):
        freqs = precompute_rope_frequencies(head_dim=64, max_seq_len=128)
        assert freqs.is_complex(), "Output must be complex-valued"

    def test_unit_magnitude(self):
        """Every entry is e^(iθ) and should have |z| = 1."""
        freqs = precompute_rope_frequencies(head_dim=64, max_seq_len=128)
        magnitudes = freqs.abs()
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)

    def test_position_zero_is_one(self):
        """At position 0 all angles are 0 → e^(i·0) = 1+0i."""
        freqs = precompute_rope_frequencies(head_dim=64, max_seq_len=128)
        expected = torch.ones(32, dtype=torch.complex64)
        assert torch.allclose(freqs[0], expected, atol=1e-5)


class TestApplyRope:
    def test_output_shape(self):
        head_dim = 64
        freqs = precompute_rope_frequencies(head_dim=head_dim, max_seq_len=32)
        x = torch.randn(2, 16, 8, head_dim)
        out = apply_rope(x, freqs[:16])
        assert out.shape == x.shape

    def test_dtype_preservation(self):
        head_dim = 32
        freqs = precompute_rope_frequencies(head_dim=head_dim, max_seq_len=16)
        x = torch.randn(1, 8, 4, head_dim)
        out = apply_rope(x, freqs[:8])
        assert out.dtype == x.dtype

    def test_different_positions_give_different_outputs(self):
        """The same vector placed at two different positions should be rotated differently."""
        head_dim = 32
        freqs = precompute_rope_frequencies(head_dim=head_dim, max_seq_len=64)
        vec = torch.randn(1, 1, 1, head_dim)
        out_pos0 = apply_rope(vec, freqs[0:1])
        out_pos10 = apply_rope(vec, freqs[10:11])
        assert not torch.allclose(out_pos0, out_pos10, atol=1e-5)

    def test_position_zero_is_identity(self):
        """Rotation at position 0 should be the identity (angle = 0 everywhere)."""
        head_dim = 32
        freqs = precompute_rope_frequencies(head_dim=head_dim, max_seq_len=16)
        x = torch.randn(1, 1, 4, head_dim)
        out = apply_rope(x, freqs[0:1])
        assert torch.allclose(out, x, atol=1e-5)
