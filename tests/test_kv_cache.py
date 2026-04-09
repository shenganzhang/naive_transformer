"""Exercise 6 — KV cache tests.

Prerequisites: RMSNorm, RoPE, attention, causal mask, and SwiGLU must be implemented.

Run:  pytest tests/test_kv_cache.py -v
"""

import torch

from config import ModelConfig
from model import Transformer


class TestKVCache:
    def setup_method(self):
        torch.manual_seed(42)
        self.config = ModelConfig.tiny()
        self.model = Transformer(self.config)
        self.model.eval()

    def test_prefill_matches_no_cache(self):
        """Processing the full prompt with cache should give the same logits as without cache."""
        tokens = torch.randint(0, self.config.vocab_size, (1, 16))

        logits_ref = self.model(tokens, start_pos=0, use_cache=False)

        self.model.init_cache(1, tokens.device)
        logits_cached = self.model(tokens, start_pos=0, use_cache=True)
        self.model.reset_cache()

        assert torch.allclose(logits_ref, logits_cached, atol=1e-5), (
            f"Max diff: {(logits_ref - logits_cached).abs().max().item()}"
        )

    def test_incremental_decode_matches(self):
        """Token-by-token decoding with cache must produce the same logits
        as a single full-sequence forward pass."""
        seq_len = 12
        tokens = torch.randint(0, self.config.vocab_size, (1, seq_len))

        logits_ref = self.model(tokens, start_pos=0, use_cache=False)

        split = 8
        self.model.init_cache(1, tokens.device)

        logits_prefill = self.model(
            tokens[:, :split], start_pos=0, use_cache=True
        )
        assert torch.allclose(logits_ref[:, :split], logits_prefill, atol=1e-5), (
            "Prefill logits don't match"
        )

        for i in range(split, seq_len):
            logits_step = self.model(
                tokens[:, i : i + 1], start_pos=i, use_cache=True
            )
            assert torch.allclose(
                logits_ref[:, i : i + 1], logits_step, atol=1e-4
            ), (
                f"Position {i} logits don't match.  "
                f"Max diff: {(logits_ref[:, i:i+1] - logits_step).abs().max().item()}"
            )

        self.model.reset_cache()
