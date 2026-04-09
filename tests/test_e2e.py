"""End-to-end tests — require ALL exercises to be implemented.

Run:  pytest tests/test_e2e.py -v
"""

import torch

from config import ModelConfig
from model import Transformer
from generate import generate, generate_text
from tokenizer import CharTokenizer


class TestEndToEnd:
    def setup_method(self):
        torch.manual_seed(42)
        self.config = ModelConfig.tiny()
        self.model = Transformer(self.config)
        self.model.eval()

    def test_forward_logits_shape(self):
        tokens = torch.randint(0, self.config.vocab_size, (2, 16))
        logits = self.model(tokens)
        assert logits.shape == (2, 16, self.config.vocab_size)

    def test_generate_produces_correct_length(self):
        prompt = torch.randint(0, self.config.vocab_size, (1, 8))
        output = generate(self.model, prompt, max_new_tokens=5, temperature=0)
        assert output.shape == (1, 13)

    def test_greedy_is_deterministic(self):
        prompt = torch.randint(0, self.config.vocab_size, (1, 8))
        out1 = generate(self.model, prompt, max_new_tokens=10, temperature=0)
        out2 = generate(self.model, prompt, max_new_tokens=10, temperature=0)
        assert torch.equal(out1, out2), "Greedy decoding must be deterministic"

    def test_text_generation_pipeline(self):
        tokenizer = CharTokenizer(vocab_size=self.config.vocab_size)
        text = generate_text(
            self.model, tokenizer, "hello", max_new_tokens=10, temperature=0.8
        )
        assert isinstance(text, str)
