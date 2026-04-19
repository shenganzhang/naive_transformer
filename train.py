"""
Training loop for the decoder-only Transformer.

Trains on next-token prediction (standard language model objective):
    loss = cross_entropy(logits[:, :-1], tokens[:, 1:])

For each position, the model predicts the *next* token using only the
tokens before it (enforced by the causal mask in model.py).

Usage (local, CPU):
    python3 train.py

Usage (Colab, GPU):
    See notebook.ipynb → "Part 3: Training"
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from config import ModelConfig
from model import Transformer
from generate import generate_text
from tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Built-in tiny dataset (no download needed)
# ---------------------------------------------------------------------------

TINY_TEXT = """
The transformer architecture was introduced in the paper "Attention is All You Need"
by Vaswani et al. in 2017. It relies entirely on self-attention mechanisms, dispensing
with recurrence and convolutions entirely. The model achieves state of the art results
on machine translation tasks.

The key innovation is the attention mechanism, which allows the model to focus on
different parts of the input sequence when making predictions. Unlike recurrent neural
networks, transformers process all tokens in parallel during training, making them
much faster to train on modern hardware.

A transformer consists of an encoder and a decoder. The encoder reads the input
sequence and produces a sequence of hidden states. The decoder uses these hidden
states along with the previously generated tokens to produce the output sequence.
Modern language models like GPT use only the decoder part, trained to predict the
next token given all previous tokens. This simple objective, applied to massive
amounts of text, produces models capable of remarkable language understanding and
generation.

The attention mechanism computes a weighted sum of values, where the weights are
determined by the compatibility of queries with keys. Scaled dot product attention
divides the dot products by the square root of the key dimension to prevent
extremely small gradients in the softmax function.

Multi-head attention runs several attention functions in parallel and concatenates
the results. This allows the model to jointly attend to information from different
representation subspaces at different positions.

Position information is injected using positional encodings. Rotary position
embeddings encode position by rotating the query and key vectors, allowing the
attention scores to naturally reflect relative positions between tokens.

The feed-forward network applies two linear transformations with a nonlinear
activation in between. The SwiGLU variant uses a gating mechanism that multiplies
two linear projections, one passed through a sigmoid linear unit activation.

Layer normalization is applied before each sub-layer. RMS normalization omits the
mean subtraction step and has been found to work equally well with less computation.

Training uses the Adam optimizer with a learning rate schedule that includes a
warmup phase followed by cosine decay. Gradient clipping prevents the loss from
exploding during early training steps when the model parameters are far from optimal.

The key value cache stores previously computed key and value tensors during
inference, avoiding redundant computation when generating tokens one at a time.
During prefill the entire prompt is processed at once. During decode each new token
attends to all previous tokens via the cache.
""".strip()


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    seq_len: int = 128          # context length per training sample
    # Optimization
    batch_size: int = 8
    max_steps: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 50
    # Logging / checkpointing
    log_every: int = 50
    eval_every: int = 100
    checkpoint_path: str = "checkpoint.pt"
    # Device
    device: str = "auto"        # "auto" picks cuda > mps > cpu

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset:
    """Sliding-window character-level dataset over a raw text string."""

    def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        # Encode without BOS so the raw token stream is contiguous
        self.tokens = torch.tensor(
            tokenizer.encode(text, add_bos=False), dtype=torch.long
        )
        print(f"Dataset: {len(self.tokens):,} tokens from {len(text):,} chars")

    def get_batch(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (input, target) pairs."""
        max_start = len(self.tokens) - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.tokens[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.tokens[s + 1 : s + self.seq_len + 1] for s in starts])
        return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    text: str = TINY_TEXT,
) -> tuple[Transformer, CharTokenizer]:
    """
    Train the transformer on a text string and return the trained model + tokenizer.

    Args:
        model_cfg: Model architecture config. Defaults to ModelConfig.small().
        train_cfg: Training hyperparameters. Defaults to TrainConfig().
        text: Raw training text.

    Returns:
        (model, tokenizer) ready for inference.
    """
    if model_cfg is None:
        model_cfg = ModelConfig.small()
    if train_cfg is None:
        train_cfg = TrainConfig()

    device = train_cfg.resolve_device()
    print(f"Device: {device}")

    tokenizer = CharTokenizer(vocab_size=model_cfg.vocab_size)
    dataset = TextDataset(text, tokenizer, train_cfg.seq_len)

    model = Transformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    t0 = time.perf_counter()
    losses = []

    for step in range(1, train_cfg.max_steps + 1):
        # Update learning rate
        lr = get_lr(step, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = dataset.get_batch(train_cfg.batch_size, device)

        # Forward: logits over all positions except the last
        logits = model(x)                         # [B, seq_len, vocab_size]
        # Shift: predict token t+1 from tokens 0..t
        logits = logits[:, :-1, :].contiguous()   # [B, seq_len-1, vocab_size]
        y = y[:, :-1].contiguous()                # [B, seq_len-1]

        loss = loss_fn(logits.view(-1, model_cfg.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        losses.append(loss.item())

        if step % train_cfg.log_every == 0:
            avg_loss = sum(losses[-train_cfg.log_every:]) / train_cfg.log_every
            elapsed = time.perf_counter() - t0
            print(
                f"step {step:4d}/{train_cfg.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"{elapsed:.1f}s"
            )

        if step % train_cfg.eval_every == 0:
            model.eval()
            sample = generate_text(
                model, tokenizer, "The transformer",
                max_new_tokens=60, temperature=0.8, top_k=20,
            )
            print(f"  sample: {sample[:120]!r}")
            model.train()

    # Save checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "model_cfg": model_cfg,
        "step": train_cfg.max_steps,
        "final_loss": losses[-1],
    }, train_cfg.checkpoint_path)
    print(f"\nCheckpoint saved → {train_cfg.checkpoint_path}")
    print(f"Final loss: {losses[-1]:.4f}")

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str = "checkpoint.pt",
    device: str = "auto",
) -> tuple[Transformer, ModelConfig]:
    """Load a trained model from a checkpoint file."""
    dev = TrainConfig(device=device).resolve_device()
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    model_cfg: ModelConfig = ckpt["model_cfg"]
    model = Transformer(model_cfg).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from {path} (step={ckpt['step']}, loss={ckpt['final_loss']:.4f})")
    return model, model_cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick local run with tiny config so it finishes in ~30s on CPU
    model_cfg = ModelConfig.tiny()
    train_cfg = TrainConfig(
        batch_size=4,
        max_steps=300,
        seq_len=64,
        log_every=50,
        eval_every=100,
    )
    model, tokenizer = train(model_cfg, train_cfg)

    print("\n--- Inference ---")
    for prompt in ["The transformer", "Attention is", "The key"]:
        out = generate_text(model, tokenizer, prompt, max_new_tokens=80, temperature=0.8, top_k=20)
        print(f"[{prompt!r}] {out!r}")
