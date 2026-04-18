import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import ModelConfig
from rope import precompute_rope_frequencies, apply_rope


# ---------------------------------------------------------------------------
# Exercise 1: RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA / Mistral)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement RMS Normalization.

        Unlike LayerNorm, RMSNorm does NOT re-center (no mean subtraction).
        It only rescales by the root-mean-square, then applies a learnable
        per-dimension scale factor.

        Formula:
            rms(x) = sqrt( mean(x^2, dim=-1, keepdim=True) + eps )
            output = (x / rms(x)) * self.weight

        Steps:
        1. Cast x to float32 for numerical stability.
        2. Square each element: x_float ** 2
        3. Mean over the last dimension (keepdim=True).
        4. Add self.eps and take sqrt → this is the RMS value.
        5. Divide x_float by the RMS.
        6. Multiply by self.weight (learnable scale).
        7. Cast the result back to the original dtype of x.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Normalized tensor, same shape and dtype as x.
        """
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight

# ---------------------------------------------------------------------------
# Exercise 4: Causal mask
# ---------------------------------------------------------------------------


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    TODO: Create a causal (autoregressive) attention mask.

    In decoder-only transformers, each position can attend to itself and all
    previous positions, but NOT to future positions. This mask enforces that
    constraint in the attention score matrix.

    The result is a boolean tensor where True means "mask out" (cannot attend):

        seq_len=4 example:
        [[False,  True,  True,  True],   # pos 0 → sees [0]
         [False, False,  True,  True],   # pos 1 → sees [0, 1]
         [False, False, False,  True],   # pos 2 → sees [0, 1, 2]
         [False, False, False, False]]   # pos 3 → sees [0, 1, 2, 3]

    Args:
        seq_len: Sequence length.
        device: Device to create the tensor on.

    Returns:
        Boolean tensor of shape (seq_len, seq_len). True = position is masked.

    Hint: torch.triu with diagonal=1 gives a matrix of ones above the main
          diagonal — exactly the positions that should be masked.
    """
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    if device is not None:
        mask = mask.to(device)
    return mask


# ---------------------------------------------------------------------------
# Exercise 3: Scaled dot-product attention
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    TODO: Implement scaled dot-product attention.

    This is the core of the attention mechanism:

        Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

    Steps:
    1. Compute the scale factor: sqrt(d_k) where d_k = q.shape[-1].
    2. Compute raw attention scores:
         scores = (Q @ K^T) / scale
       Use torch.matmul(q, k.transpose(-2, -1)).
       Shape: (batch, n_heads, seq_len_q, seq_len_k)
    3. Apply the mask (if provided):
       The mask is boolean — True means "do not attend to this position".
       Set those positions to -infinity so they become 0 after softmax:
         scores = scores.masked_fill(mask, float("-inf"))
       The mask broadcasts: shape (seq_q, seq_k) broadcasts to
       (batch, n_heads, seq_q, seq_k) automatically.
    4. Compute attention weights via softmax over the last dim (seq_len_k).
       Use float32 for numerical stability, then cast back:
         weights = softmax(scores.float(), dim=-1).type_as(q)
    5. Compute the output:
         output = weights @ V
       Shape: (batch, n_heads, seq_len_q, head_dim)

    Args:
        q: Queries  — (batch, n_heads, seq_len_q, head_dim).
        k: Keys     — (batch, n_heads, seq_len_k, head_dim).
        v: Values   — (batch, n_heads, seq_len_k, head_dim).
        mask: Optional boolean mask broadcastable to (batch, n_heads, seq_q, seq_k).
              True = do not attend.

    Returns:
        Attention output — (batch, n_heads, seq_len_q, head_dim).
    """
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores.float(), dim=-1).type_as(q)
    return weights @ v



# ---------------------------------------------------------------------------
# Attention module (scaffolding provided; Exercise 6 is the KV cache method)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) and KV cache support."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_heads // config.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)

        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None

    # ---- Cache management (provided) ----

    def init_cache(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        """Allocate zeroed KV cache tensors."""
        self.cache_k = torch.zeros(
            batch_size, max_seq_len, self.n_kv_heads, self.head_dim,
            device=device, dtype=dtype,
        )
        self.cache_v = torch.zeros(
            batch_size, max_seq_len, self.n_kv_heads, self.head_dim,
            device=device, dtype=dtype,
        )

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    # ---- Exercise 6: KV cache update ----

    def update_kv_cache(
        self, k: torch.Tensor, v: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Store new K/V vectors in the cache and return the accumulated
        keys and values needed for the current attention computation.

        During autoregressive generation we cache previously computed K and V
        so we only run the projections for the *new* tokens, not the full
        history.

        Shapes:
            self.cache_k / self.cache_v : (batch, max_seq_len, n_kv_heads, head_dim)
            k / v (input)               : (batch, new_seq_len, n_kv_heads, head_dim)

        Steps:
        1. Compute seq_len of the new keys: seq_len = k.shape[1]
        2. Write the new k into the cache at positions
           [start_pos, start_pos + seq_len):
             self.cache_k[:, start_pos : start_pos + seq_len] = k
           Do the same for v.
        3. Compute end position: end_pos = start_pos + seq_len
        4. Return the slice of the cache from 0 to end_pos:
             return self.cache_k[:, :end_pos], self.cache_v[:, :end_pos]

        Args:
            k: New key vectors   — (batch, new_seq_len, n_kv_heads, head_dim).
            v: New value vectors — (batch, new_seq_len, n_kv_heads, head_dim).
            start_pos: Starting position in the sequence.

        Returns:
            (keys, values) covering positions [0, start_pos + new_seq_len),
            each of shape (batch, end_pos, n_kv_heads, head_dim).
        """
        seq_len = k.shape[1]
        self.cache_k[:, start_pos : start_pos + seq_len] = k
        self.cache_v[:, start_pos : start_pos + seq_len] = v
        end_pos = start_pos + seq_len
        return self.cache_k[:, :end_pos], self.cache_v[:, :end_pos]
        

    # ---- Forward (provided — glues the exercises together) ----

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if use_cache:
            k, v = self.update_kv_cache(k, v, start_pos)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)  # (batch, n_heads, seq_len_q, head_dim)
        k = k.transpose(1, 2)  # (batch, n_heads, seq_len_k, head_dim)
        v = v.transpose(1, 2)

        output = scaled_dot_product_attention(q, k, v, mask)

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo(output)


# ---------------------------------------------------------------------------
# Exercise 5: SwiGLU Feed-Forward
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """SwiGLU feed-forward network (used in LLaMA / Mistral)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement the SwiGLU feed-forward computation.

        SwiGLU is a gated feed-forward variant used in modern LLMs. It uses
        three weight matrices and the SiLU (Swish) activation function:

            output = w2( SiLU(w1(x)) * w3(x) )

        Breakdown:
        - w1 projects x to hidden_dim, then applies SiLU → this is the "gate"
        - w3 projects x to hidden_dim                    → this is the "value"
        - Element-wise multiply gate * value
        - w2 projects the result back to dim

        Steps:
        1. gate  = F.silu(self.w1(x))       # shape: (..., hidden_dim)
        2. value = self.w3(x)                # shape: (..., hidden_dim)
        3. return self.w2(gate * value)      # shape: (..., dim)

        Note: SiLU(x) = x * sigmoid(x), also called the Swish activation.
              Use torch.nn.functional.silu().

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Output tensor of shape (..., dim).
        """
        gate = F.silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)
        


# ---------------------------------------------------------------------------
# Transformer blocks (scaffolding provided)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Single transformer layer: pre-norm → attention → residual → pre-norm → FFN → residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int = 0,
        mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        h = x + self.attention(
            self.attention_norm(x), freqs_cis, start_pos, mask, use_cache
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """Decoder-only Transformer (GPT / LLaMA architecture)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        freqs = precompute_rope_frequencies(config.head_dim, config.max_seq_len)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            tokens: Input token IDs — (batch, seq_len).
            start_pos: Position offset for RoPE / KV cache (used in generation).
            use_cache: If True, read/write KV caches inside each Attention layer.

        Returns:
            Logits over the vocabulary — (batch, seq_len, vocab_size).
        """
        _batch, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        mask = None
        if seq_len > 1:
            mask = create_causal_mask(seq_len, device=tokens.device)

        for layer in self.layers:
            h = layer(h, freqs_cis, start_pos, mask, use_cache)

        h = self.norm(h)
        return self.output(h)

    def init_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        """Allocate KV caches in every attention layer."""
        for layer in self.layers:
            layer.attention.init_cache(
                batch_size, self.config.max_seq_len, device, dtype
            )

    def reset_cache(self):
        """Free KV caches in every attention layer."""
        for layer in self.layers:
            layer.attention.reset_cache()
