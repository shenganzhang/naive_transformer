import torch
import torch.nn.functional as F

from model import Transformer
from tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Exercise 7: Top-k / top-p (nucleus) sampling
# ---------------------------------------------------------------------------


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    TODO: Filter a logits distribution using top-k and/or top-p (nucleus) filtering.

    Top-k: keep only the k highest logits; set the rest to -inf.
    Top-p: keep the smallest set of tokens whose cumulative probability >= p;
           set the rest to -inf.
    When both are specified, apply top-k first, then top-p.

    Steps for top-k (if top_k > 0):
    1. Clamp top_k to at most vocab_size: top_k = min(top_k, logits.size(-1))
    2. Find the k-th largest value in each row:
         threshold = torch.topk(logits, top_k).values[..., -1, None]
       (topk returns sorted descending, so [..., -1] is the k-th largest)
    3. Mask out everything below the threshold:
         logits = logits.masked_fill(logits < threshold, float("-inf"))

    Steps for top-p / nucleus (if top_p < 1.0):
    1. Sort logits in descending order:
         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    2. Compute cumulative probabilities of the sorted logits:
         cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    3. Build a mask of tokens to REMOVE — those whose cumulative probability
       exceeds top_p. Shift the mask right by 1 position so the first token
       that crosses the threshold is *kept*:
         sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
       (subtracting the current token's prob gives the cum. prob *before* it)
    4. Set masked sorted logits to -inf:
         sorted_logits[sorted_mask] = float("-inf")
    5. Scatter the filtered logits back to the original order:
         logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    Args:
        logits: Raw logits — (batch, vocab_size).
        top_k:  Number of top tokens to keep. 0 = disabled.
        top_p:  Cumulative probability threshold. 1.0 = disabled.

    Returns:
        Filtered logits (same shape), with excluded positions set to -inf.
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
        logits = logits.masked_fill(logits < top_k_values[..., -1, None], float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        # scatter mask back to original token order before applying
        mask = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))

    return logits


# ---------------------------------------------------------------------------
# Sampling helper (provided)
# ---------------------------------------------------------------------------


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample a single token from the logits distribution."""
    if temperature == 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    filtered = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# Generation loop (provided — uses KV cache from Exercise 6)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Autoregressive text generation with KV caching.

    Phase 1 — Prefill:
        Process the entire prompt at once, populating the KV cache in every
        attention layer.

    Phase 2 — Decode:
        Generate tokens one at a time. Each step feeds only the newest token
        through the model; the KV cache provides the history.

    Args:
        model: A Transformer instance.
        prompt_tokens: Token IDs — (batch, prompt_len).
        max_new_tokens: How many new tokens to generate.
        temperature: Sampling temperature (0 = greedy argmax).
        top_k: Top-k filtering (0 = disabled).
        top_p: Nucleus filtering (1.0 = disabled).

    Returns:
        Full sequence (prompt + generated) — (batch, prompt_len + max_new_tokens).
    """
    device = next(model.parameters()).device
    tokens = prompt_tokens.to(device)
    batch_size, prompt_len = tokens.shape

    model.init_cache(batch_size, device)

    # --- Prefill ---
    logits = model(tokens, start_pos=0, use_cache=True)
    next_logits = logits[:, -1, :]

    generated: list[torch.Tensor] = []
    for i in range(max_new_tokens):
        next_token = sample(next_logits, temperature, top_k, top_p)
        generated.append(next_token)

        # --- Decode ---
        cur_pos = prompt_len + i
        logits = model(next_token.unsqueeze(1), start_pos=cur_pos, use_cache=True)
        next_logits = logits[:, -1, :]

    model.reset_cache()

    if generated:
        return torch.cat([tokens] + [t.unsqueeze(1) for t in generated], dim=1)
    return tokens


# ---------------------------------------------------------------------------
# Convenience wrapper (provided)
# ---------------------------------------------------------------------------


def generate_text(
    model: Transformer,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """Text-in, text-out generation."""
    device = next(model.parameters()).device
    tokens = torch.tensor([tokenizer.encode(prompt)], device=device)
    output_tokens = generate(
        model, tokens, max_new_tokens, temperature, top_k, top_p
    )
    return tokenizer.decode(output_tokens[0].tolist())
