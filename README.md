# Transformer from Scratch

A decoder-only transformer (GPT/LLaMA style) implemented in PyTorch, structured
as 7 fill-in-the-blank exercises with a full test suite.

## Environment Setup

### Option A — Local (macOS / Linux)

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option B — Google Colab

1. Upload all `.py` files to `/content/` and the `tests/` folder to `/content/tests/`.
   Or clone your repo into Colab.
2. Run the install cell in `notebook.ipynb`, or manually:

```bash
!pip install torch pytest
```

No GPU is required — the tiny model config runs on CPU in seconds.

## Running Tests

All commands assume you are in the project root directory.

### Run a single exercise's tests

```bash
pytest tests/test_rmsnorm.py -v       # Exercise 1
pytest tests/test_rope.py -v          # Exercise 2
pytest tests/test_attention.py -v     # Exercise 3
pytest tests/test_mask.py -v          # Exercise 4
pytest tests/test_feedforward.py -v   # Exercise 5
pytest tests/test_kv_cache.py -v      # Exercise 6
pytest tests/test_sampling.py -v      # Exercise 7
```

### Run the end-to-end test (requires all 7 exercises)

```bash
pytest tests/test_e2e.py -v
```

### Run the full suite

```bash
pytest tests/ -v
```

### Useful pytest flags

| Flag | Description |
|------|-------------|
| `-v` | Verbose — show each test name and pass/fail |
| `-x` | Stop on first failure |
| `-s` | Show print/stdout output |
| `--tb=short` | Shorter tracebacks |
| `-k "test_name"` | Run only tests matching a pattern |

Example — run only the causal mask tests, stop on first failure:

```bash
pytest tests/test_mask.py -v -x
```

## Exercise Order

Work through these in order — later exercises may depend on earlier ones.

| # | Exercise | File to edit | What to implement |
|---|----------|-------------|-------------------|
| 1 | RMSNorm | `model.py` | `RMSNorm.forward` |
| 2 | RoPE | `rope.py` | `precompute_rope_frequencies` + `apply_rope` |
| 3 | Scaled dot-product attention | `model.py` | `scaled_dot_product_attention` |
| 4 | Causal mask | `model.py` | `create_causal_mask` |
| 5 | SwiGLU feed-forward | `model.py` | `FeedForward.forward` |
| 6 | KV cache | `model.py` | `Attention.update_kv_cache` |
| 7 | Top-k / top-p sampling | `generate.py` | `top_k_top_p_filtering` |

Each stub has a `TODO` comment with the math, expected tensor shapes, and
step-by-step instructions. Search for `raise NotImplementedError` to find them.
