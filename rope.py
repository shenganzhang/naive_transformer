import torch


def precompute_rope_frequencies(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    TODO: Precompute the complex-valued frequency tensor for Rotary Position Embeddings.

    RoPE encodes position information by rotating pairs of dimensions in the
    query and key vectors. Each consecutive pair of dimensions (2i, 2i+1) is
    rotated by an angle that depends on the token's position and the dimension
    index i. This makes the dot product between Q and K naturally encode
    relative position information.

    Steps:
    1. Compute frequency bands:
         freqs[i] = 1.0 / (theta ^ (2i / head_dim))
       for i = 0, 1, ..., head_dim/2 - 1.
       Use torch.arange(0, head_dim, 2) to get the 2i values, then:
         freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
       Result shape: (head_dim // 2,)

    2. Create position indices:
         t = torch.arange(max_seq_len)
       Shape: (max_seq_len,)

    3. Compute the outer product of positions and frequencies:
         angles = torch.outer(t, freqs)
       Shape: (max_seq_len, head_dim // 2)

    4. Convert to complex exponentials:
         freqs_cis = torch.polar(torch.ones_like(angles), angles)
       This gives cos(angles) + i*sin(angles), i.e., e^(i * angles).
       Shape: (max_seq_len, head_dim // 2), dtype=complex64

    Args:
        head_dim: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        theta: Base for the frequency computation (default 10000.0).

    Returns:
        Complex tensor of shape (max_seq_len, head_dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    angles = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    TODO: Apply rotary positional embeddings to the input tensor.

    This rotates consecutive pairs of dimensions in x by position-dependent
    angles encoded in freqs_cis. Complex multiplication performs 2D rotation:
    if x_pair = (a, b) treated as a+bi, and freq = cos(θ)+i*sin(θ), then
    x_pair * freq rotates (a, b) by angle θ.

    Steps:
    1. Reshape x from (..., head_dim) to (..., head_dim//2, 2):
         x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)

    2. Convert each pair to a complex number:
         x_complex = torch.view_as_complex(x_pairs)
       Each pair (a, b) becomes a + bi.
       Shape: (batch, seq_len, n_heads, head_dim // 2)

    3. Reshape freqs_cis for broadcasting:
         freqs_cis has shape (seq_len, head_dim // 2).
         Reshape to (1, seq_len, 1, head_dim // 2) so it broadcasts over
         batch and n_heads dimensions.

    4. Multiply element-wise (this performs the rotation):
         x_rotated = x_complex * freqs_cis_reshaped

    5. Convert back to real-valued tensor:
         result = torch.view_as_real(x_rotated)  ->  (..., head_dim//2, 2)
         result = result.flatten(-2)              ->  (..., head_dim)
       Then cast back to the original dtype of x.

    Args:
        x: Input tensor of shape (batch, seq_len, n_heads, head_dim).
        freqs_cis: Precomputed complex frequencies, shape (seq_len, head_dim // 2).

    Returns:
        Rotated tensor, same shape and dtype as x.
    """
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_pairs)
    seq_len, head_dim_half = freqs_cis.shape
    freqs_cis_reshaped = freqs_cis.reshape(1, seq_len, 1, head_dim_half)
    x_rotated = x_complex * freqs_cis_reshaped
    result = torch.view_as_real(x_rotated)
    result = result.flatten(-2)
    return result.type_as(x)

