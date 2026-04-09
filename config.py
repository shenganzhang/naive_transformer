from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the decoder-only Transformer model."""

    vocab_size: int = 256
    dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4  # for Grouped Query Attention; set == n_heads for standard MHA
    hidden_dim: int = 1376  # FFN intermediate size (~2.67x dim, typical for SwiGLU)
    max_seq_len: int = 512
    norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @staticmethod
    def tiny() -> "ModelConfig":
        """Tiny config for fast unit tests (runs in seconds on CPU)."""
        return ModelConfig(
            vocab_size=64,
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            hidden_dim=172,
            max_seq_len=128,
        )

    @staticmethod
    def small() -> "ModelConfig":
        """Small config for local experimentation."""
        return ModelConfig(
            vocab_size=256,
            dim=256,
            n_layers=4,
            n_heads=8,
            n_kv_heads=4,
            hidden_dim=688,
            max_seq_len=256,
        )
