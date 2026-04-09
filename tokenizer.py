class CharTokenizer:
    """Character-level tokenizer mapping characters to/from integer IDs.

    Uses Unicode code points (modulo vocab_size) as token IDs.
    IDs 0, 1, 2 are reserved for BOS, EOS, and PAD respectively.
    Simple but sufficient for testing the full generation pipeline.
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.bos_id = 0
        self.eos_id = 1
        self.pad_id = 2

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        tokens = [(ord(c) % (self.vocab_size - 3)) + 3 for c in text]
        if add_bos:
            tokens = [self.bos_id] + tokens
        return tokens

    def decode(self, tokens: list[int]) -> str:
        special = {self.bos_id, self.eos_id, self.pad_id}
        return "".join(chr(t) for t in tokens if t not in special and 32 <= t < 127)

    def __len__(self) -> int:
        return self.vocab_size
