"""Qwen2 forward pass in NumPy. Phase 3: correctness only, speed comes later.

Nothing here is imported from PyTorch or transformers. Every operation is
written out so the arithmetic is inspectable, which is the whole point of the
phase — a fast wrong answer teaches nothing.

Architecture (Qwen2.5-0.5B-Instruct, read off config.json in phase 1):

    24 blocks, hidden 896, 14 query heads / 2 kv heads (GQA 7:1), head_dim 64,
    SwiGLU FFN with intermediate 4864, RMSNorm (eps 1e-6), RoPE theta 1e6,
    vocab 151936, embeddings tied to the output projection.

Two details Qwen gets wrong-footed by other implementations:
  - q/k/v projections carry biases; o_proj and the MLP do not.
  - RoPE uses the "rotate half" layout (first half of the head paired with the
    second half), not the interleaved even/odd layout. Getting this backwards
    still produces fluent text — just subtly wrong text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .safetensors import SafeTensors


@dataclass(frozen=True)
class Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_groups(self) -> int:
        """How many query heads share each kv head."""
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text())
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__})


# --------------------------------------------------------------------- ops


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Root-mean-square norm. No mean subtraction, no bias — unlike LayerNorm.

    Computed in float32 even when the input is lower precision, because the
    sum of squares is where precision loss actually shows up.
    """
    x32 = x.astype(np.float32)
    variance = np.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * (1.0 / np.sqrt(variance + eps))) * weight


def silu(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x), computed so large-magnitude inputs cannot overflow."""
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Max-subtracted for stability. Without it, exp() overflows on long contexts."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def rope_tables(seq_len: int, head_dim: int, theta: float,
                offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Precompute cos/sin for rotary position embeddings.

    `offset` is the absolute position of the first token, which matters once a
    KV cache is in play and each step feeds in a single token at position N.
    """
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    pos = np.arange(offset, offset + seq_len, dtype=np.float32)[:, None]
    freqs = pos * inv_freq[None, :]              # (seq, head_dim/2)
    emb = np.concatenate([freqs, freqs], axis=-1)  # (seq, head_dim) — rotate-half layout
    return np.cos(emb), np.sin(emb)


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x is (heads, seq, head_dim); cos/sin are (seq, head_dim)."""
    return x * cos[None, :, :] + rotate_half(x) * sin[None, :, :]


def repeat_kv(x: np.ndarray, groups: int) -> np.ndarray:
    """Expand kv heads to match query heads for grouped-query attention.

    (kv_heads, seq, dim) -> (kv_heads * groups, seq, dim), each kv head repeated
    contiguously so head i of the query maps to kv head i // groups.
    """
    if groups == 1:
        return x
    return np.repeat(x, groups, axis=0)


# ------------------------------------------------------------------- model


class Qwen2:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self.config = Config.load(self.dir / "config.json")
        self.st = SafeTensors(self.dir / "model.safetensors")
        self._cache: dict[str, np.ndarray] = {}

    def w(self, name: str) -> np.ndarray:
        """Fetch a weight, decoding bf16 -> f32 once and keeping it resident."""
        arr = self._cache.get(name)
        if arr is None:
            arr = np.ascontiguousarray(self.st.get(name))
            self._cache[name] = arr
        return arr

    def _block(self, h: np.ndarray, i: int, cos: np.ndarray,
               sin: np.ndarray, mask: np.ndarray) -> np.ndarray:
        cfg = self.config
        p = f"model.layers.{i}."
        seq = h.shape[0]
        n_q, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

        # ---- self-attention (pre-norm) ----
        x = rms_norm(h, self.w(p + "input_layernorm.weight"), cfg.rms_norm_eps)

        q = x @ self.w(p + "self_attn.q_proj.weight").T + self.w(p + "self_attn.q_proj.bias")
        k = x @ self.w(p + "self_attn.k_proj.weight").T + self.w(p + "self_attn.k_proj.bias")
        v = x @ self.w(p + "self_attn.v_proj.weight").T + self.w(p + "self_attn.v_proj.bias")

        # (seq, heads*dim) -> (heads, seq, dim)
        q = q.reshape(seq, n_q, hd).transpose(1, 0, 2)
        k = k.reshape(seq, n_kv, hd).transpose(1, 0, 2)
        v = v.reshape(seq, n_kv, hd).transpose(1, 0, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = repeat_kv(k, cfg.kv_groups)
        v = repeat_kv(v, cfg.kv_groups)

        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)
        scores = scores + mask
        attn = softmax(scores, axis=-1) @ v                      # (heads, seq, dim)

        attn = attn.transpose(1, 0, 2).reshape(seq, n_q * hd)
        h = h + attn @ self.w(p + "self_attn.o_proj.weight").T

        # ---- SwiGLU feed-forward (pre-norm) ----
        x = rms_norm(h, self.w(p + "post_attention_layernorm.weight"), cfg.rms_norm_eps)
        gate = silu(x @ self.w(p + "mlp.gate_proj.weight").T)
        up = x @ self.w(p + "mlp.up_proj.weight").T
        h = h + (gate * up) @ self.w(p + "mlp.down_proj.weight").T

        return h

    def forward(self, token_ids: list[int]) -> np.ndarray:
        """Run the full stack. Returns logits of shape (seq, vocab)."""
        cfg = self.config
        seq = len(token_ids)

        h = self.w("model.embed_tokens.weight")[token_ids].astype(np.float32)

        cos, sin = rope_tables(seq, cfg.head_dim, cfg.rope_theta)

        # Causal mask: -inf above the diagonal, added to scores before softmax.
        mask = np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), k=1)

        for i in range(cfg.num_hidden_layers):
            h = self._block(h, i, cos, sin, mask)

        h = rms_norm(h, self.w("model.norm.weight"), cfg.rms_norm_eps)

        # Tied embeddings: the output projection is the transposed input table.
        head = ("model.embed_tokens.weight" if cfg.tie_word_embeddings
                else "lm_head.weight")
        return h @ self.w(head).T
