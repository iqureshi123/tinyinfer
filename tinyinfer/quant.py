"""Weight-only group-wise quantization to INT8 and INT4.

Weights are stored in fewer bits; activations stay fp32. That is the standard
tradeoff for inference on a memory-bound model — the 0.5B checkpoint spends far
more time moving weights than doing arithmetic, so shrinking the weights is what
actually buys throughput.

Quantization is applied per *group* of consecutive elements along the input
dimension rather than per tensor. A single scale for a whole 896x4864 matrix is
dominated by its largest outlier and wastes most of the range on values that do
not occur; a scale per 128 elements tracks the local distribution instead. Group
size is the knob that trades metadata overhead against fidelity.

Two schemes:

  symmetric  — scale only, zero maps to zero. Range [-2^(b-1)+1, 2^(b-1)-1].
  asymmetric — scale + zero point, fits [min, max] exactly. Range [0, 2^b - 1].

Asymmetric costs one extra parameter per group and is meaningfully better when
a group's values are lopsided, which is common in the gate/up projections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantConfig:
    bits: int = 8
    group_size: int = 128
    symmetric: bool = False

    def __post_init__(self):
        if self.bits not in (2, 3, 4, 8):
            raise ValueError(f"unsupported bit width {self.bits}")


@dataclass
class QuantizedTensor:
    """Quantized weights plus the parameters needed to reconstruct them."""

    q: np.ndarray            # integer codes, uint8 (unpacked, one code per byte)
    scale: np.ndarray        # (rows, n_groups) float32
    zero: np.ndarray | None  # (rows, n_groups) float32, None if symmetric
    shape: tuple[int, ...]
    config: QuantConfig

    @property
    def stored_bytes(self) -> int:
        """Size if codes were bit-packed, which is what a real engine stores.

        The codes live one-per-byte in memory here for clarity; the honest
        footprint is the packed size plus fp16 scales and zeros.
        """
        n = int(np.prod(self.shape))
        packed = (n * self.config.bits + 7) // 8
        meta = self.scale.size * 2 + (self.zero.size * 2 if self.zero is not None else 0)
        return packed + meta

    @property
    def fp32_bytes(self) -> int:
        return int(np.prod(self.shape)) * 4


def quantize(w: np.ndarray, config: QuantConfig) -> QuantizedTensor:
    """Quantize a 2D weight matrix group-wise along its last axis."""
    if w.ndim != 2:
        raise ValueError(f"expected 2D weight, got {w.shape}")

    rows, cols = w.shape
    gs = config.group_size if config.group_size > 0 else cols

    # Pad the last axis up to a multiple of the group size so every group is
    # full. The padding is dropped again at dequantize time.
    pad = (-cols) % gs
    if pad:
        w = np.concatenate([w, np.zeros((rows, pad), dtype=w.dtype)], axis=1)

    grouped = w.reshape(rows, -1, gs).astype(np.float32)
    qmax = (1 << config.bits) - 1

    if config.symmetric:
        # Symmetric: one scale, codes centred on 2^(b-1).
        amax = np.abs(grouped).max(axis=2)
        half = (1 << (config.bits - 1)) - 1
        scale = np.where(amax == 0, 1.0, amax / half).astype(np.float32)
        zero = None
        codes = np.rint(grouped / scale[:, :, None]) + (1 << (config.bits - 1))
    else:
        # Asymmetric: fit [min, max] exactly. Degenerate (constant) groups get
        # scale 1 so the division is safe and reconstruction is still exact.
        gmin = grouped.min(axis=2)
        gmax = grouped.max(axis=2)
        span = gmax - gmin
        scale = np.where(span == 0, 1.0, span / qmax).astype(np.float32)
        zero = gmin.astype(np.float32)
        codes = np.rint((grouped - zero[:, :, None]) / scale[:, :, None])

    codes = np.clip(codes, 0, qmax).astype(np.uint8)

    return QuantizedTensor(
        q=codes, scale=scale, zero=zero, shape=(rows, cols), config=config
    )


def dequantize(qt: QuantizedTensor) -> np.ndarray:
    """Reconstruct float32 weights. Lossy by construction."""
    cfg = qt.config
    if cfg.symmetric:
        centre = 1 << (cfg.bits - 1)
        out = (qt.q.astype(np.float32) - centre) * qt.scale[:, :, None]
    else:
        out = qt.q.astype(np.float32) * qt.scale[:, :, None] + qt.zero[:, :, None]

    rows, cols = qt.shape
    return out.reshape(rows, -1)[:, :cols]


def roundtrip(w: np.ndarray, config: QuantConfig) -> np.ndarray:
    """Quantize then immediately dequantize — the error this injects is the
    entire quality cost of the scheme."""
    return dequantize(quantize(w, config))


def error_stats(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    """Reconstruction error for one tensor.

    Relative Frobenius error is the headline: it is scale-free, so it can be
    compared across tensors of wildly different magnitude, which per-layer
    sensitivity analysis requires.
    """
    diff = original.astype(np.float32) - reconstructed
    denom = float(np.linalg.norm(original)) or 1.0
    return {
        "rel_fro": float(np.linalg.norm(diff)) / denom,
        "max_abs": float(np.abs(diff).max()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }
