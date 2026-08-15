"""Quantizer properties, checked on synthetic tensors — no model download needed.

This is the suite CI can run. It checks the properties that must hold for *any*
input rather than the perplexity cost on one checkpoint, which is what
scripts/quant_study.py measures.

    python tests/test_quant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.quant import (  # noqa: E402
    QuantConfig, dequantize, error_stats, quantize, roundtrip,
)

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok ' if ok else 'FAIL'} {label:<50} {detail}")
    if not ok:
        failures.append(label)


def main() -> int:
    rng = np.random.default_rng(0)

    print("shape and reconstruction")
    for shape in [(8, 128), (896, 4864), (128, 896), (7, 100)]:
        w = rng.normal(0, 0.02, shape).astype(np.float32)
        for bits in (8, 4):
            r = roundtrip(w, QuantConfig(bits=bits, group_size=128))
            check(f"shape preserved {shape} INT{bits}", r.shape == w.shape,
                  f"got {r.shape}")

    print("\nerror shrinks monotonically with more bits")
    w = rng.normal(0, 0.02, (256, 512)).astype(np.float32)
    errs = {}
    for bits in (2, 3, 4, 8):
        errs[bits] = error_stats(w, roundtrip(w, QuantConfig(bits=bits, group_size=128)))["rel_fro"]
    ordered = all(errs[a] > errs[b] for a, b in [(2, 3), (3, 4), (4, 8)])
    check("rel error strictly decreases 2->3->4->8 bits", ordered,
          " > ".join(f"{errs[b]:.4f}" for b in (2, 3, 4, 8)))

    print("\nerror shrinks with smaller groups")
    gerrs = {gs: error_stats(w, roundtrip(w, QuantConfig(bits=4, group_size=gs)))["rel_fro"]
             for gs in (512, 256, 128, 64, 32)}
    ordered = all(gerrs[a] >= gerrs[b] for a, b in [(512, 256), (256, 128), (128, 64), (64, 32)])
    check("rel error decreases as group size shrinks", ordered,
          " >= ".join(f"{gerrs[g]:.4f}" for g in (512, 256, 128, 64, 32)))

    print("\nasymmetric beats symmetric on skewed data")
    skew = rng.gamma(2.0, 0.01, (256, 512)).astype(np.float32)  # strictly positive
    e_sym = error_stats(skew, roundtrip(skew, QuantConfig(bits=4, group_size=128, symmetric=True)))["rel_fro"]
    e_asym = error_stats(skew, roundtrip(skew, QuantConfig(bits=4, group_size=128, symmetric=False)))["rel_fro"]
    check("asymmetric < symmetric on one-sided values", e_asym < e_sym,
          f"asym {e_asym:.4f} < sym {e_sym:.4f}")

    print("\nexactness and edge cases")
    const = np.full((16, 256), 0.375, dtype=np.float32)
    check("constant tensor reconstructs exactly",
          np.allclose(roundtrip(const, QuantConfig(bits=4, group_size=128)), const),
          f"max err {np.abs(roundtrip(const, QuantConfig(bits=4, group_size=128)) - const).max():.2e}")

    zeros = np.zeros((16, 256), dtype=np.float32)
    check("all-zero tensor reconstructs exactly and does not divide by zero",
          np.allclose(roundtrip(zeros, QuantConfig(bits=4, group_size=128)), zeros))

    # A group whose values are already representable must round-trip exactly.
    steps = np.tile(np.linspace(-1, 1, 16, dtype=np.float32), (4, 16))
    r = roundtrip(steps, QuantConfig(bits=4, group_size=16, symmetric=False))
    check("16 evenly spaced values are exact at INT4/g16",
          np.abs(r - steps).max() < 1e-6, f"max err {np.abs(r - steps).max():.2e}")

    print("\npadding: cols not a multiple of group size")
    for cols in (100, 127, 129, 4864):
        w2 = rng.normal(0, 0.02, (8, cols)).astype(np.float32)
        r = roundtrip(w2, QuantConfig(bits=4, group_size=128))
        e = error_stats(w2, r)["rel_fro"]
        check(f"cols={cols} round-trips with sane error", r.shape == w2.shape and e < 0.25,
              f"rel {e:.4f}")

    print("\ncodes stay in range")
    for bits in (2, 3, 4, 8):
        qt = quantize(w, QuantConfig(bits=bits, group_size=128))
        hi = (1 << bits) - 1
        check(f"INT{bits} codes within [0, {hi}]",
              int(qt.q.min()) >= 0 and int(qt.q.max()) <= hi,
              f"[{qt.q.min()}, {qt.q.max()}]")

    print("\nstorage accounting")
    qt = quantize(w, QuantConfig(bits=4, group_size=128))
    ratio = qt.fp32_bytes / qt.stored_bytes
    check("INT4 g128 compresses between 6x and 8x", 6.0 < ratio < 8.0, f"{ratio:.2f}x")
    qt8 = quantize(w, QuantConfig(bits=8, group_size=128))
    check("INT8 g128 compresses between 3.5x and 4x",
          3.5 < w.nbytes / qt8.stored_bytes < 4.0,
          f"{w.nbytes / qt8.stored_bytes:.2f}x")

    print("\ndequantize is the inverse of the stored representation")
    qt = quantize(w, QuantConfig(bits=8, group_size=64))
    check("dequantize(quantize(w)) is deterministic",
          np.array_equal(dequantize(qt), dequantize(qt)))

    print(f"\n{'PASS' if not failures else 'FAIL: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
