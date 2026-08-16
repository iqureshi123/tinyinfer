"""Does the Metal kernel actually beat the CPU path? Measured, not assumed.

Three things are compared for each of the model's real matrix shapes, at
batch=1 (the decode shape):

  1. fp32 BLAS matvec        -- the current production path
  2. CPU dequant + BLAS matvec -- what apply_quantization() does today:
                                   materialize fp32, then matmul
  3. Metal INT4 kernel        -- fused dequant+matvec on the GPU, weights
                                   already resident

(2) exists to separate two different claims. "GPU beats CPU" and "fused
dequant beats materialize-then-multiply" are different wins, and collapsing
them into one comparison would hide which one is actually doing the work.

    python scripts/bench_metal.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.quant import QuantConfig, dequantize, quantize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# (label, out_features, in_features) -- the three distinct matmul shapes that
# appear in every block: attention projections, and the two MLP directions.
SHAPES = [
    ("q/k/v/o_proj", 896, 896),
    ("gate/up_proj", 4864, 896),
    ("down_proj", 896, 4864),
]
N_TRIALS = 200


def timed(fn, n=N_TRIALS):
    fn()  # warm up (allocations, first-call overhead)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> int:
    from tinyinfer.metal_backend import MetalInt4Linear, available

    RESULTS.mkdir(exist_ok=True)
    if not available():
        print("SKIP: no Metal device available on this machine")
        return 0

    rng = np.random.default_rng(0)
    cfg = QuantConfig(bits=4, group_size=128, symmetric=False)

    print(f"Apple M5 · batch=1 (decode shape) · {N_TRIALS} trials per cell\n")
    print(f"{'shape':<16}{'fp32 BLAS':>12}{'CPU dequant':>14}{'Metal INT4':>13}"
          f"{'vs fp32':>10}{'vs dequant':>12}")
    print("-" * 79)

    out = {"trials": N_TRIALS, "shapes": {}}

    for label, out_f, in_f in SHAPES:
        W = rng.normal(0, 0.02, (out_f, in_f)).astype(np.float32)
        x = rng.normal(0, 1.0, in_f).astype(np.float32)
        qt = quantize(W, cfg)

        t_fp32 = timed(lambda: x @ W.T)

        # The path apply_quantization() actually takes today: dequantize once
        # per call, matches how it would be used in a naive integration.
        t_dequant = timed(lambda: x @ dequantize(qt).T)

        layer = MetalInt4Linear(W, cfg)
        t_metal = timed(lambda: layer(x))

        out["shapes"][label] = {
            "out_features": out_f, "in_features": in_f,
            "fp32_ms": t_fp32 * 1e3, "cpu_dequant_ms": t_dequant * 1e3,
            "metal_ms": t_metal * 1e3,
            "speedup_vs_fp32": t_fp32 / t_metal,
            "speedup_vs_dequant": t_dequant / t_metal,
        }
        print(f"{label:<16}{t_fp32*1e3:>10.3f}ms{t_dequant*1e3:>12.3f}ms"
              f"{t_metal*1e3:>11.3f}ms{t_fp32/t_metal:>9.2f}x{t_dequant/t_metal:>11.2f}x")

    (RESULTS / "bench_metal.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/bench_metal.json")
    print("\nNote: 'CPU dequant' materializes the full fp32 matrix every call --")
    print("that is what apply_quantization() does today. If Metal beats fp32 BLAS")
    print("but loses to CPU dequant, the win is memory bandwidth, not the GPU;")
    print("if it beats both, fusing dequant into the matmul is what's paying off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
