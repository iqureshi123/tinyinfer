"""Does batching fix what killed the kernel at batch=1?

bench_metal.py found the Metal INT4 kernel losing to fp32 BLAS at batch=1, and
traced it to ~0.24ms of fixed per-dispatch overhead that a near-empty kernel
pays just as much as a full one. The batched kernel (int4_matmat.metal) pays
that cost once per call instead of once per row -- this sweeps batch size to
find out where, if anywhere, that actually wins.

    python scripts/bench_metal_batched.py
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

SHAPES = [
    ("q/k/v/o_proj", 896, 896),
    ("gate/up_proj", 4864, 896),
    ("down_proj", 896, 4864),
]
BATCHES = [1, 4, 16, 64, 256]
N_TRIALS = 30


def timed(fn, n=N_TRIALS):
    fn()
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

    print(f"Apple M5 · {N_TRIALS} trials per cell\n")
    out = {"trials": N_TRIALS, "shapes": {}}

    for label, out_f, in_f in SHAPES:
        print(f"{label} ({out_f}x{in_f})")
        print(f"  {'batch':>6}{'fp32 BLAS':>13}{'Metal batched':>16}{'speedup':>10}")
        out["shapes"][label] = {}
        W = rng.normal(0, 0.02, (out_f, in_f)).astype(np.float32)
        layer = MetalInt4Linear(W, cfg)

        for B in BATCHES:
            X = rng.normal(0, 1.0, (B, in_f)).astype(np.float32)
            t_fp32 = timed(lambda: X @ W.T)
            t_metal = timed(lambda: layer.batched(X))
            speedup = t_fp32 / t_metal
            out["shapes"][label][B] = {
                "fp32_ms": t_fp32 * 1e3, "metal_ms": t_metal * 1e3, "speedup": speedup,
            }
            flag = "  <- wins" if speedup > 1.0 else ""
            print(f"  {B:>6}{t_fp32*1e3:>11.3f}ms{t_metal*1e3:>14.3f}ms"
                  f"{speedup:>9.2f}x{flag}")
        print()

    (RESULTS / "bench_metal_batched.json").write_text(json.dumps(out, indent=2))
    print(f"wrote results/bench_metal_batched.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
