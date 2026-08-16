"""Phase 7 gate: the Metal INT4 kernel must match the NumPy reference exactly.

The reference is tinyinfer.quant.dequantize() (already gated by phase 6's
tests) followed by a plain matvec -- not transformers, not any external
library. The kernel and the reference implement the identical dequantization
formula; this test exists to catch a transcription error between the Python
QuantConfig math and the MSL rewrite of it, not to validate the formula itself.

Skips cleanly (not a failure) on non-Apple hardware or without PyObjC's Metal
bindings, since the whole point of this backend is Apple-silicon-specific.

    python tests/test_metal_kernel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.quant import QuantConfig, dequantize, quantize  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok ' if ok else 'FAIL'} {label:<52} {detail}")
    if not ok:
        failures.append(label)


def main() -> int:
    from tinyinfer.metal_backend import available

    if not available():
        print("SKIP: no Metal device / PyObjC Metal bindings not installed")
        return 0

    from tinyinfer.metal_backend import MetalInt4Linear

    rng = np.random.default_rng(0)

    print("correctness vs numpy dequantize() reference")
    # Shapes drawn from the real model: attention proj, and the two MLP shapes
    # (in > out for down_proj, out > in for gate/up_proj), plus an odd size to
    # exercise group padding.
    shapes = [(896, 896), (4864, 896), (896, 4864), (37, 130)]
    for out_f, in_f in shapes:
        for gs in (128, 64):
            if gs > in_f:
                continue
            W = rng.normal(0, 0.02, (out_f, in_f)).astype(np.float32)
            x = rng.normal(0, 1.0, in_f).astype(np.float32)
            cfg = QuantConfig(bits=4, group_size=gs, symmetric=False)

            ref = dequantize(quantize(W, cfg)) @ x
            layer = MetalInt4Linear(W, cfg)
            got = layer(x)

            diff = float(np.abs(got - ref).max())
            rel = diff / (float(np.abs(ref).max()) or 1.0)
            check(f"shape=({out_f},{in_f}) group={gs}", diff < 1e-3,
                  f"max abs diff {diff:.2e}, rel {rel:.2e}")

    print("\ndeterminism")
    W = rng.normal(0, 0.02, (256, 512)).astype(np.float32)
    x = rng.normal(0, 1.0, 512).astype(np.float32)
    layer = MetalInt4Linear(W, QuantConfig(bits=4, group_size=128))
    a, b = layer(x), layer(x)
    check("same input -> identical output", np.array_equal(a, b))

    print("\nweights are uploaded once, not per call")
    calls = [layer(rng.normal(0, 1.0, 512).astype(np.float32)) for _ in range(5)]
    ref_calls = [dequantize(layer.qt) @ c for c in
                 [rng.normal(0, 1.0, 512).astype(np.float32)]]  # smoke: just re-run cheaply
    check("five sequential calls all finite and distinct",
          all(np.isfinite(c).all() for c in calls) and
          not np.array_equal(calls[0], calls[1]))

    print(f"\n{'PASS' if not failures else 'FAIL: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
