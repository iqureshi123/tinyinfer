"""Regression test for silent float64 promotion.

Phase 7 profiling found that `scores / np.sqrt(head_dim)` promoted the whole
attention path to float64, because np.sqrt returns a float64 *numpy scalar* and
under NEP 50 those promote a float32 array. The output stayed correct — float64
is strictly more precise — so every correctness gate passed while decode ran at
half speed, because the mixed-dtype matmul that followed lost the fast BLAS path.

That is the worst shape a bug can have: invisible to correctness tests, visible
only in a profile. This test makes it visible to the test suite.

    python tests/test_dtypes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer import model as M  # noqa: E402

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok ' if ok else 'FAIL'} {label:<52} {detail}")
    if not ok:
        failures.append(label)


def main() -> int:
    f32 = np.float32

    print("primitives preserve float32")
    x = np.ones((4, 16), dtype=f32)
    w = np.ones(16, dtype=f32)
    check("rms_norm", M.rms_norm(x, w, 1e-6).dtype == f32,
          str(M.rms_norm(x, w, 1e-6).dtype))
    check("silu", M.silu(x).dtype == f32, str(M.silu(x).dtype))
    check("softmax", M.softmax(x).dtype == f32, str(M.softmax(x).dtype))

    cos, sin = M.rope_tables(8, 64, 1e6)
    check("rope_tables cos", cos.dtype == f32, str(cos.dtype))
    check("rope_tables sin", sin.dtype == f32, str(sin.dtype))

    q = np.ones((14, 8, 64), dtype=f32)
    check("apply_rope", M.apply_rope(q, cos, sin).dtype == f32,
          str(M.apply_rope(q, cos, sin).dtype))
    check("repeat_kv", M.repeat_kv(np.ones((2, 8, 64), dtype=f32), 7).dtype == f32)

    print("\nthe specific trap: scaling by a numpy scalar")
    scores = np.ones((14, 8, 8), dtype=f32)
    check("scores * np.float32(1/sqrt(d)) stays f32",
          (scores * np.float32(1.0 / np.sqrt(64))).dtype == f32)
    check("scores / np.sqrt(d) would promote — trap still exists in numpy",
          (scores / np.sqrt(64)).dtype == np.float64,
          "documented, not a failure")

    print("\nfull forward pass end to end")
    model_dir = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct"
    if not model_dir.exists():
        print("  -- skipped (weights not present; CI runs the checks above)")
    else:
        from tinyinfer.model import KVCache, Qwen2

        m = Qwen2(model_dir)
        logits = m.forward([1, 2, 3, 4])
        check("forward() returns float32", logits.dtype == f32, str(logits.dtype))

        cache = KVCache(m.config, capacity=8)
        check("forward() with cache returns float32",
              m.forward([1, 2, 3, 4], cache).dtype == f32)
        check("KV cache stores float32", cache.k[0].dtype == f32)

        # Walk the intermediate values of one block by re-running its ops.
        h = m.w("model.embed_tokens.weight")[[1, 2]].astype(f32)
        c2, s2 = M.rope_tables(2, m.config.head_dim, m.config.rope_theta)
        mask = np.triu(np.full((2, 2), -np.inf, dtype=f32), k=1)
        check("_block output is float32",
              m._block(h, 0, c2, s2, mask).dtype == f32,
              str(m._block(h, 0, c2, s2, mask).dtype))

        # Every resident weight must be float32 — a bf16 or float64 leak here
        # would silently promote whatever it touches.
        bad = [n for n, a in m._cache.items() if a.dtype != f32]
        check("every cached weight is float32", not bad, f"{len(m._cache)} cached")

    print(f"\n{'PASS' if not failures else 'FAIL: ' + ', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
