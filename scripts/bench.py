"""Benchmark harness: throughput, latency percentiles, and a stage breakdown.

Reports p50/p99 per-token latency rather than a mean. A mean hides the stalls
that actually determine whether generation feels smooth, and it is the number
most hobby benchmarks quote precisely because it flatters them.

Also measures the KV cache speedup as a function of context length, because the
single-point figure from phase 4 (1.78x at 37 positions) understates it — the
cache removes quadratic work, so the win is a curve, not a constant.

    python scripts/bench.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.model import KVCache, Qwen2  # noqa: E402
from tinyinfer.quant import QuantConfig  # noqa: E402
from tinyinfer.tokenizer import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
RESULTS = ROOT / "results"
PROMPT = "The history of computing began long before the invention of the transistor, and"


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(xs), p))


def decode_latencies(model, prompt_ids, n_new: int) -> tuple[float, list[float]]:
    """Time prefill and each decode step separately.

    They are different regimes: prefill is compute-bound over the whole prompt,
    decode is memory-bound on a single token, and averaging them together
    produces a number that describes neither.
    """
    cache = KVCache(model.config, capacity=len(prompt_ids) + n_new)

    t0 = time.perf_counter()
    logits = model.forward(prompt_ids, cache)
    prefill = time.perf_counter() - t0

    steps: list[float] = []
    for _ in range(n_new):
        nxt = int(logits[-1].argmax())
        t0 = time.perf_counter()
        logits = model.forward([nxt], cache)
        steps.append(time.perf_counter() - t0)
    return prefill, steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    tok = Tokenizer(MODEL_DIR / "tokenizer.json")
    model = Qwen2(MODEL_DIR)
    prompt_ids = tok.encode(PROMPT)
    n_new = 24 if args.quick else 64

    out: dict = {"machine": "Apple M5, 24 GB", "backend": "NumPy fp32",
                 "prompt_tokens": len(prompt_ids), "decode_tokens": n_new}

    # Warm up: first call pays bf16->f32 decode for all 290 tensors, which is a
    # one-time cost and would otherwise be charged to prefill.
    model.forward(prompt_ids[:2])

    print(f"machine        Apple M5, 24 GB")
    print(f"backend        NumPy fp32 (Accelerate BLAS)")
    print(f"prompt         {len(prompt_ids)} tokens, decoding {n_new}\n")

    # ---- throughput and latency percentiles ----
    prefill, steps = decode_latencies(model, prompt_ids, n_new)
    out["fp32"] = {
        "prefill_s": prefill,
        "prefill_tok_s": len(prompt_ids) / prefill,
        "decode_tok_s": 1.0 / float(np.mean(steps)),
        "p50_ms": pct(steps, 50) * 1e3,
        "p99_ms": pct(steps, 99) * 1e3,
        "min_ms": min(steps) * 1e3,
        "max_ms": max(steps) * 1e3,
    }
    f = out["fp32"]
    print(f"{'':16}{'tok/s':>9} {'p50':>9} {'p99':>9} {'max':>9}")
    print("-" * 55)
    print(f"{'prefill':<16}{f['prefill_tok_s']:>9.1f} {'—':>9} {'—':>9} {'—':>9}")
    print(f"{'decode fp32':<16}{f['decode_tok_s']:>9.1f} {f['p50_ms']:>8.1f}ms "
          f"{f['p99_ms']:>8.1f}ms {f['max_ms']:>8.1f}ms")

    # ---- same measurement under quantization ----
    out["quantized"] = {}
    schemes = ([(8, 128)] if args.quick else [(8, 128), (4, 128), (4, 64)])
    for bits, gs in schemes:
        model.reset_weights()
        info = model.apply_quantization(QuantConfig(bits=bits, group_size=gs))
        model.forward(prompt_ids[:2])
        _, s = decode_latencies(model, prompt_ids, n_new)
        label = f"decode INT{bits} g{gs}"
        out["quantized"][label] = {
            "decode_tok_s": 1.0 / float(np.mean(s)),
            "p50_ms": pct(s, 50) * 1e3,
            "p99_ms": pct(s, 99) * 1e3,
            "compression": info["compression"],
        }
        q = out["quantized"][label]
        print(f"{label:<16}{q['decode_tok_s']:>9.1f} {q['p50_ms']:>8.1f}ms "
              f"{q['p99_ms']:>8.1f}ms {'':>9}  ({q['compression']:.2f}x smaller)")
    model.reset_weights()

    print("\n  Note: quantized weights are dequantized to fp32 before the matmul,")
    print("  so these rows measure quality-preserving compression, not speed.")
    print("  Real throughput gains need the INT4 kernel — phase 7.")

    # ---- KV cache speedup as a function of context length ----
    print(f"\n{'context':>9} {'cached':>10} {'uncached':>10} {'speedup':>9}")
    print("-" * 42)
    out["cache_scaling"] = {}
    lengths = [32, 128] if args.quick else [32, 128, 256, 512]
    for ctx in lengths:
        ids = (prompt_ids * (ctx // len(prompt_ids) + 1))[:ctx]
        k = 8
        t0 = time.perf_counter(); model.generate(ids, k, use_cache=True)
        t_c = time.perf_counter() - t0
        t0 = time.perf_counter(); model.generate(ids, k, use_cache=False)
        t_u = time.perf_counter() - t0
        out["cache_scaling"][ctx] = {"cached_s": t_c, "uncached_s": t_u,
                                     "speedup": t_u / t_c}
        print(f"{ctx:>9} {k/t_c:>9.1f}/s {k/t_u:>9.1f}/s {t_u/t_c:>8.2f}x")

    path = RESULTS / ("bench_quick.json" if args.quick else "bench.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
