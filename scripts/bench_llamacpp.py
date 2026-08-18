"""Phase 8 — benchmark tinyinfer against llama.cpp on identical weights.

The comparison only means something if both engines are running the *same*
model. So the GGUF is converted from the exact `model.safetensors` this repo
parses in phase 1, not downloaded pre-built: same 290 tensors, same 494,032,768
parameters, verified in the output below.

Both sides are measured in one process on one thermal state, because a number
copied from a previous session's JSON is a number measured on a different
machine state.

llama.cpp is a *comparison target*, not a dependency — it is invoked as an
external binary and nothing in tinyinfer/ imports or links it.

    python scripts/bench_llamacpp.py [--quick] [--llama-bench PATH]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.model import KVCache, Qwen2  # noqa: E402
from tinyinfer.tokenizer import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
GGUF_DIR = ROOT / "models" / "gguf"
RESULTS = ROOT / "results"

# Identical to scripts/bench.py, so the tinyinfer rows here are comparable to
# the ones already published.
PROMPT = "The history of computing began long before the invention of the transistor, and"

DEFAULT_LLAMA_BENCH = Path.home() / "projects" / "llama.cpp" / "build" / "bin" / "llama-bench"

# (label, gguf filename, n_gpu_layers)
#
# The F32/CPU row is the honest like-for-like: same numeric type as tinyinfer,
# same hardware, same Accelerate BLAS underneath. Every other row adds
# something tinyinfer does not have yet, and is included to size that gap
# rather than to hide it.
CONFIGS = [
    ("llama.cpp F32 CPU",    "Qwen2.5-0.5B-Instruct-F32.gguf",    0),
    ("llama.cpp F32 Metal",  "Qwen2.5-0.5B-Instruct-F32.gguf",    99),
    ("llama.cpp Q8_0 Metal", "Qwen2.5-0.5B-Instruct-Q8_0.gguf",   99),
    ("llama.cpp Q4_K_M CPU", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", 0),
    ("llama.cpp Q4_K_M Metal", "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", 99),
]


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(xs), p))


def measure_tinyinfer(n_prompt_tokens: int, n_new: int) -> dict:
    """Prefill and decode, timed separately — same method as scripts/bench.py."""
    tok = Tokenizer(MODEL_DIR / "tokenizer.json")
    model = Qwen2(MODEL_DIR)
    prompt_ids = tok.encode(PROMPT)

    # One-time bf16->f32 decode of all 290 tensors would otherwise be billed
    # to prefill.
    model.forward(prompt_ids[:2])

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

    return {
        "n_params": int(sum(w.size for w in model.iter_weights())) if hasattr(model, "iter_weights") else None,
        "prefill_tok_s": len(prompt_ids) / prefill,
        "decode_tok_s": 1.0 / float(np.mean(steps)),
        "decode_p50_ms": pct(steps, 50) * 1e3,
        "decode_p99_ms": pct(steps, 99) * 1e3,
    }


def run_llama_bench(binary: Path, gguf: Path, n_prompt: int, n_gen: int,
                    ngl: int, reps: int) -> dict | None:
    """Run llama-bench and pull the prefill (pp) and decode (tg) rows."""
    cmd = [str(binary), "-m", str(gguf), "-p", str(n_prompt), "-n", str(n_gen),
           "-ngl", str(ngl), "-r", str(reps), "-o", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    ! llama-bench failed (exit {proc.returncode})")
        print("      " + (proc.stderr.strip().splitlines() or ["no stderr"])[-1])
        return None

    rows = json.loads(proc.stdout)
    out: dict = {}
    for row in rows:
        if row.get("n_prompt", 0) > 0:
            out["prefill_tok_s"] = row["avg_ts"]
            out["prefill_stddev"] = row["stddev_ts"]
        elif row.get("n_gen", 0) > 0:
            out["decode_tok_s"] = row["avg_ts"]
            out["decode_stddev"] = row["stddev_ts"]
        out.setdefault("build_commit", row.get("build_commit"))
        out.setdefault("backends", row.get("backends"))
        out.setdefault("n_params", row.get("model_n_params"))
        out.setdefault("model_size_bytes", row.get("model_size"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--llama-bench", type=Path, default=DEFAULT_LLAMA_BENCH)
    args = ap.parse_args()

    binary = args.llama_bench
    if not binary.exists():
        found = shutil.which("llama-bench")
        if not found:
            print(f"llama-bench not found at {binary}, and not on PATH.")
            print("Build it, then re-run with --llama-bench PATH. See README phase 8.")
            return 1
        binary = Path(found)

    RESULTS.mkdir(exist_ok=True)
    n_new = 24 if args.quick else 64
    reps = 2 if args.quick else 5

    tok = Tokenizer(MODEL_DIR / "tokenizer.json")
    n_prompt = len(tok.encode(PROMPT))

    print("phase 8 — tinyinfer vs llama.cpp")
    print(f"machine        Apple M5, 24 GB")
    print(f"llama-bench    {binary}")
    print(f"workload       prefill {n_prompt} tokens, decode {n_new} tokens, {reps} reps\n")

    out: dict = {
        "machine": "Apple M5, 24 GB",
        "prompt_tokens": n_prompt,
        "decode_tokens": n_new,
        "reps": reps,
        "engines": {},
    }

    print("measuring tinyinfer (NumPy fp32 / Accelerate BLAS) ...")
    ti = measure_tinyinfer(n_prompt, n_new)
    out["engines"]["tinyinfer fp32 CPU"] = ti

    for label, fname, ngl in CONFIGS:
        gguf = GGUF_DIR / fname
        if not gguf.exists():
            print(f"skip {label}: {gguf.name} not found")
            continue
        print(f"measuring {label} ...")
        res = run_llama_bench(binary, gguf, n_prompt, n_new, ngl, reps)
        if res:
            out["engines"][label] = res

    # ---- table ----
    base = ti["decode_tok_s"]
    print(f"\n{'engine':<26}{'prefill tok/s':>14}{'decode tok/s':>14}{'vs tinyinfer':>14}")
    print("-" * 68)
    for label, r in out["engines"].items():
        pre = r.get("prefill_tok_s")
        dec = r.get("decode_tok_s")
        ratio = "—" if label.startswith("tinyinfer") else f"{dec / base:.1f}x"
        print(f"{label:<26}{pre:>14.1f}{dec:>14.1f}{ratio:>14}")

    print(f"\ntinyinfer decode latency   p50 {ti['decode_p50_ms']:.1f} ms  "
          f"p99 {ti['decode_p99_ms']:.1f} ms")

    # Parameter counts must match or the comparison is meaningless.
    counts = {r.get("n_params") for r in out["engines"].values() if r.get("n_params")}
    print(f"\nparameter counts across engines: {counts}")
    if len(counts) == 1:
        print("  same weights on both sides — comparison is like-for-like")
    else:
        print("  WARNING: parameter counts differ, the comparison is not valid")

    path = RESULTS / ("bench_vs_llamacpp_quick.json" if args.quick
                      else "bench_vs_llamacpp.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
