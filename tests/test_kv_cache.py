"""Phase 4 gate: the KV cache must be an optimization, not an approximation.

Two things are checked, and the first matters more:

  1. Cached and uncached greedy decoding produce *identical* token ids, and the
     per-step logits agree to within float-reassociation noise. Not similar —
     identical. A cache that is 99% right drifts into different text after a few
     dozen tokens and looks like a sampling bug.
  2. The speedup is measured rather than assumed.

    python tests/test_kv_cache.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.model import KVCache, Qwen2  # noqa: E402
from tinyinfer.tokenizer import Tokenizer  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct"
N_NEW = 32
PROMPT = "The capital of France is"


def main() -> int:
    tok = Tokenizer(MODEL_DIR / "tokenizer.json")
    m = Qwen2(MODEL_DIR)
    prompt_ids = tok.encode(PROMPT)

    # ---- exactness: same ids, and same logits at every step ----
    cache = KVCache(m.config, capacity=len(prompt_ids) + N_NEW)
    cached_logits = [m.forward(prompt_ids, cache)[-1]]
    cached_ids: list[int] = []
    for _ in range(N_NEW):
        nxt = int(cached_logits[-1].argmax())
        cached_ids.append(nxt)
        cached_logits.append(m.forward([nxt], cache)[-1])

    ids = list(prompt_ids)
    plain_logits = [m.forward(ids)[-1]]
    plain_ids: list[int] = []
    for _ in range(N_NEW):
        nxt = int(plain_logits[-1].argmax())
        plain_ids.append(nxt)
        ids.append(nxt)
        plain_logits.append(m.forward(ids)[-1])

    ids_match = cached_ids == plain_ids
    max_diff = max(float(np.abs(a - b).max())
                   for a, b in zip(cached_logits, plain_logits))

    print(f"prompt          {PROMPT!r} ({len(prompt_ids)} tokens)")
    print(f"generated       {N_NEW} tokens")
    print(f"token ids       {'IDENTICAL' if ids_match else 'DIVERGED'}")
    print(f"logit max diff  {max_diff:.2e}  (float reassociation only)")
    print(f"text            {tok.decode(cached_ids)!r}")
    if not ids_match:
        for j, (a, b) in enumerate(zip(cached_ids, plain_ids)):
            if a != b:
                print(f"  first divergence at step {j}: cached={a} plain={b}")
                break

    # ---- speed: measured, not assumed ----
    def timed(use_cache: bool) -> tuple[float, list[int]]:
        t0 = time.perf_counter()
        out = m.generate(prompt_ids, N_NEW, use_cache=use_cache)
        return time.perf_counter() - t0, out

    t_cached, out_cached = timed(True)
    t_plain, out_plain = timed(False)

    print(f"\n{'':16}{'wall':>8} {'tok/s':>8}")
    print(f"{'with cache':<16}{t_cached:>7.2f}s {N_NEW / t_cached:>7.1f}")
    print(f"{'without cache':<16}{t_plain:>7.2f}s {N_NEW / t_plain:>7.1f}")
    print(f"{'speedup':<16}{t_plain / t_cached:>7.2f}x")
    print(f"\ncache size      {KVCache(m.config, len(prompt_ids) + N_NEW).nbytes / 2**20:.1f} MB "
          f"for {len(prompt_ids) + N_NEW} positions")

    ok = ids_match and max_diff < 1e-3 and out_cached == out_plain
    print("\nPHASE 4 " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
