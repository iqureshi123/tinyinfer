"""Phase 7 step 1: find the bottleneck before optimizing anything.

Doc-1 rule, and it is the right one: do not touch a kernel until there is data
saying which kernel matters. This instruments a single decode step and attributes
wall time to each operation class, then converts that into the arithmetic that
would have to move to the GPU to matter.

    python scripts/profile_forward.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer import model as M  # noqa: E402
from tinyinfer.model import KVCache, Qwen2  # noqa: E402
from tinyinfer.tokenizer import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
RESULTS = ROOT / "results"

timings: dict[str, float] = defaultdict(float)
counts: dict[str, int] = defaultdict(int)


@contextmanager
def stage(name: str):
    t0 = time.perf_counter()
    yield
    timings[name] += time.perf_counter() - t0
    counts[name] += 1


class ProfiledQwen2(Qwen2):
    """Same arithmetic, wrapped in timers.

    Subclassed rather than edited in place so the profiler cannot slow down or
    subtly alter the code path that ships.
    """

    def _block(self, h, i, cos, sin, mask, cache=None):
        cfg = self.config
        p = f"model.layers.{i}."
        seq = h.shape[0]
        n_q, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

        with stage("rmsnorm"):
            x = M.rms_norm(h, self.w(p + "input_layernorm.weight"), cfg.rms_norm_eps)

        with stage("qkv_proj"):
            q = x @ self.w(p + "self_attn.q_proj.weight").T + self.w(p + "self_attn.q_proj.bias")
            k = x @ self.w(p + "self_attn.k_proj.weight").T + self.w(p + "self_attn.k_proj.bias")
            v = x @ self.w(p + "self_attn.v_proj.weight").T + self.w(p + "self_attn.v_proj.bias")
            q = q.reshape(seq, n_q, hd).transpose(1, 0, 2)
            k = k.reshape(seq, n_kv, hd).transpose(1, 0, 2)
            v = v.reshape(seq, n_kv, hd).transpose(1, 0, 2)

        with stage("rope"):
            q = M.apply_rope(q, cos, sin)
            k = M.apply_rope(k, cos, sin)

        with stage("kv_cache"):
            if cache is not None:
                k, v = cache.append(i, k, v)
            k = M.repeat_kv(k, cfg.kv_groups)
            v = M.repeat_kv(v, cfg.kv_groups)

        with stage("attention"):
            scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)
            scores = scores + mask
            attn = M.softmax(scores, axis=-1) @ v
            attn = attn.transpose(1, 0, 2).reshape(seq, n_q * hd)

        with stage("o_proj"):
            h = h + attn @ self.w(p + "self_attn.o_proj.weight").T

        with stage("rmsnorm"):
            x = M.rms_norm(h, self.w(p + "post_attention_layernorm.weight"),
                           cfg.rms_norm_eps)

        with stage("mlp"):
            gate = M.silu(x @ self.w(p + "mlp.gate_proj.weight").T)
            up = x @ self.w(p + "mlp.up_proj.weight").T
            h = h + (gate * up) @ self.w(p + "mlp.down_proj.weight").T

        return h

    def forward(self, token_ids, cache=None):
        cfg = self.config
        seq = len(token_ids)
        past = cache.length if cache is not None else 0

        with stage("embed"):
            h = self.w("model.embed_tokens.weight")[token_ids].astype(np.float32)
            cos, sin = M.rope_tables(seq, cfg.head_dim, cfg.rope_theta, offset=past)
            mask = np.zeros((seq, past + seq), dtype=np.float32)
            mask[:, past:] = np.triu(
                np.full((seq, seq), -np.inf, dtype=np.float32), k=1)

        for i in range(cfg.num_hidden_layers):
            h = self._block(h, i, cos, sin, mask, cache)

        if cache is not None:
            cache.commit(seq)

        with stage("final_norm"):
            h = M.rms_norm(h, self.w("model.norm.weight"), cfg.rms_norm_eps)

        with stage("lm_head"):
            head = ("model.embed_tokens.weight" if cfg.tie_word_embeddings
                    else "lm_head.weight")
            return h @ self.w(head).T


def flops(cfg, seq: int, past: int) -> dict[str, float]:
    """Multiply-accumulate counts per decode step, x2 for FLOPs.

    Reported alongside wall time so the profile distinguishes 'slow because it
    is a lot of arithmetic' from 'slow because the implementation is bad'.
    """
    H, L = cfg.hidden_size, cfg.num_hidden_layers
    I, V = cfg.intermediate_size, cfg.vocab_size
    hd, nq, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    ctx = past + seq
    return {
        "qkv_proj": 2 * L * seq * H * (nq * hd + 2 * nkv * hd),
        "attention": 2 * L * seq * nq * hd * ctx * 2,   # scores + weighted sum
        "o_proj": 2 * L * seq * H * H,
        "mlp": 2 * L * seq * H * I * 3,
        "lm_head": 2 * seq * H * V,
    }


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    tok = Tokenizer(MODEL_DIR / "tokenizer.json")
    m = ProfiledQwen2(MODEL_DIR)
    ids = tok.encode("The history of computing began long before the transistor, and")

    # Warm up: first call decodes bf16->f32 for all 290 tensors.
    warm = KVCache(m.config, capacity=len(ids) + 4)
    m.forward(ids, warm)
    timings.clear(); counts.clear()

    n_steps = 32
    cache = KVCache(m.config, capacity=len(ids) + n_steps + 1)
    t0 = time.perf_counter()
    logits = m.forward(ids, cache)
    prefill_wall = time.perf_counter() - t0
    prefill = dict(timings)
    timings.clear(); counts.clear()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        nxt = int(logits[-1].argmax())
        logits = m.forward([nxt], cache)
    decode_wall = time.perf_counter() - t0

    print(f"model     Qwen2.5-0.5B ({m.config.num_hidden_layers} layers, "
          f"hidden {m.config.hidden_size})")
    print(f"prompt    {len(ids)} tokens | decode {n_steps} steps\n")

    def table(label, t, wall, seq, past, steps):
        total = sum(t.values())
        fl = flops(m.config, seq, past)
        print(f"{label}  ({wall * 1e3:.1f} ms wall, {total * 1e3:.1f} ms accounted)")
        print(f"  {'stage':<14}{'ms':>9} {'%':>7} {'GFLOP/s':>10}")
        print("  " + "-" * 42)
        rows = {}
        for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
            pct = v / total * 100
            gf = (fl.get(k, 0) * steps / v / 1e9) if k in fl and v > 0 else None
            print(f"  {k:<14}{v * 1e3:>9.1f} {pct:>6.1f}% "
                  f"{gf:>10.1f}" if gf else
                  f"  {k:<14}{v * 1e3:>9.1f} {pct:>6.1f}% {'—':>10}")
            rows[k] = {"ms": v * 1e3, "pct": pct, "gflops": gf}
        print(f"  {'unaccounted':<14}{(wall - total) * 1e3:>9.1f} "
              f"{(wall - total) / wall * 100:>6.1f}%\n")
        return rows

    out = {
        "prefill": table("PREFILL", prefill, prefill_wall, len(ids), 0, 1),
        "decode": table(f"DECODE ({n_steps} steps)", dict(timings), decode_wall,
                        1, len(ids), n_steps),
        "decode_tok_s": n_steps / decode_wall,
    }

    # The headline: what fraction of decode is matmul, and is it worth a kernel?
    d = out["decode"]
    matmul = sum(d[k]["ms"] for k in ("qkv_proj", "o_proj", "mlp", "lm_head") if k in d)
    total_ms = sum(v["ms"] for v in d.values())
    print("conclusion")
    print(f"  matmul is {matmul / total_ms * 100:.1f}% of accounted decode time")
    print(f"  largest single stage: {max(d, key=lambda k: d[k]['ms'])} "
          f"({max(v['pct'] for v in d.values()):.1f}%)")
    print(f"  -> a GPU kernel is only worth writing for the matmul path; "
          f"Amdahl caps any\n     other optimization at "
          f"{100 - matmul / total_ms * 100:.1f}% of the time.")
    out["matmul_pct"] = matmul / total_ms * 100

    (RESULTS / "profile.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/profile.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
