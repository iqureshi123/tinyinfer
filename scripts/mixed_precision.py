"""Phase 6b: does the sensitivity result actually buy anything?

The per-layer study says `k_proj` tolerates INT4 and `down_proj` does not. That
is only interesting if acting on it beats uniform quantization at the same
average bit width. This tests that directly: keep the fragile matrix types at
INT8, push the robust ones to INT4, and compare against uniform INT4 on both
axes at once — perplexity and stored bytes.

A mixed scheme that is better on quality but worse on size has proved nothing.
The claim only holds if it wins on the tradeoff.

    python scripts/mixed_precision.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.eval import perplexity  # noqa: E402
from tinyinfer.model import Qwen2  # noqa: E402
from tinyinfer.quant import QuantConfig, quantize  # noqa: E402
from tinyinfer.tokenizer import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
RESULTS = ROOT / "results"

# Ranked worst-to-best by the phase 6 per-layer study.
BY_FRAGILITY = ["mlp.down_proj.weight", "mlp.up_proj.weight",
                "self_attn.v_proj.weight", "mlp.gate_proj.weight",
                "self_attn.o_proj.weight", "self_attn.q_proj.weight",
                "self_attn.k_proj.weight"]

INT4 = QuantConfig(bits=4, group_size=128)
INT8 = QuantConfig(bits=8, group_size=128)


def names_for(suffixes, n_layers):
    return [f"model.layers.{i}.{s}" for i in range(n_layers) for s in suffixes]


def measure(model, tokens, plan, window):
    """plan: list of (config, suffixes). Disjoint sets, applied in order."""
    model.reset_weights()
    stored = fp32 = 0
    for cfg, suffixes in plan:
        names = names_for(suffixes, model.config.num_hidden_layers)
        for name in names:
            qt = quantize(model.w(name), cfg)
            stored += qt.stored_bytes
            fp32 += qt.fp32_bytes
        model.apply_quantization(cfg, names)
    r = perplexity(model, tokens, window=window)
    model.reset_weights()
    # Everything not in the plan stays fp32 and is counted at full size.
    r["stored_mb"] = stored / 2**20
    r["compression"] = fp32 / stored if stored else 1.0
    r["avg_bits"] = 32.0 / r["compression"]
    return r


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    tok = Tokenizer(MODEL_DIR / "tokenizer.json")
    model = Qwen2(MODEL_DIR)
    tokens = tok.encode((ROOT / "data" / "heldout.txt").read_text())[:3072]
    window = 512

    base = perplexity(model, tokens, window=window)
    print(f"held-out {len(tokens):,} tokens, fp32 baseline ppl {base['perplexity']:.4f}\n")
    print(f"{'scheme':<40}{'ppl':>9} {'delta':>9} {'avg bits':>9} {'compress':>10}")
    print("-" * 79)

    out = {"baseline": base, "schemes": {}}

    def row(label, plan):
        r = measure(model, tokens, plan, window)
        d = (r["perplexity"] - base["perplexity"]) / base["perplexity"] * 100
        r["delta_pct"] = d
        out["schemes"][label] = r
        print(f"{label:<40}{r['perplexity']:>9.4f} {d:>+8.2f}% "
              f"{r['avg_bits']:>9.2f} {r['compression']:>9.2f}x")
        return r

    all_types = BY_FRAGILITY
    row("uniform INT8", [(INT8, all_types)])
    uniform4 = row("uniform INT4 g128", [(INT4, all_types)])

    # Promote the k most fragile types to INT8, leave the rest at INT4.
    for k in (1, 2, 3, 4):
        fragile, robust = BY_FRAGILITY[:k], BY_FRAGILITY[k:]
        short = ", ".join(s.split(".")[-2] for s in fragile)
        row(f"INT4 + INT8 on {short}", [(INT8, fragile), (INT4, robust)])

    # A smaller group on the robust half is the alternative way to spend bits.
    row("uniform INT4 g64", [(QuantConfig(bits=4, group_size=64), all_types)])

    print("\ninterpretation")
    best = min(
        (v for k, v in out["schemes"].items() if k.startswith("INT4 + INT8")),
        key=lambda v: v["delta_pct"],
    )
    label = next(k for k, v in out["schemes"].items() if v is best)
    print(f"  uniform INT4  {uniform4['delta_pct']:+.2f}% at {uniform4['avg_bits']:.2f} avg bits")
    print(f"  best mixed    {best['delta_pct']:+.2f}% at {best['avg_bits']:.2f} avg bits"
          f"   ({label})")
    won = best["delta_pct"] < uniform4["delta_pct"]
    print(f"  -> mixed precision {'wins' if won else 'does NOT win'} on quality; "
          f"costs {best['avg_bits'] - uniform4['avg_bits']:+.2f} bits/weight")

    (RESULTS / "mixed_precision.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote results/mixed_precision.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
