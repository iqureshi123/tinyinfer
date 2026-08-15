"""Phase 6: quantization as a study, not a checkbox.

Four questions, each answered with a measurement rather than an assertion:

  1. What does perplexity cost at INT8 and INT4, and how does group size move it?
  2. Symmetric or asymmetric — does the extra zero-point earn its storage?
  3. Which *layer types* absorb quantization and which do not?
  4. Which *depths* are most sensitive — early, middle, or late blocks?

Questions 3 and 4 are the interesting ones. Uniform quantization treats every
matrix as equally robust, and it is not.

    python scripts/quant_study.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.eval import perplexity  # noqa: E402
from tinyinfer.model import Qwen2  # noqa: E402
from tinyinfer.quant import QuantConfig  # noqa: E402
from tinyinfer.tokenizer import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
RESULTS = ROOT / "results"

LAYER_GROUPS = {
    "attn.q": ["self_attn.q_proj.weight"],
    "attn.k": ["self_attn.k_proj.weight"],
    "attn.v": ["self_attn.v_proj.weight"],
    "attn.o": ["self_attn.o_proj.weight"],
    "mlp.gate": ["mlp.gate_proj.weight"],
    "mlp.up": ["mlp.up_proj.weight"],
    "mlp.down": ["mlp.down_proj.weight"],
    "attn.all": ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                 "self_attn.v_proj.weight", "self_attn.o_proj.weight"],
    "mlp.all": ["mlp.gate_proj.weight", "mlp.up_proj.weight",
                "mlp.down_proj.weight"],
}


def run(model, tokens, quant_cfg, names, window):
    """Quantize a named subset, measure, then restore full precision."""
    model.reset_weights()
    info = model.apply_quantization(quant_cfg, names) if names else None
    t0 = time.perf_counter()
    ppl = perplexity(model, tokens, window=window)
    ppl["seconds"] = time.perf_counter() - t0
    if info:
        ppl["compression"] = info["compression"]
        ppl["rel_fro_mean"] = sum(
            t["rel_fro"] for t in info["per_tensor"].values()
        ) / len(info["per_tensor"])
    model.reset_weights()
    return ppl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer tokens, fewer configs")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    tok = Tokenizer(MODEL_DIR / "tokenizer.json")
    model = Qwen2(MODEL_DIR)

    text = (ROOT / "data" / "heldout.txt").read_text(encoding="utf-8")
    n_tokens = 1024 if args.quick else 3072
    window = 256 if args.quick else 512
    tokens = tok.encode(text)[:n_tokens]
    print(f"held-out corpus  Alice in Wonderland (public domain), "
          f"{len(tokens):,} tokens, window {window}\n")

    out: dict = {"tokens": len(tokens), "window": window}

    # ---- baseline ----
    base = run(model, tokens, None, None, window)
    out["baseline"] = base
    print(f"{'config':<34}{'ppl':>9} {'delta':>9} {'compress':>10} {'relerr':>9}")
    print("-" * 74)
    print(f"{'fp32 baseline':<34}{base['perplexity']:>9.4f} {'—':>9} "
          f"{'1.00x':>10} {'—':>9}")

    def line(label, r):
        d = (r["perplexity"] - base["perplexity"]) / base["perplexity"] * 100
        print(f"{label:<34}{r['perplexity']:>9.4f} {d:>+8.2f}% "
              f"{r.get('compression', 0):>9.2f}x {r.get('rel_fro_mean', 0):>9.4f}")
        r["delta_pct"] = d
        return r

    # ---- Q1/Q2: bit width x group size x scheme ----
    all_names = model.quantizable_names()
    sweeps = ([(8, 128, False), (4, 128, False), (4, 64, False)] if args.quick
              else [(8, 128, False), (8, 128, True),
                    (4, 256, False), (4, 128, False), (4, 64, False), (4, 32, False),
                    (4, 128, True), (4, 64, True),
                    (3, 64, False), (2, 64, False)])

    out["sweep"] = {}
    for bits, gs, sym in sweeps:
        cfg = QuantConfig(bits=bits, group_size=gs, symmetric=sym)
        label = f"INT{bits} g{gs} {'sym' if sym else 'asym'}"
        out["sweep"][label] = line(label, run(model, tokens, cfg, all_names, window))

    # ---- Q3: which layer types tolerate INT4? ----
    print(f"\n{'INT4 g128 asym applied to only:':<34}")
    cfg4 = QuantConfig(bits=4, group_size=128, symmetric=False)
    out["by_layer_type"] = {}
    for label, suffixes in LAYER_GROUPS.items():
        names = [f"model.layers.{i}.{s}"
                 for i in range(model.config.num_hidden_layers) for s in suffixes]
        out["by_layer_type"][label] = line(f"  {label}", run(model, tokens, cfg4, names, window))

    # ---- Q4: which depths are most sensitive? ----
    n_layers = model.config.num_hidden_layers
    thirds = {"blocks 0-7 (early)": range(0, 8),
              "blocks 8-15 (middle)": range(8, 16),
              "blocks 16-23 (late)": range(16, n_layers)}
    print(f"\n{'INT4 g128 asym applied to only:':<34}")
    out["by_depth"] = {}
    for label, rng in thirds.items():
        names = [f"model.layers.{i}.{s}" for i in rng for s in Qwen2.QUANTIZABLE]
        out["by_depth"][label] = line(f"  {label}", run(model, tokens, cfg4, names, window))

    path = RESULTS / ("quant_study_quick.json" if args.quick else "quant_study.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
