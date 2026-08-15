"""Phase 1: prove we can read the weight file.

Prints every tensor in the checkpoint with its dtype, shape, and size, plus a
summary of the architecture inferred purely from the weights. No inference.

    python scripts/inspect_weights.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.safetensors import SafeTensors  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}GB"


def main() -> int:
    st = SafeTensors(MODEL_DIR / "model.safetensors")
    config = json.loads((MODEL_DIR / "config.json").read_text())

    print(f"file        {st.path.name}")
    print(f"tensors     {len(st)}")
    print(f"parameters  {st.total_params:,}")
    print(f"weight data {human(st.total_bytes)}")
    if st.metadata:
        print(f"metadata    {st.metadata}")
    print()

    # Group the 24 repeated blocks so the listing stays readable: show layer 0
    # in full, then confirm every other layer has an identical shape signature.
    layer_re = re.compile(r"^model\.layers\.(\d+)\.(.*)$")
    top_level, by_layer = [], {}
    for name in st.keys():
        m = layer_re.match(name)
        if m:
            by_layer.setdefault(int(m.group(1)), {})[m.group(2)] = name
        else:
            top_level.append(name)

    def show(name: str, label: str | None = None) -> None:
        t = st.tensors[name]
        shape = "x".join(str(d) for d in t.shape)
        print(f"  {label or name:<44} {t.dtype:<5} {shape:<16} {human(t.nbytes):>8}")

    print("top level")
    for name in sorted(top_level):
        show(name)

    print(f"\nlayer 0 of {len(by_layer)}  (suffix shown, 'model.layers.N.' stripped)")
    for suffix in sorted(by_layer[0]):
        show(by_layer[0][suffix], suffix)

    # Every block must be structurally identical, otherwise the forward pass
    # cannot be a single loop body.
    sig0 = {s: st.tensors[n].shape for s, n in by_layer[0].items()}
    mismatched = [
        i
        for i, mapping in by_layer.items()
        if {s: st.tensors[n].shape for s, n in mapping.items()} != sig0
    ]
    print(
        f"\nlayers identical to layer 0: "
        f"{len(by_layer) - len(mismatched)}/{len(by_layer)}"
        + (f"  MISMATCHED: {mismatched}" if mismatched else "")
    )

    # Cross-check the weights against config.json. If these disagree, the config
    # is not describing this checkpoint and everything downstream is wrong.
    emb = st.tensors["model.embed_tokens.weight"].shape
    hidden = config["hidden_size"]
    n_heads = config["num_attention_heads"]
    n_kv = config["num_key_value_heads"]
    head_dim = hidden // n_heads
    q = st.tensors["model.layers.0.self_attn.q_proj.weight"].shape
    k = st.tensors["model.layers.0.self_attn.k_proj.weight"].shape

    checks = [
        ("vocab_size", emb[0], config["vocab_size"]),
        ("hidden_size", emb[1], hidden),
        ("n_layers", len(by_layer), config["num_hidden_layers"]),
        ("q_proj out", q[0], n_heads * head_dim),
        ("k_proj out", k[0], n_kv * head_dim),
        ("mlp inter", st.tensors["model.layers.0.mlp.gate_proj.weight"].shape[0],
         config["intermediate_size"]),
    ]

    print("\nweights vs config.json")
    ok = True
    for label, got, want in checks:
        flag = "ok " if got == want else "BAD"
        ok &= got == want
        print(f"  {flag} {label:<14} weights={got:<8} config={want}")

    tied = "lm_head.weight" not in st
    print(f"  ok  tied embeddings  lm_head absent={tied} "
          f"config={config['tie_word_embeddings']}")
    ok &= tied == config["tie_word_embeddings"]

    print(f"\narchitecture: {config['num_hidden_layers']} layers, hidden {hidden}, "
          f"{n_heads} heads / {n_kv} kv heads (GQA {n_heads // n_kv}:1), "
          f"head_dim {head_dim}, ffn {config['intermediate_size']}, "
          f"vocab {config['vocab_size']}")

    # Read one tensor end to end to prove decoding works, not just header parsing.
    w = st.get("model.embed_tokens.weight")
    print(f"\nembed_tokens decoded: {w.dtype} {w.shape} "
          f"mean={w.mean():+.5f} std={w.std():.5f} "
          f"min={w.min():+.4f} max={w.max():+.4f}")
    if not np.isfinite(w).all():
        print("  BAD non-finite values in embedding table")
        ok = False

    print("\nPHASE 1 " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import numpy as np  # noqa: E402  (used in main)

    raise SystemExit(main())
