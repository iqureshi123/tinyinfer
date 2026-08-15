"""Phase 5 gate: sampling must be reproducible and the truncations must be exact.

Reproducibility is the property that matters. An unseeded sampler makes every
later benchmark unfalsifiable, because you can never re-run the case that broke.

    python tests/test_sampling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.model import Qwen2, softmax  # noqa: E402
from tinyinfer.sampling import Sampler, SamplerConfig  # noqa: E402
from tinyinfer.tokenizer import Tokenizer  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct"


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'ok ' if ok else 'FAIL'} {label:<46} {detail}")
    return ok


def unit_tests() -> bool:
    """Synthetic distributions where the correct answer is known by hand."""
    ok = True
    print("truncation behaviour (synthetic logits)")

    # A distribution with a known, deliberately uneven shape.
    logits = np.log(np.array([0.5, 0.25, 0.15, 0.06, 0.04], dtype=np.float32))

    # top_k=1 must be deterministic regardless of seed.
    picks = {Sampler(SamplerConfig(top_k=1, seed=s))(logits) for s in range(20)}
    ok &= check("top_k=1 always picks argmax", picks == {0}, f"picked {picks}")

    # top_p=0.5 keeps only the first token (cum=0.5 reaches p at index 0).
    picks = {Sampler(SamplerConfig(top_p=0.5, seed=s))(logits) for s in range(20)}
    ok &= check("top_p=0.5 keeps only token 0", picks == {0}, f"picked {sorted(picks)}")

    # top_p=0.8 keeps tokens 0,1 (cum 0.5, 0.75, 0.90 -> index 2 crosses).
    picks = {Sampler(SamplerConfig(top_p=0.8, seed=s))(logits) for s in range(300)}
    ok &= check("top_p=0.8 keeps tokens {0,1,2}", picks <= {0, 1, 2} and len(picks) > 1,
                f"picked {sorted(picks)}")

    # top_k=2 can never emit token 2+.
    picks = {Sampler(SamplerConfig(top_k=2, seed=s))(logits) for s in range(300)}
    ok &= check("top_k=2 never emits token >= 2", picks <= {0, 1}, f"picked {sorted(picks)}")

    # Stricter of k and p wins when both are set.
    picks = {Sampler(SamplerConfig(top_k=4, top_p=0.5, seed=s))(logits) for s in range(50)}
    ok &= check("k=4 + p=0.5 -> p wins", picks == {0}, f"picked {sorted(picks)}")

    # temperature=0 is argmax, not a division by zero.
    s = Sampler(SamplerConfig(temperature=0.0, seed=1))
    ok &= check("temperature=0 is greedy", s(logits) == 0)

    # Empirical frequencies must track the true probabilities at temp=1.
    s = Sampler(SamplerConfig(temperature=1.0, seed=7))
    counts = np.bincount([s(logits) for _ in range(20000)], minlength=5) / 20000
    want = softmax(logits)
    max_err = float(np.abs(counts - want).max())
    ok &= check("untruncated freqs match softmax", max_err < 0.02, f"max err {max_err:.4f}")

    # High temperature must flatten the distribution toward uniform.
    hot = Sampler(SamplerConfig(temperature=100.0, seed=3))
    hot_counts = np.bincount([hot(logits) for _ in range(20000)], minlength=5) / 20000
    ok &= check("temperature=100 approaches uniform",
                float(np.abs(hot_counts - 0.2).max()) < 0.03,
                f"spread {hot_counts.round(3)}")
    return ok


def model_tests() -> bool:
    """End-to-end: the same seed must reproduce the same text, bit for bit."""
    ok = True
    print("\nreproducibility (real model)")

    tok = Tokenizer(MODEL_DIR / "tokenizer.json")
    m = Qwen2(MODEL_DIR)
    ids = tok.encode("Once upon a time")
    cfg = SamplerConfig(temperature=0.8, top_k=50, top_p=0.95, seed=1234)

    a = m.generate(ids, 24, sampler=Sampler(cfg))
    b = m.generate(ids, 24, sampler=Sampler(cfg))
    ok &= check("same seed -> identical ids", a == b)

    other = SamplerConfig(temperature=0.8, top_k=50, top_p=0.95, seed=999)
    c = m.generate(ids, 24, sampler=Sampler(other))
    ok &= check("different seed -> different ids", a != c)

    # Greedy via the sampler must equal greedy via argmax.
    g1 = m.generate(ids, 16, sampler=Sampler(SamplerConfig(temperature=0.0)))
    g2 = m.generate(ids, 16)
    ok &= check("temperature=0 == argmax path", g1 == g2)

    # Cached and uncached sampling must agree given the same seed.
    s1 = m.generate(ids, 16, use_cache=True, sampler=Sampler(cfg))
    s2 = m.generate(ids, 16, use_cache=False, sampler=Sampler(cfg))
    ok &= check("cached == uncached at same seed", s1 == s2)

    print(f"\n  seed 1234: {tok.decode(a)!r}")
    print(f"  seed  999: {tok.decode(c)!r}")
    print(f"  greedy   : {tok.decode(g2)!r}")
    return ok


def main() -> int:
    ok = unit_tests() and model_tests()
    print("\nPHASE 5 " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
