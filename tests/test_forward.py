"""Phase 3 gate: our fp32 logits must match the reference within 1e-3.

transformers/torch are the oracle and appear nowhere in tinyinfer/. The check is
run at step one on several prompts rather than only on the final token, because
an attention-mask or RoPE error can leave the last position looking plausible
while every earlier position is wrong.

    python tests/test_forward.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.model import Qwen2  # noqa: E402
from tinyinfer.tokenizer import Tokenizer  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct"
TOL = 1e-3

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "1, 2, 3, 4,",
    "Once upon a time, in a land far away, there lived a",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
]


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM

    tok = Tokenizer(MODEL_DIR / "tokenizer.json")
    ours = Qwen2(MODEL_DIR)

    print("loading reference model...")
    ref = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager"
    )
    ref.eval()

    all_ok = True
    print(f"\n{'prompt':<44} {'seq':>4} {'max abs diff':>13} {'top-1':>7}")
    print("-" * 74)

    for text in PROMPTS:
        ids = tok.encode(text)
        t0 = time.perf_counter()
        got = ours.forward(ids)
        elapsed = time.perf_counter() - t0

        with torch.no_grad():
            want = ref(torch.tensor([ids])).logits[0].numpy()

        diff = np.abs(got - want).max()
        # Argmax agreement at every position, not just the last — a mask bug
        # often leaves the final token correct and earlier ones wrong.
        top1 = (got.argmax(-1) == want.argmax(-1)).mean()

        ok = diff < TOL and top1 == 1.0
        all_ok &= ok
        label = (text[:40] + "...") if len(text) > 43 else text
        print(f"{label!r:<44} {len(ids):>4} {diff:>13.2e} {top1:>6.0%} "
              f"{'ok' if ok else 'FAIL'}  ({elapsed:.2f}s)")

    # Greedy continuation as a human-readable sanity check on top of the numbers.
    ids = tok.encode("The capital of France is")
    for _ in range(8):
        ids.append(int(ours.forward(ids)[-1].argmax()))
    print(f"\ngreedy sample: {tok.decode(ids)!r}")

    print(f"\ntolerance    {TOL:.0e}")
    print("PHASE 3 " + ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
