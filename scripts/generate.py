"""Run the engine. This is the demo entry point.

    python scripts/generate.py "The capital of France is"
    python scripts/generate.py --chat "Explain a KV cache in one sentence."
    python scripts/generate.py --temperature 0.8 --top-p 0.95 --seed 42 "Once upon a time"
    python scripts/generate.py --quant int4 "The capital of France is"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinyinfer.model import KVCache, Qwen2  # noqa: E402
from tinyinfer.quant import QuantConfig  # noqa: E402
from tinyinfer.sampling import Sampler, SamplerConfig  # noqa: E402
from tinyinfer.tokenizer import Tokenizer  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct"

QUANT = {
    "none": None,
    "int8": QuantConfig(bits=8, group_size=128),
    "int4": QuantConfig(bits=4, group_size=128),
    "int4-g64": QuantConfig(bits=4, group_size=64),
}

CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run tinyinfer on a prompt.")
    ap.add_argument("prompt", nargs="+")
    ap.add_argument("-n", "--max-tokens", type=int, default=64)
    ap.add_argument("--chat", action="store_true",
                    help="wrap the prompt in the instruct chat template")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 is greedy (default)")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quant", choices=sorted(QUANT), default="none")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    prompt = " ".join(args.prompt)
    if args.chat:
        prompt = CHAT_TEMPLATE.format(prompt)

    tok = Tokenizer(MODEL_DIR / "tokenizer.json")
    model = Qwen2(MODEL_DIR)

    if QUANT[args.quant] is not None:
        t0 = time.perf_counter()
        info = model.apply_quantization(QUANT[args.quant])
        print(f"[{args.quant}: {info['tensors']} tensors, "
              f"{info['compression']:.2f}x smaller, "
              f"{time.perf_counter() - t0:.1f}s]", file=sys.stderr)

    sampler = None
    if args.temperature > 0:
        sampler = Sampler(SamplerConfig(
            temperature=args.temperature,
            top_k=args.top_k or None,
            top_p=args.top_p if args.top_p < 1.0 else None,
            seed=args.seed,
        ))

    ids = tok.encode(prompt)
    # <|im_end|> and <|endoftext|> both terminate a turn.
    stop = {tok.special.get("<|im_end|>"), tok.special.get("<|endoftext|>")} - {None}

    print(prompt, end="", flush=True)

    # Stream token by token so the demo shows generation happening rather than
    # pausing and dumping a paragraph.
    cache = None if args.no_cache else KVCache(
        model.config, capacity=len(ids) + args.max_tokens)
    pick = sampler if sampler is not None else (lambda lg: int(lg.argmax()))

    t0 = time.perf_counter()
    n = 0
    if cache is not None:
        logits = model.forward(ids, cache)
        for _ in range(args.max_tokens):
            nxt = pick(logits[-1])
            n += 1
            if nxt in stop:
                break
            print(tok.decode([nxt]), end="", flush=True)
            logits = model.forward([nxt], cache)
    else:
        cur = list(ids)
        for _ in range(args.max_tokens):
            nxt = pick(model.forward(cur)[-1])
            n += 1
            if nxt in stop:
                break
            print(tok.decode([nxt]), end="", flush=True)
            cur.append(nxt)

    dt = time.perf_counter() - t0
    print(f"\n\n[{n} tokens in {dt:.2f}s — {n / dt:.1f} tok/s, "
          f"cache={'off' if args.no_cache else 'on'}, quant={args.quant}]",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
