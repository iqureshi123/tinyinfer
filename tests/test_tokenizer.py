"""Phase 2 gate: our tokenizer must produce byte-identical ids to the reference.

The `tokenizers` library is the oracle here and nowhere else in the project.
Doc-1 rule: exact equality on 10,000 strings before the model is touched, because
a tokenizer that is 99.9% right produces output that looks almost correct and
costs days to track down later.

    python tests/test_tokenizer.py
"""

from __future__ import annotations

import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenizers import Tokenizer as RefTokenizer  # noqa: E402

from tinyinfer.tokenizer import Tokenizer  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct"

WORDS = (
    "the quick brown fox jumps over a lazy dog while parsing tensors and "
    "computing attention scores in float32 precision on apple silicon".split()
)
CJK = "日本語のテキスト中文字符한국어테스트"
EMOJI = "😀🎉🚀👨‍👩‍👧‍👦🇨🇦🏳️‍🌈"
ACCENTS = "café naïve résumé Ångström ölçü żółć"
CODE = [
    "def f(x): return x**2",
    "SELECT * FROM t WHERE id = 1;",
    "int main(){printf(\"%d\\n\", 42);}",
    "const a = {b: [1,2,3]};",
    "#include <stdio.h>",
]
EDGE = [
    "", " ", "  ", "\n", "\n\n", "\t", " \n ", "\r\n", "   \n\n   ",
    "a", " a", "a ", "  a  ", "0", "007", "3.14159", "1,000,000",
    "!!!", "...", "?!", "—", "…", "«»", "\\", "//", "/*", "<|>",
    "\x00", "\x1b[0m", "​", "﻿", "é" , "é",  # NFC vs NFD
]
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"]


def build_corpus(n: int, seed: int = 0) -> list[str]:
    """Deterministic mixed corpus: edge cases first, then generated variety."""
    rng = random.Random(seed)
    out: list[str] = []
    out.extend(EDGE)
    out.extend(CODE)
    out.extend(SPECIALS)
    out.extend([CJK, EMOJI, ACCENTS])

    while len(out) < n:
        kind = rng.random()
        if kind < 0.40:  # plain english
            s = " ".join(rng.choices(WORDS, k=rng.randint(1, 24)))
        elif kind < 0.55:  # punctuation and casing chaos
            s = "".join(
                rng.choice([rng.choice(WORDS), rng.choice(" ,.!?;:'\"()[]{}—\n\t"),
                            str(rng.randint(0, 10**6))])
                for _ in range(rng.randint(1, 20))
            )
        elif kind < 0.68:  # unicode mixed into latin
            s = "".join(rng.choice([rng.choice(WORDS), rng.choice(CJK),
                                    rng.choice(EMOJI), rng.choice(ACCENTS), " "])
                        for _ in range(rng.randint(1, 18)))
        elif kind < 0.78:  # raw bytes -> text, exercises byte-level fallback
            b = bytes(rng.randrange(256) for _ in range(rng.randint(1, 40)))
            s = b.decode("utf-8", errors="replace")
        elif kind < 0.86:  # whitespace runs, the classic pretokenizer trap
            s = "".join(rng.choice([" ", "  ", "\n", "\t", rng.choice(WORDS)])
                        for _ in range(rng.randint(1, 16)))
        elif kind < 0.94:  # code-like
            s = rng.choice(CODE) + rng.choice(["", "\n", "  ", ";"])
        else:  # chat template with special tokens embedded
            s = (f"<|im_start|>{rng.choice(['user','assistant','system'])}\n"
                 f"{' '.join(rng.choices(WORDS, k=rng.randint(1,10)))}<|im_end|>\n")
        out.append(s)
    return out[:n]


def main() -> int:
    ours = Tokenizer(MODEL_DIR / "tokenizer.json")
    ref = RefTokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))

    corpus = build_corpus(10_000)
    print(f"corpus       {len(corpus):,} strings")
    print(f"vocab        ours={len(ours):,}  ref={ref.get_vocab_size(True):,}")

    enc_fail, dec_fail = [], []
    total_tokens = 0

    for s in corpus:
        got = ours.encode(s)
        want = ref.encode(s, add_special_tokens=False).ids
        total_tokens += len(got)
        if got != want:
            if len(enc_fail) < 5:
                enc_fail.append((s, got, want))
            continue

        # Round-trip: decode must reproduce the NFC-normalized input exactly.
        back = ours.decode(got)
        expect = unicodedata.normalize("NFC", s)
        if back != expect and len(dec_fail) < 5:
            dec_fail.append((s, back, expect))

    n_enc_ok = len(corpus) - sum(1 for _ in enc_fail) if not enc_fail else None
    print(f"tokens       {total_tokens:,} produced")

    if enc_fail:
        print(f"\nENCODE MISMATCH (showing {len(enc_fail)}):")
        for s, got, want in enc_fail:
            print(f"  input {s!r}")
            print(f"    ours {got}")
            print(f"    ref  {want}")
    else:
        print("encode       10000/10000 exact match vs reference")

    if dec_fail:
        print(f"\nDECODE MISMATCH (showing {len(dec_fail)}):")
        for s, back, expect in dec_fail:
            print(f"  input  {s!r}\n    got  {back!r}\n    want {expect!r}")
    else:
        print("decode       10000/10000 exact round-trip")

    # Every id in the vocab must survive id -> token -> id.
    id_fail = [i for i in range(0, len(ours.vocab), 97)
               if ours.vocab.get(ours.id_to_token.get(i, "\x00\x00"), -1) != i]
    print(f"vocab sweep  {'ok' if not id_fail else f'FAIL {id_fail[:5]}'} "
          f"({len(range(0, len(ours.vocab), 97)):,} ids sampled)")

    ok = not enc_fail and not dec_fail and not id_fail
    print("\nPHASE 2 " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
