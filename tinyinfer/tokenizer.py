"""Byte-level BPE tokenizer for Qwen2, implemented from scratch.

The `tokenizers` / `transformers` libraries are used only as a test oracle in
`tests/`, never here. `regex` is used for the pre-tokenization pattern because it
needs Unicode property escapes (\\p{L}, \\p{N}) that the stdlib `re` lacks — that
is a regex engine, not a tokenizer.

The pipeline, read straight off tokenizer.json:

    1. NFC normalize
    2. split out any special tokens (<|im_start|> etc.) so BPE never sees them
    3. pre-tokenize with the Qwen2 regex — splits on word/number/punct/whitespace
       boundaries so merges can never cross them
    4. byte-level encode: UTF-8 bytes -> a reversible 1:1 unicode alphabet
    5. BPE: repeatedly merge the adjacent pair with the lowest merge rank
    6. map the resulting pieces to ids through vocab

Decoding runs it backwards, and the byte-level alphabet is what makes that
lossless for arbitrary bytes — including partial UTF-8 sequences that a single
token can end in the middle of.
"""

from __future__ import annotations

import functools
import json
import unicodedata
from pathlib import Path

import regex

# Qwen2's pre-tokenizer pattern, verbatim from tokenizer.json.
_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


@functools.lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Map all 256 byte values to unique printable unicode code points.

    GPT-2's trick, inherited by Qwen. BPE operates on characters, but text is
    bytes, and many byte values are control characters or invalid on their own.
    So every byte gets a distinct printable stand-in: the already-printable
    ranges map to themselves, and the remaining 68 bytes are shifted up into
    U+0100+. The mapping is a bijection, so it round-trips exactly.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapped = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            mapped.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(printable, mapped)}


class Tokenizer:
    def __init__(self, tokenizer_json: str | Path):
        spec = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))

        self.normalizer = (spec.get("normalizer") or {}).get("type")
        if self.normalizer not in (None, "NFC"):
            raise ValueError(f"unsupported normalizer {self.normalizer!r}")

        model = spec["model"]
        if model["type"] != "BPE":
            raise ValueError(f"unsupported model type {model['type']!r}")

        self.vocab: dict[str, int] = model["vocab"]
        self.id_to_token: dict[int, str] = {i: t for t, i in self.vocab.items()}

        # merges may be "a b" strings or ["a", "b"] pairs depending on version
        self.merge_ranks: dict[tuple[str, str], int] = {}
        for rank, m in enumerate(model["merges"]):
            a, b = m.split(" ", 1) if isinstance(m, str) else m
            self.merge_ranks[(a, b)] = rank

        # Special tokens are matched before BPE and are never split.
        self.special: dict[str, int] = {}
        for entry in spec.get("added_tokens", []):
            self.special[entry["content"]] = entry["id"]
            self.id_to_token.setdefault(entry["id"], entry["content"])
        # Longest-first so <|im_start|> wins over any shorter prefix.
        self._special_re = (
            regex.compile(
                "|".join(regex.escape(s) for s in sorted(self.special, key=len, reverse=True))
            )
            if self.special
            else None
        )

        self._split_re = regex.compile(_SPLIT_PATTERN)
        self._b2u = bytes_to_unicode()
        self._u2b = {c: b for b, c in self._b2u.items()}
        self._bpe_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ BPE

    def _bpe(self, piece: str) -> list[str]:
        """Merge the lowest-ranked adjacent pair until none remain.

        Rank is merge priority, not frequency — merges.txt is ordered, and rank 0
        must always be applied before rank 1. Picking the *minimum* rank each
        pass is what makes the result deterministic and match the reference.
        """
        cached = self._bpe_cache.get(piece)
        if cached is not None:
            return cached

        symbols = list(piece)
        while len(symbols) >= 2:
            best, best_rank = None, None
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = i, rank
            if best is None:
                break
            symbols[best : best + 2] = [symbols[best] + symbols[best + 1]]

        self._bpe_cache[piece] = symbols
        return symbols

    # --------------------------------------------------------------- encode

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        if self.normalizer == "NFC":
            text = unicodedata.normalize("NFC", text)

        if not allowed_special or self._special_re is None:
            return self._encode_ordinary(text)

        ids: list[int] = []
        pos = 0
        for m in self._special_re.finditer(text):
            ids.extend(self._encode_ordinary(text[pos : m.start()]))
            ids.append(self.special[m.group()])
            pos = m.end()
        ids.extend(self._encode_ordinary(text[pos:]))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self._split_re.findall(text):
            mapped = "".join(self._b2u[b] for b in piece.encode("utf-8"))
            for token in self._bpe(mapped):
                ids.append(self.vocab[token])
        return ids

    # --------------------------------------------------------------- decode

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        out = bytearray()
        for i in ids:
            token = self.id_to_token[i]
            if i in self.special.values() or token in self.special:
                if skip_special:
                    continue
                # Special tokens are literal text, not byte-level encoded.
                out.extend(token.encode("utf-8"))
            else:
                out.extend(self._u2b[c] for c in token)
        return out.decode("utf-8", errors="replace")

    # ----------------------------------------------------------------- misc

    def __len__(self) -> int:
        return len(self.vocab) + len(self.special)
