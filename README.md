# tinyinfer

An LLM inference engine written from scratch — no PyTorch `generate()`, no `transformers`, no `llama.cpp`. The safetensors parser, the tokenizer, the forward pass, the KV cache, the sampler, and the quantizer are all implemented here.

Target model: **Qwen2.5-0.5B-Instruct**. Target hardware: **Apple M5, 24 GB**.

```
$ python -c "from tinyinfer.model import Qwen2; from tinyinfer.tokenizer import Tokenizer; ..."
'The capital of France is' -> ' Paris. It is the largest city in Europe and the third
 largest city in the world. It is located in the south of France, on the banks of the'
```

---

## Status

| Phase | State | Result |
|---|---|---|
| 1. Parse safetensors | ✅ | 290 tensors, 494,032,768 params, every shape cross-checked against config |
| 2. BPE tokenizer | ✅ | 10000/10000 exact id match and exact round-trip vs reference |
| 3. fp32 forward pass | ✅ | logits within 4.44e-04 of reference (tol 1e-3), top-1 100% at every position |
| 4. KV cache | ✅ | output ids **identical** to uncached, 1.78× at 37 positions |
| 5. Sampling | ✅ | temperature / top-k / top-p, seeded and reproducible |
| 6. Quantization + study | ✅ | INT8 free, INT4 costly, and the cost is **not uniform** — see below |
| 7. Metal compute kernels | ⛔ | not started |
| 8. Benchmark vs llama.cpp | ⛔ | not started |

Phases 1–6 are Python + NumPy, correctness first. Phases 7–8 move the hot path to C++/Metal.

## The finding

Quantization is usually reported as one number per bit width. That hides the more useful result: **weight matrices do not degrade equally.** Quantizing only `k_proj` to INT4 is nearly free; quantizing only `down_proj` costs almost eighty times more perplexity, for the same bits saved.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/quant_sensitivity_dark.svg">
  <img alt="Perplexity cost of INT4 quantization by weight matrix type" src="results/quant_sensitivity_light.svg">
</picture>

Three things fall out of it:

- **INT8 is free.** Perplexity moves by a fraction of a percent at ~4× compression. There is no reason to ship fp32 weights.
- **INT4 is not free, and group size is the lever.** Halving the group from 128 to 64 recovers a large share of the loss for a few percent more metadata.
- **MLP matrices are the fragile ones.** Attention as a whole absorbs INT4 far better than the feed-forward block, and `down_proj` is the single worst matrix in the model — which is the argument for mixed-precision rather than uniform quantization.
- **Depth barely matters.** Early, middle, and late blocks land within about a percentage point of each other. This is a *negative* result and is reported because it rules out a plausible and commonly assumed strategy.

Full numbers: [`results/quant_study.json`](results/quant_study.json). Reproduce with `python scripts/quant_study.py`.

## How it works

```
tinyinfer/
├── safetensors.py  — hand-rolled reader: u64 header length, JSON header, raw
│                     data region. bf16 -> f32 by a 16-bit left shift, because
│                     bf16 is exactly the top half of an f32.
├── tokenizer.py    — byte-level BPE: NFC, special-token split, the Qwen2
│                     pre-tokenization regex, the GPT-2 byte<->unicode
│                     bijection, rank-ordered merging.
├── model.py        — RMSNorm, rotate-half RoPE, grouped-query attention,
│                     SwiGLU, causal masking, tied embeddings, KV cache.
├── sampling.py     — temperature, top-k, top-p, seedable RNG.
├── quant.py        — group-wise INT2/3/4/8, symmetric and asymmetric.
└── eval.py         — sliding-window perplexity.
```

### Details worth knowing

**RoPE uses the rotate-half layout, not interleaved.** Getting this backwards still produces fluent text — just subtly wrong text. It is caught by comparing logits at step one, not by reading the output.

**The KV cache stores rotated keys.** That is what makes it correct: RoPE depends on absolute position, so a key rotated when it was written stays valid forever.

**Cache storage is preallocated, not grown.** Appending with `np.concatenate` recopies the whole history on every token, which quietly reintroduces the quadratic cost the cache exists to remove.

**Quantization is group-wise, not per-tensor.** One scale for an 896×4864 matrix is set by its largest outlier and wastes most of its range on values that never occur.

**Perplexity windows overlap.** Only newly-exposed tokens are scored, so tokens near a window boundary are not penalised for having no context, and the result does not depend on window size.

## Verification

Every phase has a gate that had to pass before the next one started. Reference libraries appear **only** in `tests/`, never in `tinyinfer/`.

```bash
python scripts/inspect_weights.py   # phase 1
python tests/test_tokenizer.py      # phase 2 — 10,000 strings, exact
python tests/test_forward.py        # phase 3 — logits vs transformers
python tests/test_kv_cache.py       # phase 4 — cached == uncached
python tests/test_sampling.py       # phase 5 — reproducibility
python scripts/quant_study.py       # phase 6 — the study
```

The tokenizer corpus covers whitespace runs, CJK, ZWJ emoji sequences, NFC/NFD pairs, raw-byte garbage, code, and embedded chat-template tokens — the cases where a byte-level BPE actually breaks.

## Limitations

- **No GPU path yet.** Everything runs on NumPy over Accelerate BLAS. The tokens/sec figures are what a correctness-first CPU implementation gives you, and are not competitive with llama.cpp. That comparison is phase 8 and will be published whether or not it wins.
- **Quantized weights are dequantized to fp32 before the matmul.** That measures the quality cost exactly, which is what the study is for, but it means quantization currently buys compression and not speed. Real throughput needs the INT4 kernel.
- **The 1.78× cache speedup is measured at 37 positions**, where the quadratic term barely dominates. `scripts/bench.py` measures the curve across context lengths; the single number understates the win at realistic lengths.
- **One model, one machine.** Everything here is Qwen2.5-0.5B on an M5. Nothing has been checked against a second architecture or a non-Apple target.
- **Perplexity is measured on one corpus** (Alice in Wonderland). It is public-domain, shipped in the repo, and reproducible offline — but a single domain, and the quantization result could differ on code or multilingual text.

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install numpy regex
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/Qwen2.5-0.5B-Instruct')"
```

`tokenizers`, `torch`, and `transformers` are needed only to run the tests, where they serve as oracles.

## Build log

[`LOG.md`](LOG.md) — every bug that cost more than an hour, what it looked like, and what it actually was.

## Prior work

A direct sequel to [chip8](https://github.com/iqureshi123/chip8): parse a binary format, implement a spec exactly, verify headlessly against a reference, then optimize. Same discipline, pointed at a transformer instead of a 1970s VM.

## License

MIT
