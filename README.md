# tinyinfer

An LLM inference engine written from scratch — no PyTorch `generate()`, no `transformers`, no `llama.cpp`. The tokenizer, the forward pass, the KV cache, the sampling loop, and the quantization are all implemented here.

Target model: **Qwen2.5-0.5B-Instruct**. Target hardware: **Apple M5, 24 GB**.

---

## Status — read this first

**It generates text.** Phases 1–4 are done and verified; the engine runs Qwen2.5-0.5B end to end in NumPy at ~19 tok/s. Everything after phase 4 is unwritten.

| Phase | State | Result |
|---|---|---|
| 1. Parse safetensors, print layer names and shapes | ✅ | 290 tensors, 494,032,768 params, all shapes match config |
| 2. BPE tokenizer, exact round-trip on 10,000 strings | ✅ | 10000/10000 exact id match and exact round-trip |
| 3. Naive fp32 forward pass, logits match reference | ✅ | 4.44e-04 worst case (tol 1e-3), top-1 100% at every position |
| 4. KV cache — identical output, measured speedup | ✅ | identical ids, 1.78× at 37 positions |
| 5. Sampling: temperature, top-k, top-p, seeded RNG | ⛔ | — |
| 6. INT8 → INT4 quantization, perplexity at each level | ⛔ | — |
| 7. Metal compute kernels for the hot loop | ⛔ | — |
| 8. Benchmark vs. llama.cpp, identical model and hardware | ⛔ | — |

Phases 1–5 are Python + NumPy, correctness only. Phases 6–8 move the hot path to C++ and Metal.

```
$ python tests/test_forward.py
'The capital of France is' -> ' Paris. It is the largest city in'
```

**On the cache speedup:** 1.78× is measured at 37 positions, where the quadratic
term barely dominates yet. The win scales with context length — that number should
be re-measured at 512 and 2048 tokens, and the curve is more honest than any single
figure. Not done yet.

## What gets measured

The point of the project is the numbers, so they are declared before any of them exist:

| Metric | fp32 | INT8 | INT4 |
|---|---|---|---|
| Tokens/sec | — | — | — |
| Memory footprint | — | — | — |
| Perplexity (held-out) | — | — | — |
| Δ perplexity vs. fp32 | baseline | — | — |

Plus a head-to-head against `llama.cpp` on the same model, prompt, and machine — **published whether or not it wins**, with an explanation of why.

## The quantization study

Phase 6 is not a checkbox. Beyond "INT8 and INT4 both work," this repo answers:

- Which layers tolerate aggressive quantization and which don't — attention vs. MLP vs. embeddings degrade differently.
- Where quality falls off, and whether the cliff is sharp or gradual.
- Whether that answer changes with prompt length.

Results as a chart and a written analysis, not a single number.

## Correctness discipline

Every phase is verified against a reference implementation before the next one starts. Specifically:

- The tokenizer must produce **byte-identical token IDs** to the reference on 10,000 strings before the model is touched. Tokenizer mismatches produce output that looks almost right and are near-impossible to debug later.
- The fp32 forward pass must match reference logits within `1e-3` at step one, not step fifty. Attention masking off by one still produces fluent text — just subtly wrong text.
- The KV cache must produce **identical** output to the uncached path, not merely similar.

A benchmark script exists from phase 1 so tokens/sec is tracked across every phase, not measured once at the end.

## Build log

Every bug that costs more than an hour gets an entry in [`LOG.md`](LOG.md): what the cause appeared to be, and what it actually was.

## Prior work

This is a direct sequel to [chip8](https://github.com/iqureshi123/chip8) — same discipline (parse a binary format, implement a spec exactly, verify headlessly, then optimize), pointed at a transformer instead of a 1970s VM.

## License

MIT
