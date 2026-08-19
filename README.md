# tinyinfer

An LLM inference engine written from scratch — no PyTorch `generate()`, no `transformers`, no `llama.cpp`. The safetensors parser, the tokenizer, the forward pass, the KV cache, the sampler, and the quantizer are all implemented here.

Target model: **Qwen2.5-0.5B-Instruct**. Target hardware: **Apple M5, 24 GB**.

```
$ python scripts/generate.py --chat "Explain what a KV cache does in one sentence."
A KV cache is a data structure that stores frequently accessed data in memory,
allowing for quick retrieval of information.

[23 tokens in 0.87s — 26.4 tok/s, cache=on, quant=none]
```

---

## Status

| Phase | State | Result |
|---|---|---|
| 1. Parse safetensors | ✅ | 290 tensors, 494,032,768 params, every shape cross-checked against config |
| 2. BPE tokenizer | ✅ | 10000/10000 exact id match and exact round-trip vs reference |
| 3. fp32 forward pass | ✅ | logits within 4.44e-04 of reference (tol 1e-3), top-1 100% at every position |
| 4. KV cache | ✅ | output ids **identical** to uncached, 1.33×→6.29× as context grows 32→512 |
| 5. Sampling | ✅ | temperature / top-k / top-p, seeded and reproducible |
| 6. Quantization + study | ✅ | INT8 free, INT4 costly, and the cost is **not uniform** — see below |
| 7. Optimize the hot path | ✅ | **1.95× shipped** from a float64 fix; Metal INT4 kernel built, verified, **loses to fp32 BLAS at every batch size tested** — see below |
| 8. Benchmark vs llama.cpp | ✅ | within **1.14×** on matched fp32/CPU; **6.6×** behind its Q4_K_M Metal path — and the gap is quantized kernels, not the GPU |

Phases 1–6 are Python + NumPy, correctness first. Phases 7–8 move the hot path to C++/Metal.

## The finding

Quantization is usually reported as one number per bit width. That hides the more useful result: **weight matrices do not degrade equally.** Quantizing only `k_proj` to INT4 costs +0.44% perplexity. Quantizing only `down_proj` — same bit width, same group size, same bits saved — costs **16× more**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/quant_sensitivity_dark.svg">
  <img alt="Perplexity cost of INT4 quantization by weight matrix type" src="results/quant_sensitivity_light.svg">
</picture>

Measured on 3,072 held-out tokens, fp32 baseline perplexity **12.7891**.

### 1. INT8 is free. INT4 is not, and the cliff is right below it.

| Scheme | Perplexity | Δ | Compression |
|---|---|---|---|
| fp32 | 12.7891 | — | 1.00× |
| INT8 g128 asym | 12.7754 | **−0.11%** | 3.88× |
| INT4 g32 asym | 14.6141 | +14.27% | 6.40× |
| INT4 g128 asym | 16.4615 | +28.72% | 7.53× |
| INT3 g64 asym | 36.6353 | +186.46% | 9.14× |
| INT2 g64 asym | 608,112 | +4,754,844% | 12.80× |

The degradation is **not gradual**. INT8 is indistinguishable from fp32, INT4 is expensive but usable, INT3 nearly triples perplexity, and INT2 destroys the model outright — 608,112 is not a degraded model, it is noise. Anyone planning a bit-width sweep should know the useful range ends at 4.

### 2. Group size is the main lever, and asymmetric is not optional

| INT4 config | Δ perplexity |
|---|---|
| g256 asym | +36.02% |
| g128 asym | +28.72% |
| g64 asym | +19.33% |
| g32 asym | +14.27% |
| g128 **sym** | +42.38% |
| g64 **sym** | +30.10% |

Shrinking groups from 256 to 32 cuts the loss by more than half for ~17% less compression. And at equal group size, dropping the zero point costs about as much as doubling the group — symmetric g64 (+30.10%) is barely better than asymmetric g128 (+28.72%) while storing more metadata. The extra zero point pays for itself.

### 3. MLP is twice as fragile as attention

Quantizing the whole attention block costs +9.23%; the whole MLP block costs +18.49%. Per matrix, the spread runs from `k_proj` (+0.44%) to `down_proj` (+7.25%) — a 16× range at identical bit width.

The obvious move is mixed precision: keep the fragile matrices at INT8, push the robust ones to INT4. **It does not pay off, and that is the more useful finding.**

| Scheme | Perplexity | Δ | Avg bits/weight | Compression |
|---|---|---|---|---|
| uniform INT4 g128 | 16.4615 | +28.72% | 4.25 | 7.53× |
| uniform INT4 **g64** | 15.2611 | **+19.33%** | **4.50** | **7.11×** |
| INT4 + INT8 on `down_proj` | 15.1453 | +18.42% | 5.42 | 5.90× |
| INT4 + INT8 on `down`, `up` | 14.4008 | +12.60% | 6.59 | 4.86× |
| INT4 + INT8 on `down`, `up`, `v` | 13.6471 | +6.71% | 6.62 | 4.83× |

Compare rows 2 and 3. They reach effectively the same quality (+19.33% vs +18.42%), but mixed precision spends **5.42 bits per weight to get there and uniform INT4 at group 64 spends 4.50**. Shrinking the group is simply a more efficient way to spend the same bits than promoting whole matrix types to INT8.

So the sensitivity result is real — the matrices genuinely differ by 16× — but acting on it *at the granularity of matrix type* is the wrong lever. Group size dominates. A finer-grained scheme (per-channel or outlier-aware, as GPTQ and AWQ do) might still win; promoting whole tensors does not.

This was written into the README as a prediction before it was measured, and the measurement contradicted it. Reproduce with `python scripts/mixed_precision.py`.

### 4. Depth barely matters — a negative result

| Blocks quantized to INT4 | Δ perplexity |
|---|---|
| 0–7 (early) | +8.13% |
| 8–15 (middle) | +7.34% |
| 16–23 (late) | +9.30% |

Within about two percentage points of each other. "Protect the early layers" is a plausible and commonly assumed strategy, and on this model it is not supported. Reported because a strategy ruled out is worth as much as one confirmed.

Full numbers: [`results/quant_study.json`](results/quant_study.json). Reproduce with `python scripts/quant_study.py` (~20 min).

## Performance

Apple M5, 24 GB, NumPy fp32 over Accelerate BLAS. Percentiles, not means — a mean hides the stalls that decide whether generation reads as smooth.

| | tok/s | p50 | p99 |
|---|---|---|---|
| prefill | 131.2 | — | — |
| decode | 26.2 | 34.9 ms | 79.7 ms |

Prefill and decode are reported separately because they are different regimes: prefill is compute-bound across the whole prompt, decode is memory-bound on a single token. Averaging them produces a number that describes neither.

**These numbers are 1.95× what they were before profiling.** Phase 7 began with a profile rather than a kernel, and the profile found a one-line bug worth more than any kernel would have been: `scores / np.sqrt(head_dim)` was silently promoting the entire attention path to float64, because `np.sqrt` returns a float64 *numpy scalar* and those promote float32 arrays under NEP 50. The mixed-dtype matmul that followed lost the fast BLAS path — 947 µs against 13.6 µs on that op alone.

Every correctness test passed the whole time. Float64 is *more* precise, so the bug made the reference comparison agree slightly better (4.44e-04 before, 4.71e-04 after) while halving throughput. A defect that improves your accuracy metric and halves your speed is invisible to correctness testing by construction. Full account in [`LOG.md`](LOG.md); [`tests/test_dtypes.py`](tests/test_dtypes.py) now asserts float32 through every primitive, the cache, and all 290 weights.

**The KV cache speedup is a curve, not a constant.** Phase 4 measured 1.78× at 37 positions, which understates it badly — the cache removes quadratic work, so the win grows with context:

| Context | Cached | Uncached | Speedup |
|---|---|---|---|
| 32 | 10.5 tok/s | 7.9 tok/s | 1.33× |
| 128 | 13.6 tok/s | 3.0 tok/s | 4.52× |
| 256 | 10.1 tok/s | 2.0 tok/s | 5.13× |
| 512 | 5.6 tok/s | 0.9 tok/s | **6.29×** |

Quantized decode measures within noise of fp32, because weights are dequantized to fp32 before the matmul. That is deliberate: it isolates the *quality* cost, which is what the study measures. Throughput gains need a real INT4 kernel.

Where the time actually goes, per decode token ([`results/profile.json`](results/profile.json)): **96.9% is matmul**, and the three MLP projections are 65% on their own. That is the case for a GPU kernel, and it also caps every non-matmul optimization at 3.1% of the budget.

Reproduce with `python scripts/bench.py` and `python scripts/profile_forward.py`.

### The Metal kernel, and losing to fp32 BLAS honestly

Following that profile, [`tinyinfer/kernels/int4_matvec.metal`](tinyinfer/kernels/int4_matvec.metal) fuses INT4 dequantization directly into the matvec — it reads each weight byte once and accumulates, rather than materializing a full fp32 matrix and calling BLAS on it. No Xcode is needed: `xcrun metal` requires a full Xcode install, but the Metal *framework* compiles shader source at runtime via `newLibraryWithSource:`, the same mechanism llama.cpp's Metal backend uses. [`tinyinfer/metal_backend.py`](tinyinfer/metal_backend.py) drives it through PyObjC, with weights uploaded to the GPU once at construction and reused every call.

It is correct — verified against the phase-6 `dequantize()` reference on every real matrix shape in the model, max diff ~4e-6 ([`tests/test_metal_kernel.py`](tests/test_metal_kernel.py)) — and it clearly beats the naive "dequantize-then-BLAS" path:

| Shape | fp32 BLAS | CPU dequant→BLAS | Metal INT4 | vs fp32 | vs CPU dequant |
|---|---|---|---|---|---|
| q/k/v/o_proj (896×896) | 0.013 ms | 0.748 ms | 0.702 ms | 0.02× | 1.07× |
| gate/up_proj (4864×896) | 0.221 ms | 4.618 ms | 0.333 ms | 0.66× | **13.86×** |
| down_proj (896×4864) | 0.215 ms | 4.465 ms | 0.825 ms | 0.26× | 5.41× |

**It loses to plain fp32 BLAS on every shape.** Doc-1 rule: if your benchmark loses to the established library, publish the loss and explain why, so here is why — measured, not guessed. Dispatching the identical kernel against an 8×64 matrix with almost nothing to compute still costs 0.24 ms; the full 896×896 shape costs 0.29 ms. The real arithmetic is ~0.04 ms; the other ~0.24 ms is fixed cost from creating a command buffer, encoding, committing, and `waitUntilCompleted()` synchronously. fp32 BLAS at this size is an in-process library call with none of that — 0.013 ms.

At batch=1, a single decode token is too little work to amortize a GPU dispatch. This isn't specific to Metal: fp32 BLAS itself goes from 214 µs/row at batch 1 down to 11 µs/row at batch 128 — even the CPU path is dominated by fixed costs at this size.

**The obvious next hypothesis — batch many rows into one dispatch, pay the fixed cost once — was tested and refuted.** [`tinyinfer/kernels/int4_matmat.metal`](tinyinfer/kernels/int4_matmat.metal) processes a whole batch in a single command buffer (verified correct: 40/40 against the reference across every shape and five batch sizes up to 128). Swept against fp32 BLAS from batch 1 to 256:

| Shape | batch 1 | batch 4 | batch 16 | batch 64 | batch 256 |
|---|---|---|---|---|---|
| q/k/v/o_proj | 0.01× | 0.06× | 0.05× | 0.08× | 0.14× |
| gate/up_proj | 0.42× | 0.92× | 0.38× | 0.18× | 0.12× |
| down_proj | 0.24× | 0.71× | 0.32× | 0.16× | 0.12× |

The gap doesn't close — it widens. The kernel's per-row cost does improve with batching (the fixed dispatch cost genuinely amortizes, confirming half the earlier diagnosis), but fp32 BLAS improves faster, because it isn't just avoiding dispatch overhead — it's running an AMX-tuned, register-blocked matrix-matrix multiply. This kernel is one thread per (output feature, batch row), each doing an independent scalar loop with no data reuse across threads: more parallel work at larger batch, not more *efficient* work. Closing that gap needs Metal's `simdgroup_matrix` tiling or MPSGraph, not a bigger batch — a materially different piece of work than what's here.

**Where phase 7 actually lands:** the profiling-driven float64 fix is a real, shipped 1.95× win. The GPU kernel is a correctly-built, correctly-verified experiment whose result is that a naive per-thread Metal kernel is the wrong tool against Accelerate's BLAS on this hardware at these sizes — and the benchmark says exactly why, which is more useful than a kernel that happened to win without anyone knowing the reason. Reproduce with `python scripts/bench_metal.py` and `python scripts/bench_metal_batched.py`; full account in [`LOG.md`](LOG.md).

### Phase 8 — against llama.cpp, on identical weights

A comparison only means anything if both engines run the same model, so the GGUF is converted from the exact `model.safetensors` this repo parses in phase 1, not downloaded pre-built. Both engines report **494,032,768 parameters**; [`scripts/bench_llamacpp.py`](scripts/bench_llamacpp.py) asserts that and fails loudly if it ever drifts.

Every row below comes from one run, on one machine, in one session. That matters more than it sounds: comparing today's llama.cpp against the fp32 figures in the table above — measured in a different session — would not be a valid comparison, so all of it is re-measured together. llama.cpp at `60adddd`, Metal enabled, prefill 14 tokens, decode 64, 5 repetitions.

| engine | prefill tok/s | decode tok/s | vs tinyinfer |
|---|---|---|---|
| **tinyinfer fp32 CPU** | 157.7 | 37.2 | — |
| llama.cpp F32 CPU | 190.7 | 42.4 | 1.14× |
| llama.cpp F32 Metal | 651.1 | 61.0 | 1.6× |
| llama.cpp Q8_0 Metal | 1007.3 | 205.3 | 5.5× |
| llama.cpp Q4_K_M CPU | 410.9 | 179.4 | 4.8× |
| llama.cpp Q4_K_M Metal | 981.0 | 243.5 | **6.6×** |

Two results, and the second is the one worth having.

**On the like-for-like comparison the gap is small.** Same fp32, same CPU, same Accelerate BLAS underneath: 37.2 tok/s against 42.4, so llama.cpp decodes 1.14× faster. Re-measuring tinyinfer *after* all five llama.cpp passes gives 37.7 tok/s, so the bias from measurement order is 1.12×–1.14× — not enough to move the conclusion. A from-scratch NumPy engine lands within about 14% of llama.cpp when neither side gets quantized kernels or a GPU. Prefill is the weaker half at 1.21× behind, which is where llama.cpp's batching shows.

**The 6.6× headline is not the GPU.** Compare llama.cpp against itself: Q4_K_M on the *CPU* decodes at 179.4 tok/s while F32 on *Metal* manages 61.0. Quantized CPU kernels beat the fp32 GPU path by 2.9×. Moving F32 from CPU to Metal buys 1.4×; quantizing F32→Q4_K_M on the same CPU buys 4.2×. The device is the smaller variable. The kernel operating on quantized data directly is the larger one.

That is the phase 7 finding arriving from the other direction. The hand-written Metal INT4 kernel lost to Accelerate BLAS because it was a naive scalar kernel with no tiling and no data reuse; what actually wins is a dequant-free *blocked* matmul, and llama.cpp shows a CPU doing it well enough to beat an untuned GPU. tinyinfer still dequantizes INT4 to fp32 before every matmul, which is exactly why its quantization buys compression and no speed — and that one architectural choice, not the missing GPU, is most of the remaining 6.6×.

```bash
python scripts/bench_llamacpp.py --llama-bench /path/to/llama.cpp/build/bin/llama-bench
```

llama.cpp is invoked as an external binary for comparison. Nothing in `tinyinfer/` imports or links it.

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
python tests/test_dtypes.py         # phase 7 — no silent float64 promotion
python tests/test_metal_kernel.py   # phase 7 — Metal kernel vs numpy reference (skips off-Apple)
python scripts/quant_study.py       # phase 6 — the study
```

The tokenizer corpus covers whitespace runs, CJK, ZWJ emoji sequences, NFC/NFD pairs, raw-byte garbage, code, and embedded chat-template tokens — the cases where a byte-level BPE actually breaks.

## Limitations

- **No GPU path in the generation loop.** Everything runs on NumPy over Accelerate BLAS. Phase 8 measured the cost precisely: 1.14× behind llama.cpp on matched fp32/CPU, and 6.6× behind its Q4_K_M Metal path.
- **Quantized weights are dequantized to fp32 before the matmul.** That measures the quality cost exactly, which is what the study is for, but it means quantization currently buys compression and not speed. Phase 8 puts a number on it: llama.cpp gets 4.2× on the same CPU purely from keeping the matmul in quantized space.
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
