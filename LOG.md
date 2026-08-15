# Build log

Every bug that cost more than an hour. What I thought the cause was, and what it actually was.

This file is the point. It is where the interview answers come from.

---

## Nothing has crossed the hour mark yet

Phases 1–6 all passed their gate on the first or second run. That is not luck and
it is worth writing down *why*, because the reason is the actual lesson:

**Every phase was verified against a reference before the next one started.** The
tokenizer was checked for byte-exact equality on 10,000 strings before the model
was touched. The forward pass was checked against reference logits at step one,
not step fifty. The KV cache was checked for *identical* output, not similar
output. Each gate was cheap to build and caught its class of error immediately,
so no error ever got a chance to compound into the multi-day kind.

The failure mode this avoids is specific: a tokenizer that is 99.9% correct, or a
RoPE convention that is subtly wrong, produces *fluent* output. There is nothing
to notice. You find it a week later when quantization results make no sense, and
by then three layers of work are built on top of it.

Entries below are near-misses — things that went wrong but were caught in minutes
because a check existed. They are recorded because the cost of *not* having the
check was the multi-hour version.

---

## 2026-08-15 — Gutenberg extraction grabbed the table of contents

**Symptom:** the held-out corpus came out as 167 characters instead of ~144,000.

**Thought it was:** a truncated download or a bad `--max-time`.

**Actually was:** the slice started at the first occurrence of `CHAPTER I.`, which
is in the table of contents, and ended at the first `CHAPTER V.`, also in the
table of contents. Both markers matched, the slice was valid, and the result was
four lines of contents. Fixed by searching for the *second* occurrence.

**Cost:** ~2 minutes, because the script printed the character count. Without that
print it would have surfaced as an inexplicably meaningless perplexity number
much later, with the corpus being the last place anyone would look.

---

## 2026-08-15 — `git rebase --root --exec` silently did not rewrite authorship

**Symptom:** after running `git rebase --root --exec 'git commit --amend
--reset-author'`, `git log` showed the original commit hashes and the original
author on every commit. No error was printed.

**Thought it was:** the `--reset-author` flag not applying without an explicit
`user.email`.

**Actually was:** the rebase aborted immediately because the working tree had
unstaged changes, and the failure went to stderr which was being discarded. The
command "succeeded" from the shell's point of view. `git filter-branch
--env-filter` was the working approach, and it reports plainly what it rewrote.

**Cost:** ~5 minutes. The lesson generalises: a git command that rewrites history
and prints nothing has probably not run.

---

<!--
Format:

## YYYY-MM-DD — one-line title

**Symptom:** what I saw
**Thought it was:** the first hypothesis
**Actually was:** the real cause
**Cost:** how long
-->

## 2026-08-15 — the mixed-precision hypothesis was wrong

**Symptom:** not a bug. The README predicted, in writing, that mixed precision
would beat uniform INT4 — keep the fragile matrices at INT8, push the robust
ones to INT4, recover most of the quality at most of the compression.

**Thought it was:** obviously correct. The per-layer study showed a 16x spread
between `k_proj` (+0.44%) and `down_proj` (+7.25%) at identical bit width. If
some matrices are that much more fragile, spending more bits on them should be
the efficient allocation.

**Actually was:** it wins on quality and loses on the tradeoff. Mixed INT4/INT8
with `down_proj` promoted reaches +18.42% at 5.42 bits/weight. Uniform INT4 at
group size 64 reaches +19.33% at **4.50** bits. Same quality, nearly a full bit
per weight cheaper. Group size dominates matrix-type selection.

The error was comparing schemes on perplexity alone. Any scheme that spends
more bits looks better on that axis, which makes the comparison meaningless
unless size is held roughly fixed. Once both axes were plotted the conclusion
inverted.

**Cost:** no debugging time, but it would have shipped as a false claim in the
README if the prediction had not been tested. Written down because the habit
that caught it — state the prediction, then measure it — is the transferable
part, not the result.

## 2026-08-15 — a float64 promotion halved throughput, and every test passed

**Symptom:** the phase 7 profile attributed 42.7% of decode time to `o_proj`, a
896x896 matmul. The MLP, with 16x the FLOPs, took about the same. Isolated, the
same `o_proj` line ran in 0.014 ms; in situ the profiler measured 1.18 ms.

**Thought it was:** three wrong guesses, in order. First, a profiler artifact —
so I reimplemented one decode step inline with raw timers, and it ran at 38 ms/token
against production's 67 ms, which made it a real gap rather than a measurement
error. Second, BLAS thread thrashing on tiny matmuls — `VECLIB_MAXIMUM_THREADS`
at 1, 2, 4 and 10 moved nothing. Third, cold-cache weight streaming — but that
could not explain why `o_proj` (3.2 MB/layer) cost the same as the MLP (25 MB/layer).

**Actually was:** one line.

    scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(hd)

`np.sqrt(64)` returns a **float64 numpy scalar**, not a Python float. Under
NEP 50 a Python float leaves a float32 array alone, but a float64 *numpy scalar*
promotes it. So `scores` became float64, and that cascaded: softmax in float64,
the weighted sum in float64, and then a float64 `attn` multiplied against a
float32 weight matrix in `o_proj`. That mixed-dtype matmul has no fast BLAS path
and measured **947 us against 13.6 us** for the float32 version — 70x, which is
exactly the anomaly the profile was pointing at.

Fixed by folding the reciprocal and casting once:

    scores = (q @ k.transpose(0, 2, 1)) * np.float32(1.0 / np.sqrt(hd))

**Result:** decode went from 14.7 to 28.6 tok/s. A 1.95x speedup, no algorithm
change, one line.

**Cost:** about 40 minutes, and it would never have been found by the test suite.
That is the part worth keeping. Float64 is *more* precise than float32, so the
promotion made every correctness gate pass slightly more comfortably — phase 3
logits agreed with the reference to 4.44e-04 while silently doing double-width
arithmetic. After the fix that figure is 4.71e-04: marginally less precise,
still far inside the 1e-3 tolerance, and twice as fast.

A bug that improves your accuracy metric while halving your throughput is
invisible to correctness testing by construction. It shows up in a profile or
it does not show up at all. `tests/test_dtypes.py` now asserts the dtype of
every primitive, every cached weight, and the full forward pass, so the next
one fails a test instead of a benchmark.
