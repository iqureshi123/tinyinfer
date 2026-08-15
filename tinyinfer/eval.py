"""Perplexity on held-out text.

Perplexity is exp of the mean negative log-likelihood the model assigns to text
it did not generate. It is the standard way to price a quantization scheme,
because it is sensitive to distribution-wide damage that a handful of greedy
completions will not reveal — a model can still say "Paris" correctly while
being measurably worse everywhere else.

The corpus is public-domain text shipped in the repo, so the number is
reproducible without a download and cannot silently change between runs.
"""

from __future__ import annotations

import numpy as np


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Computed via the shifted log-sum-exp identity rather than log(softmax(x)),
    which underflows to -inf for confidently-wrong tokens — exactly the tokens
    that dominate a perplexity score."""
    m = np.max(x, axis=-1, keepdims=True)
    shifted = x - m
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def perplexity(model, token_ids: list[int], window: int = 512,
               stride: int | None = None) -> dict[str, float]:
    """Sliding-window perplexity over a token sequence.

    With `stride < window`, each window carries context from the previous one and
    only the newly-exposed tokens are scored. That avoids penalising tokens which
    appear at the very start of a window with almost no context to predict from,
    which would otherwise inflate the score and make it depend on window size.
    """
    stride = stride or window // 2
    n = len(token_ids)
    if n < 2:
        raise ValueError("need at least 2 tokens")

    total_nll = 0.0
    total_counted = 0
    prev_end = 0
    start = 0

    while start < n - 1:
        end = min(start + window, n)
        chunk = token_ids[start:end]

        logits = model.forward(chunk)
        logprobs = log_softmax(logits.astype(np.float32))

        # Position i predicts token i+1, so targets are the chunk shifted left.
        targets = np.asarray(chunk[1:], dtype=np.int64)
        picked = logprobs[np.arange(len(targets)), targets]

        # Score only tokens not already scored by a previous window.
        first_new = max(0, prev_end - start - 1)
        scored = picked[first_new:]

        total_nll += float(-scored.sum())
        total_counted += scored.size

        prev_end = end
        if end == n:
            break
        start += stride

    mean_nll = total_nll / total_counted
    return {
        "perplexity": float(np.exp(mean_nll)),
        "nll": mean_nll,
        "tokens_scored": total_counted,
    }
