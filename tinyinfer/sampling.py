"""Token sampling: temperature, top-k, top-p, with a seedable RNG.

Order matters and is not arbitrary. Temperature scales the *logits* before the
softmax; top-k and top-p then truncate the resulting distribution and it is
renormalized over what survives. Applying temperature after truncation would
change which tokens were eligible, which is a different (and wrong) sampler.

top-k and top-p compose: k caps the candidate count, p caps the cumulative
probability mass, and when both are set the stricter one wins at each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import softmax


@dataclass
class SamplerConfig:
    temperature: float = 1.0
    top_k: int | None = None      # None or 0 disables
    top_p: float | None = None    # None or >=1.0 disables
    seed: int | None = None

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


class Sampler:
    def __init__(self, config: SamplerConfig | None = None):
        self.config = config or SamplerConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def reset(self) -> None:
        """Re-seed to the configured seed so a run can be replayed exactly."""
        self.rng = np.random.default_rng(self.config.seed)

    def __call__(self, logits: np.ndarray) -> int:
        """Pick one token id from a (vocab,) logit vector."""
        cfg = self.config

        # Temperature 0 means argmax. Dividing by zero would be inf/nan, so this
        # is a special case rather than a limit.
        if cfg.is_greedy:
            return int(logits.argmax())

        logits = logits.astype(np.float32) / cfg.temperature
        probs = softmax(logits)

        # Sort descending once; both truncations are prefix operations on it.
        order = np.argsort(-probs)
        sorted_probs = probs[order]

        keep = len(sorted_probs)

        if cfg.top_k:
            keep = min(keep, cfg.top_k)

        if cfg.top_p is not None and cfg.top_p < 1.0:
            cumulative = np.cumsum(sorted_probs)
            # searchsorted finds the first index where cumsum >= p; +1 keeps that
            # token, so the retained mass is always >= p rather than falling just
            # short of it. Guarantees at least one candidate.
            n_p = int(np.searchsorted(cumulative, cfg.top_p) + 1)
            keep = min(keep, n_p)

        keep = max(1, keep)
        candidates = order[:keep]
        weights = probs[candidates]
        weights = weights / weights.sum()

        return int(self.rng.choice(candidates, p=weights))
