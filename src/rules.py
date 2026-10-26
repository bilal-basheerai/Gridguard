"""Rule-based detectors and the hybrid combiner for GridGuard.

Rules look for patterns that persist: hours of zeros, a stuck meter, usage far
below normal for the time of day. They share the interface of the other
detectors: score(df) returns a table where a higher value means "more anomalous".
"""

import numpy as np
import pandas as pd

from src.evaluate import count_episodes
from src.features import _slot_keys


def trailing_run_length(mask):
    """For a (time, meter) boolean array: length of the True-run ending at each row."""
    out = np.zeros(mask.shape, dtype=np.int32)
    run = np.zeros(mask.shape[1], dtype=np.int32)
    for t in range(mask.shape[0]):
        run = np.where(mask[t], run + 1, 0)
        out[t] = run
    return out


class RunLengthRule:
    """How long (in half-hours) a simple pattern has lasted.

    kind="zero"   exact zero readings in a row
    kind="stuck"  the same non-zero reading repeated
    """

    def __init__(self, kind):
        assert kind in ("zero", "stuck")
        self.kind = kind
        self.name = "Zero-run rule" if kind == "zero" else "Stuck-reading rule"

    def score(self, df):
        x = df.to_numpy()
        if self.kind == "zero":
            mask = x == 0
        else:
            mask = np.zeros(x.shape, dtype=bool)
            mask[1:] = (x[1:] == x[:-1]) & (x[1:] != 0)
        runs = trailing_run_length(mask).astype("float32")
        return pd.DataFrame(runs, index=df.index, columns=df.columns)


class SustainedLowRule:
    """Usage stays far below what is normal for this household and time of day (6 h)."""

    name = "Sustained-low rule"

    def __init__(self, profile, window=12):
        self.profile = profile
        self.window = window

    def score(self, df):
        keys = pd.MultiIndex.from_arrays(list(_slot_keys(df.index)))
        expected = pd.DataFrame(self.profile.reindex(keys)[df.columns].to_numpy(),
                                index=df.index, columns=df.columns)
        peak = df.rolling(self.window, min_periods=self.window).max()
        expected_peak = expected.rolling(self.window, min_periods=self.window).max()
        ratio = peak / expected_peak.clip(lower=1e-3)
        return (1 - ratio).clip(0, 1).fillna(0.0)


class LowVariabilityRule:
    """Readings barely change for 6 hours (a stuck meter that still jitters)."""

    name = "Low-variability rule"

    def __init__(self, level, window=12):
        self.level = level
        self.window = window

    def score(self, df):
        spread = df.rolling(self.window, min_periods=self.window).std()
        spread = spread / self.level.reindex(df.columns)
        return (1.0 / (spread + 0.01)).fillna(0.0)


def calibrate_by_budget(clean_scores, candidates, budget, gap=12):
    """Smallest threshold whose false-alarm episodes per meter-week stay within budget.

    clean_scores: (time, meter) scores on clean data (no anomalies, no labels).
    candidates:   thresholds to try, in ascending order.
    """
    s = np.nan_to_num(np.asarray(clean_scores, dtype="float64"))
    meter_weeks = s.shape[1] * s.shape[0] / (48 * 7)
    for cand in candidates:
        if count_episodes(s > cand, gap) / meter_weeks <= budget:
            return float(cand)
    return float(candidates[-1])


def combine_max(parts):
    """Hybrid detector: alarm when ANY component alarms.

    parts: list of (scores, threshold). Each score table is divided by its own
    threshold, so a value above 1 means that component fired; the hybrid takes the
    largest ratio. Returns (combined_scores, 1.0).
    """
    ratios = [s.to_numpy() / max(t, 1e-6) for s, t in parts]
    combined = np.maximum.reduce(ratios)
    first = parts[0][0]
    return pd.DataFrame(combined, index=first.index, columns=first.columns), 1.0