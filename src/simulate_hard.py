"""A harder set of synthetic faults for GridGuard.

The standard simulator zeroes readings exactly and freezes them exactly, which is
the easiest case for simple rules. These variants are noisier: usage drops to a
few percent of normal instead of exactly zero, and a stuck meter still jitters
slightly instead of repeating one number. They show where exact rules break.
"""

import numpy as np
import pandas as pd

HARD_TYPES = ["near_zero", "jitter_flat"]

# (min, max) duration in half-hour steps
DURATIONS = {
    "near_zero": (12, 72),     # 6 to 36 hours at 1 to 5% of normal usage
    "jitter_flat": (24, 96),   # 12 to 48 hours at a constant value, +/- 3% noise
}


def inject_hard_anomalies(df, events_per_meter=4, first_step=0, seed=0, gap=48):
    """Return (df_with_anomalies, labels, events); the input is not modified."""
    rng = np.random.default_rng(seed)
    values = df.to_numpy(dtype="float32", copy=True)
    labels = np.zeros(values.shape, dtype=bool)
    n_steps, n_meters = values.shape
    events = []

    for j in range(n_meters):
        occupied = np.zeros(n_steps, dtype=bool)
        placed, attempts = 0, 0
        while placed < events_per_meter and attempts < 300:
            attempts += 1
            kind = HARD_TYPES[int(rng.integers(len(HARD_TYPES)))]
            low, high = DURATIONS[kind]
            length = int(rng.integers(low, high + 1))
            if first_step + length + gap >= n_steps:
                continue
            start = int(rng.integers(first_step, n_steps - length - gap))
            end = start + length
            if occupied[max(0, start - gap):end + gap].any():
                continue

            segment = values[start:end, j]
            if kind == "near_zero":
                new = segment * np.float32(rng.uniform(0.01, 0.05))
            else:  # jitter_flat: stuck near the segment's average, with small noise
                new = float(segment.mean()) * (1 + rng.uniform(-0.03, 0.03, length))
            values[start:end, j] = new.astype("float32")
            labels[start:end, j] = True
            occupied[start:end] = True
            events.append((df.columns[j], kind, df.index[start], df.index[end - 1]))
            placed += 1

    df_out = pd.DataFrame(values, index=df.index, columns=df.columns)
    labels = pd.DataFrame(labels, index=df.index, columns=df.columns)
    events = pd.DataFrame(events, columns=["meter", "type", "start", "end"])
    return df_out, labels, events