"""Synthetic anomaly injection for evaluating GridGuard detectors.

Smart meter readings are altered in controlled, labelled ways so detectors can
be scored against known ground truth. This is standard practice for defensive
research: the injected faults mimic sensor failures, tampering and data errors.
"""

import numpy as np
import pandas as pd

ANOMALY_TYPES = ["zero_out", "flatline", "scale_down", "drift", "spike"]

# (min, max) duration in half-hour steps
DURATIONS = {
    "zero_out": (12, 72),            # 6 to 36 hours
    "flatline": (24, 96),            # 12 to 48 hours
    "scale_down": (3 * 48, 14 * 48), # 3 to 14 days
    "drift": (5 * 48, 14 * 48),      # 5 to 14 days
    "spike": (1, 3),                 # 0.5 to 1.5 hours
}


def inject_anomalies(df, events_per_meter=6, first_step=0, seed=0, gap=48):
    """Return (df_with_anomalies, labels, events); the input is not modified.

    df           time x meter table of readings
    first_step   earliest row where an anomaly may start
    gap          minimum number of steps between two anomalies of a meter
    labels       boolean table, True where a reading was altered
    events       one row per anomaly: meter, type, start, end (inclusive)
    """
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
            kind = ANOMALY_TYPES[int(rng.integers(len(ANOMALY_TYPES)))]
            low, high = DURATIONS[kind]
            length = int(rng.integers(low, high + 1))
            if first_step + length + gap >= n_steps:
                continue
            start = int(rng.integers(first_step, n_steps - length - gap))
            end = start + length
            if occupied[max(0, start - gap):end + gap].any():
                continue

            segment = values[start:end, j]
            if kind == "zero_out":
                new = np.zeros_like(segment)
            elif kind == "flatline":
                new = np.full_like(segment, values[start, j])
            elif kind == "scale_down":
                new = segment * np.float32(rng.uniform(0.3, 0.7))
            elif kind == "drift":
                final = rng.uniform(0.3, 0.6)
                new = segment * np.linspace(1.0, final, length, dtype="float32")
            else:  # spike
                new = segment * np.float32(rng.uniform(5, 10))

            values[start:end, j] = new
            labels[start:end, j] = True
            occupied[start:end] = True
            events.append((df.columns[j], kind, df.index[start], df.index[end - 1]))
            placed += 1

    df_out = pd.DataFrame(values, index=df.index, columns=df.columns)
    labels = pd.DataFrame(labels, index=df.index, columns=df.columns)
    events = pd.DataFrame(events, columns=["meter", "type", "start", "end"])
    return df_out, labels, events X