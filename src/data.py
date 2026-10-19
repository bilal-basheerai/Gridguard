"""Data loading utilities for GridGuard (London Smart Meters dataset)."""

import numpy as np
import pandas as pd

START = pd.Timestamp("2013-01-01 00:00")
N_STEPS = 365 * 48                 # half-hourly readings in 2013
EXPECTED_HOUSEHOLDS = 5560


def load_raw_sample(ts_path, n_meters=300, seed=42):
    """Build a (time x household) table from the raw .ts file.

    Keeps only households with complete readings for all of 2013, picked in
    random order with a fixed seed so the sample is reproducible.
    """
    with open(ts_path) as f:
        text = f.read()
    lines = text.split("@data", 1)[1].strip().splitlines()
    if len(lines) != EXPECTED_HOUSEHOLDS:
        raise ValueError(
            f"Expected {EXPECTED_HOUSEHOLDS} households, found {len(lines)}"
        )

    index = pd.date_range(START, periods=N_STEPS, freq="30min", name="timestamp")
    rng = np.random.default_rng(seed)
    data = {}
    for i in rng.permutation(len(lines)):
        name, ts, vals = lines[i].split(":", 2)
        start = pd.to_datetime(ts, format="%Y-%m-%d %H-%M-%S").floor("30min")
        offset = int((START - start) / pd.Timedelta("30min"))
        if offset < 0:
            continue                       # starts after 1 Jan 2013
        values = vals.split(",")
        if len(values) < offset + N_STEPS:
            continue                       # ends before 31 Dec 2013
        data[name] = np.array(values[offset:offset + N_STEPS], dtype="float32")
        if len(data) == n_meters:
            break
    return pd.DataFrame(data, index=index)


def load_sample(path):
    """Load the saved parquet sample (time x household)."""
    return pd.read_parquet(path)