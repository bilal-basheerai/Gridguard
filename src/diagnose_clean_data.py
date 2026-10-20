"""Do zero-outs and flat-lines already occur naturally in the clean data?

If households normally have long zero readings (empty homes, holidays) or long
constant stretches (for example gaps filled by repeating the last value), then
injected zero-outs and flat-lines look like ordinary data, and no detector can
treat them as anomalies without flooding users with false alarms.

Run from the project root:
    python -m src.diagnose_clean_data
"""

import numpy as np
import pandas as pd

from src.data import load_sample
from src.run_baselines import DATA_PATH, OUT_DIR

ZERO_MIN_STEPS = 12       # 6 hours of zeros
FLAT_MIN_STEPS = 24       # 12 hours of an identical non-zero value


def run_lengths(values):
    """(value, length) of every run of identical consecutive readings."""
    change = np.flatnonzero(np.diff(values) != 0)
    starts = np.r_[0, change + 1]
    ends = np.r_[change + 1, len(values)]
    return values[starts], ends - starts


def count_runs(df):
    """Per meter: number of long zero runs, long constant runs, and the longest of each."""
    rows = []
    for meter in df.columns:
        vals, lens = run_lengths(df[meter].to_numpy())
        zero = (vals == 0) & (lens >= ZERO_MIN_STEPS)
        flat = (vals != 0) & (lens >= FLAT_MIN_STEPS)
        rows.append({
            "meter": meter,
            "zero_runs": int(zero.sum()),
            "flat_runs": int(flat.sum()),
            "longest_zero_h": float(lens[vals == 0].max() / 2) if (vals == 0).any() else 0.0,
            "longest_flat_h": float(lens[vals != 0].max() / 2),
        })
    return pd.DataFrame(rows).set_index("meter")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_sample(DATA_PATH)
    table = count_runs(df)
    meter_weeks = df.shape[1] * len(df) / (48 * 7)

    print(f"Natural patterns in the clean data ({df.shape[1]} households, one year)\n")
    for label, col, steps in [("Zero readings for 6 h or more", "zero_runs", ZERO_MIN_STEPS),
                              ("Identical non-zero readings for 12 h or more", "flat_runs", FLAT_MIN_STEPS)]:
        total = int(table[col].sum())
        share = (table[col] > 0).mean()
        print(f"{label}:")
        print(f"  {total} episodes in total | {total / meter_weeks:.3f} per meter-week | "
              f"{share:.0%} of households have at least one")
    print(f"\nLongest zero run: {table['longest_zero_h'].max():.0f} h | "
          f"longest constant run: {table['longest_flat_h'].max():.0f} h")
    table.to_csv(OUT_DIR / "clean_data_runs.csv")


if __name__ == "__main__":
    main()