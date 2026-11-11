"""Frozen hold-out evaluation of GridGuard. Run once per dataset and do not tune on the result.

    python -m src.run_holdout data/meters_holdout.parquet holdout
    python -m src.run_holdout data/meters_2013.parquet 2013     (same settings, for comparison)

Settings are fixed in FROZEN below: false-alarm budget 0.5, 70/30 time split, 3 simulated
faults per meter, seeds 0 to 2. The sustained-drop rule is evaluated both off and on.
Saves results/holdout_<label>.csv.
"""

import sys

import pandas as pd

from src.data import load_sample
from src.engine import prepare, self_benchmark

FROZEN = dict(model_budget=0.5, test_fraction=0.3, events_per_meter=3, seeds=(0, 1, 2))
MAX_METERS = 150
TYPES = ["drift", "scale_down", "spike", "zero_out", "flatline"]


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    path, label = sys.argv[1], sys.argv[2]
    df = prepare(load_sample(path)).iloc[:, :MAX_METERS]
    print(f"{label}: {df.shape[1]} meters, {df.index[0]} to {df.index[-1]}", flush=True)

    rows = []
    for use_drop_rule in (False, True):
        print(f"running with drop rule {use_drop_rule} ...", flush=True)
        res = self_benchmark(df, use_drop_rule=use_drop_rule, **FROZEN)
        recall = res["by_type"]["event_recall"]
        row = {
            "dataset": label,
            "meters": df.shape[1],
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
            "budget": FROZEN["model_budget"],
            "drop_rule": use_drop_rule,
            "event_recall": res["event_recall"],
            "false_alarms_clean": res["false_alarms_per_meter_week_on_clean"],
            "typical_delay_h": res["typical_delay_h"],
        }
        row.update({t: float(recall.get(t, float("nan"))) for t in TYPES})
        rows.append(row)

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\n" + out.to_string(index=False))
    out.to_csv(f"results/holdout_{label}.csv", index=False)
    print(f"\nSaved results/holdout_{label}.csv")


if __name__ == "__main__":
    main()