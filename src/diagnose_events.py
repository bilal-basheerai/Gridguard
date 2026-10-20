"""How visible are the injected anomalies?

Some injected faults barely change the readings (for example, zeroing a household
at 3 a.m. when it uses almost nothing). No detector can reliably catch those, so
this report shows how large each fault really is, as a share of the household's
usual half-hourly usage. It uses the same split and seed as the experiments.

Run from the project root:
    python -m src.diagnose_events
"""

import numpy as np
import pandas as pd

from src.data import load_sample
from src.run_baselines import DATA_PATH, OUT_DIR, SEED, TEST_START
from src.simulate import inject_anomalies


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_sample(DATA_PATH)
    rng = np.random.default_rng(SEED)
    meters = list(rng.permutation(df.columns.to_numpy()))
    test_meters = meters[int(0.7 * len(meters)):]

    history = df.loc[:TEST_START - pd.Timedelta(minutes=30)]
    level = history.mean().clip(lower=1e-3)
    original = df[test_meters]
    first_step = int(df.index.get_loc(TEST_START))
    injected, _, events = inject_anomalies(
        original, events_per_meter=6, first_step=first_step, seed=SEED
    )

    rows = []
    for ev in events.itertuples(index=False):
        change = (injected.loc[ev.start:ev.end, ev.meter]
                  - original.loc[ev.start:ev.end, ev.meter]).abs().mean()
        rows.append({"type": ev.type, "visibility": change / level[ev.meter]})
    table = pd.DataFrame(rows)

    summary = table.groupby("type")["visibility"].agg(
        events="size",
        median_visibility="median",
        share_below_10pct=lambda v: (v < 0.10).mean(),
        share_below_25pct=lambda v: (v < 0.25).mean(),
    )
    pd.set_option("display.width", 140)
    print("Visibility = average change in the readings, as a share of the "
          "household's usual half-hourly usage\n")
    print(summary.round(3).to_string())
    print(f"\nAll events: {(table['visibility'] < 0.10).mean():.1%} change the readings by "
          f"less than 10% of usual usage")
    summary.round(4).to_csv(OUT_DIR / "event_visibility.csv")


if __name__ == "__main__":
    main()