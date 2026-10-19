"""Day 2: inject anomalies, run the baseline detectors, report results.

Run from the project root:
    python -m src.run_baselines
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.baselines import IsolationForestDetector, RollingZScore
from src.data import load_sample
from src.evaluate import calibrate_threshold, evaluate
from src.features import deviation_from_profile, fit_typical_profile
from src.simulate import inject_anomalies

DATA_PATH = "data/meters_2013.parquet"
OUT_DIR = Path("results")
TEST_START = pd.Timestamp("2013-10-01 00:00")      # Jan-Sep = history, Oct-Dec = test
WARMUP_START = TEST_START - pd.Timedelta(days=11)  # rolling features need warm-up
QUANTILE = 0.99                                    # false-alarm budget
SEED = 42


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_sample(DATA_PATH)

    # 1) Split meters: detectors are fitted on some meters and tested on unseen ones
    rng = np.random.default_rng(SEED)
    meters = list(rng.permutation(df.columns.to_numpy()))
    n_train = int(0.7 * len(meters))
    train_meters, test_meters = meters[:n_train], meters[n_train:]
    print(f"Train meters: {len(train_meters)} | test meters: {len(test_meters)}")

    # 2) Normal behaviour, learned only from clean history (before the test period)
    history = df.loc[:TEST_START - pd.Timedelta(minutes=30)]
    profile = fit_typical_profile(history)
    level = history.mean().clip(lower=1e-3)
    dev_scale = deviation_from_profile(history, profile).std().clip(lower=1e-3)
    train_hist = history[train_meters]

    # 3) Test data: unseen meters with synthetic anomalies injected in Oct-Dec only
    first_step = int(df.index.get_loc(TEST_START))
    test_full, labels_full, events = inject_anomalies(
        df[test_meters], events_per_meter=6, first_step=first_step, seed=SEED
    )
    print(f"Injected {len(events)} anomalies:")
    print(events["type"].value_counts().to_string(), "\n")

    test_in = test_full.loc[WARMUP_START:]
    labels = labels_full.loc[TEST_START:]

    # 4) Detectors
    detectors = [
        RollingZScore(),
        IsolationForestDetector(profile, level, dev_scale).fit(train_hist),
    ]

    summaries, by_type_tables, test_scores = [], [], {}
    for det in detectors:
        print(f"Running {det.name} ...")
        # threshold from clean training data only (no labels, no test data)
        clean = det.score(train_hist.iloc[:, :60])
        clean = clean.loc[clean.index >= clean.index[0] + pd.Timedelta(days=7)]
        threshold = calibrate_threshold(clean.to_numpy(), QUANTILE)

        scores = det.score(test_in).loc[TEST_START:]
        test_scores[det.name] = (scores, threshold)
        summary, by_type = evaluate(scores, labels, events, threshold, det.name)
        summaries.append(summary)
        by_type["detector"] = det.name
        by_type_tables.append(by_type.reset_index())

    results = pd.DataFrame(summaries).set_index("detector")
    by_type = pd.concat(by_type_tables).set_index(["detector", "type"])

    pd.set_option("display.width", 160)
    print("\n=== Overall results (test meters, Oct-Dec) ===")
    print(results.round(3).to_string())
    print("\n=== Per anomaly type ===")
    print(by_type.round(3).to_string())

    results.round(4).to_csv(OUT_DIR / "baselines_overall.csv")
    by_type.round(4).to_csv(OUT_DIR / "baselines_by_type.csv")
    plot_example(test_full, events, test_scores, OUT_DIR / "example_detections.png")
    print(f"\nSaved tables and figure to {OUT_DIR}/")


def plot_example(test_full, events, test_scores, path):
    """Plot one test meter with its injected anomalies and detector flags."""
    meter = events["meter"].value_counts().index[0]
    series = test_full.loc[TEST_START:, meter]
    fig, axes = plt.subplots(len(test_scores), 1, figsize=(14, 3.2 * len(test_scores)),
                             sharex=True)
    for ax, (name, (scores, threshold)) in zip(np.atleast_1d(axes), test_scores.items()):
        ax.plot(series.index, series.values, lw=0.6, color="tab:blue")
        for ev in events[events["meter"] == meter].itertuples():
            ax.axvspan(ev.start, ev.end, color="tab:red", alpha=0.25)
        flagged = scores[meter] > threshold
        ax.scatter(series.index[flagged], series[flagged], s=8, color="black", zorder=3)
        ax.set_title(f"{name}: meter {meter} (red = injected anomaly, black = flagged)")
        ax.set_ylabel("kWh per half hour")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


if __name__ == "__main__":
    main()X