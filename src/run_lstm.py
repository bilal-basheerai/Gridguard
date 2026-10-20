"""Day 3: train the LSTM autoencoder and compare it with the baselines.

Run from the project root (after python -m src.run_baselines):
    python -m src.run_lstm
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_sample
from src.evaluate import calibrate_threshold, evaluate
from src.lstm_ae import LSTMAutoencoderDetector
from src.run_baselines import (DATA_PATH, OUT_DIR, QUANTILE, SEED, TEST_START,
                               WARMUP_START, plot_example)
from src.simulate import inject_anomalies


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_sample(DATA_PATH)

    # Same split, history and injected anomalies as run_baselines (same seed),
    # so the comparison with the baselines is like for like.
    rng = np.random.default_rng(SEED)
    meters = list(rng.permutation(df.columns.to_numpy()))
    n_train = int(0.7 * len(meters))
    train_meters, test_meters = meters[:n_train], meters[n_train:]

    history = df.loc[:TEST_START - pd.Timedelta(minutes=30)]
    level = history.mean().clip(lower=1e-3)
    train_hist = history[train_meters]

    first_step = int(df.index.get_loc(TEST_START))
    test_full, labels_full, events = inject_anomalies(
        df[test_meters], events_per_meter=6, first_step=first_step, seed=SEED
    )
    test_in = test_full.loc[WARMUP_START:]
    labels = labels_full.loc[TEST_START:]
    print(f"Train meters: {len(train_meters)} | test meters: {len(test_meters)} | "
          f"anomalies: {len(events)}")

    # Train on clean history of the training meters (the last 20 are held out)
    start = time.time()
    det = LSTMAutoencoderDetector(level)
    det.fit(train_hist)
    print(f"Training took {(time.time() - start) / 60:.1f} minutes")
    det.save(OUT_DIR / "lstm_ae.pt")

    # Threshold from the held-out clean meters only (no labels, no test data)
    held_out = train_hist.iloc[:, -det.n_val_meters:]
    clean = det.score(held_out, step=2)
    threshold = calibrate_threshold(clean.to_numpy(), QUANTILE)

    print("Scoring the test meters ...")
    scores = det.score(test_in).loc[TEST_START:]
    summary, by_type = evaluate(scores, labels, events, threshold, det.name)

    overall = pd.DataFrame([summary]).set_index("detector")
    by_type["detector"] = det.name
    by_type = by_type.reset_index().set_index(["detector", "type"])

    # Combine with the baseline results saved by run_baselines
    all_overall = pd.concat([
        pd.read_csv(OUT_DIR / "baselines_overall.csv", index_col=0), overall.round(4)
    ])
    all_by_type = pd.concat([
        pd.read_csv(OUT_DIR / "baselines_by_type.csv", index_col=[0, 1]), by_type.round(4)
    ])

    pd.set_option("display.width", 160)
    print("\n=== Overall results (test meters, Oct-Dec) ===")
    print(all_overall.round(3).to_string())
    print("\n=== Event recall by anomaly type ===")
    print(all_by_type["event_recall"].unstack(0).round(3).to_string())
    print("\n=== Median detection delay (hours) by anomaly type ===")
    print(all_by_type["median_delay_h"].unstack(0).round(1).to_string())

    all_overall.to_csv(OUT_DIR / "all_detectors_overall.csv")
    all_by_type.to_csv(OUT_DIR / "all_detectors_by_type.csv")
    plot_example(test_full, events, {det.name: (scores, threshold)},
                 OUT_DIR / "example_detections_lstm.png")
    plot_training(det.history, OUT_DIR / "lstm_training_curve.png")
    print(f"\nSaved tables, model and figures to {OUT_DIR}/")


def plot_training(history, path):
    plt.figure(figsize=(6, 3.5))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="validation (held-out meters)")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.title("LSTM autoencoder training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


if __name__ == "__main__":
    main()