"""Export hybrid detector output (z-score + rules) for the dashboard.

Run from the project root:
    python -m src.export_dashboard_data
"""

import numpy as np
import pandas as pd

from src.baselines import RollingZScore
from src.data import load_sample
from src.evaluate import calibrate_threshold
from src.features import fit_typical_profile
from src.rules import (LowVariabilityRule, RunLengthRule, SustainedLowRule,
                       calibrate_by_budget)
from src.run_baselines import DATA_PATH, OUT_DIR, QUANTILE, TEST_START, WARMUP_START
from src.run_day4 import MIN_RUN_STEPS, RULE_BUDGET
from src.simulate import inject_anomalies

SEED = 42


def main():
    df = load_sample(DATA_PATH)

    # Same split and normal-behaviour fit as run_day4.py (seed 42)
    rng = np.random.default_rng(SEED)
    meters = list(rng.permutation(df.columns.to_numpy()))
    n_train = int(0.7 * len(meters))
    train_meters, test_meters = meters[:n_train], meters[n_train:]
    history = df.loc[:TEST_START - pd.Timedelta(minutes=30)]
    profile = fit_typical_profile(history)
    level = history.mean().clip(lower=1e-3)
    train_hist = history[train_meters]
    first_step = int(df.index.get_loc(TEST_START))

    zscore = RollingZScore()
    rules = [RunLengthRule("zero"), RunLengthRule("stuck"),
             SustainedLowRule(profile), LowVariabilityRule(level)]

    # Thresholds from clean training data only
    clean_cols = train_hist.iloc[:, :60]
    clean_z = zscore.score(clean_cols).iloc[7 * 48:]
    z_thr = calibrate_threshold(clean_z.to_numpy(), QUANTILE)
    rule_thr = {}
    for rule in rules:
        clean = rule.score(clean_cols).to_numpy()
        if isinstance(rule, RunLengthRule):
            candidates = np.arange(MIN_RUN_STEPS - 0.5, 96.5, 1.0)
        else:
            sample = clean[::5].ravel()
            candidates = np.unique(np.quantile(sample, np.linspace(0.90, 0.9999, 80)))
        rule_thr[rule.name] = calibrate_by_budget(clean, candidates, RULE_BUDGET)

    # Standard test set, same injection as the benchmark
    test_full, labels_full, events = inject_anomalies(
        df[test_meters], events_per_meter=6, first_step=first_step, seed=SEED)
    test_in = test_full.loc[WARMUP_START:]
    labels = labels_full.loc[TEST_START:]
    reading = test_full.loc[TEST_START:]

    z = zscore.score(test_in).loc[TEST_START:]
    z_ratio = z.to_numpy() / max(z_thr, 1e-6)
    rule_ratios = np.stack([
        rule.score(test_in).loc[TEST_START:].to_numpy() / max(rule_thr[rule.name], 1e-6)
        for rule in rules])                                   # (rules, time, meter)

    combined = np.maximum(z_ratio, rule_ratios.max(axis=0))
    alert = combined > 1.0

    short = np.array([r.name.replace(" rule", "").lower() for r in rules])
    reason = np.where((rule_ratios > 1.0).any(axis=0), short[rule_ratios.argmax(axis=0)], "")

    lab = labels.to_numpy()
    truth = (lab != 0) & ~pd.isna(lab)

    n_t, n_m = z.shape
    out = pd.DataFrame({
        "ts": np.repeat(z.index.to_numpy(), n_m),
        "meter": np.tile(z.columns.astype(str).to_numpy(), n_t),
        "reading": reading.to_numpy().astype("float32").ravel(),
        "model_score": np.nan_to_num(z.to_numpy()).astype("float32").ravel(),
        "rule_reason": reason.ravel(),
        "alert": alert.astype("int8").ravel(),
        "anomaly_type": np.where(truth, "injected", "").ravel(),
    })

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "dashboard_data.parquet"
    out.to_parquet(path, index=False)

    print(f"Saved {len(out):,} rows for {n_m} meters to {path}")
    print(f"Alerted share of injected readings: {out.loc[truth.ravel(), 'alert'].mean():.3f}")
    print(f"Alerted share of clean readings:    {out.loc[~truth.ravel(), 'alert'].mean():.4f}")
    print("Alerts by rule reason:")
    print(out.loc[out['alert'] == 1, 'rule_reason'].replace('', 'model only').value_counts())


if __name__ == "__main__":
    main()