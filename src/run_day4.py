"""Day 4: rule-based detectors, hybrid detectors and a harder test set.

Everything is repeated over several random seeds (each seed changes the meter
split and where the anomalies are injected) so results can be reported as mean
and spread. No threshold is tuned on test data: ML thresholds come from clean
training data at a fixed false-alarm budget, and rule thresholds come from clean
training data at a fixed budget of false alarms per meter-week.

Run from the project root (takes a while):
    python -m src.run_day4
"""

import numpy as np
import pandas as pd

from src.baselines import IsolationForestDetector, RollingZScore
from src.data import load_sample
from src.evaluate import calibrate_threshold, evaluate
from src.features import deviation_from_profile, fit_typical_profile
from src.rules import (LowVariabilityRule, RunLengthRule, SustainedLowRule,
                       calibrate_by_budget, combine_max)
from src.run_baselines import (DATA_PATH, OUT_DIR, QUANTILE, TEST_START,
                               WARMUP_START, plot_example)
from src.simulate import inject_anomalies
from src.simulate_hard import inject_hard_anomalies

SEEDS = [42, 1, 2, 3, 4]       # 42 reproduces the Day 2 and Day 3 benchmark
RULE_BUDGET = 0.05             # allowed false-alarm episodes per meter-week, per rule
MIN_RUN_STEPS = 6              # a run rule never fires on less than 3 hours (6 half-hours)
SWEEP_QUANTILES = [0.98, 0.99, 0.995, 0.999]
TEST_SETS = [("standard", inject_anomalies, 6), ("hard", inject_hard_anomalies, 4)]


def run_seed(df, seed, first_seed):
    # 1) Split meters and learn normal behaviour from clean history only
    rng = np.random.default_rng(seed)
    meters = list(rng.permutation(df.columns.to_numpy()))
    n_train = int(0.7 * len(meters))
    train_meters, test_meters = meters[:n_train], meters[n_train:]
    history = df.loc[:TEST_START - pd.Timedelta(minutes=30)]
    profile = fit_typical_profile(history)
    level = history.mean().clip(lower=1e-3)
    dev_scale = deviation_from_profile(history, profile).std().clip(lower=1e-3)
    train_hist = history[train_meters]
    first_step = int(df.index.get_loc(TEST_START))

    # 2) Detectors
    zscore = RollingZScore()
    iforest = IsolationForestDetector(profile, level, dev_scale).fit(train_hist)
    rules = [RunLengthRule("zero"), RunLengthRule("stuck"),
             SustainedLowRule(profile), LowVariabilityRule(level)]
    detectors = [zscore, iforest] + rules

    # 3) Thresholds from clean training data only
    clean_cols = train_hist.iloc[:, :60]
    thresholds, clean_z = {}, None
    for det in (zscore, iforest):
        clean = det.score(clean_cols).iloc[7 * 48:]
        thresholds[det.name] = calibrate_threshold(clean.to_numpy(), QUANTILE)
        if det is zscore:
            clean_z = clean
    for rule in rules:
        clean = rule.score(clean_cols).to_numpy()
        if isinstance(rule, RunLengthRule):
            candidates = np.arange(MIN_RUN_STEPS - 0.5, 96.5, 1.0)   # run lengths 6 to 96
        else:
            sample = clean[::5].ravel()
            candidates = np.unique(np.quantile(sample, np.linspace(0.90, 0.9999, 80)))
        thresholds[rule.name] = calibrate_by_budget(clean, candidates, RULE_BUDGET)
    print("  rule thresholds: " + ", ".join(
        f"{r.name} {thresholds[r.name]:.3g}" for r in rules))

    # 4) Evaluate on the standard and the hard test set
    overall_rows, type_rows, sweep_rows = [], [], []
    for set_name, injector, per_meter in TEST_SETS:
        test_full, labels_full, events = injector(
            df[test_meters], events_per_meter=per_meter, first_step=first_step, seed=seed)
        test_in = test_full.loc[WARMUP_START:]
        labels = labels_full.loc[TEST_START:]

        scores = {d.name: d.score(test_in).loc[TEST_START:] for d in detectors}
        parts = {name: (scores[name], thresholds[name]) for name in scores}
        rule_parts = [parts[r.name] for r in rules]

        table = {name: parts[name] for name in scores}
        table["Rules only"] = combine_max(rule_parts)
        table["Hybrid: z-score + rules"] = combine_max([parts[zscore.name]] + rule_parts)
        table["Hybrid: Isolation Forest + rules"] = combine_max([parts[iforest.name]] + rule_parts)

        for name, (s, thr) in table.items():
            summary, by_type = evaluate(s, labels, events, thr, name)
            overall_rows.append({"seed": seed, "test_set": set_name, **summary})
            for typ, r in by_type.iterrows():
                type_rows.append({"seed": seed, "test_set": set_name, "detector": name,
                                  "type": typ, "events": int(r["events"]),
                                  "event_recall": r["event_recall"],
                                  "median_delay_h": r["median_delay_h"]})

        if seed == first_seed and set_name == "standard":
            for q in SWEEP_QUANTILES:                      # false-alarm budget sweep
                thr_q = calibrate_threshold(clean_z.to_numpy(), q)
                s, thr = combine_max([(scores[zscore.name], thr_q)] + rule_parts)
                summary, _ = evaluate(s, labels, events, thr, "Hybrid: z-score + rules")
                sweep_rows.append({"quantile": q, **summary})
            s, thr = table["Hybrid: z-score + rules"]
            plot_example(test_full, events, {"Hybrid: z-score + rules": (s.loc[TEST_START:], thr)},
                         OUT_DIR / "example_detections_hybrid.png")
    return overall_rows, type_rows, sweep_rows


def pm(mean, std, digits=3):
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = load_sample(DATA_PATH)
    overall_rows, type_rows, sweep_rows = [], [], []
    for seed in SEEDS:
        print(f"Seed {seed} ...")
        o, t, s = run_seed(df, seed, SEEDS[0])
        overall_rows += o
        type_rows += t
        sweep_rows += s

    overall = pd.DataFrame(overall_rows)
    by_type = pd.DataFrame(type_rows)
    sweep = pd.DataFrame(sweep_rows)
    overall.round(4).to_csv(OUT_DIR / "day4_all_seeds_overall.csv", index=False)
    by_type.round(4).to_csv(OUT_DIR / "day4_all_seeds_by_type.csv", index=False)
    sweep.round(4).to_csv(OUT_DIR / "day4_budget_sweep.csv", index=False)

    pd.set_option("display.width", 200)
    metrics = ["event_recall", "median_delay_h", "false_alarms_per_meter_week", "pr_auc"]
    order = list(dict.fromkeys(overall["detector"]))
    for set_name, _, _ in TEST_SETS:
        sub = overall[overall["test_set"] == set_name]
        g = sub.groupby("detector")[metrics].agg(["mean", "std"]).reindex(order)
        shown = pd.DataFrame({m: [pm(g.loc[d, (m, "mean")], g.loc[d, (m, "std")]) for d in order]
                              for m in metrics}, index=order)
        print(f"\n=== {set_name.upper()} test set: mean ± std over {len(SEEDS)} seeds ===")
        print(shown.to_string())

        sub_t = by_type[by_type["test_set"] == set_name]
        recall = sub_t.groupby(["detector", "type"])["event_recall"].mean().unstack("type")
        print(f"\nEvent recall by anomaly type ({set_name} set, mean over seeds):")
        print(recall.reindex(order).round(3).to_string())

    print(f"\n=== False-alarm budget sweep (hybrid z-score + rules, standard set, seed {SEEDS[0]}) ===")
    print(sweep[["quantile", "event_recall", "median_delay_h",
                 "false_alarms_per_meter_week"]].round(3).to_string(index=False))
    print(f"\nSaved tables and figure to {OUT_DIR}/")


if __name__ == "__main__":
    main()