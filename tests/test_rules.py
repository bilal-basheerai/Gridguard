"""Tests for the rule-based detectors, the hybrid combiner and the hard simulator."""

import numpy as np
import pandas as pd

from src.evaluate import evaluate
from src.features import fit_typical_profile
from src.rules import (LowVariabilityRule, RunLengthRule, SustainedLowRule,
                       calibrate_by_budget, combine_max, trailing_run_length)
from src.simulate_hard import inject_hard_anomalies
from tests.test_harness import make_data


def test_trailing_run_length_counts_consecutive_trues():
    mask = np.array([[0], [1], [1], [0], [1], [1], [1]], dtype=bool)
    assert trailing_run_length(mask)[:, 0].tolist() == [0, 1, 2, 0, 1, 2, 3]


def test_zero_rule_counts_zero_runs():
    index = pd.date_range("2013-01-01", periods=8, freq="30min")
    df = pd.DataFrame({"T0": [0.3, 0, 0, 0, 0.2, 0, 0, 0.1]}, index=index, dtype="float32")
    assert RunLengthRule("zero").score(df)["T0"].tolist() == [0, 1, 2, 3, 0, 1, 2, 0]


def test_stuck_rule_counts_repeats_and_ignores_zeros():
    index = pd.date_range("2013-01-01", periods=8, freq="30min")
    df = pd.DataFrame({"T0": [0.3, 0.3, 0.3, 0.2, 0.0, 0.0, 0.0, 0.5]}, index=index,
                      dtype="float32")
    assert RunLengthRule("stuck").score(df)["T0"].tolist() == [0, 1, 2, 0, 0, 0, 0, 0]


def test_all_rules_do_not_look_ahead():
    df = make_data()
    history = df.iloc[:48 * 60]
    profile = fit_typical_profile(history)
    rules = [RunLengthRule("zero"), RunLengthRule("stuck"),
             SustainedLowRule(profile), LowVariabilityRule(history.mean())]
    changed = df.copy()
    changed.iloc[3000:] = 0.0                   # alter only the future
    for rule in rules:
        a = rule.score(df).iloc[:3000].to_numpy()
        b = rule.score(changed).iloc[:3000].to_numpy()
        np.testing.assert_allclose(a, b, err_msg=rule.name)


def test_calibrate_by_budget_respects_the_budget():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 4, size=(2000, 5)).astype(float)
    scores[500:520, 0] = 20                      # one long run in one meter
    threshold = calibrate_by_budget(scores, np.arange(0.5, 30.5), budget=0.01)
    from src.evaluate import count_episodes
    weeks = 5 * 2000 / (48 * 7)
    assert count_episodes(scores > threshold) / weeks <= 0.01


def test_hybrid_flags_exactly_the_union_of_its_parts():
    rng = np.random.default_rng(1)
    index = pd.date_range("2013-01-01", periods=300, freq="30min")
    a = pd.DataFrame(rng.random((300, 3)), index=index, columns=list("xyz"))
    b = pd.DataFrame(rng.random((300, 3)) * 10, index=index, columns=list("xyz"))
    combined, thr = combine_max([(a, 0.9), (b, 9.0)])
    expected = (a.to_numpy() > 0.9) | (b.to_numpy() > 9.0)
    np.testing.assert_array_equal(combined.to_numpy() > thr, expected)


def test_hybrid_recall_is_at_least_that_of_each_part():
    df = make_data()
    out, labels, events = inject_hard_anomalies(df, events_per_meter=3, first_step=48 * 30, seed=2)
    rng = np.random.default_rng(3)
    a = pd.DataFrame(rng.random(labels.shape), index=labels.index, columns=labels.columns)
    b = labels.astype(float) * 0.5 + pd.DataFrame(rng.random(labels.shape) * 0.4,
                                                  index=labels.index, columns=labels.columns)
    ra, _ = evaluate(a, labels, events, 0.97, "a")
    rb, _ = evaluate(b, labels, events, 0.5, "b")
    combined, thr = combine_max([(a, 0.97), (b, 0.5)])
    rc, _ = evaluate(combined, labels, events, thr, "hybrid")
    assert rc["event_recall"] >= max(ra["event_recall"], rb["event_recall"]) - 1e-12


def test_hard_simulator_labels_match_changes_and_use_only_hard_types():
    df = make_data()
    out, labels, events = inject_hard_anomalies(df, events_per_meter=3, first_step=48 * 30, seed=1)
    changed = out.to_numpy() != df.to_numpy()
    assert not (changed & ~labels.to_numpy()).any()
    assert set(events["type"]) <= {"near_zero", "jitter_flat"}
    assert not labels.iloc[:48 * 30].to_numpy().any()
    assert (out.to_numpy()[labels.to_numpy()] > 0).all()      # not exact zeros: hard for exact rules