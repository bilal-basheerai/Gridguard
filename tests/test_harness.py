"""Sanity tests for the GridGuard evaluation harness.

These tests check that the measuring stick is sound, so that detector results
can be trusted: labels match the injected faults, nothing looks into the
future, and the metrics behave correctly on cases with known answers.

Run from the project root:  python -m pytest -v
"""

import numpy as np
import pandas as pd

from src.baselines import IsolationForestDetector, RollingZScore
from src.evaluate import calibrate_threshold, count_episodes, evaluate
from src.features import fit_typical_profile
from src.simulate import inject_anomalies


def make_data(n_steps=48 * 120, n_meters=8, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2013-01-01", periods=n_steps, freq="30min", name="timestamp")
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    shape = 1 + 0.6 * np.exp(-((hour - 19) / 2.5) ** 2)
    base = rng.uniform(0.15, 0.5, n_meters)
    values = shape[:, None] * base[None, :] * rng.gamma(6, 1 / 6, (n_steps, n_meters))
    return pd.DataFrame(values.astype("float32"), index=index,
                        columns=[f"T{i}" for i in range(n_meters)])


def make_injected(first_step=48 * 30):
    df = make_data()
    out, labels, events = inject_anomalies(df, events_per_meter=3,
                                           first_step=first_step, seed=1)
    return df, out, labels, events


# ---- 1. The simulator and its labels ---------------------------------------

def test_labels_match_changes():
    """Readings change only where labelled, and the label count equals event lengths."""
    df, out, labels, events = make_injected()
    changed = (out.to_numpy() != df.to_numpy())
    assert not (changed & ~labels.to_numpy()).any(), "unlabelled change found"
    lengths = sum(
        out.index.get_loc(e.end) - out.index.get_loc(e.start) + 1
        for e in events.itertuples()
    )
    assert int(labels.to_numpy().sum()) == lengths


def test_no_events_before_first_step():
    _, out, labels, _ = make_injected(first_step=48 * 30)
    assert not labels.iloc[:48 * 30].to_numpy().any()


def test_input_is_not_modified():
    df = make_data()
    before = df.copy()
    inject_anomalies(df, events_per_meter=3, first_step=100, seed=1)
    pd.testing.assert_frame_equal(df, before)


def test_events_do_not_overlap_per_meter():
    _, out, _, events = make_injected()
    for _, g in events.groupby("meter"):
        g = g.sort_values("start")
        assert (g["start"].iloc[1:].to_numpy() > g["end"].iloc[:-1].to_numpy()).all()


# ---- 2. The metrics on cases with known answers -----------------------------

def test_oracle_detector_scores_perfectly():
    """A detector that is handed the labels must score 100%."""
    _, _, labels, events = make_injected()
    scores = labels.astype(float)
    summary, _ = evaluate(scores, labels, events, threshold=0.5, name="oracle")
    assert summary["precision"] == 1.0 and summary["recall"] == 1.0
    assert summary["event_recall"] == 1.0
    assert summary["false_alarms_per_meter_week"] == 0.0
    assert abs(summary["pr_auc"] - 1.0) < 1e-9


def test_random_detector_is_near_chance():
    """Random scores must give PR-AUC close to the share of anomalous readings."""
    _, _, labels, events = make_injected()
    rng = np.random.default_rng(0)
    scores = pd.DataFrame(rng.random(labels.shape), index=labels.index,
                          columns=labels.columns)
    summary, _ = evaluate(scores, labels, events,
                          threshold=calibrate_threshold(scores.to_numpy(), 0.99), name="random")
    base_rate = labels.to_numpy().mean()
    assert abs(summary["pr_auc"] - base_rate) < 0.03


def test_detection_delay_is_measured_from_event_start():
    index = pd.date_range("2013-01-01", periods=200, freq="30min")
    labels = pd.DataFrame(False, index=index, columns=["T0"])
    labels.iloc[50:100, 0] = True
    events = pd.DataFrame([("T0", "drift", index[50], index[99])],
                          columns=["meter", "type", "start", "end"])
    scores = pd.DataFrame(0.0, index=index, columns=["T0"])
    scores.iloc[55, 0] = 1.0                     # first alarm 5 steps = 2.5 h late
    summary, _ = evaluate(scores, labels, events, threshold=0.5, name="delay")
    assert summary["median_delay_h"] == 2.5


def test_false_alarm_episodes_are_grouped():
    near = np.zeros((100, 1), dtype=bool); near[[10, 20], 0] = True    # gap 10: one alarm
    far = np.zeros((100, 1), dtype=bool); far[[10, 40], 0] = True      # gap 30: two alarms
    assert count_episodes(near) == 1
    assert count_episodes(far) == 2


def test_threshold_is_the_requested_quantile_of_clean_scores():
    clean = np.arange(1, 1001, dtype=float)
    assert abs(calibrate_threshold(clean, 0.99) - np.quantile(clean, 0.99)) < 1e-9


# ---- 3. No look-ahead: a score may not depend on the future -----------------

def test_rolling_zscore_does_not_look_ahead():
    df = make_data()
    changed = df.copy()
    changed.iloc[3000:] *= 5.0                   # alter only the future
    a = RollingZScore().score(df).iloc[:3000]
    b = RollingZScore().score(changed).iloc[:3000]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


def test_isolation_forest_features_do_not_look_ahead():
    df = make_data()
    profile = fit_typical_profile(df.iloc[:48 * 60])
    level = df.iloc[:48 * 60].mean()
    scale = (df.iloc[:48 * 60] - df.iloc[:48 * 60].mean()).std()
    det = IsolationForestDetector(profile, level, scale)
    changed = df.copy()
    changed.iloc[3000:] *= 5.0
    a = det._features(df).reshape(len(df), df.shape[1], -1)[:3000]
    b = det._features(changed).reshape(len(df), df.shape[1], -1)[:3000]
    np.testing.assert_allclose(a, b)