"""Evaluation harness shared by every GridGuard detector."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def calibrate_threshold(clean_scores, quantile=0.999, max_rows=2_000_000, seed=0):
    """Threshold = chosen quantile of scores on clean data (no labels needed)."""
    values = np.asarray(clean_scores, dtype="float64").ravel()
    values = values[np.isfinite(values)]
    if len(values) > max_rows:
        values = np.random.default_rng(seed).choice(values, max_rows, replace=False)
    return float(np.quantile(values, quantile))


def count_episodes(flags, gap=12):
    """Count alarm episodes: flags closer than `gap` steps (6 h) form one alarm."""
    total = 0
    for j in range(flags.shape[1]):
        idx = np.flatnonzero(flags[:, j])
        if idx.size:
            total += 1 + int((np.diff(idx) > gap).sum())
    return total


def evaluate(scores, labels, events, threshold, name="detector"):
    """Score one detector on the test period.

    Returns (summary dict, per-anomaly-type DataFrame).
    scores / labels: time x meter tables for the test period
    events: anomaly table with meter, type, start, end
    """
    s = scores.to_numpy()
    y = labels.to_numpy()
    flags = s > threshold

    tp = int((flags & y).sum())
    fp = int((flags & ~y).sum())
    fn = int((~flags & y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pr_auc = float(average_precision_score(y.ravel(), np.nan_to_num(s).ravel()))

    weeks = len(scores) / (48 * 7)
    false_alarms = count_episodes(flags & ~y)
    false_per_meter_week = false_alarms / (scores.shape[1] * weeks)

    rows = []
    for ev in events.itertuples(index=False):
        j = scores.columns.get_loc(ev.meter)
        a = scores.index.get_loc(ev.start)
        b = scores.index.get_loc(ev.end)
        hit = np.flatnonzero(flags[a:b + 1, j])
        rows.append({
            "type": ev.type,
            "detected": hit.size > 0,
            "delay_hours": hit[0] / 2 if hit.size else np.nan,
        })
    per_event = pd.DataFrame(rows)

    by_type = per_event.groupby("type").agg(
        events=("detected", "size"),
        event_recall=("detected", "mean"),
        median_delay_h=("delay_hours", "median"),
    )
    summary = {
        "detector": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "event_recall": float(per_event["detected"].mean()),
        "median_delay_h": float(per_event["delay_hours"].median()),
        "false_alarms_per_meter_week": false_per_meter_week,   # alarm episodes
    }
    return summary, by_type X