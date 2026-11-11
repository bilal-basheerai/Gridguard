"""Tests for the sustained-drop rule (src/slow_rules.py) and its use in the engine."""

import numpy as np
import pandas as pd

from src.engine import GridGuard, prepare
from src.features import fit_typical_profile
from src.slow_rules import SustainedDropRule


def make_households(n_meters=8, days=140, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2013-01-01", periods=days * 48, freq="30min")
    slot = (idx.hour * 2 + idx.minute // 30).to_numpy()
    day = np.arange(len(idx)) // 48
    base = 0.15 + 0.25 * np.exp(-((slot - 16) / 3.0) ** 2) + 0.6 * np.exp(-((slot - 37) / 4.0) ** 2)
    cols = {}
    for m in range(n_meters):
        daily = np.exp(rng.normal(0, 0.12, size=days))[day]
        noise = rng.gamma(2.0, 0.5, size=len(idx)) / 2.0
        cols[f"M{m:02d}"] = rng.uniform(0.6, 1.8) * base * noise * daily + 0.02
    return pd.DataFrame(cols, index=idx)


def test_drop_rule_scores_a_multi_day_scale_down_higher_than_normal_days():
    df = prepare(make_households())
    history = df.iloc[:100 * 48]
    rule = SustainedDropRule(fit_typical_profile(history)).fit(history)

    test = df.iloc[90 * 48:].copy()
    test.iloc[20 * 48:26 * 48, 0] *= 0.4                     # six days at 40 percent of normal
    score = rule.score(test)
    faulty = score.iloc[24 * 48:26 * 48, 0].max()
    normal = score.iloc[: 15 * 48, 1:].to_numpy().max()
    assert faulty > normal


def test_engine_adds_the_rule_only_when_asked():
    df = prepare(make_households(days=70))
    plain = GridGuard().fit(df)
    extended = GridGuard(use_drop_rule=True).fit(df)
    assert len(plain.report()) == 5
    assert len(extended.report()) == 6
    assert "Sustained-drop rule" in set(extended.report()["component"])


def test_engine_flags_a_scale_down_with_the_drop_rule_reason():
    df = prepare(make_households(days=140))
    guard = GridGuard(model_budget=0.5, use_drop_rule=True).fit(df.iloc[:100 * 48])
    test = df.copy()
    test.iloc[110 * 48:118 * 48, 0] *= 0.35                  # eight days at 35 percent of normal
    eps = guard.detect(test).episodes()
    mine = eps[(eps.meter == "M00") & (eps.start >= test.index[108 * 48]) & (eps.start < test.index[122 * 48])]
    assert (mine.reason == "Sustained-drop rule").any()