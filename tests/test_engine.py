"""Tests for the GridGuard engine (src/engine.py), run on synthetic households."""
 
import numpy as np
import pandas as pd
import pytest
 
from src.engine import GridGuard, prepare, quality_report, self_benchmark
 
 
def make_households(n_meters=12, days=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2013-01-01", periods=days * 48, freq="30min")
    slot = (idx.hour * 2 + idx.minute // 30).to_numpy()
    base = 0.15 + 0.25 * np.exp(-((slot - 16) / 3.0) ** 2) + 0.6 * np.exp(-((slot - 37) / 4.0) ** 2)
    cols = {}
    for m in range(n_meters):
        noise = rng.gamma(2.0, 0.5, size=len(idx)) / 2.0
        cols[f"M{m:02d}"] = rng.uniform(0.6, 1.8) * base * noise + 0.02
    return pd.DataFrame(cols, index=idx)
 
 
def test_prepare_accepts_half_hourly_and_fills_short_gaps():
    df = make_households(n_meters=2, days=10)
    df.iloc[100, 0] = np.nan
    out = prepare(df)
    assert out.shape == df.shape
    assert not np.isnan(out.iloc[100, 0])
 
 
def test_prepare_sums_15_minute_data_to_half_hours():
    idx = pd.date_range("2013-01-01", periods=96 * 3, freq="15min")
    raw = pd.DataFrame({"A": np.ones(len(idx))}, index=idx)
    out = prepare(raw)
    assert out.index.freq is not None
    assert float(out["A"].iloc[0]) == 2.0
 
 
def test_prepare_rejects_hourly_data_with_clear_message():
    idx = pd.date_range("2013-01-01", periods=200, freq="60min")
    with pytest.raises(ValueError, match="half-hourly"):
        prepare(pd.DataFrame({"A": np.ones(200)}, index=idx))
 
 
def test_prepare_rejects_non_timestamp_index():
    with pytest.raises(ValueError, match="timestamps"):
        prepare(pd.DataFrame({"A": [1.0, 2.0, 3.0]}))
 
 
def test_quality_report_counts_basics():
    df = prepare(make_households(n_meters=3, days=14))
    rep = quality_report(df)
    assert rep["meters"] == 3 and rep["days"] == 14.0 and rep["negative_readings"] == 0
 
 
def test_fit_needs_enough_history():
    df = prepare(make_households(days=10))
    with pytest.raises(ValueError, match="at least"):
        GridGuard().fit(df)
 
 
def test_detects_injected_zero_out_and_flatline_with_reasons():
    df = prepare(make_households(days=120))
    cut = 90 * 48
    guard = GridGuard(model_budget=0.5).fit(df.iloc[:cut])
 
    test = df.copy()
    test.iloc[100 * 48:100 * 48 + 24, 0] = 0.0                  # 12 h of zeros, meter 0
    test.iloc[105 * 48:105 * 48 + 40, 1] = test.iloc[105 * 48, 1]  # 20 h flat-line, meter 1
    eps = guard.detect(test).episodes()
 
    zero = eps[(eps.meter == "M00") & (eps.start >= test.index[100 * 48]) & (eps.start < test.index[101 * 48])]
    flat = eps[(eps.meter == "M01") & (eps.start >= test.index[105 * 48]) & (eps.start < test.index[106 * 48])]
    assert (zero.reason == "Zero-run rule").any()
    assert (flat.reason == "Stuck-reading rule").any()
    assert {"meter", "start", "end", "hours", "reason", "explanation", "strength"} <= set(eps.columns)
 
 
def test_calibration_report_has_one_row_per_component():
    df = prepare(make_households(days=60))
    rep = GridGuard().fit(df).report()
    assert len(rep) == 5 and {"component", "threshold", "budget met"} <= set(rep.columns)
 
 
def test_detect_rejects_unknown_meters():
    df = prepare(make_households(days=60))
    guard = GridGuard().fit(df)
    other = df.copy()
    other.columns = [f"X{i}" for i in range(other.shape[1])]
    with pytest.raises(ValueError, match="not seen"):
        guard.detect(other)
 
 
def test_self_benchmark_reports_sensible_numbers():
    df = prepare(make_households(n_meters=10, days=150))
    res = self_benchmark(df)
    assert res["events_tested"] > 0
    assert 0.0 <= res["event_recall"] <= 1.0
    assert res["event_recall"] > 0.5                    # zero-outs and flat-lines at least
    assert {"zero_out", "flatline"} <= set(res["by_type"].index)
 
 
def test_self_benchmark_rejects_short_history():
    df = prepare(make_households(days=50))
    with pytest.raises(ValueError):
        self_benchmark(df)
 