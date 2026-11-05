"""Analyse your own data: upload half-hourly smart meter readings and get alerts with reasons."""
 
import io
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, so `src` imports work
 
from src.engine import (MIN_HISTORY_DAYS, WARMUP_STEPS, GridGuard, prepare,  # noqa: E402
                        quality_report, self_benchmark)
from src.loaders import wide_from_columns, wide_from_long  # noqa: E402
from src.simulate import inject_anomalies  # noqa: E402
 
MAX_METERS = 200
WIDE = "Wide: one timestamp column, one column per meter"
LONG = "Long: one row per reading (timestamp, meter, value)"
 
st.set_page_config(page_title="Analyse your data - GridGuard", layout="wide")
 
 
@st.cache_data
def read_csv_bytes(data):
    return pd.read_csv(io.BytesIO(data))
 
 
@st.cache_data
def demo_data():
    """Synthetic households with a few simulated faults near the end (for trying the page)."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2013-01-01", periods=150 * 48, freq="30min")
    slot = (idx.hour * 2 + idx.minute // 30).to_numpy()
    base = 0.15 + 0.25 * np.exp(-((slot - 16) / 3.0) ** 2) + 0.6 * np.exp(-((slot - 37) / 4.0) ** 2)
    cols = {f"DEMO{m:02d}": rng.uniform(0.6, 1.8) * base * (rng.gamma(2.0, 0.5, len(idx)) / 2.0) + 0.02
            for m in range(12)}
    faulty, _, _ = inject_anomalies(pd.DataFrame(cols, index=idx), events_per_meter=2,
                                    first_step=110 * 48, seed=3)
    return faulty
 
 
def plot_meter(det, meter, since):
    j = det.columns.index(meter)
    keep = det.index >= since
    x, y = det.index[keep], det.readings[keep, j]
    hit = det.ratio[keep, j] > 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Reading"))
    fig.add_trace(go.Scatter(x=x[hit], y=y[hit], mode="markers", name="Alert",
                             marker=dict(color="red", size=6)))
    fig.update_layout(height=420, xaxis_title="Time", yaxis_title="Consumption per half-hour")
    return fig
 
 
# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
st.title("Analyse your own smart meter data")
st.write(
    "Upload half-hourly readings and GridGuard learns what is normal for **each of your meters**, "
    "then flags readings that look faulty or tampered with and says why. "
    "It can also estimate how well it would work on your data.")
st.info(
    "Privacy: your file is processed in the app's memory for this session and GridGuard does not "
    "store it. This is a public demo, so do not upload confidential data; for private data, run "
    "the app on your own computer.")
 
use_demo = st.checkbox("Use synthetic example data instead of my own file", value=False)
 
if use_demo:
    wide, source = demo_data(), "example"
else:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.markdown(
            "**Expected format.** Half-hourly readings (15-minute data is summed to 30 minutes), "
            "at least about 5 weeks per meter. Either layout works:\n\n"
            "- *Wide*: `timestamp, meter1, meter2, ...`\n"
            "- *Long*: `timestamp, meter_id, kwh`")
        st.stop()
    raw = read_csv_bytes(uploaded.getvalue())
    layout = st.radio("File layout", [WIDE, LONG])
    dayfirst = st.checkbox("Dates are day-first (for example 31/12/2013)", value=False)
    cols = list(raw.columns)
    try:
        if layout == WIDE:
            ts_col = st.selectbox("Timestamp column", cols, index=0)
            wide = wide_from_columns(raw, ts_col, dayfirst)
        else:
            ts_col = st.selectbox("Timestamp column", cols, index=0)
            meter_col = st.selectbox("Meter ID column", cols, index=min(1, len(cols) - 1))
            value_col = st.selectbox("Reading column", cols, index=min(2, len(cols) - 1))
            wide = wide_from_long(raw, ts_col, meter_col, value_col, dayfirst)
    except ValueError as err:
        st.error(str(err))
        st.stop()
    source = uploaded.name
 
try:
    df = prepare(wide)
except ValueError as err:
    st.error(str(err))
    st.stop()
if df.shape[1] > MAX_METERS:
    st.warning(f"Using the first {MAX_METERS} meters of {df.shape[1]} to keep the app responsive.")
    df = df.iloc[:, :MAX_METERS]
 
# ----------------------------------------------------------------------------
# 2. Data quality
# ----------------------------------------------------------------------------
q = quality_report(df)
st.subheader("Your data")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Meters", q["meters"])
c2.metric("Days of data", q["days"])
c3.metric("Missing readings", f"{q['missing_pct']}%")
c4.metric("Zero readings", f"{q['zero_pct']}%")
st.caption(f"{q['start']:%Y-%m-%d} to {q['end']:%Y-%m-%d}")
if q["negative_readings"]:
    st.warning(f"{q['negative_readings']} negative readings found. Check your units or data source.")
if q["missing_pct"] > 5:
    st.warning("More than 5% of readings are missing, which may reduce detection quality.")
if q["zero_pct"] > 10:
    st.warning("Many zero readings. If these are real faults inside the history period, "
               "GridGuard will learn them as normal and become less sensitive.")
 
# ----------------------------------------------------------------------------
# 3. Settings
# ----------------------------------------------------------------------------
st.subheader("Settings")
start, end = df.index[0], df.index[-1]
min_cut = start + pd.Timedelta(days=MIN_HISTORY_DAYS)
max_cut = end - pd.Timedelta(days=3)
if max_cut <= min_cut:
    st.error(f"Not enough data: you need at least {MIN_HISTORY_DAYS} days of normal history "
             "plus a few more days to check.")
    st.stop()
default_cut = min(max(start + (end - start) * 0.7, min_cut), max_cut)
cut_date = st.slider("Normal history ends on", min_value=min_cut.date(), max_value=max_cut.date(),
                     value=default_cut.date(),
                     help="GridGuard learns normal behaviour from data before this date and "
                          "checks data after it. The history should be mostly normal.")
budget = st.slider("False-alarm budget (alarms per meter per week)", 0.1, 2.0, 0.5, 0.1,
                   help="Lower is stricter: fewer false alarms but more subtle faults are missed.")
cut = int(df.index.searchsorted(pd.Timestamp(cut_date)))
cut_time = df.index[cut]
sig = (source, str(cut_date), float(budget))
 
# ----------------------------------------------------------------------------
# 4. Detection
# ----------------------------------------------------------------------------
st.subheader("Find faulty or tampered readings")
if st.button("Run detection", type="primary"):
    with st.spinner("Learning normal behaviour and scanning your data..."):
        guard = GridGuard(model_budget=budget).fit(df.iloc[:cut])
        det = guard.detect(df.iloc[cut - WARMUP_STEPS:])
        eps = det.episodes()
        eps = eps[eps["end"] >= cut_time].reset_index(drop=True)
    st.session_state["run"] = {"sig": sig, "guard": guard, "det": det, "eps": eps}
 
run = st.session_state.get("run")
if run and run["sig"] != sig:
    st.info("Settings changed. Press **Run detection** again to update the results.")
elif run:
    eps, det, guard = run["eps"], run["det"], run["guard"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Alert episodes", len(eps))
    m2.metric("Meters with alerts", int(eps["meter"].nunique()) if len(eps) else 0)
    m3.metric("Meters checked", df.shape[1])
    if eps.empty:
        st.success("No alerts at this sensitivity. Try a higher false-alarm budget to be more sensitive.")
    else:
        reasons = sorted(eps["reason"].unique())
        chosen = st.multiselect("Show these reasons", reasons, default=reasons)
        view = eps[eps["reason"].isin(chosen)]
        st.dataframe(view, width="stretch", hide_index=True)
        st.download_button("Download alerts (CSV)", view.to_csv(index=False).encode("utf-8"),
                           "gridguard_alerts.csv", "text/csv")
        flagged = list(view["meter"].unique())
        options = flagged + [m for m in det.columns if m not in flagged]
        meter = st.selectbox("Meter to plot", options)
        st.plotly_chart(plot_meter(det, meter, cut_time), width="stretch")
    with st.expander("What GridGuard learned from your history"):
        st.dataframe(guard.report(), width="stretch", hide_index=True)
        st.caption("'Budget met = False' means that component could not reach its false-alarm "
                   "budget on your history, so it is running at its strictest setting.")
 
# ----------------------------------------------------------------------------
# 5. Self-benchmark
# ----------------------------------------------------------------------------
st.subheader("How well would this work on your data?")
st.caption(
    "GridGuard learns from the earlier part of your history, then tests itself on the later part: "
    "once as-is (to count false alarms) and once with simulated faults added (to count how many "
    "it catches). Faults are simulated, so treat this as an estimate, not a guarantee.")
if st.button("Estimate accuracy on my data"):
    try:
        with st.spinner("Running the self-test..."):
            res = self_benchmark(df.iloc[:cut], model_budget=budget)
        st.session_state["bench"] = {"sig": sig, "res": res}
    except ValueError as err:
        st.warning(str(err))
 
bench = st.session_state.get("bench")
if bench and bench["sig"] != sig:
    st.info("Settings changed. Press **Estimate accuracy on my data** again to update.")
elif bench:
    res = bench["res"]
    b1, b2, b3 = st.columns(3)
    b1.metric("Simulated faults caught", f"{res['event_recall']:.0%}")
    b2.metric("False alarms per meter-week (normal data)", res["false_alarms_per_meter_week_on_clean"])
    b3.metric("Typical delay to first alert (hours)", res["typical_delay_h"])
    table = res["by_type"].copy()
    table["event_recall"] = (table["event_recall"] * 100).round(0).astype(int).astype(str) + "%"
    table.index.name = "fault type"
    st.dataframe(table.rename(columns={"events": "faults tested", "event_recall": "caught",
                                       "typical_delay_h": "typical delay (h)"}), width="stretch")
    st.caption("Faults that last hours (zero-outs, flat-lines) are usually easiest. Slow changes "
               "(drift, scale-down) are harder at strict budgets: raise the false-alarm budget to "
               "catch more of them, at the cost of more false alarms.")
 
st.divider()
st.caption("GridGuard is decision support, not proof of theft. An alert means a meter's readings "
           "look unusual and deserve a check, not that anyone did anything wrong.")
 