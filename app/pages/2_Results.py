"""Results: the evidence behind GridGuard (reads the benchmark CSV files in results/)."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RESULTS = Path(__file__).resolve().parents[2] / "results"

TYPE_LABELS = {
    "drift": "Drift", "flatline": "Flat-line", "scale_down": "Scale-down", "spike": "Spike",
    "zero_out": "Zero-out", "jitter_flat": "Flat with jitter", "near_zero": "Near-zero",
}
TYPE_ORDER = {
    "standard": ["zero_out", "flatline", "spike", "drift", "scale_down"],
    "hard": ["jitter_flat", "near_zero"],
}
SET_LABELS = {"standard": "Standard test set (5 fault types)",
              "hard": "Hard test set (disguised faults)"}
DEFAULT_DETECTORS = ["Rolling z-score", "Isolation Forest", "Rules only", "Hybrid: z-score + rules"]
METRICS = ["event_recall", "median_delay_h", "false_alarms_per_meter_week", "pr_auc"]

st.set_page_config(page_title="Results - GridGuard", layout="wide")


@st.cache_data
def load(name):
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


def pct(series):
    return (series * 100).round(1).astype(str) + "%"


st.title("How well does GridGuard work?")
st.write(
    "All numbers below come from a leakage-free benchmark on real London household data with "
    "**simulated faults**: detectors learn normal behaviour from January to September, and faults "
    "are injected only in October to December on households the detectors never saw. Every "
    "threshold is set from clean data only, never from the test faults.")

# ----------------------------------------------------------------------------
# 1. Statistical models vs rules vs hybrid
# ----------------------------------------------------------------------------
st.header("1. Statistical models, rules and the hybrid")
overall, by_type = load("day4_all_seeds_overall.csv"), load("day4_all_seeds_by_type.csv")
if overall is None or by_type is None:
    st.warning("The benchmark files (day4_all_seeds_overall.csv and day4_all_seeds_by_type.csv) "
               "were not found in the results folder.")
else:
    test_set = st.radio("Test set", ["standard", "hard"], horizontal=True,
                        format_func=lambda k: SET_LABELS[k])
    means = (overall[overall["test_set"] == test_set].groupby("detector")[METRICS].mean())

    if test_set == "standard" and {"Rolling z-score", "Hybrid: z-score + rules"} <= set(means.index):
        z, h = means.loc["Rolling z-score"], means.loc["Hybrid: z-score + rules"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Faults caught: statistical model alone", f"{z['event_recall']:.0%}")
        c2.metric("Faults caught: model + rules (hybrid)", f"{h['event_recall']:.0%}",
                  delta=f"{(h['event_recall'] - z['event_recall']) * 100:+.0f} points")
        c3.metric("False alarms per meter-week (hybrid)",
                  f"{h['false_alarms_per_meter_week']:.2f}",
                  delta=f"{h['false_alarms_per_meter_week'] - z['false_alarms_per_meter_week']:+.2f} vs model alone",
                  delta_color="inverse")

    detectors = st.multiselect("Detectors to compare", list(means.index),
                               default=[d for d in DEFAULT_DETECTORS if d in means.index])
    types_df = (by_type[by_type["test_set"] == test_set]
                .groupby(["detector", "type"])["event_recall"].mean().unstack("type"))
    types = [t for t in TYPE_ORDER[test_set] if t in types_df.columns]
    fig = go.Figure()
    for d in detectors:
        if d in types_df.index:
            fig.add_trace(go.Bar(name=d, x=[TYPE_LABELS[t] for t in types],
                                 y=(types_df.loc[d, types] * 100).round(1)))
    fig.update_layout(barmode="group", height=420, yaxis_title="Faults caught (%)",
                      xaxis_title="Fault type", legend_title_text="Detector")
    st.plotly_chart(fig, width="stretch")

    table = means.loc[[d for d in detectors if d in means.index]].copy()
    table["event_recall"] = pct(table["event_recall"])
    table["median_delay_h"] = table["median_delay_h"].round(1)
    table["false_alarms_per_meter_week"] = table["false_alarms_per_meter_week"].round(2)
    table["pr_auc"] = table["pr_auc"].round(3)
    st.dataframe(table.rename(columns={
        "event_recall": "faults caught", "median_delay_h": "typical delay (h)",
        "false_alarms_per_meter_week": "false alarms / meter-week", "pr_auc": "PR-AUC"}),
        width="stretch")
    st.caption("Mean over 5 random seeds. Statistical models catch gradual and extreme changes; "
               "simple rules catch zero-outs and flat-lines almost perfectly. The hybrid combines both. "
               "The hard set uses disguised faults and is mostly missed: treat it as a stress test.")

# ----------------------------------------------------------------------------
# 2. Trading false alarms for recall
# ----------------------------------------------------------------------------
st.header("2. The false-alarm trade-off")
sweep = load("day4_budget_sweep.csv")
if sweep is None:
    st.warning("day4_budget_sweep.csv was not found in the results folder.")
else:
    fig = go.Figure(go.Scatter(
        x=sweep["false_alarms_per_meter_week"], y=(sweep["event_recall"] * 100).round(1),
        mode="lines+markers+text", text=[f"{q}" for q in sweep["quantile"]],
        textposition="top center"))
    fig.update_layout(height=380, xaxis_title="False alarms per meter per week",
                      yaxis_title="Faults caught (%)")
    st.plotly_chart(fig, width="stretch")
    st.caption("Hybrid (z-score + rules), standard test set, one seed. Labels show the z-score "
               "quantile used for the threshold. A stricter threshold means fewer false alarms but "
               "fewer faults caught. This is why GridGuard lets you choose your own budget.")

# ----------------------------------------------------------------------------
# 3. Slow faults
# ----------------------------------------------------------------------------
st.header("3. Catching slow faults: the sustained-drop rule")
drop = load("drop_rule_comparison.csv")
if drop is None:
    st.warning("drop_rule_comparison.csv was not found in the results folder.")
else:
    st.write(
        "Gradual drift and multi-day scale-downs were the weak spot: the short-window rules cannot "
        "see a meter that quietly reads 30 to 60% low for days. A rule comparing the last 3 days of "
        "usage with each household's normal level fixes most of that, with almost no extra false alarms.")
    budgets = sorted(drop["budget"].unique())
    budget = st.radio("False-alarm budget (alarms per meter per week)", budgets, horizontal=True,
                      index=min(1, len(budgets) - 1))
    on = drop["drop_rule"].astype(str) == "True"
    before = drop[(drop["budget"] == budget) & ~on].iloc[0]
    after = drop[(drop["budget"] == budget) & on].iloc[0]

    d1, d2, d3 = st.columns(3)
    d1.metric("All faults caught", f"{after['event_recall']:.0%}",
              delta=f"{(after['event_recall'] - before['event_recall']) * 100:+.0f} points")
    d2.metric("Drift caught", f"{after['drift']:.0%}",
              delta=f"{(after['drift'] - before['drift']) * 100:+.0f} points")
    d3.metric("Scale-down caught", f"{after['scale_down']:.0%}",
              delta=f"{(after['scale_down'] - before['scale_down']) * 100:+.0f} points")

    cols = [c for c in ["zero_out", "flatline", "spike", "drift", "scale_down"] if c in drop.columns]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Without drop rule", x=[TYPE_LABELS[c] for c in cols],
                         y=[round(before[c] * 100, 1) for c in cols]))
    fig.add_trace(go.Bar(name="With drop rule", x=[TYPE_LABELS[c] for c in cols],
                         y=[round(after[c] * 100, 1) for c in cols]))
    fig.update_layout(barmode="group", height=400, yaxis_title="Faults caught (%)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"False alarms on normal data: {before['false_alarms_clean']:.3f} without the rule, "
               f"{after['false_alarms_clean']:.3f} with it (per meter per week). "
               "Measured with GridGuard's self-benchmark on 40 households, learning from the earlier "
               "part of the year and testing on the later part.")

# ----------------------------------------------------------------------------
# 4. Hold-out check
# ----------------------------------------------------------------------------
st.header("4. Does it hold on a different year and different households?")
h13, h12 = load("holdout_2013.csv"), load("holdout_2012.csv")
if h13 is None or h12 is None:
    st.info("The hold-out results have not been generated yet "
            "(results/holdout_2013.csv and results/holdout_2012.csv).")
else:
    st.write(
        "All settings were **frozen before this run** (false-alarm budget 0.5, 3-day drop window, "
        "70/30 time split, 3 simulated faults per meter, 3 seeds) and the check was run once. "
        "The 2012 households are different from the 2013 sample, and the data is from a different year.")
    on13 = h13[h13["drop_rule"].astype(str) == "True"].iloc[0]
    on12 = h12[h12["drop_rule"].astype(str) == "True"].iloc[0]
    e1, e2, e3 = st.columns(3)
    e1.metric("Faults caught: 2013 sample", f"{on13['event_recall']:.0%}")
    e2.metric("Faults caught: 2012 hold-out", f"{on12['event_recall']:.0%}",
              delta=f"{(on12['event_recall'] - on13['event_recall']) * 100:+.0f} points")
    e3.metric("False alarms per meter-week: 2012 hold-out", f"{on12['false_alarms_clean']:.2f}",
              delta=f"{on12['false_alarms_clean'] - on13['false_alarms_clean']:+.2f} vs 2013",
              delta_color="inverse")

    cols = [c for c in ["zero_out", "flatline", "spike", "drift", "scale_down"] if c in h12.columns]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"2013 sample ({int(on13['meters'])} meters)",
                         x=[TYPE_LABELS[c] for c in cols], y=[round(on13[c] * 100, 1) for c in cols]))
    fig.add_trace(go.Bar(name=f"2012 hold-out ({int(on12['meters'])} meters)",
                         x=[TYPE_LABELS[c] for c in cols], y=[round(on12[c] * 100, 1) for c in cols]))
    fig.update_layout(barmode="group", height=400, yaxis_title="Faults caught (%)",
                      title_text="With the sustained-drop rule, budget 0.5")
    st.plotly_chart(fig, width="stretch")

    both = pd.concat([h13, h12])
    both["drop_rule"] = both["drop_rule"].astype(str).map({"True": "on", "False": "off"})
    show = both[["dataset", "meters", "drop_rule", "event_recall", "false_alarms_clean",
                 "typical_delay_h"] + cols].copy()
    for c in ["event_recall"] + cols:
        show[c] = pct(show[c])
    st.dataframe(show.rename(columns={
        "dataset": "data", "drop_rule": "drop rule", "event_recall": "faults caught",
        "false_alarms_clean": "false alarms / meter-week", "typical_delay_h": "typical delay (h)"}),
        width="stretch", hide_index=True)
    st.caption("Same code, same settings, same number of meters per run. Whatever the numbers show "
               "is reported as it came out.")

# ----------------------------------------------------------------------------
# 5. Example and limits
# ----------------------------------------------------------------------------
st.header("5. Example detections")
image = RESULTS / "example_detections_hybrid.png"
if image.exists():
    st.image(str(image), caption="Hybrid detector on test households (simulated faults).")

st.header("Limits of this evidence")
st.markdown(
    "- Faults are **simulated**, not real theft or attacks. Real tampering may look different.\n"
    "- One city of residential data. The hold-out check uses a different year and different "
    "households from the same dataset, not a different utility or country.\n"
    "- The drop rule was designed after seeing which faults the first detectors missed; the hold-out "
    "check above is the test on data it had not seen.\n"
    "- Spikes and disguised faults are still weakly detected.\n"
    "- GridGuard is decision support, not proof of theft.")