from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="GridGuard", layout="wide")
DATA = Path(__file__).resolve().parent.parent / "results" / "dashboard_data.parquet"
COLUMNS = ["Start", "End", "Hours", "Reason", "Peak score", "Injected fault?"]


@st.cache_data
def load():
    return pd.read_parquet(DATA)


def episodes(m):
    """Group consecutive alert readings into alert episodes."""
    m = m.reset_index(drop=True)
    m["grp"] = (m["alert"] != m["alert"].shift()).cumsum()
    a = m[m["alert"] == 1]
    if a.empty:
        return pd.DataFrame(columns=COLUMNS)
    g = a.groupby("grp")
    out = pd.DataFrame({
        "Start": g["ts"].min(),
        "End": g["ts"].max(),
        "Hours": g.size() / 2,
        "Peak score": g["model_score"].max().round(2),
        "Rule": g["rule_reason"].agg(lambda s: next((x for x in s if x), "")),
        "Injected fault?": g["anomaly_type"].agg(lambda s: "Yes" if (s != "").any() else "No"),
    })
    out["Reason"] = out["Rule"].map(lambda r: f"Rule: {r}" if r else "Model: unusual pattern")
    return out[COLUMNS].reset_index(drop=True)


df = load()

st.title("GridGuard: smart meter anomaly detection")
st.caption("Hybrid detector: rolling z-score model plus physics-informed rules. "
           "Faults in this demo are simulated and injected into real London household data.")

meter = st.sidebar.selectbox("Meter", sorted(df["meter"].unique()))
show_truth = st.sidebar.checkbox("Show injected (true) faults", value=False)

m = df[df["meter"] == meter].sort_values("ts")
alert_pts = m[m["alert"] == 1]
eps = episodes(m)

c1, c2, c3 = st.columns(3)
c1.metric("Readings", f"{len(m):,}")
c2.metric("Alert episodes", f"{len(eps):,}")
c3.metric("Episodes matching an injected fault", f"{(eps['Injected fault?'] == 'Yes').sum():,}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=m["ts"], y=m["reading"], mode="lines", name="Reading"))
fig.add_trace(go.Scatter(x=alert_pts["ts"], y=alert_pts["reading"], mode="markers",
                         name="Alert", marker=dict(color="red", size=6)))
if show_truth:
    t = m[m["anomaly_type"] != ""]
    fig.add_trace(go.Scatter(x=t["ts"], y=t["reading"], mode="markers", name="Injected fault",
                             marker=dict(color="orange", symbol="x", size=5)))
fig.update_layout(height=450, xaxis_title="Time", yaxis_title="Consumption (kWh per half-hour)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Alert episodes")
st.caption("An episode is a run of consecutive flagged readings. "
           "'Injected fault? = No' means a false alarm on normal behaviour.")
st.dataframe(eps, use_container_width=True, hide_index=True)