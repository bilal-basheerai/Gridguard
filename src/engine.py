"""GridGuard engine: learn from a user's own meter history, detect, explain and self-benchmark.
 
Typical use (wide table: half-hourly DatetimeIndex, one column per meter):
 
    df = prepare(raw_df)                          # validate and tidy
    guard = GridGuard(model_budget=0.5).fit(df.iloc[:cut])
    result = guard.detect(df)
    result.episodes()                             # one row per alert, with a reason
    self_benchmark(df.iloc[:cut])                 # how well would this work on YOUR data?
 
Thresholds are never tuned on labels. The statistical model and each rule get a
false-alarm budget (alarm episodes per meter-week) measured on the user's own
history, which is assumed to be mostly normal behaviour.
"""
 
from dataclasses import dataclass
 
import numpy as np
import pandas as pd
 
from src.baselines import RollingZScore
from src.evaluate import count_episodes, evaluate
from src.features import fit_typical_profile
from src.rules import LowVariabilityRule, RunLengthRule, SustainedLowRule
from src.simulate import inject_anomalies
 
STEPS_PER_DAY = 48
STEPS_PER_WEEK = 7 * STEPS_PER_DAY
WARMUP_STEPS = 7 * STEPS_PER_DAY       # the rolling z-score needs a week of context
MIN_HISTORY_DAYS = 28                  # shortest history we accept for fitting
MIN_RUN_STEPS = 6                      # run rules never fire on less than 3 hours
CALIBRATION_METERS = 100               # cap on meters used for calibration (speed)
EPISODE_GAP = 12                       # alarms closer than 6 hours form one episode
 
MODEL_NAME = "Statistical model"
 
EXPLANATIONS = {
    MODEL_NAME: "Reading is unusual compared with this meter's own recent 7-day pattern.",
    "Zero-run rule": "The meter reported exactly zero for hours.",
    "Stuck-reading rule": "The meter repeated the same non-zero value for hours.",
    "Sustained-low rule": "Usage stayed far below what is normal for this household at this time of day.",
    "Low-variability rule": "Readings barely changed for hours, as if the meter were stuck.",
}
 
 
# ----------------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------------
 
def prepare(df):
    """Validate and tidy an uploaded wide table (DatetimeIndex x meter columns).
 
    Accepts 30-minute data, or 15-minute data (summed to 30 minutes). Gaps of up to
    one hour are interpolated; longer gaps stay missing (NaN). Raises ValueError with
    a plain-language message if the table cannot be used.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The first column must be timestamps (dates and times).")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] == 0:
        raise ValueError("No numeric meter columns were found.")
    if len(df) < 3:
        raise ValueError("Not enough rows.")
 
    step = df.index.to_series().diff().median()
    if step == pd.Timedelta(minutes=15):
        df = df.resample("30min").sum(min_count=1)
    elif step != pd.Timedelta(minutes=30):
        raise ValueError(
            f"Readings are {step} apart. GridGuard currently needs half-hourly "
            "(30-minute) data, or 15-minute data that it can sum to 30 minutes.")
    df = df.asfreq("30min").interpolate(limit=2)
    df.columns = df.columns.astype(str)
    return df.astype("float32")
 
 
def quality_report(df):
    """Simple data-quality summary to show the user before fitting."""
    values = df.to_numpy()
    observed = np.isfinite(values)
    return {
        "meters": int(df.shape[1]),
        "days": round(len(df) / STEPS_PER_DAY, 1),
        "start": df.index[0],
        "end": df.index[-1],
        "missing_pct": round(100 * float((~observed).mean()), 2),
        "negative_readings": int((values < 0).sum()),
        "zero_pct": round(100 * float((values[observed] == 0).mean()), 2),
    }
 
 
# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
 
def _calibrate(clean, candidates, budget):
    """Smallest threshold whose false-alarm episodes per meter-week fit the budget.
 
    Returns (threshold, budget_met). If no candidate meets the budget, the strictest
    candidate is returned with budget_met=False.
    """
    s = np.nan_to_num(np.asarray(clean, dtype="float64"))
    meter_weeks = s.shape[1] * s.shape[0] / STEPS_PER_WEEK
    for cand in candidates:
        if count_episodes(s > cand, EPISODE_GAP) / meter_weeks <= budget:
            return float(cand), True
    return float(candidates[-1]), False
 
 
# ----------------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------------
 
@dataclass
class Detection:
    """Result of GridGuard.detect: per-reading ratios, where >1 means alarm."""
    index: pd.DatetimeIndex
    columns: list
    names: list
    stack: np.ndarray        # (components, time, meter) score / threshold, float32
    readings: np.ndarray     # (time, meter)
 
    @property
    def ratio(self):
        return self.stack.max(axis=0)
 
    @property
    def alert(self):
        return pd.DataFrame(self.ratio > 1.0, index=self.index, columns=self.columns)
 
    def ratio_frame(self):
        return pd.DataFrame(self.ratio, index=self.index, columns=self.columns)
 
    def episodes(self, gap=EPISODE_GAP):
        """One row per alert episode, with the reason that caused it."""
        flags = self.ratio > 1.0
        rows = []
        for j, meter in enumerate(self.columns):
            idx = np.flatnonzero(flags[:, j])
            if not idx.size:
                continue
            for seg in np.split(idx, np.flatnonzero(np.diff(idx) > gap) + 1):
                a, b = int(seg[0]), int(seg[-1])
                peaks = self.stack[:, a:b + 1, j].max(axis=1)
                # Name the most specific rule that fired (rules are listed from most to
                # least specific); fall back to the statistical model. Ratios are not
                # compared across rules, because their scales differ.
                k = next((i for i in range(1, len(peaks)) if peaks[i] > 1.0), 0)
                rows.append({
                    "meter": meter,
                    "start": self.index[a],
                    "end": self.index[b],
                    "hours": (b - a + 1) / 2,
                    "reason": self.names[k],
                    "explanation": EXPLANATIONS[self.names[k]],
                    "strength": round(float(peaks[k]), 2),
                })
        cols = ["meter", "start", "end", "hours", "reason", "explanation", "strength"]
        return pd.DataFrame(rows, columns=cols).sort_values(["start", "meter"]).reset_index(drop=True)
 
 
class GridGuard:
    """Hybrid detector: a statistical model plus four physics-informed rules.
 
    model_budget  false-alarm episodes per meter-week allowed for the statistical model
    rule_budget   the same, for each rule
    """
 
    def __init__(self, model_budget=0.5, rule_budget=0.05):
        self.model_budget = model_budget
        self.rule_budget = rule_budget
 
    def fit(self, history):
        """Learn normal behaviour and calibrate thresholds from mostly-normal history."""
        if len(history) < MIN_HISTORY_DAYS * STEPS_PER_DAY:
            raise ValueError(f"Need at least {MIN_HISTORY_DAYS} days of history to learn "
                             f"normal behaviour; got {len(history) / STEPS_PER_DAY:.0f}.")
        self.meters = list(history.columns)
        self.profile = fit_typical_profile(history)
        self.level = history.mean().clip(lower=1e-3)
 
        self.zscore = RollingZScore()
        self.rules = [RunLengthRule("zero"), RunLengthRule("stuck"),
                      SustainedLowRule(self.profile), LowVariabilityRule(self.level)]
        self.names = [MODEL_NAME] + [r.name for r in self.rules]
 
        cal = history.iloc[:, :CALIBRATION_METERS]
        self.thresholds, self.budget_met = {}, {}
 
        clean_z = self.zscore.score(cal).iloc[WARMUP_STEPS:].to_numpy()
        z_candidates = np.unique(np.quantile(clean_z[::3].ravel(), np.linspace(0.90, 0.99999, 120)))
        self.thresholds[MODEL_NAME], self.budget_met[MODEL_NAME] = _calibrate(
            clean_z, z_candidates, self.model_budget)
 
        for rule in self.rules:
            clean = rule.score(cal).to_numpy()
            if isinstance(rule, RunLengthRule):
                candidates = np.arange(MIN_RUN_STEPS - 0.5, 96.5, 1.0)
            else:
                candidates = np.unique(np.quantile(clean[::5].ravel(), np.linspace(0.90, 0.9999, 80)))
            self.thresholds[rule.name], self.budget_met[rule.name] = _calibrate(
                clean, candidates, self.rule_budget)
        return self
 
    def report(self):
        """Calibration summary: which thresholds were learned and whether budgets were met."""
        return pd.DataFrame([{
            "component": n,
            "threshold": round(self.thresholds[n], 3),
            "budget (alarms/meter-week)": self.model_budget if n == MODEL_NAME else self.rule_budget,
            "budget met": self.budget_met[n],
        } for n in self.names])
 
    def detect(self, df):
        """Score every reading. df must contain only meters seen during fit."""
        unknown = [c for c in df.columns if c not in self.profile.columns]
        if unknown:
            raise ValueError(f"Meters not seen during fit: {unknown[:5]}")
        parts = [self.zscore.score(df).to_numpy() / max(self.thresholds[MODEL_NAME], 1e-6)]
        for rule in self.rules:
            parts.append(rule.score(df).to_numpy() / max(self.thresholds[rule.name], 1e-6))
        stack = np.nan_to_num(np.stack(parts)).astype("float32")
        return Detection(index=df.index, columns=list(df.columns), names=self.names,
                         stack=stack, readings=df.to_numpy())
 
 
# ----------------------------------------------------------------------------
# Self-benchmark
# ----------------------------------------------------------------------------
 
def self_benchmark(history, model_budget=0.5, test_fraction=0.3, events_per_meter=3,
                   seeds=(0, 1, 2)):
    """Estimate how well GridGuard works on THIS user's data.
 
    The earlier part of the history is used to learn normal behaviour. The later part is
    (a) checked as-is, which measures false alarms if it is normal, and (b) corrupted with
    simulated faults (zero-out, flat-line, scale-down, drift, spike), which measures recall.
    Faults are simulated, so treat the result as an estimate, not a guarantee.
    """
    n = len(history)
    cut = int(n * (1 - test_fraction))
    if cut < MIN_HISTORY_DAYS * STEPS_PER_DAY or cut < WARMUP_STEPS:
        raise ValueError("History is too short to hold out a test period. "
                         "Provide roughly 60 days or more.")
    if n - cut < 30 * STEPS_PER_DAY:
        raise ValueError("The held-out test period is shorter than 30 days, which is too "
                         "short to place simulated faults. Use more history.")
 
    guard = GridGuard(model_budget=model_budget).fit(history.iloc[:cut])
    test_in = history.iloc[cut - WARMUP_STEPS:]
    n_test = len(test_in) - WARMUP_STEPS
    weeks = n_test / STEPS_PER_WEEK
 
    clean_flags = guard.detect(test_in).ratio[WARMUP_STEPS:] > 1.0
    clean_fa = count_episodes(clean_flags, EPISODE_GAP) / (test_in.shape[1] * weeks)
 
    summaries, by_types = [], []
    for seed in seeds:
        injected, labels, events = inject_anomalies(
            test_in, events_per_meter=events_per_meter, first_step=WARMUP_STEPS, seed=seed)
        if events.empty:
            continue
        ratio = guard.detect(injected).ratio_frame().iloc[WARMUP_STEPS:]
        summary, by_type = evaluate(ratio, labels.iloc[WARMUP_STEPS:], events, 1.0, "GridGuard")
        summaries.append(summary)
        by_types.append(by_type.reset_index())
    if not summaries:
        raise ValueError("Could not place any simulated faults; provide more history.")
 
    overall = pd.DataFrame(summaries)
    allt = pd.concat(by_types)
    allt["detected"] = allt["event_recall"] * allt["events"]
    grouped = allt.groupby("type").agg(events=("events", "sum"), detected=("detected", "sum"),
                                       typical_delay_h=("median_delay_h", "mean"))
    by_type = pd.DataFrame({
        "events": grouped["events"].astype(int),
        "event_recall": (grouped["detected"] / grouped["events"]).round(3),
        "typical_delay_h": grouped["typical_delay_h"].round(1),
    })
    return {
        "event_recall": round(float(overall["event_recall"].mean()), 3),
        "false_alarms_per_meter_week_on_clean": round(float(clean_fa), 3),
        "false_alarms_per_meter_week_with_faults": round(float(overall["false_alarms_per_meter_week"].mean()), 3),
        "typical_delay_h": round(float(overall["median_delay_h"].mean()), 1),
        "events_tested": int(by_type["events"].sum()),
        "by_type": by_type,
        "calibration": guard.report(),
    }
 