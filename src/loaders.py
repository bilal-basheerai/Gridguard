"""Turn an uploaded CSV table into the wide format the engine expects.
 
Wide format: DatetimeIndex (timestamps) x one numeric column per meter.
Two upload layouts are supported:
  * wide: one timestamp column, one column per meter
  * long: one row per reading, with timestamp, meter id and value columns
"""
 
import pandas as pd
 
 
def _timestamps(series, dayfirst, column):
    times = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    if times.notna().mean() < 0.9:
        raise ValueError(
            f"Column '{column}' does not look like timestamps. Pick the column that holds "
            "the date and time of each reading (for example 2013-10-01 00:30:00).")
    return times
 
 
def wide_from_columns(raw, timestamp_col, dayfirst=False):
    """One timestamp column plus one column per meter."""
    times = _timestamps(raw[timestamp_col], dayfirst, timestamp_col)
    wide = raw.drop(columns=[timestamp_col]).copy()
    wide.index = pd.DatetimeIndex(times)
    return wide[wide.index.notna()]
 
 
def wide_from_long(raw, timestamp_col, meter_col, value_col, dayfirst=False):
    """One row per reading: timestamp, meter id, value."""
    if len({timestamp_col, meter_col, value_col}) < 3:
        raise ValueError("Pick three different columns for timestamp, meter and value.")
    times = _timestamps(raw[timestamp_col], dayfirst, timestamp_col)
    long = pd.DataFrame({
        "t": times,
        "meter": raw[meter_col].astype(str),
        "value": pd.to_numeric(raw[value_col], errors="coerce"),
    }).dropna(subset=["t"])
    wide = long.pivot_table(index="t", columns="meter", values="value", aggfunc="mean")
    wide.index = pd.DatetimeIndex(wide.index)
    wide.columns.name = None
    return wide
 
