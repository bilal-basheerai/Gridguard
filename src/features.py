"""Feature engineering for GridGuard."""

import pandas as pd


def calendar_features(index):
    """Calendar features for a DatetimeIndex."""
    return pd.DataFrame(
        {
            "hour": index.hour + index.minute / 60,
            "day_of_week": index.dayofweek,
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "month": index.month,
        },
        index=index,
    )


def rolling_mean(df, days=7):
    """Per-meter rolling mean over the previous `days` days."""
    return df.rolling(window=days * 48, min_periods=48).mean()


def _slot_keys(index):
    """Half-hour slot (0 to 47) and weekend flag for each timestamp."""
    slot = index.hour * 2 + index.minute // 30
    is_weekend = index.dayofweek >= 5
    return is_weekend, slot


def fit_typical_profile(df):
    """Each meter's average reading per (weekend flag, half-hour slot).

    Fit this on clean training data only, to avoid leaking test data.
    """
    is_weekend, slot = _slot_keys(df.index)
    return df.groupby([is_weekend, slot]).mean()


def deviation_from_profile(df, profile):
    """How far each reading is from that meter's typical value for that time."""
    keys = pd.MultiIndex.from_arrays(list(_slot_keys(df.index)))
    expected = profile.reindex(keys)[df.columns].to_numpy()
    return df - expected