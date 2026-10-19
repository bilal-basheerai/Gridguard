"""Baseline anomaly detectors for GridGuard.

Every detector exposes score(df) -> DataFrame (same shape as df) where a higher
value means "more anomalous". Thresholds are chosen elsewhere.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features import deviation_from_profile


class RollingZScore:
    """Flags readings far from the meter's own trailing 7-day mean."""

    name = "Rolling z-score"

    def __init__(self, window_days=7):
        self.window = window_days * 48

    def score(self, df):
        past = df.shift(1)                       # a reading never sees itself
        mean = past.rolling(self.window, min_periods=48).mean()
        std = past.rolling(self.window, min_periods=48).std()
        z = ((df - mean) / (std + 1e-3)).abs()
        return z.fillna(0.0)


class IsolationForestDetector:
    """Isolation Forest on context features that capture sustained changes."""

    name = "Isolation Forest"

    def __init__(self, profile, level, dev_scale, n_estimators=200, seed=42,
                 max_train_rows=300_000):
        self.profile = profile        # per-meter typical profile
        self.level = level            # per-meter mean reading
        self.dev_scale = dev_scale    # per-meter std of normal deviations
        self.max_train_rows = max_train_rows
        self.seed = seed
        self.model = IsolationForest(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1
        )

    def _features(self, df):
        level = self.level.reindex(df.columns)
        scale = self.dev_scale.reindex(df.columns)
        z = deviation_from_profile(df, self.profile) / scale
        rel = df / level
        feats = [
            z,                                             # deviation from usual
            rel,                                           # level vs meter average
            z.rolling(24, min_periods=1).mean(),           # 12 h sustained shift
            z.rolling(48 * 3, min_periods=1).mean(),       # 3 day slow drift
            df.rolling(12, min_periods=2).std() / level,   # 6 h variability
        ]
        stacked = np.stack([f.to_numpy() for f in feats], axis=-1)
        return np.nan_to_num(stacked, nan=0.0).reshape(-1, len(feats))

    def fit(self, df):
        X = self._features(df)
        rng = np.random.default_rng(self.seed)
        if len(X) > self.max_train_rows:
            X = X[rng.choice(len(X), self.max_train_rows, replace=False)]
        self.model.fit(X)
        return self

    def score(self, df):
        raw = -self.model.score_samples(self._features(df))
        return pd.DataFrame(raw.reshape(df.shape), index=df.index, columns=df.columns)