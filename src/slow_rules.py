"""Rules for slow faults: gradual drift and multi-day scale-downs.

Short-window rules (zero-run, stuck, sustained-low) cannot see a meter that quietly
under-reports by 30 to 60 percent for days. This rule compares each meter's usage over
the last few days with what is normal for that household at those times of day, and
scores the drop relative to how much that meter's own usage normally fluctuates.
"""

import pandas as pd

from src.features import _slot_keys


class SustainedDropRule:
    """Usage over the last `window_days` is well below this household's normal level."""

    name = "Sustained-drop rule"

    def __init__(self, profile, window_days=3):
        self.profile = profile
        self.window = window_days * 48
        self.scale = None

    def _drop(self, df):
        keys = pd.MultiIndex.from_arrays(list(_slot_keys(df.index)))
        expected = pd.DataFrame(self.profile.reindex(keys)[df.columns].to_numpy(),
                                index=df.index, columns=df.columns)
        actual = df.rolling(self.window, min_periods=self.window).mean()
        normal = expected.rolling(self.window, min_periods=self.window).mean()
        return (1 - actual / normal.clip(lower=1e-3)).clip(lower=0)

    def fit(self, history):
        """Learn how much each meter's multi-day usage normally dips."""
        self.scale = self._drop(history).std().clip(lower=0.05)
        return self

    def score(self, df):
        drop = self._drop(df) / self.scale.reindex(df.columns)
        return drop.fillna(0.0)