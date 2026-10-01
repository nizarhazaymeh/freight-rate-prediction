"""Loading and cleaning of the raw load files."""
from __future__ import annotations

import pandas as pd

from . import config


def load_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Fix input-level data-quality issues. Never touches the target.

    - Negative weights are sign flips: their absolute values follow the same
      distribution as positive weights, so we take abs().
    - Missing weights stay NaN; the model gets an explicit missing flag.
    - Missing market_index is filled downstream with that day's mean, because
      market_index is a near-daily market factor (within-day std 0.025 vs
      between-day std 0.17).
    """
    out = frame.copy()
    out["weight"] = out["weight"].abs()
    return out


def daily_mean(frames: list[pd.DataFrame], column: str) -> pd.Series:
    """Per-date mean of a feature, pooled over every frame that has it."""
    stacked = pd.concat([f[["date", column]] for f in frames if column in f])
    return stacked.groupby("date")[column].mean()


def city_coordinates(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """City -> (lat, lon) lookup. Each city has exactly one coordinate pair."""
    parts = []
    for f in frames:
        for side in ("pickup", "delivery"):
            if f"{side}_lat" in f:
                parts.append(
                    f[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
                )
    return pd.concat(parts).drop_duplicates("city").set_index("city")
