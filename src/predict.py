"""Write validation_predictions.csv and fill data/december_chart_inputs.csv.

Usage: python -m src.predict
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import city_coordinates, load_csv


def december_features(december: pd.DataFrame, labeled: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """The December chart file only has pickup, delivery, distance, equipment,
    weight and date. Fill the missing model inputs:

    - coordinates: looked up per city (each city has one fixed coordinate pair);
    - market_index: that day's mean across the December loads in validation.csv
      (market_index is a daily market factor, so this is the day's market level);
    - quote_signal: that day's median across validation loads (a typical quote).
    """
    out = december.copy()
    coords = city_coordinates([labeled, validation])
    for side in ("pickup", "delivery"):
        out[f"{side}_lat"] = out[side].map(coords["lat"])
        out[f"{side}_lon"] = out[side].map(coords["lon"])

    daily = validation.groupby("date").agg(
        market_index=("market_index", "mean"), quote_signal=("quote_signal", "median")
    )
    out["market_index"] = out["date"].map(daily["market_index"])
    out["quote_signal"] = out["date"].map(daily["quote_signal"])

    missing = out[["pickup_lat", "delivery_lat", "market_index", "quote_signal"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"could not build features for {int(missing.sum())} December rows")
    return out


def main() -> None:
    model = joblib.load(config.MODEL_PATH)
    labeled = load_csv(config.TRAIN_PATH)
    validation = load_csv(config.VALIDATION_PATH)

    # Final predictions, in template order.
    template = pd.read_csv(config.TEMPLATE_PATH)
    preds = pd.Series(model.predict(validation), index=validation["load_id"])
    template["predicted_rate"] = preds.reindex(template["load_id"]).round(2).to_numpy()
    if template["predicted_rate"].isna().any() or (template["predicted_rate"] <= 0).any():
        raise ValueError("invalid validation predictions")
    template.to_csv(config.PREDICTIONS_PATH, index=False)
    print(f"Wrote {len(template):,} predictions to {config.PREDICTIONS_PATH.name}")

    # December chart: fill predicted_rate in place, keeping the original columns.
    raw = pd.read_csv(config.DECEMBER_PATH)
    december = raw.assign(date=pd.to_datetime(raw["date"]))
    rates = model.predict(december_features(december, labeled, validation))
    raw["predicted_rate"] = np.round(rates, 2)
    raw.to_csv(config.DECEMBER_PATH, index=False)
    print(
        f"Filled {len(raw)} December rows in {config.DECEMBER_PATH.relative_to(config.ROOT)} "
        f"(range ${raw['predicted_rate'].min():,.2f} - ${raw['predicted_rate'].max():,.2f})"
    )


if __name__ == "__main__":
    main()
