"""Rolling-origin backtest, then fit the final model on all labeled data.

Usage: python -m src.train
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config
from .data import clean, load_csv
from .model import FreightRateModel


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    ape = np.abs(err) / y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(100 * ape.mean()),
        "MedianAPE_%": float(100 * np.median(ape)),
        "n": int(len(y)),
    }


def baseline_rate_per_mile(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Median rate-per-mile by equipment x distance decile, times distance."""
    edges = np.quantile(train["distance"], np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    rpm = (train[config.TARGET] / train["distance"]).groupby(
        [train["equipment"], pd.cut(train["distance"], edges)], observed=True
    ).median()
    keys = pd.MultiIndex.from_arrays([test["equipment"], pd.cut(test["distance"], edges)])
    return rpm.reindex(keys).to_numpy() * test["distance"].to_numpy()


def baseline_plain_gbm(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Off-the-shelf gradient boosting on raw columns with a log target."""
    cols = ["distance", "weight", "market_index", "quote_signal", *config.COORD_COLS]

    def design(f):
        X = clean(f)[cols].copy()
        X["equipment"] = f["equipment"].map({e: i for i, e in enumerate(config.EQUIPMENT)})
        X["dow"] = f["date"].dt.dayofweek
        return X

    gbm = HistGradientBoostingRegressor(
        max_iter=800, learning_rate=0.04, categorical_features=[len(cols)], random_state=config.RANDOM_STATE
    ).fit(design(train), np.log(train[config.TARGET]))
    return np.exp(gbm.predict(design(test)))


def backtest(labeled: pd.DataFrame) -> dict:
    results = {}
    for test_start, test_end in config.BACKTEST_FOLDS:
        train = labeled[labeled["date"] < test_start]
        test = labeled[(labeled["date"] >= test_start) & (labeled["date"] < test_end)]
        # Unlabeled test features are visible at prediction time (as in the
        # real validation set), so they may inform the daily market level.
        model = FreightRateModel().fit(train, extra_feature_frames=[test.drop(columns=config.TARGET)])

        preds = {
            "baseline_rate_per_mile": baseline_rate_per_mile(train, test),
            "baseline_plain_gbm": baseline_plain_gbm(train, test),
            "stage1_linear_only": model.predict_linear_only(test),
            "full_model": model.predict(test),
        }
        y = test[config.TARGET].to_numpy()
        # Corrupted test labels (same rule as training) are reported separately:
        # no model can predict a 3x data-entry error.
        clean_mask = np.abs(np.log(y / preds["full_model"])) < np.log(config.OUTLIER_FACTOR)

        fold = f"train<{test_start} | test {test_start}..{test_end}"
        results[fold] = {
            name: {"all_rows": metrics(y, p), "clean_rows": metrics(y[clean_mask], p[clean_mask])}
            for name, p in preds.items()
        }
        results[fold]["corrupted_label_share_%"] = float(100 * (~clean_mask).mean())
        results[fold]["train_rows"] = int(len(train))
        print(f"\n{fold}  (train n={len(train):,}, test n={len(test):,})")
        for name, p in preds.items():
            a, c = results[fold][name]["all_rows"], results[fold][name]["clean_rows"]
            print(
                f"  {name:24s} clean MAPE {c['MAPE_%']:5.2f}%  MAE {c['MAE']:6.1f}  RMSE {c['RMSE']:6.1f}"
                f"   | all rows MAPE {a['MAPE_%']:5.2f}%  RMSE {a['RMSE']:6.1f}"
            )
    return results


def summarize(results: dict) -> dict:
    folds = [k for k in results]
    models = [k for k in results[folds[0]] if isinstance(results[folds[0]][k], dict)]
    summary = {}
    for m in models:
        summary[m] = {
            subset: {
                metric: float(np.mean([results[f][m][subset][metric] for f in folds]))
                for metric in ("MAE", "RMSE", "MAPE_%", "MedianAPE_%")
            }
            for subset in ("all_rows", "clean_rows")
        }
    return summary


def main() -> None:
    labeled = load_csv(config.TRAIN_PATH)
    validation = load_csv(config.VALIDATION_PATH)

    print("Rolling-origin backtest (2-month horizon, mirrors Nov-Dec validation)")
    results = backtest(labeled)
    summary = summarize(results)
    print("\nMean over folds (clean rows):")
    for m, s in summary.items():
        c = s["clean_rows"]
        print(f"  {m:24s} MAPE {c['MAPE_%']:5.2f}%  MAE {c['MAE']:6.1f}  RMSE {c['RMSE']:6.1f}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps({"folds": results, "mean": summary}, indent=2))

    print("\nFitting final model on all labeled data (Jan-Oct 2025)...")
    model = FreightRateModel().fit(labeled, extra_feature_frames=[validation])
    dropped = int((~model.inlier_mask_).sum())
    print(f"  excluded {dropped} corrupted labels ({100 * dropped / len(labeled):.2f}%)")
    coefs = dict(zip(model._linear_design(model._prepare(labeled.head(1), model.daily_mi_)).columns,
                     model.linear_.coef_))
    print(f"  stage-1 time trend: {100 * coefs['t']:.2f}% per 30 days")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    print(f"  saved {config.MODEL_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
