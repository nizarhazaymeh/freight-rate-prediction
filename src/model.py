"""Three-stage freight rate model.

Stage 1  Structural log-linear model
         log(rate) ~ log(distance), equipment, log(market_index), log(weight),
         quote_signal and a linear time trend. It captures the multiplicative
         pricing structure and is the only stage that sees time, so the
         month-over-month upward drift extrapolates into unseen months
         (tree models would flatten it at the last training value).

Stage 2  Lane target encodings
         Out-of-fold, shrunken means of the stage-1 residual per lane x
         equipment, lane, and city. Lane premiums are large (~5%) and stable
         over time, and 88% of validation loads run on lanes seen in training.

Stage 3  Gradient boosting on the stage-1 residual
         Learns non-linear effects and interactions (distance taper, weight
         curve, geography, lane encodings). Final prediction = stage 1 + stage 3.

Rows whose label is more than OUTLIER_FACTOR away from the stage-1 estimate
are treated as corrupted labels and excluded from training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from . import config
from .data import clean

TE_KEYS = {
    "lane_eq": ["pickup", "delivery", "equipment"],
    "lane": ["pickup", "delivery"],
    "pickup_eq": ["pickup", "equipment"],
    "delivery_eq": ["delivery", "equipment"],
    "pickup": ["pickup"],
    "delivery": ["delivery"],
}
TE_SMOOTHING = 5.0
TE_FOLDS = 5


def _haversine_miles(df: pd.DataFrame) -> pd.Series:
    lat1, lon1, lat2, lon2 = (np.radians(df[c]) for c in config.COORD_COLS)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))


class FreightRateModel:
    def __init__(self, gbm_params: dict | None = None):
        self.gbm_params = dict(
            max_iter=800,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=config.RANDOM_STATE,
        )
        self.gbm_params.update(gbm_params or {})

    # ------------------------------------------------------------------ prep
    def _prepare(self, df: pd.DataFrame, daily_mi: pd.Series) -> pd.DataFrame:
        d = clean(df)
        mi_day = d["date"].map(daily_mi)
        d["market_index"] = d["market_index"].fillna(mi_day).fillna(self.mi_median_)
        d["weight_missing"] = d["weight"].isna().astype(float)
        d["weight"] = d["weight"].fillna(self.weight_median_)
        d["t"] = (d["date"] - pd.Timestamp(config.EPOCH)).dt.days.astype(float)
        return d

    @staticmethod
    def _linear_design(d: pd.DataFrame) -> pd.DataFrame:
        ld = np.log(d["distance"])
        lq = np.log(d["quote_signal"])
        return pd.DataFrame(
            {
                "log_dist": ld,
                "log_dist_sq": ld**2,
                "log_mi": np.log(d["market_index"]),
                "log_qs": lq,
                "log_qs_sq": lq**2,
                "log_weight": np.log(d["weight"]),
                "weight_missing": d["weight_missing"],
                "reefer": (d["equipment"] == "Reefer").astype(float),
                "flatbed": (d["equipment"] == "Flatbed").astype(float),
                "t": d["t"] / 30.0,
            },
            index=d.index,
        )

    def _tree_design(self, d: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(
            {
                "log_dist": np.log(d["distance"]),
                "equipment": d["equipment"].map({e: i for i, e in enumerate(config.EQUIPMENT)}),
                "weight": d["weight"],
                "weight_missing": d["weight_missing"],
                "market_index": d["market_index"],
                "quote_signal": d["quote_signal"],
                "circuity": d["distance"] / _haversine_miles(d).clip(lower=1.0),
                "dow": d["date"].dt.dayofweek,
            },
            index=d.index,
        )
        for c in config.COORD_COLS:
            X[c] = d[c]
        return pd.concat([X, te], axis=1)

    # ------------------------------------------------- target encoding utils
    @staticmethod
    def _te_table(d: pd.DataFrame, resid: pd.Series, keys: list[str]) -> pd.DataFrame:
        g = resid.groupby([d[k] for k in keys])
        return pd.DataFrame({"sum": g.sum(), "n": g.size()})

    @staticmethod
    def _te_apply(d: pd.DataFrame, table: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
        idx = pd.MultiIndex.from_frame(d[keys]) if len(keys) > 1 else pd.Index(d[keys[0]])
        joined = table.reindex(idx)
        n = joined["n"].fillna(0.0).to_numpy()
        s = joined["sum"].fillna(0.0).to_numpy()
        return pd.DataFrame(
            {f"te_{name}": s / (n + TE_SMOOTHING), f"n_{name}": n}, index=d.index
        )

    def _te_features(self, d: pd.DataFrame, tables: dict) -> pd.DataFrame:
        return pd.concat(
            [self._te_apply(d, tables[name], keys, name) for name, keys in TE_KEYS.items()], axis=1
        )

    def _te_out_of_fold(self, d: pd.DataFrame, resid: pd.Series) -> pd.DataFrame:
        parts = []
        folds = KFold(TE_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        for fit_idx, enc_idx in folds.split(d):
            fit_d, enc_d = d.iloc[fit_idx], d.iloc[enc_idx]
            tables = {
                name: self._te_table(fit_d, resid.iloc[fit_idx], keys) for name, keys in TE_KEYS.items()
            }
            parts.append(self._te_features(enc_d, tables))
        return pd.concat(parts).loc[d.index]

    # -------------------------------------------------------------- fit/pred
    def fit(self, df: pd.DataFrame, extra_feature_frames: list[pd.DataFrame] | None = None):
        """Fit on labeled loads.

        `extra_feature_frames` are unlabeled frames (e.g. the loads to be
        predicted) whose market_index values help fill the daily market level.
        Only features are used from them, never labels.
        """
        df = df.reset_index(drop=True)
        self.weight_median_ = df["weight"].abs().median()
        self.mi_median_ = df["market_index"].median()
        frames = [df] + list(extra_feature_frames or [])
        self.daily_mi_ = pd.concat([f[["date", "market_index"]] for f in frames]).groupby("date")[
            "market_index"
        ].mean()

        d = self._prepare(df, self.daily_mi_)
        y = np.log(d[config.TARGET])
        X_lin = self._linear_design(d)

        # Stage 1 with one round of outlier removal.
        self.linear_ = Ridge(alpha=1e-3).fit(X_lin, y)
        resid = y - self.linear_.predict(X_lin)
        self.inlier_mask_ = np.abs(resid) < np.log(config.OUTLIER_FACTOR)
        self.linear_ = Ridge(alpha=1e-3).fit(X_lin[self.inlier_mask_], y[self.inlier_mask_])

        dc = d[self.inlier_mask_]
        yc = y[self.inlier_mask_]
        resid = yc - self.linear_.predict(X_lin[self.inlier_mask_])

        # Stage 2: out-of-fold encodings for training; full tables for inference.
        te_train = self._te_out_of_fold(dc, resid)
        self.te_tables_ = {name: self._te_table(dc, resid, keys) for name, keys in TE_KEYS.items()}

        # Stage 3.
        X_tree = self._tree_design(dc, te_train)
        self.tree_columns_ = list(X_tree.columns)
        self.gbm_ = HistGradientBoostingRegressor(
            categorical_features=[self.tree_columns_.index("equipment")], **self.gbm_params
        ).fit(X_tree, resid)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        daily = df.groupby("date")["market_index"].mean().combine_first(self.daily_mi_)
        d = self._prepare(df, daily)
        base = self.linear_.predict(self._linear_design(d))
        X_tree = self._tree_design(d, self._te_features(d, self.te_tables_))[self.tree_columns_]
        return np.exp(base + self.gbm_.predict(X_tree))

    def predict_linear_only(self, df: pd.DataFrame) -> np.ndarray:
        daily = df.groupby("date")["market_index"].mean().combine_first(self.daily_mi_)
        return np.exp(self.linear_.predict(self._linear_design(self._prepare(df, daily))))
