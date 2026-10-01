# Freight Rate Prediction

Predicts the posted rate of a freight load from its lane, distance, equipment, weight, date and market signals. Assessment instructions are in `freight-rate-ml-assessment.pdf`.

## Quick start

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m src.eda        # data-quality findings + figures in reports/figures/
python -m src.train      # rolling backtest, then fits the final model -> artifacts/model.joblib
python -m src.predict    # writes validation_predictions.csv, fills data/december_chart_inputs.csv

python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python -m src.report     # builds reports/freight_rate_report.pdf
```

`src.train` takes about one minute on a laptop. The scorer validates both files and writes `scorer_results/candidate_december.png`.

## Approach

**Data quality**
- **Corrupted labels (1.4%)**: about 670 rates sit at 0.2-0.5x or 2.5-4.6x the normal rate for their lane, with an almost empty gap between them and genuine noise (~5%). They are spread evenly across equipment, months, cities and distances, so no feature can predict them. Rows more than 1.6x away from the structural model's estimate are excluded from training.
- **Negative weights (292 train / 145 validation)**: their absolute values follow the normal weight distribution, so they are treated as sign flips and fixed with `abs()`.
- **Missing weight / market_index (<1%)**: weight gets the median plus a missing flag. `market_index` is a daily market factor (within-day std 0.025 vs between-day std 0.17), so a missing value takes that day's mean.
- **Unseen cities**: 8 validation cities (1,447 rows) never appear in training, so the model uses coordinates rather than city IDs as its geographic features.

**Model** (`src/model.py`): three stages, all fitted on the log of the rate.
1. **Structural log-linear model**: log distance, equipment, log market_index, log weight, quote_signal and a **linear time trend**. Rates drift upward by about 0.6% a month, independent of market_index. This is the only stage that sees time, so the trend extrapolates into Nov-Dec instead of flattening out the way a tree model would.
2. **Lane target encodings**: out-of-fold, shrunken means of the stage-1 residual per lane x equipment, lane and city. Lane premiums (~5%) are stable over time, and 88% of validation loads run on a lane seen in training.
3. **Gradient boosting** (`HistGradientBoostingRegressor`) on the stage-1 residual. It captures non-linear effects and interactions such as the distance taper, the weight curve, geography and the lane encodings.

**Validation**: a rolling-origin, time-based backtest. Each fold trains on all data before a cutoff and tests on the following two months, which mirrors the real task (train Jan-Oct, predict Nov-Dec). A random split would leak future market conditions and lane prices into training.

| Model (mean of 3 folds, clean labels) | MAPE | MAE | RMSE |
|---|---|---|---|
| Rate-per-mile baseline (equipment x distance bucket) | 3.71% | $80 | $111 |
| Plain gradient boosting on raw columns | 3.43% | $79 | $114 |
| Stage 1 only (log-linear + trend) | 3.14% | $71 | $101 |
| **Full model** | **2.31%** | **$56** | **$86** |

On the latest fold (train through August, test Sep-Oct), the full model reaches 1.73% MAPE. Full per-fold results, including metrics on all rows, are in `reports/backtest_metrics.json`.

**December chart**: `december_chart_inputs.csv` provides only lane, distance, equipment, weight and date. Coordinates come from the city lookup. market_index and quote_signal take that day's mean and median from the December loads in `validation.csv`, so the chart follows the real daily market level (including its weekly cycle, which peaks on Thursdays) plus the time trend.

## Layout

```
data/                    input CSVs (december_chart_inputs.csv is filled in by src.predict)
src/config.py            paths and constants
src/data.py              loading and input cleaning
src/model.py             three-stage FreightRateModel
src/train.py             backtest + final fit
src/predict.py           validation predictions + December chart inputs
src/eda.py               data-quality checks and figures
src/report.py            builds the PDF report
score.py                 provided scorer
validation_predictions.csv
reports/                 PDF report, backtest metrics, figures
scorer_results/          chart produced by score.py
```
