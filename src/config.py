from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = REPORTS_DIR / "backtest_metrics.json"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

TARGET = "posted_rate"
EQUIPMENT = ["Dry Van", "Reefer", "Flatbed"]
COORD_COLS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]

# Reference date for the linear time trend (days since this date).
EPOCH = "2025-01-01"

# A label is treated as corrupted when it is more than this factor away from
# the structural model's estimate. Corrupted rows sit at roughly 0.2-0.5x or
# 2.5-4.6x the expected rate, while genuine noise is ~5%, so 1.6x separates
# the two groups cleanly.
OUTLIER_FACTOR = 1.6

# Rolling-origin backtest: each fold trains on everything before `test_start`
# and evaluates on the following two months, mirroring the Nov-Dec horizon
# of the final validation set.
BACKTEST_FOLDS = [
    ("2025-07-01", "2025-09-01"),
    ("2025-08-01", "2025-10-01"),
    ("2025-09-01", "2025-11-01"),
]

RANDOM_STATE = 42
