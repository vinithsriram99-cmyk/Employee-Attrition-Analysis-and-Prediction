"""Shared paths and model settings for the Employee Attrition project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Employee-Attrition.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "employee_attrition_clean.csv"
DATABASE_PATH = DATA_DIR / "employee_attrition.db"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "attrition_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
RISK_PREDICTIONS_PATH = ARTIFACTS_DIR / "employee_risk_predictions.csv"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_set_predictions.csv"

RANDOM_STATE = 42
RISK_THRESHOLDS = {"high": 0.50, "medium": 0.30}
