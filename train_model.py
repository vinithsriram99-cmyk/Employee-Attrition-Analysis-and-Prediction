"""Train, evaluate, and save an employee attrition classifier and risk scores."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import (
    ARTIFACTS_DIR,
    DATABASE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RISK_PREDICTIONS_PATH,
    RISK_THRESHOLDS,
    TEST_PREDICTIONS_PATH,
)

TARGET_COLUMN = "AttritionFlag"
IDENTIFIER_COLUMNS = ["EmployeeNumber"]
NON_FEATURE_COLUMNS = [
    "Attrition", "AttritionFlag", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours",
]


def select_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Separate model features from the target and deterministic identifier columns."""
    features = data.drop(columns=NON_FEATURE_COLUMNS, errors="ignore").copy()
    numeric_features = features.select_dtypes(include="number").columns.tolist()
    categorical_features = features.select_dtypes(exclude="number").columns.tolist()
    return features, numeric_features, categorical_features


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def score_model(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, object], list[float], Pipeline]:
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics: dict[str, object] = {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    return metrics, probabilities.tolist(), pipeline


def risk_tier(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["high"]:
        return "High"
    if probability >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Clean dataset not found at {path}. Run clean_and_load.py first.")
    return pd.read_csv(path)


def write_predictions_to_database(predictions: pd.DataFrame) -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        predictions.to_sql("risk_predictions", connection, if_exists="replace", index=False)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_risk_predictions_probability "
            "ON risk_predictions(attrition_probability)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_PATH, help="Path to the cleaned CSV.")
    args = parser.parse_args()

    data = load_data(args.data)
    x, numeric_features, categorical_features = select_features(data)
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    model_results: dict[str, dict[str, object]] = {}
    fitted_candidates: dict[str, Pipeline] = {}
    test_probabilities: dict[str, list[float]] = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            [("preprocessor", make_preprocessor(numeric_features, categorical_features)), ("model", estimator)]
        )
        metrics, probabilities, fitted_pipeline = score_model(
            name, pipeline, x_train, y_train, x_test, y_test
        )
        model_results[name] = metrics
        fitted_candidates[name] = fitted_pipeline
        test_probabilities[name] = probabilities

    best_name = max(model_results, key=lambda model_name: model_results[model_name]["roc_auc"])
    best_model = fitted_candidates[best_name]
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    best_test_predictions = best_model.predict(x_test)
    test_frame = data.loc[x_test.index, IDENTIFIER_COLUMNS + ["Attrition"]].copy()
    test_frame["actual_attrition"] = y_test
    test_frame["predicted_attrition"] = best_test_predictions
    test_frame["attrition_probability"] = test_probabilities[best_name]
    test_frame["risk_tier"] = test_frame["attrition_probability"].map(risk_tier)
    test_frame.sort_values("attrition_probability", ascending=False).to_csv(TEST_PREDICTIONS_PATH, index=False)

    all_probabilities = best_model.predict_proba(x)[:, 1]
    risk_predictions = data[
        ["EmployeeNumber", "Department", "JobRole", "OverTime", "JobSatisfaction", "WorkLifeBalance", "MonthlyIncome", "YearsAtCompany"]
    ].copy()
    risk_predictions["attrition_probability"] = all_probabilities
    risk_predictions["risk_tier"] = risk_predictions["attrition_probability"].map(risk_tier)
    risk_predictions.sort_values("attrition_probability", ascending=False).to_csv(RISK_PREDICTIONS_PATH, index=False)
    write_predictions_to_database(risk_predictions)

    metrics_payload = {
        "selected_model": best_name,
        "selection_metric": "roc_auc",
        "test_set_size": int(len(x_test)),
        "training_set_size": int(len(x_train)),
        "risk_thresholds": RISK_THRESHOLDS,
        "models": model_results,
        "feature_counts": {"numeric": len(numeric_features), "categorical": len(categorical_features)},
        "note": "Model scores are decision-support signals. HR must review context before any retention action.",
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Selected model: {best_name}")
    print(json.dumps(model_results[best_name], indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved risk predictions: {RISK_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
