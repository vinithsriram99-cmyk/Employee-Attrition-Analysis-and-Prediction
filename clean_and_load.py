"""Validate, enrich, save, and load the employee attrition dataset into SQLite."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd

from config import DATABASE_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH

REQUIRED_COLUMNS = {
    "Age", "Attrition", "BusinessTravel", "Department", "DistanceFromHome",
    "EmployeeNumber", "JobRole", "JobSatisfaction", "MonthlyIncome", "OverTime",
    "PerformanceRating", "TotalWorkingYears", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
}


def add_business_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create readable bands that support analysis and HR review in the dashboard."""
    clean = data.copy()
    clean["AttritionFlag"] = clean["Attrition"].map({"No": 0, "Yes": 1})
    clean["AgeBand"] = pd.cut(
        clean["Age"], bins=[0, 29, 39, 49, float("inf")],
        labels=["Under 30", "30-39", "40-49", "50+"], include_lowest=True,
    ).astype(str)
    clean["CommuteBand"] = pd.cut(
        clean["DistanceFromHome"], bins=[-1, 5, 15, float("inf")],
        labels=["0-5 miles", "6-15 miles", "16+ miles"], include_lowest=True,
    ).astype(str)
    clean["TenureBand"] = pd.cut(
        clean["YearsAtCompany"], bins=[-1, 1, 5, 10, float("inf")],
        labels=["0-1 years", "2-5 years", "6-10 years", "11+ years"], include_lowest=True,
    ).astype(str)
    clean["PromotionWaitBand"] = pd.cut(
        clean["YearsSinceLastPromotion"], bins=[-1, 1, 3, 5, float("inf")],
        labels=["0-1 years", "2-3 years", "4-5 years", "6+ years"], include_lowest=True,
    ).astype(str)
    clean["IncomeBand"] = pd.qcut(
        clean["MonthlyIncome"], q=4,
        labels=["Lower income", "Lower-middle income", "Upper-middle income", "Higher income"],
        duplicates="drop",
    ).astype(str)
    satisfaction = ["JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]
    if set(satisfaction).issubset(clean.columns):
        clean["EngagementScore"] = clean[satisfaction].mean(axis=1).round(2)  # average of five 1-4 ratings
    return clean


OUTLIER_COLUMNS = ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
                   "YearsSinceLastPromotion", "YearsWithCurrManager"]


def count_outliers(data: pd.DataFrame) -> dict[str, int]:
    """Number of values above Q3 + 1.5 x IQR. They are kept in the data and clipped inside the model pipeline."""
    result = {}
    for column in OUTLIER_COLUMNS:
        if column in data.columns:
            q1, q3 = data[column].quantile([0.25, 0.75])
            result[column] = int((data[column] > q3 + 1.5 * (q3 - q1)).sum())
    return result


def prepare_dataset(source_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the raw CSV, validate it, and return a clean enriched DataFrame plus a report."""
    if not source_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {source_path}. Pass --source with the CSV path."
        )

    data = pd.read_csv(source_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if not set(data["Attrition"].dropna().unique()).issubset({"Yes", "No"}):
        raise ValueError("Attrition must contain only Yes and No values.")
    if not data["EmployeeNumber"].is_unique:
        raise ValueError("EmployeeNumber must uniquely identify each employee.")

    before_rows = len(data)
    duplicate_rows = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    missing_before = int(data.isna().sum().sum())
    numeric_columns = data.select_dtypes(include="number").columns
    categorical_columns = data.select_dtypes(exclude="number").columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    data[categorical_columns] = data[categorical_columns].fillna("Unknown")
    clean = add_business_features(data)
    constant_columns = [column for column in data.columns if data[column].nunique(dropna=False) <= 1]

    report: dict[str, object] = {
        "source_rows": before_rows,
        "rows_after_deduplication": len(clean),
        "duplicate_rows_removed": duplicate_rows,
        "missing_values_before_cleaning": missing_before,
        "missing_values_after_cleaning": int(clean.isna().sum().sum()),
        "constant_columns": constant_columns,
        "outliers_detected_iqr": count_outliers(data),
        "outlier_handling": "kept in the clean data; clipped to the IQR limits inside the model pipeline (training data only)",
        "attrition_counts": clean["Attrition"].value_counts().to_dict(),
        "attrition_rate": round(float(clean["AttritionFlag"].mean()), 4),
    }
    return clean, report


def load_to_sqlite(clean: pd.DataFrame, database_path: Path = DATABASE_PATH) -> None:
    """Replace the analysis table and create indexes for dashboard filters and SQL queries."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        clean.to_sql("employee_attrition", connection, if_exists="replace", index=False)
        connection.executescript(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_employee_number
                ON employee_attrition(EmployeeNumber);
            CREATE INDEX IF NOT EXISTS idx_attrition_department
                ON employee_attrition(Attrition, Department);
            CREATE INDEX IF NOT EXISTS idx_job_role
                ON employee_attrition(JobRole);
            CREATE INDEX IF NOT EXISTS idx_overtime
                ON employee_attrition(OverTime);
            """
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAW_DATA_PATH, help="Path to the raw CSV dataset.")
    args = parser.parse_args()

    clean, report = prepare_dataset(args.source)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_DATA_PATH, index=False)
    load_to_sqlite(clean)
    report_path = PROCESSED_DATA_PATH.parent / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved cleaned data: {PROCESSED_DATA_PATH}")
    print(f"Loaded SQLite database: {DATABASE_PATH}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
