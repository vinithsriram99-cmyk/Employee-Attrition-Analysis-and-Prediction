"""Focused checks for source-data validation and feature engineering."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from clean_and_load import add_business_features, prepare_dataset


class PipelineTests(unittest.TestCase):
    def test_add_business_features_maps_attrition(self):
        frame = pd.DataFrame(
            {
                "Attrition": ["Yes", "No"], "Age": [25, 42], "DistanceFromHome": [2, 20],
                "YearsAtCompany": [1, 12], "YearsSinceLastPromotion": [0, 6], "MonthlyIncome": [3000, 9000],
            }
        )
        enriched = add_business_features(frame)
        self.assertEqual(enriched["AttritionFlag"].tolist(), [1, 0])
        self.assertIn("TenureBand", enriched.columns)
        self.assertIn("IncomeBand", enriched.columns)

    def test_prepare_dataset_rejects_duplicate_employee_id(self):
        required = {
            "Age": [30, 31], "Attrition": ["No", "Yes"], "BusinessTravel": ["Travel_Rarely"] * 2,
            "Department": ["Sales"] * 2, "DistanceFromHome": [1, 2], "EmployeeNumber": [7, 7],
            "JobRole": ["Sales Executive"] * 2, "JobSatisfaction": [3, 2], "MonthlyIncome": [4000, 5000],
            "OverTime": ["No", "Yes"], "PerformanceRating": [3, 3], "TotalWorkingYears": [5, 6],
            "WorkLifeBalance": [3, 3], "YearsAtCompany": [2, 3], "YearsInCurrentRole": [2, 2],
            "YearsSinceLastPromotion": [1, 1], "YearsWithCurrManager": [2, 2],
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "employees.csv"
            pd.DataFrame(required).to_csv(source, index=False)
            with self.assertRaises(ValueError):
                prepare_dataset(source)


if __name__ == "__main__":
    unittest.main()
