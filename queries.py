"""Reusable SQL insights for the Employee Attrition dashboard."""

from __future__ import annotations

from collections import OrderedDict

import pandas as pd

from db import read_query


QUERY_CATALOG = OrderedDict(
    [
        (
            "1. Attrition by department",
            """
            SELECT Department,
                   COUNT(*) AS employees,
                   SUM(AttritionFlag) AS employees_left,
                   ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
            FROM employee_attrition
            GROUP BY Department
            ORDER BY attrition_rate_pct DESC;
            """,
        ),
        (
            "2. Overtime impact",
            """
            SELECT OverTime,
                   COUNT(*) AS employees,
                   SUM(AttritionFlag) AS employees_left,
                   ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
            FROM employee_attrition
            GROUP BY OverTime
            ORDER BY attrition_rate_pct DESC;
            """,
        ),
        (
            "3. Attrition by job role",
            """
            SELECT JobRole,
                   COUNT(*) AS employees,
                   SUM(AttritionFlag) AS employees_left,
                   ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
            FROM employee_attrition
            GROUP BY JobRole
            ORDER BY attrition_rate_pct DESC, employees DESC;
            """,
        ),
        (
            "4. Travel frequency and attrition",
            """
            SELECT BusinessTravel,
                   COUNT(*) AS employees,
                   ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
            FROM employee_attrition
            GROUP BY BusinessTravel
            ORDER BY attrition_rate_pct DESC;
            """,
        ),
        (
            "5. Job satisfaction and attrition",
            """
            SELECT JobSatisfaction,
                   COUNT(*) AS employees,
                   ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
            FROM employee_attrition
            GROUP BY JobSatisfaction
            ORDER BY JobSatisfaction;
            """,
        ),
        (
            "6. Work-life balance and attrition",
            """
            SELECT WorkLifeBalance,
                   COUNT(*) AS employees,
                   ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
            FROM employee_attrition
            GROUP BY WorkLifeBalance
            ORDER BY WorkLifeBalance;
            """,
        ),
        (
            "7. Tenure bands with highest attrition",
            """
            SELECT TenureBand,
                   COUNT(*) AS employees,
                   ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
            FROM employee_attrition
            GROUP BY TenureBand
            ORDER BY attrition_rate_pct DESC;
            """,
        ),
        (
            "8. Promotion wait and attrition",
            """
            SELECT PromotionWaitBand,
                   COUNT(*) AS employees,
                   ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
            FROM employee_attrition
            GROUP BY PromotionWaitBand
            ORDER BY attrition_rate_pct DESC;
            """,
        ),
        (
            "9. Commute distance and attrition",
            """
            SELECT CommuteBand,
                   COUNT(*) AS employees,
                   ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
            FROM employee_attrition
            GROUP BY CommuteBand
            ORDER BY attrition_rate_pct DESC;
            """,
        ),
        (
            "10. Highest-risk employees from the model",
            """
            SELECT EmployeeNumber, Department, JobRole, OverTime, JobSatisfaction,
                   WorkLifeBalance, MonthlyIncome, YearsAtCompany,
                   ROUND(attrition_probability * 100, 1) AS attrition_probability_pct,
                   risk_tier
            FROM risk_predictions
            ORDER BY attrition_probability DESC
            LIMIT 25;
            """,
        ),
    ]
)


def run_named_query(name: str) -> pd.DataFrame:
    """Run a catalogued SQL query by its dashboard label."""
    if name not in QUERY_CATALOG:
        raise KeyError(f"Unknown SQL insight: {name}")
    return read_query(QUERY_CATALOG[name])
