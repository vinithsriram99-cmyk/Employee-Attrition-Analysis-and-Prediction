-- Employee Attrition SQL insights. Run after: python clean_and_load.py and python train_model.py

-- Q1. Department-level attrition rate
SELECT Department, COUNT(*) AS employees, SUM(AttritionFlag) AS employees_left,
       ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
FROM employee_attrition GROUP BY Department ORDER BY attrition_rate_pct DESC;

-- Q2. Overtime impact
SELECT OverTime, COUNT(*) AS employees, SUM(AttritionFlag) AS employees_left,
       ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
FROM employee_attrition GROUP BY OverTime ORDER BY attrition_rate_pct DESC;

-- Q3. Attrition by job role
SELECT JobRole, COUNT(*) AS employees, SUM(AttritionFlag) AS employees_left,
       ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
FROM employee_attrition GROUP BY JobRole ORDER BY attrition_rate_pct DESC, employees DESC;

-- Q4. Attrition by business travel frequency
SELECT BusinessTravel, COUNT(*) AS employees,
       ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
FROM employee_attrition GROUP BY BusinessTravel ORDER BY attrition_rate_pct DESC;

-- Q5. Attrition by job satisfaction score
SELECT JobSatisfaction, COUNT(*) AS employees,
       ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
FROM employee_attrition GROUP BY JobSatisfaction ORDER BY JobSatisfaction;

-- Q6. Attrition by work-life balance score
SELECT WorkLifeBalance, COUNT(*) AS employees,
       ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
FROM employee_attrition GROUP BY WorkLifeBalance ORDER BY WorkLifeBalance;

-- Q7. Attrition by tenure band
SELECT TenureBand, COUNT(*) AS employees,
       ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
FROM employee_attrition GROUP BY TenureBand ORDER BY attrition_rate_pct DESC;

-- Q8. Attrition by time since last promotion
SELECT PromotionWaitBand, COUNT(*) AS employees,
       ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
FROM employee_attrition GROUP BY PromotionWaitBand ORDER BY attrition_rate_pct DESC;

-- Q9. Attrition by commuting distance
SELECT CommuteBand, COUNT(*) AS employees,
       ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
FROM employee_attrition GROUP BY CommuteBand ORDER BY attrition_rate_pct DESC;

-- Q10. Highest model-scored attrition risk records
SELECT EmployeeNumber, Department, JobRole, OverTime, JobSatisfaction, WorkLifeBalance,
       MonthlyIncome, YearsAtCompany, ROUND(attrition_probability * 100, 1) AS attrition_probability_pct,
       risk_tier
FROM risk_predictions ORDER BY attrition_probability DESC LIMIT 25;
