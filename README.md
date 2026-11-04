# Employee Attrition Analysis and Prediction

A Python, SQL, machine-learning, and Streamlit project that analyses employee attrition, compares two classifiers, and produces a prioritised employee retention review queue.

## What this project delivers

- Clean, validated employee data in CSV and SQLite formats.
- Exploratory analysis for attrition by department, role, travel, overtime, satisfaction, tenure, promotion wait, commute, and income band.
- Two model candidates: Logistic Regression and Random Forest.
- Test-set metrics: accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- A risk probability and risk tier for each employee after model training.
- A Streamlit dashboard with overview, EDA, at-risk review, evaluation, SQL insights, and data-quality pages.

## Project structure

| Path | Purpose |
| --- | --- |
| `clean_and_load.py` | Validates the source CSV, creates business-friendly bands, saves a clean CSV, and loads SQLite. |
| `db.py` | SQLite connection and query helpers. |
| `schema.sql` | Database schema and indexes. |
| `queries.py` and `sql_queries.sql` | Ten SQL business questions used in the dashboard and available for manual review. |
| `train_model.py` | Trains, evaluates, selects, and saves the strongest model by test ROC-AUC. |
| `app.py` | Streamlit dashboard. |
| `config.py` | Central project paths, random seed, and risk-threshold settings. |
| `data/raw/Employee-Attrition.csv` | Input dataset. Copy the supplied dataset here before running. |
| `data/processed/` | Generated clean data and cleaning report. |
| `artifacts/` | Generated model, metrics, test predictions, and employee risk scores. |
| `docs/PROJECT_PLAN.md` | Implementation plan and evaluation checklist. |
| `docs/PROJECT_REPORT.md` | Project explanation and report template. |
| `RESULTS_SUMMARY.md` | Actual data quality, model, and EDA results generated from the supplied dataset. |
| `tests/test_pipeline.py` | Focused validation tests for the data preparation logic. |

## Setup and run

Use Python 3.10 or newer. From this project folder, run the following commands.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Put the supplied dataset at this exact path
# data/raw/Employee-Attrition.csv

# 4. Validate, clean, enrich, and load the SQLite database
python clean_and_load.py

# 5. Train and evaluate the models, then create risk scores
python train_model.py

# 6. Optional: run data pipeline tests
python -m unittest tests/test_pipeline.py

# 7. Start the dashboard
streamlit run app.py
```

The terminal will show a local URL such as `http://localhost:8501`. Open it in your browser.

## What happens in each run step

1. `clean_and_load.py` checks required columns and unique employee IDs, removes duplicate rows, handles missing values, adds `AttritionFlag` and five business-friendly bands, writes `data/processed/employee_attrition_clean.csv`, and creates `data/employee_attrition.db`.
2. `train_model.py` makes a stratified 80/20 train/test split, one-hot encodes categorical columns within a pipeline, evaluates Logistic Regression and Random Forest, chooses the higher ROC-AUC model, and saves the model and review files in `artifacts/`.
3. `streamlit run app.py` presents the dashboard. The At-Risk Review and Model Evaluation pages become available after model training.

## Model design

The target is `AttritionFlag`: `Yes = 1` and `No = 0`.

- `EmployeeNumber` is kept only as an identifier and is never used as a predictor.
- Constant columns (`EmployeeCount`, `Over18`, and `StandardHours`) are excluded from modelling because they add no predictive signal.
- Numeric features use median imputation. Categorical features use most-frequent imputation and one-hot encoding.
- Training and preprocessing remain inside one scikit-learn pipeline, preventing test-set leakage.
- The selected model is the candidate with the stronger test ROC-AUC. Do not claim an 85% threshold until the saved `artifacts/metrics.json` confirms it.

## Risk tiers

The dashboard classifies model probability as:

| Tier | Probability |
| --- | --- |
| High | 50% or more |
| Medium | 30% to below 50% |
| Low | Below 30% |

These cut-offs are set in `config.py` and can be changed with a clear HR policy decision. A score should begin a human retention conversation. It must not be used for automatic hiring, termination, compensation, or promotion decisions.

## Dashboard guide

- **Overview:** workforce totals and the largest observed attrition-rate differences.
- **EDA:** selected attrition-rate breakdowns, age and income distributions, and numerical correlations.
- **At-Risk Review:** filter model scores by tier, department, and role; download the selected review queue as CSV.
- **Model Evaluation:** selected model metrics, model comparison, and confusion matrix.
- **SQL Insights:** ten documented business questions with their actual SQL.
- **Data Quality:** completeness, distinct-value counts, duplicate count, and constant source columns.

## GitHub commands after you test locally

```bash
git init
git add .
git commit -m "Build employee attrition analysis mini project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/employee-attrition-analysis.git
git push -u origin main
```

Do not commit generated models or employee risk files unless your evaluator explicitly asks for them. They are intentionally covered by `.gitignore`.
