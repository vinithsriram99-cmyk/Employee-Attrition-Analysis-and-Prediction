# Employee Attrition Analysis and Prediction

HR-analytics project: cleans the IBM-style employee dataset, explores what is linked to attrition, trains two classifiers,
ranks employees by attrition risk, and presents everything in a Streamlit dashboard.

## Files
| File | Purpose |
| --- | --- |
| `Employee-Attrition.csv` | The supplied dataset (1,470 employees, 35 columns) |
| `clean_and_load.py` | Validates and cleans the data, adds engineered features, saves a clean CSV and a SQLite database |
| `transformers.py` | `IQRClipper`: outlier handling used inside the model pipeline |
| `train_model.py` | Trains Logistic Regression and Random Forest, selects by cross-validation, saves metrics, feature importance and risk scores |
| `write_results_summary.py` | Writes `RESULTS_SUMMARY.md` from your own run |
| `insights.py` | Helpers for the Key Drivers and Retention Impact pages |
| `app.py` | Streamlit dashboard (8 pages) |
| `db.py`, `config.py` | Database helper and shared settings |
| `queries.py`, `sql_queries.sql`, `schema.sql` | Ten SQL business questions and the table schema |
| `test_pipeline.py` | Tests for the data-preparation logic |

## Run
```bash
python3 -m pip install -r requirements.txt
python3 clean_and_load.py          # clean data + SQLite database
python3 train_model.py             # models, metrics, feature importance, risk scores
python3 write_results_summary.py   # RESULTS_SUMMARY.md
python3 -m streamlit run app.py
```
The raw CSV can be at `data/raw/Employee-Attrition.csv` or next to the scripts.

## Method
- **Cleaning:** schema checks, duplicate and missing-value handling (none found), target `Attrition` mapped to 1 = left, 0 = stayed.
- **Outliers:** counted with the IQR rule in the cleaning report; clipped to IQR limits inside the model pipeline, learned from training data only.
- **Feature engineering:** `AgeBand`, `TenureBand`, `CommuteBand`, `PromotionWaitBand`, `IncomeBand`, and `EngagementScore` (average of five 1-4 satisfaction ratings).
- **Models:** Logistic Regression and Random Forest (balanced class weights), stratified 80/20 split, preprocessing inside the pipeline (impute, clip, scale, one-hot).
- **Selection:** 5-fold cross-validated ROC-AUC on the training rows. The test set is used only for the reported scores.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix, training time.
- **Drivers:** permutation importance on the test set, shown with what the data says about each driver.
- **Business impact:** a what-if page estimates leavers retained, new attrition rate and replacement-cost savings from adjustable assumptions.

## Dashboard pages
Overview, EDA, Key Drivers, At-Risk Review (filter and download), Retention Impact, Model Evaluation, SQL Insights, Data Quality.

## Results
See `RESULTS_SUMMARY.md` (generated from your run). Typical values: ROC-AUC about 0.83, recall of leavers about 70%.
The brief suggests an example target of 85%. This dataset and these models do not reach 85% accuracy or AUC, and the results do not claim to.
Always predicting "stay" already gives about 84% accuracy because only 16% of employees left, so accuracy alone is not a good measure here.

## Limitations and responsible use
Small, imbalanced dataset; risk scores include training rows, so tier attrition rates are optimistic; balanced class weights make probabilities
higher than real-world rates (about 30% of employees land in the High tier while 16% actually left). Patterns are correlations, not causes.
Scores start a human retention conversation. They must never drive automatic hiring, firing, pay or promotion decisions.
