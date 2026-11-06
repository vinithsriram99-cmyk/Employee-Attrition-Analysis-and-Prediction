"""Streamlit dashboard for Employee Attrition Analysis and Prediction."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import ARTIFACTS_DIR, METRICS_PATH, PROCESSED_DATA_PATH, RISK_PREDICTIONS_PATH
from db import database_exists
from insights import driver_table, evidence_points, retention_scenario
from queries import QUERY_CATALOG, run_named_query

st.set_page_config(page_title="Employee Attrition Explorer", page_icon="📊", layout="wide")


@st.cache_data
def load_employee_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_PATH)


@st.cache_data
def load_risk_predictions() -> pd.DataFrame:
    return pd.read_csv(RISK_PREDICTIONS_PATH)


@st.cache_data
def load_metrics() -> dict[str, object]:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def attrition_rate_frame(data: pd.DataFrame, column: str) -> pd.DataFrame:
    result = (
        data.groupby(column, dropna=False)
        .agg(employees=("EmployeeNumber", "count"), attrition_rate=("AttritionFlag", "mean"))
        .reset_index()
    )
    result["attrition_rate_pct"] = result["attrition_rate"] * 100
    return result.sort_values("attrition_rate_pct", ascending=False)


def make_rate_chart(data: pd.DataFrame, column: str, title: str):
    summary = attrition_rate_frame(data, column)
    return px.bar(
        summary, x=column, y="attrition_rate_pct", text="employees", title=title,
        labels={"attrition_rate_pct": "Attrition rate (%)", column: column},
        color="attrition_rate_pct", color_continuous_scale="Reds",
    )


def render_overview(data: pd.DataFrame) -> None:
    st.title("Employee Attrition Overview")
    st.caption("Use the dashboard to explore workforce attrition patterns and review model risk signals.")
    total_employees = len(data)
    employees_left = int(data["AttritionFlag"].sum())
    overall_rate = data["AttritionFlag"].mean() * 100
    overtime_rate = data.loc[data["OverTime"] == "Yes", "AttritionFlag"].mean() * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", f"{total_employees:,}")
    c2.metric("Employees who left", f"{employees_left:,}")
    c3.metric("Overall attrition rate", f"{overall_rate:.1f}%")
    c4.metric("Overtime attrition rate", f"{overtime_rate:.1f}%")

    left, right = st.columns(2)
    left.plotly_chart(make_rate_chart(data, "Department", "Attrition rate by department"), use_container_width=True)
    right.plotly_chart(make_rate_chart(data, "OverTime", "Attrition rate by overtime status"), use_container_width=True)
    st.plotly_chart(make_rate_chart(data, "JobRole", "Attrition rate by job role"), use_container_width=True)


def render_eda(data: pd.DataFrame) -> None:
    st.title("Exploratory Data Analysis")
    st.caption("The charts describe observed patterns. They do not prove that a factor causes attrition.")
    dimension = st.selectbox(
        "Compare attrition by", ["BusinessTravel", "JobSatisfaction", "WorkLifeBalance", "TenureBand", "PromotionWaitBand", "CommuteBand", "IncomeBand"]
    )
    st.plotly_chart(make_rate_chart(data, dimension, f"Attrition rate by {dimension}"), use_container_width=True)

    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(data, x="Age", color="Attrition", barmode="overlay", nbins=24, title="Age distribution by attrition"),
        use_container_width=True,
    )
    right.plotly_chart(
        px.box(data, x="Attrition", y="MonthlyIncome", color="Attrition", title="Monthly income by attrition"),
        use_container_width=True,
    )
    correlation_columns = ["Age", "DistanceFromHome", "JobSatisfaction", "MonthlyIncome", "TotalWorkingYears", "WorkLifeBalance", "YearsAtCompany", "YearsSinceLastPromotion", "AttritionFlag"]
    correlation = data[correlation_columns].corr(numeric_only=True).round(2)
    st.plotly_chart(
        px.imshow(correlation, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric feature correlations"),
        use_container_width=True,
    )


def render_risk_review() -> None:
    st.title("At Risk Employee Review")
    st.warning("Use these scores only as a retention-review prompt. Do not make automated employment decisions from this dashboard.")
    if not RISK_PREDICTIONS_PATH.exists():
        st.info("Run `python train_model.py` to create model risk scores.")
        return

    risks = load_risk_predictions().copy()
    department_options = sorted(risks["Department"].dropna().unique())
    role_options = sorted(risks["JobRole"].dropna().unique())
    left, middle, right = st.columns(3)
    selected_tiers = left.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"])
    selected_departments = middle.multiselect("Department", department_options, default=department_options)
    selected_roles = right.multiselect("Job role", role_options, default=role_options)
    filtered = risks[
        risks["risk_tier"].isin(selected_tiers)
        & risks["Department"].isin(selected_departments)
        & risks["JobRole"].isin(selected_roles)
    ].copy()
    filtered["attrition_probability_pct"] = (filtered["attrition_probability"] * 100).round(1)
    filtered = filtered.sort_values("attrition_probability", ascending=False)
    st.metric("Employees in selected review queue", f"{len(filtered):,}")
    st.dataframe(filtered.drop(columns="attrition_probability"), use_container_width=True, hide_index=True)
    st.download_button(
        "Download selected review queue (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="employee_attrition_review_queue.csv",
        mime="text/csv",
    )


def render_model_evaluation() -> None:
    st.title("Model Evaluation")
    if not METRICS_PATH.exists():
        st.info("Run `python train_model.py` to create model evaluation results.")
        return
    metrics = load_metrics()
    selected_name = str(metrics["selected_model"])
    selected = metrics["models"][selected_name]
    st.subheader(f"Selected model: {selected_name}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{selected['accuracy']:.1%}")
    c2.metric("Precision", f"{selected['precision']:.1%}")
    c3.metric("Recall", f"{selected['recall']:.1%}")
    c4.metric("F1-score", f"{selected['f1_score']:.1%}")
    c5.metric("ROC-AUC", f"{selected['roc_auc']:.1%}")
    st.caption(f"Train/test split: {metrics['training_set_size']} training rows and {metrics['test_set_size']} test rows. Model selection is based on ROC-AUC.")

    matrix = selected["confusion_matrix"]
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted stay", "Predicted leave"],
            y=["Actual stay", "Actual leave"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )
    figure.update_layout(title="Confusion matrix", xaxis_title="Prediction", yaxis_title="Actual")
    st.plotly_chart(figure, use_container_width=True)
    model_comparison = (
        pd.DataFrame(metrics["models"]).T.drop(columns=["confusion_matrix", "model"], errors="ignore")
        .infer_objects().reset_index(names="model")
    )
    st.dataframe(model_comparison, use_container_width=True, hide_index=True)
    st.caption("train_seconds = time to fit on the training rows; cv_roc_auc = 5-fold cross-validated ROC-AUC used to pick the model.")
    st.info(str(metrics["note"]))


def render_sql_insights() -> None:
    st.title("SQL Insights")
    if not database_exists():
        st.info("Run `python clean_and_load.py` to create the SQLite database.")
        return
    selected_query = st.selectbox("Choose a business question", list(QUERY_CATALOG))
    st.code(QUERY_CATALOG[selected_query].strip(), language="sql")
    try:
        result = run_named_query(selected_query)
    except Exception as error:
        st.error(f"This query is not ready yet: {error}")
        return
    st.dataframe(result, use_container_width=True, hide_index=True)


def render_data_quality(data: pd.DataFrame) -> None:
    st.title("Data Quality")
    summary = pd.DataFrame({"column": data.columns, "missing_values": data.isna().sum().values, "distinct_values": data.nunique().values})
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.write(f"Duplicate rows: {int(data.duplicated().sum())}")
    st.write("Constant source columns: " + ", ".join(column for column in data.columns if data[column].nunique() <= 1))


def render_key_drivers(data: pd.DataFrame) -> None:
    st.title("Key Drivers of Attrition")
    importance_path = ARTIFACTS_DIR / "feature_importance.csv"
    if not importance_path.exists():
        st.info("Run `python train_model.py` to calculate feature importance.")
        return
    importance = pd.read_csv(importance_path)
    top = importance.head(12).sort_values("importance_mean")
    st.plotly_chart(
        px.bar(top, x="importance_mean", y="feature", orientation="h",
               title="Most influential features (drop in ROC-AUC when the feature is shuffled, test data)"),
        use_container_width=True,
    )
    st.dataframe(driver_table(data, importance, top=10), use_container_width=True, hide_index=True)
    st.subheader("What the data suggests for retention")
    for point in evidence_points(data):
        st.write("- " + point)
    st.caption("Importance shows what the model relies on, and the comparisons are observed patterns. Neither proves cause and effect.")


def render_retention_impact(data: pd.DataFrame) -> None:
    st.title("Retention Impact (what-if)")
    st.warning("Assumption-based scenario. Change the sliders to test ideas; this is not a forecast.")
    if not RISK_PREDICTIONS_PATH.exists():
        st.info("Run `python train_model.py` first.")
        return
    merged = load_risk_predictions().merge(data[["EmployeeNumber", "AttritionFlag"]], on="EmployeeNumber")
    tiers = st.multiselect("Employees targeted by retention actions", ["High", "Medium", "Low"], default=["High"])
    retained_pct = st.slider("Share of would-be leavers that the actions retain (%)", 0, 60, 20)
    cost_pct = st.slider("Cost to replace one employee (% of annual salary)", 20, 200, 50)
    result = retention_scenario(merged, data, tiers, retained_pct, cost_pct)
    c1, c2, c3 = st.columns(3)
    c1.metric("Employees targeted", f"{result['employees_targeted']:,}")
    c2.metric("Observed leavers in this group", f"{result['observed_leavers_in_group']:,}")
    c3.metric("Leavers retained", f"{result['leavers_retained']}")
    c4, c5 = st.columns(2)
    c4.metric("Attrition rate", f"{result['new_rate_pct']}%", f"{result['new_rate_pct'] - result['current_rate_pct']:.2f} points vs {result['current_rate_pct']}%", delta_color="inverse")
    c5.metric("Estimated replacement cost saved", f"{result['estimated_saving']:,}")
    st.caption("Salary values use the dataset's own currency units. Risk scores include employees the model was trained on, so the observed rates by tier are optimistic.")
    tier_table = merged.groupby("risk_tier").agg(employees=("EmployeeNumber", "count"), observed_attrition_rate=("AttritionFlag", "mean")).reset_index()
    tier_table["observed_attrition_rate"] = (tier_table["observed_attrition_rate"] * 100).round(1)
    st.dataframe(tier_table, use_container_width=True, hide_index=True)


def main() -> None:
    st.sidebar.title("Employee Attrition")
    page = st.sidebar.radio("Navigate", ["Overview", "EDA", "Key Drivers", "At-Risk Review", "Retention Impact", "Model Evaluation", "SQL Insights", "Data Quality"])
    if not PROCESSED_DATA_PATH.exists():
        st.title("Employee Attrition Analysis and Prediction")
        st.error("Run `python clean_and_load.py` before starting the dashboard.")
        return
    data = load_employee_data()
    if page == "Overview":
        render_overview(data)
    elif page == "EDA":
        render_eda(data)
    elif page == "Key Drivers":
        render_key_drivers(data)
    elif page == "At-Risk Review":
        render_risk_review()
    elif page == "Retention Impact":
        render_retention_impact(data)
    elif page == "Model Evaluation":
        render_model_evaluation()
    elif page == "SQL Insights":
        render_sql_insights()
    else:
        render_data_quality(data)


if __name__ == "__main__":
    main()
