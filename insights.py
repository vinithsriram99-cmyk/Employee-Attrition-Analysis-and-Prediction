"""Plain-pandas helpers for the Key Drivers and Retention Impact pages (no Streamlit code)."""
import pandas as pd


def rate(data: pd.DataFrame, column: str, value) -> float:
    subset = data[data[column] == value]
    return float(subset["AttritionFlag"].mean() * 100) if len(subset) else float("nan")


def driver_table(data: pd.DataFrame, importance: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """For the most important model features: how leavers differ from stayers."""
    rows = []
    for _, row in importance.head(top).iterrows():
        feature = row["feature"]
        if feature not in data.columns:
            continue
        if pd.api.types.is_numeric_dtype(data[feature]):
            stay = data.loc[data["AttritionFlag"] == 0, feature].median()
            left = data.loc[data["AttritionFlag"] == 1, feature].median()
            note = f"median {stay:,.1f} (stayed) vs {left:,.1f} (left)"
        else:
            rates = data.groupby(feature)["AttritionFlag"].mean().mul(100)
            note = f"highest attrition: {rates.idxmax()} ({rates.max():.1f}%) | lowest: {rates.idxmin()} ({rates.min():.1f}%)"
        rows.append({"feature": feature, "importance": round(float(row["importance_mean"]), 4), "what the data shows": note})
    return pd.DataFrame(rows)


def evidence_points(data: pd.DataFrame) -> list[str]:
    """Plain-language findings with their real numbers (used as suggested retention focus areas)."""
    points = []
    if "OverTime" in data:
        points.append(f"Overtime: {rate(data, 'OverTime', 'Yes'):.1f}% of employees working overtime left, vs {rate(data, 'OverTime', 'No'):.1f}% without overtime -> review workload and overtime policy.")
    if "TenureBand" in data:
        points.append(f"New joiners: {rate(data, 'TenureBand', '0-1 years'):.1f}% attrition in the first year, vs {rate(data, 'TenureBand', '11+ years'):.1f}% after 11 years -> strengthen onboarding and early check-ins.")
    if "JobSatisfaction" in data:
        points.append(f"Job satisfaction: score 1 has {rate(data, 'JobSatisfaction', 1):.1f}% attrition vs {rate(data, 'JobSatisfaction', 4):.1f}% at score 4 -> act on low satisfaction scores.")
    if "PromotionWaitBand" in data and "6+ years" in set(data["PromotionWaitBand"]):
        long_wait, short_wait = rate(data, "PromotionWaitBand", "6+ years"), rate(data, "PromotionWaitBand", "0-1 years")
        if long_wait > short_wait + 2:
            points.append(f"Career growth: {long_wait:.1f}% attrition after 6+ years without promotion, vs {short_wait:.1f}% within 1 year -> plan career-development conversations.")
        else:
            points.append(f"Promotion wait: no clear pattern in this data ({long_wait:.1f}% after 6+ years vs {short_wait:.1f}% within 1 year), so it is not a priority lever here.")
    if "JobRole" in data:
        top = data.groupby("JobRole")["AttritionFlag"].mean().mul(100).sort_values(ascending=False)
        points.append(f"Role hot-spot: {top.index[0]} has the highest attrition ({top.iloc[0]:.1f}%).")
    return points


def retention_scenario(merged: pd.DataFrame, data: pd.DataFrame, tiers: list, retained_pct: float, cost_pct: float) -> dict:
    """What-if estimate. merged needs: risk_tier, AttritionFlag, MonthlyIncome. All inputs are assumptions."""
    target = merged[merged["risk_tier"].isin(tiers)]
    leavers = int(target["AttritionFlag"].sum())
    retained = leavers * retained_pct / 100
    leaver_pay = target.loc[target["AttritionFlag"] == 1, "MonthlyIncome"].mean() * 12 if leavers else 0.0
    total_left = int(data["AttritionFlag"].sum())
    return {
        "employees_targeted": int(len(target)),
        "observed_leavers_in_group": leavers,
        "leavers_retained": round(retained, 1),
        "current_rate_pct": round(100 * total_left / len(data), 2),
        "new_rate_pct": round(100 * (total_left - retained) / len(data), 2),
        "estimated_saving": round(retained * leaver_pay * cost_pct / 100),
    }
