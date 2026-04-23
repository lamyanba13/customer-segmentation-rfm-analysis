from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = ["Sales Amount"]
CATEGORICAL_COLUMNS = ["Category", "Region"]


@dataclass
class ChurnThresholds:
    high_risk_recency: float
    medium_risk_recency: float
    low_frequency: float


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize and clean transaction records."""
    cleaned = df.copy()
    cleaned.columns = [col.strip() for col in cleaned.columns]
    cleaned["Order Date"] = pd.to_datetime(cleaned["Order Date"], errors="coerce")

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip().fillna("Unknown")

    cleaned["Customer ID"] = cleaned["Customer ID"].astype("string").str.strip()
    cleaned["Order ID"] = cleaned["Order ID"].astype("string").str.strip()

    cleaned = cleaned.drop_duplicates(subset=["Order ID"])
    cleaned = cleaned.dropna(subset=["Customer ID", "Order ID", "Order Date", "Sales Amount"])
    cleaned = cleaned.loc[cleaned["Sales Amount"] > 0].copy()
    cleaned["Sales Amount"] = cleaned["Sales Amount"].round(2)

    return cleaned.sort_values("Order Date").reset_index(drop=True)


def calculate_rfm(df: pd.DataFrame, analysis_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Build the RFM table from a transaction DataFrame."""
    if analysis_date is None:
        analysis_date = df["Order Date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("Customer ID")
        .agg(
            Recency=("Order Date", lambda dates: (analysis_date - dates.max()).days),
            Frequency=("Order ID", "nunique"),
            Monetary=("Sales Amount", "sum"),
            Last_Order_Date=("Order Date", "max"),
        )
        .reset_index()
    )

    return rfm


def _safe_qcut(series: pd.Series, ascending: bool = True) -> pd.Series:
    ranked = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranked, 4, labels=[1, 2, 3, 4]).astype(int)


def score_rfm(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assign quartile-based RFM scores and composite strings."""
    scored = rfm_df.copy()
    scored["R_Score"] = _safe_qcut(scored["Recency"], ascending=False)
    scored["F_Score"] = _safe_qcut(scored["Frequency"], ascending=True)
    scored["M_Score"] = _safe_qcut(scored["Monetary"], ascending=True)
    scored["RFM_Score"] = (
        scored["R_Score"].astype(str)
        + scored["F_Score"].astype(str)
        + scored["M_Score"].astype(str)
    )
    scored["RFM_Total"] = scored[["R_Score", "F_Score", "M_Score"]].sum(axis=1)
    return scored.sort_values(["RFM_Total", "Monetary"], ascending=[False, False]).reset_index(drop=True)


def derive_churn_thresholds(rfm_df: pd.DataFrame) -> ChurnThresholds:
    return ChurnThresholds(
        high_risk_recency=float(rfm_df["Recency"].quantile(0.75)),
        medium_risk_recency=float(rfm_df["Recency"].quantile(0.50)),
        low_frequency=float(rfm_df["Frequency"].quantile(0.25)),
    )


def detect_churn_risk(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Label likely churn risk from recency and purchase cadence."""
    churn = rfm_df.copy()
    thresholds = derive_churn_thresholds(churn)

    conditions = [
        (churn["Recency"] >= thresholds.high_risk_recency) & (churn["Frequency"] <= thresholds.low_frequency),
        (churn["Recency"] >= thresholds.medium_risk_recency),
    ]
    choices = ["High Risk", "Medium Risk"]
    churn["Churn Risk"] = np.select(conditions, choices, default="Low Risk")
    return churn


def segment_revenue_contribution(rfm_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rfm_df.groupby("Segment", dropna=False)
        .agg(
            Customers=("Customer ID", "nunique"),
            Revenue=("Monetary", "sum"),
            Avg_Customer_Value=("Monetary", "mean"),
        )
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    total_revenue = summary["Revenue"].sum()
    summary["Revenue Share %"] = np.where(total_revenue > 0, summary["Revenue"] / total_revenue * 100, 0.0)
    return summary


def segment_churn_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    return (
        rfm_df.groupby(["Segment", "Churn Risk"], dropna=False)
        .agg(Customers=("Customer ID", "nunique"), Revenue=("Monetary", "sum"))
        .reset_index()
        .sort_values(["Segment", "Customers"], ascending=[True, False])
    )


def kpi_snapshot(df: pd.DataFrame, rfm_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "transactions": int(len(df)),
        "customers": int(df["Customer ID"].nunique()),
        "revenue": float(df["Sales Amount"].sum()),
        "average_order_value": float(df["Sales Amount"].mean()),
        "repeat_purchase_rate": float((rfm_df["Frequency"] > 1).mean() * 100),
    }

