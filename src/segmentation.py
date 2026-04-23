from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_synthetic_transactions(
    num_rows: int = 3600,
    customer_count: int = 900,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic ecommerce transaction dataset."""
    rng = np.random.default_rng(seed)

    customer_ids = [f"CUST-{idx:04d}" for idx in range(1, customer_count + 1)]
    categories = {
        "Electronics": {"mean": 220, "sigma": 0.55},
        "Fashion": {"mean": 95, "sigma": 0.45},
        "Home & Living": {"mean": 150, "sigma": 0.50},
        "Beauty": {"mean": 70, "sigma": 0.35},
        "Sports": {"mean": 130, "sigma": 0.40},
        "Groceries": {"mean": 45, "sigma": 0.25},
    }
    category_names = list(categories.keys())
    category_weights = np.array([0.18, 0.22, 0.17, 0.13, 0.12, 0.18])

    regions = ["North", "South", "East", "West", "Central"]
    region_weights = np.array([0.22, 0.20, 0.18, 0.24, 0.16])

    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    seasonal_month_boost = {
        1: 1.00,
        2: 0.94,
        3: 1.02,
        4: 0.98,
        5: 1.04,
        6: 1.01,
        7: 0.97,
        8: 1.03,
        9: 1.08,
        10: 1.15,
        11: 1.28,
        12: 1.33,
    }

    loyalty = rng.gamma(shape=2.2, scale=1.0, size=customer_count)
    loyalty = loyalty / loyalty.sum()
    chosen_customers = rng.choice(customer_ids, size=num_rows, replace=True, p=loyalty)
    chosen_categories = rng.choice(category_names, size=num_rows, p=category_weights)
    chosen_regions = rng.choice(regions, size=num_rows, p=region_weights)
    chosen_dates = rng.choice(dates, size=num_rows, replace=True)

    records: List[Dict[str, object]] = []
    for idx in range(num_rows):
        category = chosen_categories[idx]
        dt = pd.Timestamp(chosen_dates[idx])
        category_profile = categories[category]
        month_multiplier = seasonal_month_boost[dt.month]
        region_multiplier = {"North": 1.04, "South": 0.97, "East": 0.95, "West": 1.08, "Central": 1.00}[chosen_regions[idx]]
        raw_sale = rng.lognormal(
            mean=np.log(category_profile["mean"]),
            sigma=category_profile["sigma"],
        )
        sales_amount = max(12, raw_sale * month_multiplier * region_multiplier)

        records.append(
            {
                "Customer ID": chosen_customers[idx],
                "Order ID": f"ORD-{2023000 + idx:07d}",
                "Order Date": dt.strftime("%Y-%m-%d"),
                "Sales Amount": round(float(sales_amount), 2),
                "Category": category,
                "Region": chosen_regions[idx],
            }
        )

    df = pd.DataFrame(records)

    # Add a few intentionally imperfect records so the cleaning notebook has work to do.
    imperfect_rows = df.sample(15, random_state=seed).copy()
    imperfect_rows.loc[imperfect_rows.index[:5], "Sales Amount"] = np.nan
    imperfect_rows.loc[imperfect_rows.index[5:10], "Order Date"] = None
    imperfect_rows.loc[imperfect_rows.index[10:], "Category"] = "  Fashion  "
    raw_df = pd.concat([df, imperfect_rows], ignore_index=True)

    duplicate_rows = raw_df.sample(12, random_state=seed + 1).copy()
    raw_df = pd.concat([raw_df, duplicate_rows], ignore_index=True)

    return raw_df.sample(frac=1, random_state=seed).reset_index(drop=True)


def assign_segments(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Map RFM patterns into business-friendly customer segments."""
    segmented = rfm_df.copy()

    def _segment_row(row: pd.Series) -> str:
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        if r >= 4 and f <= 2:
            return "New Customers"
        if r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        if r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        if m >= 4 and f <= 2:
            return "Big Spenders"
        return "Potential Loyalists"

    segmented["Segment"] = segmented.apply(_segment_row, axis=1)
    return segmented


def get_business_recommendations() -> Dict[str, str]:
    return {
        "Champions": "Reward with VIP perks, early product access, and referral campaigns to amplify advocacy.",
        "Loyal Customers": "Deepen retention using bundles, subscription offers, and personalized cross-sell messaging.",
        "Potential Loyalists": "Nudge toward a second or third order with limited-time offers and product education.",
        "New Customers": "Improve onboarding with welcome journeys, category discovery, and first-repeat discounts.",
        "Big Spenders": "Protect high-value shoppers with concierge-style support and premium assortment recommendations.",
        "At Risk": "Trigger win-back campaigns using urgency, replenishment reminders, and service outreach.",
        "Hibernating": "Use low-cost reactivation channels and suppress expensive paid media unless engagement returns.",
    }


def save_visuals(transactions_df: pd.DataFrame, rfm_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    plt.figure(figsize=(10, 6))
    category_sales = (
        transactions_df.groupby("Category")["Sales Amount"].sum().sort_values(ascending=False)
    )
    sns.barplot(x=category_sales.values, y=category_sales.index)
    plt.title("Revenue by Product Category")
    plt.xlabel("Revenue")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(output_dir / "category_revenue.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    segment_counts = rfm_df["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Customers"]
    sns.barplot(data=segment_counts, x="Customers", y="Segment")
    plt.title("Customer Count by Segment")
    plt.tight_layout()
    plt.savefig(output_dir / "segment_counts.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    revenue_by_segment = rfm_df.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)
    plt.pie(revenue_by_segment.values, labels=revenue_by_segment.index, autopct="%1.1f%%", startangle=140)
    plt.title("Revenue Contribution by Segment")
    plt.tight_layout()
    plt.savefig(output_dir / "segment_revenue_share.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    pivot = rfm_df.pivot_table(
        index="F_Score",
        columns="R_Score",
        values="Monetary",
        aggfunc="mean",
    )
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Average Monetary Value by R and F Scores")
    plt.tight_layout()
    plt.savefig(output_dir / "rfm_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    churn_counts = (
        rfm_df.groupby(["Segment", "Churn Risk"])["Customer ID"]
        .nunique()
        .reset_index(name="Customers")
    )
    sns.barplot(data=churn_counts, x="Customers", y="Segment", hue="Churn Risk")
    plt.title("Churn Risk by Segment")
    plt.tight_layout()
    plt.savefig(output_dir / "churn_risk_by_segment.png", dpi=200)
    plt.close()
