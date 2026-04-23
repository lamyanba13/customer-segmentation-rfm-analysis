from __future__ import annotations

import json
from itertools import count
from pathlib import Path

import pandas as pd

from src.rfm import (
    calculate_rfm,
    clean_transactions,
    detect_churn_risk,
    kpi_snapshot,
    score_rfm,
    segment_churn_summary,
    segment_revenue_contribution,
)
from src.segmentation import assign_segments, create_synthetic_transactions, get_business_recommendations, save_visuals


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "ecommerce_transactions_raw.csv"
CLEAN_PATH = ROOT / "data" / "cleaned" / "ecommerce_transactions_cleaned.csv"
RFM_PATH = ROOT / "data" / "cleaned" / "customer_rfm_segments.csv"
RFM_SCORES_PATH = ROOT / "data" / "cleaned" / "customer_rfm_scores.csv"
VISUALS_DIR = ROOT / "visuals"
NOTEBOOKS_DIR = ROOT / "notebooks"
CELL_COUNTER = count(1)


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"code-{next(CELL_COUNTER):03d}",
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"markdown-{next(CELL_COUNTER):03d}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def build_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = create_synthetic_transactions()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(RAW_PATH, index=False)

    clean_df = clean_transactions(raw_df)
    clean_df.to_csv(CLEAN_PATH, index=False)

    rfm_df = calculate_rfm(clean_df)
    rfm_df = score_rfm(rfm_df)
    rfm_df.to_csv(RFM_SCORES_PATH, index=False)
    rfm_df = assign_segments(rfm_df)
    rfm_df = detect_churn_risk(rfm_df)
    rfm_df.to_csv(RFM_PATH, index=False)

    save_visuals(clean_df, rfm_df, VISUALS_DIR)
    return clean_df, rfm_df


def build_notebooks(clean_df: pd.DataFrame, rfm_df: pd.DataFrame) -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    recommendations = get_business_recommendations()
    revenue_summary = segment_revenue_contribution(rfm_df)
    churn_summary = segment_churn_summary(rfm_df)
    kpis = kpi_snapshot(clean_df, rfm_df)

    notebook_1 = [
        _markdown_cell(
            "# Data Cleaning\n"
            "This notebook loads the raw ecommerce transaction dataset, inspects its quality, and applies cleaning steps before downstream analysis."
        ),
        _code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import pandas as pd\n"
            "\n"
            "ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
            "sys.path.insert(0, str(ROOT))\n"
            "from src.rfm import clean_transactions\n"
            "\n"
            "raw_path = ROOT / 'data' / 'raw' / 'ecommerce_transactions_raw.csv'\n"
            "df_raw = pd.read_csv(raw_path)\n"
            "df_raw.head()"
        ),
        _code_cell(
            "df_raw.info()\n"
            "df_raw.isna().sum()"
        ),
        _code_cell(
            "df_clean = clean_transactions(df_raw)\n"
            "df_clean.head()"
        ),
        _code_cell(
            "print('Rows before cleaning:', len(df_raw))\n"
            "print('Rows after cleaning:', len(df_clean))\n"
            "print('Duplicate Order IDs removed:', df_raw.duplicated(subset=['Order ID']).sum())\n"
            "print('Missing values after cleaning:')\n"
            "print(df_clean.isna().sum())"
        ),
        _code_cell(
            "clean_path = ROOT / 'data' / 'cleaned' / 'ecommerce_transactions_cleaned.csv'\n"
            "df_clean.to_csv(clean_path, index=False)\n"
            "clean_path"
        ),
    ]

    notebook_2 = [
        _markdown_cell(
            "# RFM Analysis\n"
            "This notebook calculates customer-level recency, frequency, and monetary values and adds quartile-based RFM scores."
        ),
        _code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
            "sys.path.insert(0, str(ROOT))\n"
            "from src.rfm import calculate_rfm, score_rfm\n"
            "\n"
            "clean_path = ROOT / 'data' / 'cleaned' / 'ecommerce_transactions_cleaned.csv'\n"
            "df = pd.read_csv(clean_path, parse_dates=['Order Date'])\n"
            "rfm = score_rfm(calculate_rfm(df))\n"
            "rfm.head()"
        ),
        _code_cell(
            "rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2)"
        ),
        _code_cell(
            "plt.figure(figsize=(8, 5))\n"
            "sns.histplot(rfm['Monetary'], bins=30, kde=True)\n"
            "plt.title('Distribution of Customer Monetary Value')\n"
            "plt.show()"
        ),
        _code_cell(
            "pivot = rfm.pivot_table(index='F_Score', columns='R_Score', values='Monetary', aggfunc='mean')\n"
            "plt.figure(figsize=(7, 5))\n"
            "sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu')\n"
            "plt.title('Average Monetary Value by R and F Scores')\n"
            "plt.show()"
        ),
        _code_cell(
            "rfm_path = ROOT / 'data' / 'cleaned' / 'customer_rfm_scores.csv'\n"
            "rfm.to_csv(rfm_path, index=False)\n"
            "rfm_path"
        ),
    ]

    recommendations_lines = "\\n".join([f"- **{segment}**: {text}" for segment, text in recommendations.items()])
    notebook_3 = [
        _markdown_cell(
            "# Customer Segmentation\n"
            "This notebook assigns business-facing segments, evaluates revenue concentration, flags churn risk, and summarizes recommendations.\n\n"
            f"## Suggested Actions\n{recommendations_lines}"
        ),
        _code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
            "sys.path.insert(0, str(ROOT))\n"
            "from src.rfm import detect_churn_risk, segment_revenue_contribution, segment_churn_summary\n"
            "from src.segmentation import assign_segments, get_business_recommendations\n"
            "\n"
            "clean_path = ROOT / 'data' / 'cleaned' / 'ecommerce_transactions_cleaned.csv'\n"
            "scores_path = ROOT / 'data' / 'cleaned' / 'customer_rfm_scores.csv'\n"
            "segments_path = ROOT / 'data' / 'cleaned' / 'customer_rfm_segments.csv'\n"
            "df = pd.read_csv(clean_path, parse_dates=['Order Date'])\n"
            "rfm = pd.read_csv(scores_path, parse_dates=['Last_Order_Date'])\n"
            "if 'Segment' not in rfm.columns:\n"
            "    rfm = assign_segments(rfm)\n"
            "rfm = detect_churn_risk(rfm)\n"
            "rfm.to_csv(segments_path, index=False)\n"
            "rfm.head()"
        ),
        _code_cell(
            "segment_summary = segment_revenue_contribution(rfm)\n"
            "segment_summary"
        ),
        _code_cell(
            "plt.figure(figsize=(8, 5))\n"
            "sns.barplot(data=segment_summary, x='Revenue Share %', y='Segment')\n"
            "plt.title('Segment Revenue Contribution')\n"
            "plt.show()"
        ),
        _code_cell(
            "churn_summary = segment_churn_summary(rfm)\n"
            "churn_summary"
        ),
        _markdown_cell(
            "## KPI Snapshot\n"
            f"- Transactions analyzed: **{kpis['transactions']:,}**\n"
            f"- Customers analyzed: **{kpis['customers']:,}**\n"
            f"- Revenue analyzed: **${kpis['revenue']:,.2f}**\n"
            f"- Average order value: **${kpis['average_order_value']:,.2f}**\n"
            f"- Repeat purchase rate: **{kpis['repeat_purchase_rate']:.1f}%**"
        ),
    ]

    _write_notebook(NOTEBOOKS_DIR / "01_cleaning.ipynb", notebook_1)
    _write_notebook(NOTEBOOKS_DIR / "02_rfm_analysis.ipynb", notebook_2)
    _write_notebook(NOTEBOOKS_DIR / "03_customer_segments.ipynb", notebook_3)

    (ROOT / "data" / "cleaned" / "segment_revenue_summary.csv").write_text(
        revenue_summary.to_csv(index=False),
        encoding="utf-8",
    )
    (ROOT / "data" / "cleaned" / "segment_churn_summary.csv").write_text(
        churn_summary.to_csv(index=False),
        encoding="utf-8",
    )


def main() -> None:
    clean_df, rfm_df = build_datasets()
    build_notebooks(clean_df, rfm_df)


if __name__ == "__main__":
    main()
