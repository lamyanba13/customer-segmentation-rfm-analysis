# Customer Segmentation RFM Analysis

A complete Python project that analyzes ecommerce customer behavior using Recency, Frequency, and Monetary (RFM) metrics. The repository includes a realistic synthetic transaction dataset, modular analysis code, professional visualizations, Jupyter notebooks, and a static HTML dashboard for a full workflow from raw data cleaning to business-ready customer segmentation.

## Project Overview

This project demonstrates how an ecommerce team can:

- clean transactional order data
- calculate RFM scores for each customer
- classify customers into actionable segments
- identify churn-risk customers
- quantify segment-level revenue contribution
- translate findings into business recommendations

The included dataset contains 3,000+ realistic transaction rows with the following fields:

- `Customer ID`
- `Order ID`
- `Order Date`
- `Sales Amount`
- `Category`
- `Region`

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- seaborn
- jupyter

## Folder Structure

```text
customer-segmentation-rfm-analysis/
|-- data/
|   |-- raw/
|   |   `-- ecommerce_transactions_raw.csv
|   `-- cleaned/
|       |-- ecommerce_transactions_cleaned.csv
|       |-- customer_rfm_scores.csv
|       `-- customer_rfm_segments.csv
|-- notebooks/
|   |-- 01_cleaning.ipynb
|   |-- 02_rfm_analysis.ipynb
|   `-- 03_customer_segments.ipynb
|-- src/
|   |-- __init__.py
|   |-- rfm.py
|   `-- segmentation.py
|-- visuals/
|   |-- category_revenue.png
|   |-- churn_risk_by_segment.png
|   |-- rfm_heatmap.png
|   |-- segment_counts.png
|   `-- segment_revenue_share.png
|-- index.html
|-- README.md
`-- requirements.txt
```

## Key Analyses

### 1. Data Cleaning

- removes duplicate orders
- standardizes data types
- converts order dates to datetime
- strips inconsistent category labels
- drops incomplete and invalid transactions

### 2. RFM Analysis

- `Recency`: days since latest purchase
- `Frequency`: unique order count
- `Monetary`: total spend per customer
- quartile-based scoring from 1 to 4
- combined RFM score for ranking customer value

### 3. Customer Segmentation

Customers are grouped into business-friendly segments such as:

- Champions
- Loyal Customers
- Potential Loyalists
- New Customers
- Big Spenders
- At Risk
- Hibernating

### 4. Revenue and Churn Insights

- segment-level revenue share
- churn-risk classification based on recency and frequency
- customer volume by segment
- category and region performance context

## How To Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Open the notebooks

```bash
jupyter notebook
```

Then run the notebooks in this order:

1. `01_cleaning.ipynb`
2. `02_rfm_analysis.ipynb`
3. `03_customer_segments.ipynb`

### 3. View the HTML dashboard

Open `index.html` in a browser to see a portfolio-style dashboard with KPI cards, segment snapshots, visual charts, revenue contribution, churn risk context, and business recommendations.

## Business Value

This analysis helps ecommerce stakeholders answer questions such as:

- Which customers contribute the most revenue?
- Which customer groups are most likely to churn?
- Which segments deserve retention investment?
- How should lifecycle campaigns differ by segment?

## Example Recommendations

- Retain `Champions` with VIP rewards and referral campaigns.
- Convert `Potential Loyalists` with personalized repeat-purchase offers.
- Recover `At Risk` customers using targeted win-back campaigns.
- Limit paid reactivation spend on `Hibernating` customers until engagement improves.

## Author Notes

The project is designed for portfolio use, academic submission, analytics practice, and GitHub publishing. It uses synthetic data so it can be shared publicly without privacy concerns.
