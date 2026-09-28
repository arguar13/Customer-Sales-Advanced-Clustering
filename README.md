*Read this in other languages: [Español](README_es.md)*

# Customer Sales - Advanced Clustering and Segmentation Pipeline

This project implements an end-to-end, reproducible Machine Learning pipeline that segments the customers of a real online retailer from their transaction history. Transactions are turned into customer-level behavioral features (RFM and beyond), processed with custom Scikit-Learn transformers, clustered with an automatically benchmarked and tuned algorithm, and validated **against the future**: the segments are built with data up to a cutoff date and then checked on what customers actually did in the following months.

---

# About the Dataset

**Name:** [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) — UCI Machine Learning Repository  
**Citation:** Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33  
**License:** CC BY 4.0  
**Problem Type:** Unsupervised Learning (Clustering / Customer Segmentation)

## Dataset Characteristics

- **Size:** 541,909 invoice lines between 01/12/2010 and 09/12/2011.
- **Business:** UK-based online retailer selling unique all-occasion giftware; many customers are wholesalers.
- **Columns:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.
- **Format:** the original Excel file is stored as a compressed CSV (`data/online_retail.csv.gz`) with the same columns.

---

# Project Objective

Give marketing an actionable answer to *which groups of customers exist, how valuable each one is and what to do with each of them*, with a segmentation that is reproducible, leakage-safe and validated on data the model never saw.

## Key Capabilities

- Transaction cleaning with an explicit log of every rule, and cancellations netted out of spending (net revenue).
- Temporal design: features from an observation window, validation on a later outcome window.
- Customer-level feature construction (RFM + tenure + product variety) and custom Scikit-Learn transformers.
- Skew-aware preprocessing: `log1p`, IQR clipping and scaling inside a single pipeline.
- Data-driven selection of the number of clusters (Elbow + consensus of Silhouette, Calinski-Harabasz and Davies-Bouldin).
- Automated benchmarking (K-Means, Gaussian Mixture, Agglomerative) and Optuna tuning adopted only if it beats the baseline.
- Stability analysis, temporal validation and business personas.

---

# Data Preparation

## Transaction Cleaning

| Step | Rows | Removed |
|---|---|---|
| Raw transactions | 541,909 | - |
| Without missing `customer_id` | 406,829 | 135,080 |
| Without exact duplicates | 401,604 | 5,225 |
| Without non-positive prices / non-product codes (postage, fees, adjustments) | 399,656 | 1,948 |

The 8,506 cancellation lines are **kept as negative revenue** rather than dropped: some large orders were cancelled right after being placed, and dropping only the cancellation would keep a purchase that never happened. Cancellations amount to 5.4% of gross revenue.

## Temporal Design

- **Observation window (01/12/2010 - 31/08/2011):** the only data used to build features and fit the segmentation. 3,314 customers purchased in it; 9 of them had a non-positive net revenue (everything cancelled) and are excluded, leaving **3,305 customers**.
- **Outcome window (01/09/2011 - 09/12/2011):** never seen by the model; used to check whether the segments anticipate repurchase and revenue.

## Customer Features

| Feature | Definition |
|---|---|
| `recency` | Days since the last purchase (at the cutoff date) |
| `frequency` | Number of distinct invoices |
| `monetary` | Net revenue in GBP (purchases minus cancellations) |
| `tenure` | Days since the first purchase |
| `n_products` | Number of distinct products bought |
| `avg_order_value` | `monetary / frequency` (created by `FeatureEngineer`) |
| `orders_per_month` | `frequency / max(tenure / 30, 1)` (created by `FeatureEngineer`) |

`country` is kept out of the distance (over 90% of customers are in the UK and the goal is a behavioral segmentation) and used for profiling.

---

# Custom Transformers & Preprocessing

To ensure a reproducible and leakage-free workflow, preprocessing logic is encapsulated into custom classes inheriting from `BaseEstimator` and `TransformerMixin`.

- **FeatureEngineer:** derives `avg_order_value` and `orders_per_month` from the customer table.
- **IQRTransformer:** learns robust lower and upper bounds (IQR method) during `fit` and clips values during `transform`.
- **RareCategoryEncoder:** groups infrequent categories into `Other`; used to profile the market mix of each segment.

The numerical pipeline applies median imputation, `log1p` (distances should reflect relative, not absolute, differences in spending), IQR clipping and standard scaling.

---

# Modeling

## Optimal Number of Clusters

K-Means is fitted for `k = 2..8`. Each internal index has its own bias, so `k` is chosen as the lowest average rank across Silhouette, Calinski-Harabasz and Davies-Bouldin. The consensus is **k = 3**.

## Benchmark

| Model | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|
| **K-Means** | **0.321** | **1561.8** | **1.150** |
| Agglomerative | 0.273 | 1200.3 | 1.261 |
| Gaussian Mixture | 0.152 | 798.7 | 2.225 |

## Hyperparameter Optimization (Optuna)

Optuna (seeded TPE sampler) tunes the winner's hyperparameters with `k` fixed, maximizing a Silhouette Score penalized for degenerate or heavily imbalanced solutions. The tuned configuration is adopted **only if it beats the baseline by at least 0.005**. Here the best trial (`n_init=25`, `init='k-means++'`, 0.3212) did not meaningfully improve on the baseline (0.3211), so the simpler baseline configuration was kept.

---

# Results

| Check | Result |
|---|---|
| Silhouette / Calinski-Harabasz / Davies-Bouldin | 0.321 / 1561.8 / 1.150 |
| Stability (mean ARI over 10 subsamples of 80%) | 0.9824 (min 0.9375) |
| Overall repurchase rate in the outcome window | 58.8% |

| Cluster | Persona | Customers | Median profile | Bought again next quarter | Share of next-quarter net revenue |
|---|---|---|---|---|---|
| 0 | Loyal core | 1,117 (33.8%) | 26 days since last order, 5 orders, GBP 1,749 | 84.2% | 77.7% |
| 1 | New customers | 549 (16.6%) | Active for 48 days, 1 order, GBP 330 | 51.2% | 8.2% |
| 2 | Lapsed one-timers | 1,639 (49.6%) | 143 days since last order, 1 order, GBP 312 | 44.1% | 14.1% |

One third of the customers generates more than three quarters of the next-quarter revenue. The Silhouette Score is moderate (typical for continuous behavioral data), but the segments separate future behavior sharply and are stable across subsamples.

---

# Project Structure

```
├── data/
│   └── online_retail.csv.gz
├── notebooks/
│   └── Customer Sales - Advanced Clustering.ipynb
├── README.md
├── README_es.md
└── requirements.txt
```

# How to Run

```bash
git clone https://github.com/arguar13/Customer-Sales-Advanced-Clustering.git
cd Customer-Sales-Advanced-Clustering
pip install -r requirements.txt
jupyter notebook "notebooks/Customer Sales - Advanced Clustering.ipynb"
```

The data path is resolved relative to the repository, so the notebook runs from either the project root or the `notebooks/` folder.

---

# License

The code is provided for educational and portfolio purposes. The dataset belongs to its authors and is distributed by the UCI Machine Learning Repository under the CC BY 4.0 license.

---

# Author

**Armando Guarnera**  
Data Scientist — Argentina
