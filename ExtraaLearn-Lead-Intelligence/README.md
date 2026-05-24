# ExtraaLearn: Lead Intelligence System

**MIT Applied Data Science — Elective Project**

---

## Overview

ExtraaLearn is an early-stage EdTech startup selling technology programs to
students and working professionals. As lead volume grows, the sales team needs
a reliable way to prioritize outreach. This project builds a Lead Intelligence
System that scores every incoming lead with a conversion probability and assigns
it to a priority tier.

---

## Business Problem

The sales team can't chase every lead with equal effort. Treating all leads the
same wastes time on low-probability contacts and lets high-probability customers
cool off. The question this project answers: which leads are most likely to
convert, and what do they look like?

---

## Approach

Full supervised binary classification workflow:

| Phase | Details |
|-------|---------|
| Data Understanding | 4,612 leads, 15 features, no missing values |
| EDA | Conversion rates by segment, engagement analysis, channel behavior |
| Preprocessing | IQR outlier clipping, OHE + StandardScaler pipeline, feature engineering |
| Modeling | Logistic Regression → Decision Tree → Random Forest → Gradient Boosting → XGBoost |
| Evaluation | ROC-AUC, PR-AUC, Precision, Recall, F1 |
| Explainability | SHAP beeswarm + bar plots |
| Lead Scoring | Probability-based tiers with sales action recommendations |

---

## Results

**Final model: XGBoost (tuned via RandomizedSearchCV)**

| Metric | Score |
|--------|-------|
| Recall | 0.88 |
| Precision | 0.69 |
| F1 | 0.77 |
| ROC-AUC | 0.927 |

**Lead tier performance on held-out test set:**

| Tier | Threshold | Conversion Rate | % of Pipeline |
|------|-----------|----------------|---------------|
| Hot | ≥ 0.75 | 79.9% | 27.3% |
| Warm | 0.50 – 0.74 | 40.7% | 10.8% |
| Nurture | 0.25 – 0.49 | 14.9% | 20.4% |
| Low | < 0.25 | 1.4% | 41.5% |

The Hot tier captures **73% of all actual converters** from just 27% of the pipeline.

---

## Key Findings

- **Time on site** is the strongest predictor of conversion
- **Website-first leads** convert at 4x the rate of mobile-first leads
- **High profile completion + above-median time on site** (`high_intent_flag`) converts at 54% vs 21% for the rest
- **Referral leads** convert at 67.7% — highest of any segment, only 2% of volume

---

## Tech Stack

`Python 3.14` `scikit-learn 1.8` `XGBoost 3.2` `SHAP 0.51` `pandas 3.0` `seaborn` `matplotlib` `PyTorch 2.12 (MPS)`

---

## Files

| File | Description |
|------|-------------|
| `ExtraaLearn_rshypitka.ipynb` | Full project notebook |
| `ExtraaLearn_rshypitka.html` | Rendered HTML export for review without Jupyter |

---

*MIT Applied Data Science Program — Rob Shypitka*
