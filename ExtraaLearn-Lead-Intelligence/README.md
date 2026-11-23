# ExtraaLearn: Lead Conversion Intelligence

**MIT Applied AI & Data Science - Elective Project**

---

## Overview

ExtraaLearn is an EdTech startup offering upskilling programs for working professionals. This project builds a supervised machine learning pipeline to predict which leads are most likely to convert into paying customers, enabling sales and marketing teams to prioritize high-intent prospects.

---

## Business Problem

With rapid lead growth, manual outreach across all prospects is inefficient. The goal is to identify the key behavioral and demographic drivers of lead conversion and deliver a recall-optimized classification model that flags high-potential leads while remaining interpretable for business stakeholders.

---

## Approach

| Phase | Details |
|-------|---------|
| Data Understanding & Prep | Lead-level behavioral, source, and profile dataset; preprocessing and feature engineering |
| Exploratory Data Analysis | Conversion rate analysis across interaction types, time spent on website, lead origin, and current occupation |
| Modeling & Tuning | Decision Tree, Random Forest, Gradient Boosting, and XGBoost tuned via recall-scored `GridSearchCV` |
| Business Insights | High-intent converter profile and actionable outreach recommendations |

---

## Key Results (Tuned Models)

- **Decision Tree:** Accuracy 0.79 | Precision 0.60 | Recall 0.90 | ROC-AUC 0.88
- **Random Forest:** Accuracy 0.84 | Precision 0.68 | Recall 0.87 | ROC-AUC 0.92
- **Gradient Boosting:** Accuracy 0.80 | Precision 0.61 | Recall 0.90 | ROC-AUC 0.91
- **XGBoost:** Accuracy 0.82 | Precision 0.64 | Recall 0.89 | ROC-AUC 0.93

---

## Tech Stack

`Python` `pandas` `numpy` `scikit-learn` `xgboost` `matplotlib` `seaborn`

---

## Files

| File | Description |
|------|-------------|
| `ExtraaLearn_Lead_Intelligence_rshypitka.ipynb` | Full portfolio analysis and modeling notebook |
| `ExtraaLearn_Companion_Guide.md` | Technical companion guide and methodology notes |
| `ExtraaLearn.csv` | Source dataset |

---

*MIT Applied AI & Data Science Program - Rob Shypitka*
