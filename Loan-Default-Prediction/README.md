# Loan Default Prediction

**MIT Applied AI & Data Science - Capstone Project**

---

## Overview

A retail bank seeks to automate home equity loan approval decisions while maintaining regulatory compliance under the Equal Credit Opportunity Act (ECOA). This project builds and evaluates interpretable binary classification models on the HMEQ dataset (5,960 records, 13 features) to predict loan default (`BAD = 1`) versus repayment (`BAD = 0`).

---

## Business Problem

Missing a defaulter (false negative) exposes the lender to loss of the outstanding loan balance plus collection costs, whereas flagging a repayer (false positive) incurs a manual underwriting review. Because these error costs are asymmetric and adverse credit decisions require clear, objective factor explanations, model selection prioritizes **Recall** alongside **model interpretability**.

---

## Approach

| Phase | Details |
|-------|---------|
| Data Understanding & Prep | HMEQ dataset (5,960 rows, 13 columns; ~80% repaid / ~20% default); 70/30 stratified split; missing-value imputation and outlier handling |
| Exploratory Data Analysis | Univariate and bivariate analysis of debt-to-income ratio, delinquency history, credit inquiries, and loan-to-value |
| Modeling & Tuning | Decision Tree, Random Forest, Gradient Boosting, and XGBoost with class balancing and recall-focused hyperparameter tuning |
| Explainability & Output | Feature importance analysis tied to ECOA-compliant financial factors and executive presentation deck |

---

## Tech Stack

`Python` `pandas` `numpy` `scikit-learn` `xgboost` `shap` `matplotlib` `seaborn`

---

## Files

| File | Description |
|------|-------------|
| `Loan_Default_Prediction_rshypitka.ipynb` | Full end-to-end Capstone analysis and modeling notebook |
| `Loan_Default_Presentation_rshypitka.pptx` | Executive presentation deck |
| `hmeq.csv` | Home Equity (HMEQ) source dataset |

---

*MIT Applied AI & Data Science Program - Rob Shypitka*
