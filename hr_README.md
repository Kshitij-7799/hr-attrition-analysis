# HR Analytics — Employee Attrition Analysis

## Overview
Analyze IBM HR dataset of 1,400+ employees to identify the key drivers of employee attrition. Uses SQL for data extraction and Python for EDA and machine learning.

## Dataset
- Source: [Kaggle — IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Size: 1,470 employees, 35 features
- Target: Attrition (Yes/No)

## Analysis Performed
1. SQL-based data extraction (SQLite)
2. Attrition rate by department, overtime, job satisfaction
3. Monthly income and tenure comparison
4. Random Forest classification model
5. Feature importance analysis
6. Confusion matrix

## Key Findings
- Overtime employees have significantly higher attrition
- Low job satisfaction is a strong attrition predictor
- Employees with short tenure and low income churn more

## Tech Stack
- Python, Pandas, NumPy
- SQLite (SQL queries)
- Scikit-learn
- Matplotlib, Seaborn

## How to Run
```bash
pip install -r requirements.txt
python hr_attrition_analysis.py
```
