# ============================================================
# HR Analytics — Employee Attrition Analysis
# Author: Kshitij Kumar
# Description: Identify key drivers of employee attrition
# Tools: Python, Pandas, SQL (SQLite), Seaborn, Matplotlib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: Load Dataset
# ============================================================
# Download from: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

print("Shape:", df.shape)
print("\nAttrition Distribution:")
print(df['Attrition'].value_counts())
print("\nSample Data:")
print(df.head())

# ============================================================
# STEP 2: SQL-based Data Extraction (SQLite simulation)
# ============================================================

conn = sqlite3.connect(':memory:')
df.to_sql('employees', conn, index=False, if_exists='replace')

# SQL Query: Attrition by Department
query1 = """
SELECT Department,
       COUNT(*) AS total_employees,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attrited,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate
FROM employees
GROUP BY Department
ORDER BY attrition_rate DESC
"""
dept_attrition = pd.read_sql(query1, conn)
print("\nAttrition by Department:")
print(dept_attrition)

# SQL Query: Avg salary of attrited vs retained
query2 = """
SELECT Attrition,
       ROUND(AVG(MonthlyIncome), 2) AS avg_monthly_income,
       ROUND(AVG(YearsAtCompany), 2) AS avg_tenure,
       ROUND(AVG(Age), 2) AS avg_age
FROM employees
GROUP BY Attrition
"""
income_stats = pd.read_sql(query2, conn)
print("\nIncome/Tenure Stats by Attrition:")
print(income_stats)

conn.close()

# ============================================================
# STEP 3: EDA
# ============================================================

df['Attrition_bin'] = (df['Attrition'] == 'Yes').astype(int)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('HR Attrition — Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Attrition distribution
axes[0, 0].pie(df['Attrition'].value_counts(), labels=['No', 'Yes'],
               autopct='%1.1f%%', colors=['#2196F3', '#F44336'], startangle=90)
axes[0, 0].set_title('Overall Attrition Rate')

# Attrition by Department
dept = df.groupby('Department')['Attrition_bin'].mean() * 100
axes[0, 1].bar(dept.index, dept.values, color=['steelblue', 'salmon', 'seagreen'])
axes[0, 1].set_title('Attrition Rate by Department (%)')
axes[0, 1].set_ylabel('%')

# Attrition by OverTime
ot = df.groupby('OverTime')['Attrition_bin'].mean() * 100
axes[0, 2].bar(ot.index, ot.values, color=['#4CAF50', '#F44336'])
axes[0, 2].set_title('Attrition Rate by OverTime (%)')
axes[0, 2].set_ylabel('%')

# Monthly Income distribution
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df, ax=axes[1, 0], palette='Set1')
axes[1, 0].set_title('Monthly Income vs Attrition')

# Years at Company
sns.histplot(data=df, x='YearsAtCompany', hue='Attrition', bins=20,
             ax=axes[1, 1], palette='Set1', kde=True)
axes[1, 1].set_title('Years at Company vs Attrition')

# Job Satisfaction
js = df.groupby('JobSatisfaction')['Attrition_bin'].mean() * 100
axes[1, 2].bar(js.index, js.values, color=sns.color_palette('RdYlGn', 4))
axes[1, 2].set_title('Attrition Rate by Job Satisfaction')
axes[1, 2].set_xlabel('Job Satisfaction (1=Low, 4=High)')
axes[1, 2].set_ylabel('%')

plt.tight_layout()
plt.savefig('hr_attrition_eda.png', dpi=120)
plt.show()

# ============================================================
# STEP 4: Feature Engineering
# ============================================================

# Drop non-numeric/useless columns
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'Attrition']
df_model = df.drop(drop_cols, axis=1)

# Encode categoricals
le = LabelEncoder()
cat_cols = df_model.select_dtypes(include='object').columns
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop('Attrition_bin', axis=1)
y = df_model['Attrition_bin']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=42, stratify=y)

# ============================================================
# STEP 5: Model Building
# ============================================================

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# ============================================================
# STEP 6: Feature Importance
# ============================================================

feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 7))
sns.barplot(x=feat_imp.values[:12], y=feat_imp.index[:12], palette='Reds_r')
plt.title('Top 12 Factors Driving Employee Attrition', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('attrition_feature_importance.png')
plt.show()

print("\nTop 3 Attrition Drivers:")
for i, (feat, score) in enumerate(feat_imp.head(3).items(), 1):
    print(f"  {i}. {feat} ({score:.4f})")

# ============================================================
# STEP 7: Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.title('Confusion Matrix — Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('hr_confusion_matrix.png')
plt.show()

print("\nAnalysis complete!")
