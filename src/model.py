import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("eda_data.csv")

# Drop unnecessary columns
columns_to_drop = [
    "Unnamed: 0",
    "Company Name",
    "Job Description",
    "Salary Estimate",
    "Location",
    "Headquarters",
    "Competitors"
]
df = df.drop(columns=columns_to_drop)

# Convert categorical columns to category type
categorical_columns = [
    "Size",
    "Type of ownership",
    "Industry",
    "Sector",
    "Revenue",
    "job_simp",
    "seniority",
    "job_state"
]

for col in categorical_columns:
    df[col] = df[col].astype('category')

# Drop company_txt column
df = df.drop('company_txt', axis=1)

# Separate features and target variable
X = df.drop('avg_salary', axis=1)
y = df['avg_salary']

# One-hot encode categorical features
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)