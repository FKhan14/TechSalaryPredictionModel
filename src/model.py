import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("Data/eda_data.csv")

# Drop unnecessary columns
columns_to_drop = [
    "Unnamed: 0",
    "Company Name",
    "Job Description",
    "Salary Estimate",
    "Location",
    "Headquarters",
    "Competitors",
    "company_txt"
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


# Separate features and target variable
X = df.drop('avg_salary', axis=1)
y = df['avg_salary']


# Split the data into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)



# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[

        ('cat', categorical_transformer, categorical_columns)
    ])

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=1)
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', rf_model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

# Evaluate the model
mse = mean_squared_error(y_valid, preds)
r2 = r2_score(y_valid, preds)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")