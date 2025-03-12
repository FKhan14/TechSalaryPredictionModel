import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
    "Headquarters",
    "Competitors",
    "company_txt",
    "min_salary",
    "max_salary"
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
    "job_state",
    "Location"
]

for col in categorical_columns:
    df[col] = df[col].astype('category')

# Separate features and target variable
X = df.drop('avg_salary', axis=1)
y = df['avg_salary']

# Split the data into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)

# Preprocessing for numerical data
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Define the model with improved default parameters
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=1
)

# Create the pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])

# Define hyperparameter grid for tuning
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    my_pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

# Fit the grid search
print("Training model with grid search...")
grid_search.fit(X_train, y_train)

# Print best parameters
print("\nBest parameters:", grid_search.best_params_)

# Make predictions with best model
best_model = grid_search.best_estimator_
preds = best_model.predict(X_valid)

# Calculate and print metrics
mae = mean_absolute_error(y_valid, preds)
mse = mean_squared_error(y_valid, preds)
r2 = r2_score(y_valid, preds)

print(f"\nModel Performance Metrics:")
print(f"Mean Absolute Error: {mae:,.2f}")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# Get feature names after preprocessing
feature_names = (
    numeric_features.tolist() +
    grid_search.best_estimator_.named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(categorical_columns).tolist()
)

# Calculate and print feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': grid_search.best_estimator_.named_steps['model'].feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))