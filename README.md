# Tech Salary Prediction Model

## Overview
This project implements a machine learning model that predicts salaries in the technology industry using Random Forest Regression. The model analyzes various job-related features such as job title, company size, location, and required skills to estimate salary ranges for tech positions.

## Dataset
The dataset is sourced from [Glassdoor Job Postings](https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor) on Kaggle. It contains detailed information about tech job listings including:
- Company information (size, revenue, industry)
- Job details (title, seniority, location)
- Required skills (Python, R, AWS, etc.)
- Company ratings and reviews
- Salary information

Credit: Dataset by "thedevastator" on Kaggle

## Features
The model considers several key factors including:
- Job title and seniority level
- Company characteristics (size, type of ownership, industry, sector)
- Location (state)
- Technical skills (Python, R, AWS, Excel, etc.)
- Company revenue
- Engineered features:
  - Tech skill level (combination of multiple tech skills)
  - Tech expertise (skill level × company rating)
  - Location-based features
  - Company size and sector interactions

## Model Details
- **Algorithm**: Random Forest Regressor
- **Optimized Hyperparameters** (found through extensive grid search):
  - bootstrap: False
  - criterion: 'poisson'
  - max_depth: None
  - max_features: 'sqrt'
  - min_samples_leaf: 1
  - min_samples_split: 2
  - n_estimators: 300

## Technical Implementation
- **Data Processing**:
  - Categorical encoding using OneHotEncoder
  - Missing value imputation
  - Feature scaling using StandardScaler
  - Custom feature engineering
- **Model Pipeline**:
  - Automated preprocessing
  - Feature transformation
  - Model training and prediction
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared (R²) score

## Project Structure
```
.
├── Data/
│   └── eda_data.csv    # Dataset containing job listings and salary information
├── src/
│   └── model.py        # Main model implementation with optimized parameters
└── README.md
```

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- Pipeline and ColumnTransformer for preprocessing
- SimpleImputer for handling missing values

## Usage
To use the model:
1. Ensure all dependencies are installed:
```bash
pip install pandas numpy scikit-learn
```
2. Place the dataset in the Data/ directory
3. Run the model:
```bash
python src/model.py
```

The model will output:
- Performance metrics (MAE, MSE, R²)
- Feature importance rankings
- Predictions on the test set

## Model Performance
The model achieves strong predictive performance through:
- Optimized hyperparameters via extensive grid search
- Effective feature engineering
- Robust preprocessing pipeline
- Handling of categorical variables

## Future Improvements
- Feature importance analysis visualization
- API deployment for real-time predictions
- Web interface for easy interaction
- Regular model retraining with new data
- Cross-validation with different random seeds 