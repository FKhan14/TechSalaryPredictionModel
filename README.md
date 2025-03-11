# Tech Salary Prediction Model

## Overview
This project implements a machine learning model that predicts salaries in the technology industry using Random Forest Regression. The model analyzes various job-related features such as job title, company size, location, and required skills to estimate salary ranges for tech positions.

## Features
The model takes into account several key factors including:
- Job title and seniority level
- Company characteristics (size, type of ownership, industry, sector)
- Location (state)
- Technical skills (Python, R, AWS, Excel, etc.)
- Company revenue
- Other job-related metrics

## Technical Details
- **Model**: Random Forest Regressor
- **Data Processing**:
  - Categorical encoding using OneHotEncoder
  - Missing value imputation
  - Feature preprocessing using scikit-learn Pipeline
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared (R²) score

## Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- Pipeline and ColumnTransformer for preprocessing
- SimpleImputer for handling missing values

## Project Structure
```
.
├── Data/
│   └── eda_data.csv    # Dataset containing job listings and salary information
├── src/
│   └── model.py        # Main model implementation
└── README.md
```

## How It Works
1. Data Loading and Cleaning:
   - Loads the dataset
   - Removes unnecessary columns
   - Converts categorical data to appropriate types

2. Feature Processing:
   - Handles missing values using SimpleImputer
   - Performs one-hot encoding on categorical variables
   - Preprocesses features using ColumnTransformer

3. Model Training:
   - Splits data into training (80%) and validation (20%) sets
   - Trains a Random Forest model with 100 estimators
   - Uses Pipeline to combine preprocessing and model training

4. Prediction and Evaluation:
   - Makes salary predictions on validation data
   - Evaluates model performance using multiple metrics

## Usage
To use the model:
1. Ensure all dependencies are installed
2. Place your dataset in the Data/ directory
3. Run the model:
```python
python src/model.py
```

The model will output performance metrics including MAE, MSE, and R-squared score.

## Future Improvements
- Feature importance analysis
- Hyperparameter tuning
- Cross-validation implementation
- API deployment for real-time predictions
- Web interface for easy interaction 