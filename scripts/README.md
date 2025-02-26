# Brent Oil Price Prediction Modules

This project aims to predict Brent oil prices using various time series models. It involves data preprocessing, exploratory data analysis, model training, and evaluation.
Here’s a concise and bulleted summary of your project:

### **Modules Overview**

- **train_models.py**  
  - **TimeSeriesModel** class for training time series models.
  - Methods: 
    - `fit_arima`: Trains ARIMA model.
    - `fit_garch`: Trains GARCH model.
    - `fit_random_forest`: Trains Random Forest model.
    - `evaluate_model`: Evaluates models using RMSE and MAE.

- **explore_data.py**  
  - **TimeSeriesEDA** class for exploratory data analysis.
  - Methods:
    - Data cleaning and summary.
    - Plots moving average, trends, monthly/yearly trends, distribution.
    - Detects outliers and decomposes time series.
    - ACF/PACF plots for autocorrelation analysis.

- **preprocess_data.py**  
  - Functions for data preprocessing:
    - `missing_values_proportions`: Calculates missing values proportions.
    - `handle_outliers`: Handles outliers by replacing or bounding.
    - `detect_outliers`: Detects outliers using boxplots and Z-scores.

### **Project Workflow**
1. **Data Preprocessing**: Clean data with `preprocess_data.py`.
2. **Exploratory Data Analysis**: Analyze data with `explore_data.py`.
3. **Model Training**: Train models with `train_models.py`.
4. **Model Evaluation**: Evaluate models with `train_models.py`.