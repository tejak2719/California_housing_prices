# California Housing Price Prediction – Regression Project

## Objective  
To predict median house prices in California using multiple regression models.  
This project focuses on addressing target variable skewness and comparing algorithmic performance using proper evaluation metrics.

---

## Dataset Overview

- **Source**: `sklearn.datasets.fetch_california_housing`
- **Rows**: 20,640  
- **Features**:
  - `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`  
- **Target**: `MedHouseVal` – Median house value in $100,000s  

---

## Exploratory Data Analysis (EDA)

- Checked target skewness and applied **log transformation** using `np.log1p()`
- Generated correlation heatmap to identify strong predictors
- Analyzed feature importances using Random Forest and XGBoost

### EDA Visuals
- `skewed_target_distribution.png`  
- `normalized_target_distribution.png`  
- `correlation_heatmap.png`  
- `random_forest_feature_importance.png`  
- `xgboost_feature_importance.png`

---

## Model Building & Evaluation

### Preprocessing
- Log-transformed `MedHouseVal` to address right skew
- Split data into 80/20 for training/testing

### Models Applied

| Model              | Description               |
|-------------------|---------------------------|
| Linear Regression | Baseline model            |
| Random Forest     | Ensemble with bagging     |
| XGBoost Regressor | Boosted tree-based model  |

### Evaluation Metrics  
All metrics calculated on **log-transformed target** (`Log_MedHouseVal`):

| Model              | R² Score | MAE   | RMSE  |
|-------------------|----------|-------|-------|
| Linear Regression | 0.60     | 0.34  | 0.48  |
| Random Forest     | 0.81     | 0.21  | 0.29  |
| XGBoost Regressor | 0.83     | 0.20  | 0.27  |

**Best Model**: `XGBoost Regressor`

---

## Metric Improvement Summary  

![California Housing Metric Summary](Screenshot 2025-07-05 at 1.07.42 PM.png)

- Skew handled with log transformation
- Metrics improved progressively from Linear ➝ RF ➝ XGB
- Feature importance extracted
- Plots saved for report/demo

---

## Output Files

| File Name                                | Description                                  |
|------------------------------------------|----------------------------------------------|
| `CaliforniaHousing_Regression.ipynb`     | Complete code and workflow                   |
| `best_regression_model.pkl`              | Serialized model using `joblib`              |
| `README.md`                              | This documentation                           |
| `.png` files                             | All EDA and evaluation charts                |


---