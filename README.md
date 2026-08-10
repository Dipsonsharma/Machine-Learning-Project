# Machine Learning Prediction Models

A portfolio of ten end-to-end machine-learning notebooks focused on structured, tabular data. The projects cover classification and regression problems across finance, healthcare, human resources, education, telecommunications, property, and automotive pricing.

## Projects at a glance

| # | Project | Task | Selected / highlighted approach |
|---:|---|---|---|
| 1 | Loan Approval Prediction | Classification | XGBoost; reported test accuracy: 98.78% |
| 2 | Credit Card Fraud Detection | Classification | SVC selected using ROC-AUC / recall-focused comparison |
| 3 | Customer Churn Prediction | Classification | XGBoost; reported test accuracy: 84.06% |
| 4 | Diabetes Prediction | Classification | KNN selected using recall; tuned with randomized search |
| 5 | Employee Attrition Prediction | Classification | Random Forest; reported test accuracy: 90.28% |
| 6 | Heart Disease Prediction | Classification | Tuned Random Forest; reported test accuracy: 97.44% |
| 7 | Student Performance Prediction | Classification | Tuned Random Forest; approximately 99.2% test accuracy |
| 8 | Employee Salary Prediction | Regression | Random Forest; test R²: 0.357 on the notebook’s transformed scale |
| 9 | House Price Prediction | Regression | ElasticNet baseline; test R² near zero, indicating further work is needed |
| 10 | Used Car Price Prediction | Regression | Tuned XGBoost workflow; final metrics should be saved explicitly |

## Repository structure

Predection Models/
├── Loan_approval_predection.ipynb
├── credit_card_fraud_detection.ipynb
├── customer_churn_prediction.ipynb
├── diabetes_predection.ipynb
├── employee_attrition_prediction.ipynb
├── employee_salary_predection.ipynb
├── heart_disease_predection.ipynb
├── house_price_predection.ipynb
├── student_performance_predction.ipynb
└── use_car_price_predection.ipynb
```

## Common workflow


1. Load and inspect the dataset.
2. Check missing values, duplicates, class balance, distributions, and outliers.
3. Remove identifiers or irrelevant columns and engineer selected features.
4. Encode categorical variables and standardize numerical variables with a preprocessing pipeline.
5. Split data into training and test sets.
6. Compare multiple algorithms.
7. Evaluate the selected model and, in several projects, tune it with `GridSearchCV` or `RandomizedSearchCV`.
8. Generate predictions or probabilities for unseen data where applicable.

### Classification models

The classification notebooks compare combinations of:

- K-Nearest Neighbors
- Decision Tree
- Random Forest
- XGBoost
- Logistic Regression
- Support Vector Classifier

Primary metrics include accuracy, precision, recall, F1 score, ROC-AUC, confusion matrices, and classification reports. The metric used to rank models differs by notebook—for example, fraud and diabetes projects emphasize recall, while several others use F1 score.

### Regression models

The regression notebooks compare combinations of:

- Linear Regression
- Lasso
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

Primary metrics include mean absolute error (MAE), root mean squared error (RMSE), and R².

## Setup

Create and activate a virtual environment, then install the main dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter xgboost imbalanced-learn
```

Place the datasets in the relative `datasets/` location expected by the notebooks, then start Jupyter:

```bash
jupyter notebook
```

## Notes on results

- Metrics in this README are taken from saved notebook outputs and should be reproduced before being used in a final comparison.
- Very high classification scores should be checked for data leakage, duplicate records, class imbalance, and train/test contamination.
- Healthcare, credit, and HR use cases require fairness, privacy, calibration, and human-review safeguards before any real-world use.
- The house-price model is intentionally retained as a useful diagnostic case: its near-zero R² signals the need for stronger features, data checks, and validation before more tuning.
- For the used-car model, preserve a final test-metric table and residual plot so the tuned model can be evaluated clearly.