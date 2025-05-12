# Churn Prediction Machine Learning Project

## Project Description

The Churn Prediction Machine Learning project aims to predict customer churn for a business using historical customer data. The project utilizes machine learning techniques to train a model that can predict whether a customer will leave the company, allowing businesses to implement retention strategies to minimize churn.

The model is trained using customer data such as demographic information, account type, balance, usage patterns, and more. The trained model is then served as an API using FastAPI, enabling businesses to integrate the model into their existing systems for real-time churn predictions.

---

## How to Run the API

The project uses FastAPI to serve the churn prediction model via an API. Follow the instructions below to set up and run the API:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction

# Customer Churn Prediction (ML Pipeline Project)

This project is a machine learning pipeline for predicting customer churn using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## 🔍 Objective

To build a clean and modular ML pipeline that:
- Handles feature preprocessing and encoding
- Selects and trains a classification model
- Evaluates model performance using ROC-AUC and PR curves
- Is ready for deployment with saved model artifacts

## 📁 Files Included

- `Telco-Customer-Churn.csv` - Raw dataset used for training
- `churn_model.joblib` - Trained ML model (e.g., XGBoost or Logistic Regression)
- *(optional)* `pipeline.py` or `notebook.ipynb` - Code for training and evaluation (add this when ready)

## 🧪 Model Features

- Feature engineering (target encoding, interaction features)
- Scikit-learn `Pipeline` and `ColumnTransformer`
- Cross-validation with stratified splits
- Evaluation metrics: ROC-AUC, Precision-Recall, Confusion Matrix

## 🧰 Libraries Used

- `pandas`
- `scikit-learn`
- `xgboost` or `lightgbm` *(optional)*
- `joblib`
- `matplotlib` / `seaborn`


## 📊 Example Use Case

> Telecom companies can use this model to identify customers likely to churn and take proactive steps to retain them.

---

## 🧠 Dataset Source

Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 📜 License

This project is open-source and free to use for educational and non-commercial purposes.
