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

## 🚀 Next Steps

- [ ] Add full preprocessing pipeline script
- [ ] Create FastAPI/Flask app to serve model predictions
- [ ] Containerize with Docker for deployment
- [ ] Add test inputs and documentation

## 📊 Example Use Case

> Telecom companies can use this model to identify customers likely to churn and take proactive steps to retain them.

---

## 🧠 Dataset Source

Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 📜 License

This project is open-source and free to use for educational and non-commercial purposes.
