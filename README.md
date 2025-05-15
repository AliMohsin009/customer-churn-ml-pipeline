
# 📉 Customer Churn Prediction ML Pipeline

Predict whether a telecom customer is likely to churn using machine learning. This project includes a complete ML pipeline — from data preprocessing to model training, evaluation, and deployment via an interactive Gradio app.

[![Hugging Face Space](https://img.shields.io/badge/Live%20Demo-HuggingFace-%23ff6720?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/alim7897/churnpredictator)

---

## 🚀 Live Demo

🎯 Try the model right now in your browser:  
👉 **[Customer Churn Predictor App](https://huggingface.co/spaces/alim7897/churnpredictator)** (Hosted on Hugging Face Spaces)

---

## 🧠 Project Summary

This ML pipeline helps telecom companies identify customers who are likely to churn. It leverages structured customer data to make binary classification predictions and includes:

- **Data cleaning & preprocessing**
- **Feature engineering**
- **Model training (XGBoost)**
- **Evaluation (ROC AUC)**
- **Interactive deployment using Gradio**

---

## 📂 Dataset

- **Source**: IBM Telco Customer Churn Dataset
- **Columns**: Customer demographics, service usage patterns, billing info, and churn status
- **Target Variable**: `Churn` — Yes or No

---

## 🛠️ How It Works

The model is trained using XGBoost and includes preprocessing pipelines with:

- Standardization for numerical features
- One-hot encoding for categorical variables
- Imputation for missing values

The deployed Gradio interface allows users to:
- Select a few key features (e.g., contract type, monthly charges)
- Get a real-time churn prediction and probability

---

## 💻 Run Locally

Clone the repo and install requirements:

```bash
git clone https://github.com/AliMohsin009/customer-churn-ml-pipeline.git
cd customer-churn-ml-pipeline
pip install -r requirements.txt
```

Launch the Gradio app locally:

```bash
python gradio_app.py
```

---

## 🧰 Tech Stack

- **Python**
- **Pandas**, **Scikit-learn**, **XGBoost**
- **Gradio** (UI)
- **Joblib** (model serialization)
- **FastAPI** (REST API version available)
- **Hugging Face Spaces** (live hosting)

---

## 📊 Model Performance

- **ROC AUC Score**: ~0.84
- Evaluated using a stratified train/test split (80/20)

---

## 📎 Future Improvements

- Add SHAP explanations for model interpretability
- Integrate database or cloud storage for real-time inference
- Extend to other customer behavior prediction problems

---

## 👨‍💻 Author

**Ali Mohsin**  
📫 [alim7897 on Hugging Face](https://huggingface.co/alim7897)  
🔗 [LinkedIn](https://www.linkedin.com/in/mohsinali123)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
