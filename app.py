
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, confloat
from typing import Literal
import pandas as pd
import joblib
import os
import logging

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ML & preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ------------------------------
# Model Training (if needed)
# ------------------------------

MODEL_PATH = "churn_model.joblib"

def train_model():
    print("Training churn prediction model...")

    # Load data
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Features and labels
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    model_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train)

    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Train if model doesn't exist
if not os.path.exists(MODEL_PATH):
    train_model()

# Load model for serving
model = joblib.load(MODEL_PATH)

# ------------------------------
# FastAPI Setup
# ------------------------------

app = FastAPI(title="Churn Predictor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: confloat(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: confloat(ge=0)
    TotalCharges: confloat(ge=0)

    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 75.35,
                "TotalCharges": 850.5
            }
        }
        
@app.get("/")
def root():
    return {"message": "Churn Prediction API is live"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        input_dict = data.dict()
        logger.info(f"Received input: {input_dict}")

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        logger.info(f"Prediction: {prediction}, Probability: {prediction_proba:.4f}")

        return {
            "prediction": int(prediction),
            "churn_probability": round(float(prediction_proba), 4)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}


