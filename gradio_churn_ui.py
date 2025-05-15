import gradio as gr
import requests

API_URL = "https://churn-api-390674139847.us-central1.run.app/predict"

def predict_churn(
    gender, senior, partner, dependents, tenure,
    phone, multiline, internet, online_sec, online_backup,
    device_protect, tech_support, tv, movies,
    contract, paperless, payment, monthly, total
):
    payload = {
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiline,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protect,
        "TechSupport": tech_support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        result = response.json()
        if "error" in result:
            return f"Error: {result['error']}"
        churn = "Yes" if result["prediction"] == 1 else "No"
        return f"Churn: {churn}\nProbability: {result['churn_probability']:.2%}"
    except Exception as e:
        return f"Request failed: {str(e)}"

# UI
iface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Checkbox(label="Senior Citizen"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),
        gr.Slider(0, 72, label="Tenure (months)"),
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract Type"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="Payment Method"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="Churn Predictor",
    description="Enter customer
