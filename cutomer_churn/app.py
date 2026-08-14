# ==========================================
# Customer Churn Prediction Streamlit App
# Step 2: Import Libraries & Load Model
# ==========================================
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# -----------------------------
# Load Saved Model
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    preprocessor = joblib.load(
        BASE_DIR / "churn_preprocessor.pkl"
    )

    model = joblib.load(
        BASE_DIR / "churn_model.pkl"
    )

    return preprocessor, model


preprocessor, model = load_model()

# App title
st.title("Customer Churn Prediction & Retention Analytics")

st.write(
    "Enter customer information to predict churn risk "
    "and generate retention insights."
)

st.success("Model loaded successfully!")


# -----------------------------
# App Header
# -----------------------------
st.title("📊 Customer Churn Prediction & Retention Analytics")

st.write(
    "Enter customer information below to predict churn risk "
    "and generate retention insights."
)

st.divider()


# -----------------------------
# Customer Information Section
# -----------------------------
st.subheader("👤 Customer Information")

st.info(
    "Fill in the customer details below. "
    "The trained machine learning model will estimate churn risk."
)


# Create two columns for future input fields
col1, col2 = st.columns(2)

with col1:
    st.write("### Customer & Service Details")

with col2:
    st.write("### Billing & Contract Details")


st.divider()


# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔍 Prediction Result")

st.write(
    "Prediction results will appear here after entering "
    "customer information and clicking the prediction button."
)


# -----------------------------
# Customer Information Section
# -----------------------------
st.subheader("👤 Customer Information")

st.info(
    "Fill in the customer details below. "
    "The trained machine learning model will estimate churn risk."
)

col1, col2 = st.columns(2)

with col1:
    st.write("### Customer & Service Details")

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        [
            "No",
            "Yes",
            "No phone service"
        ]
    )

    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    online_backup = st.selectbox(
        "Online Backup",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )


with col2:
    st.write("### Billing & Contract Details")

    device_protection = st.selectbox(
        "Device Protection",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )


    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0,
        step=0.1
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0,
        step=0.1
    )


    # -----------------------------
# Prepare Customer Input Data
# -----------------------------

senior_citizen_value = 1 if senior_citizen == "Yes" else 0

customer_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen_value],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})



# -----------------------------
# Prediction Section
# -----------------------------

st.divider()
st.subheader("🔍 Prediction Result")

if st.button("Predict Churn"):

    # Preprocess customer data
    customer_encoded = preprocessor.transform(customer_data)

    # Predict churn
    prediction = model.predict(customer_encoded)[0]

    # Predict churn probability
    churn_probability = model.predict_proba(
        customer_encoded
    )[0, 1]

    churn_probability_percent = round(
        churn_probability * 100,
        2
    )

    # -----------------------------
    # Display Prediction
    # -----------------------------

    if prediction == "Yes":
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    st.metric(
        label="Churn Probability",
        value=f"{churn_probability_percent}%"
    )

    # -----------------------------
    # Risk Level
    # -----------------------------

    if churn_probability >= 0.60:
        risk_level = "High Risk"

    elif churn_probability >= 0.30:
        risk_level = "Medium Risk"

    else:
        risk_level = "Low Risk"

    # Display Risk Level
    if risk_level == "High Risk":
        st.error("🔴 Risk Level: High Risk")

    elif risk_level == "Medium Risk":
        st.warning("🟠 Risk Level: Medium Risk")

    else:
        st.success("🟢 Risk Level: Low Risk")

    # -----------------------------
    # Retention Recommendation
    # -----------------------------

    st.subheader("💡 Retention Recommendation")

    if risk_level == "High Risk":

        st.write(
            "Priority retention action recommended: "
            "provide proactive support, personalized offers, "
            "and suitable contract incentives."
        )

    elif risk_level == "Medium Risk":

        st.write(
            "Monitor the customer and provide early engagement, "
            "service guidance, and targeted loyalty benefits."
        )

    else:

        st.write(
            "Customer currently has low churn risk. "
            "Maintain service quality and regular loyalty communication."
        )