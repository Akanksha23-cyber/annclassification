import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


PROJECT_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    model = load_model(PROJECT_DIR / "model.h5")

    with open(PROJECT_DIR / "onehot_encoder_geo.pkl", "rb") as file:
        geography_encoder = pickle.load(file)

    with open(PROJECT_DIR / "label_encoder_gender.pkl", "rb") as file:
        gender_encoder = pickle.load(file)

    with open(PROJECT_DIR / "scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, geography_encoder, gender_encoder, scaler


st.set_page_config(page_title="Customer Churn Prediction")

model, geography_encoder, gender_encoder, scaler = load_artifacts()

st.title("Customer Churn Prediction")
st.write("Enter customer details to estimate whether they are likely to leave.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        geography = st.selectbox("Geography", list(geography_encoder.categories_[0]))
        gender = st.selectbox("Gender", list(gender_encoder.classes_))
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=50000.0, format="%.2f")
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
        has_credit_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda value: "Yes" if value == 1 else "No")
        is_active_member = st.selectbox(
            "Is Active Member",
            [1, 0],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=60000.0,
            format="%.2f",
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    geography_data = pd.DataFrame({"Geography": [geography]})
    geography_encoded = geography_encoder.transform(geography_data).toarray()
    geography_columns = geography_encoder.get_feature_names_out(["Geography"])

    input_data = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Gender": gender_encoder.transform([gender])[0],
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_of_products,
                "HasCrCard": has_credit_card,
                "IsActiveMember": is_active_member,
                "EstimatedSalary": estimated_salary,
            }
        ]
    )

    for column, value in zip(geography_columns, geography_encoded[0]):
        input_data[column] = value

    feature_order = [
        "CreditScore",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Geography_France",
        "Geography_Germany",
        "Geography_Spain",
    ]
    input_data = input_data[feature_order]
    input_scaled = scaler.transform(input_data)

    probability = float(model.predict(input_scaled, verbose=0)[0][0])
    prediction = "Exited" if probability >= 0.5 else "Stayed"

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{probability:.1%}")

    if prediction == "Exited":
        st.error("The customer is likely to exit.")
    else:
        st.success("The customer is likely to stay.")
