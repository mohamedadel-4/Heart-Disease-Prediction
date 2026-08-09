import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("final_model.pkl")
    ohe = joblib.load("encoder.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    imputer = joblib.load("imputer.pkl")
    with open("feature_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    return model, ohe, label_encoder, imputer, config

model, ohe, label_encoder, imputer, config = load_artifacts()

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's clinical information and click **Predict**.")

st.info(
    "This is a machine-learning demonstration project and is not a medical diagnosis. "
    "Do not use the prediction to make medical decisions."
)

st.subheader("Patient Information")

c1, c2 = st.columns(2)

with c1:
    age = st.number_input(
        "Age",
        min_value=float(config["numeric_ranges"]["Age"]["min"]),
        max_value=float(config["numeric_ranges"]["Age"]["max"]),
        value=float(round((config["numeric_ranges"]["Age"]["min"] +
                           config["numeric_ranges"]["Age"]["max"]) / 2)),
        step=1.0
    )
    gender = st.selectbox("Gender", config["categorical_features"]["Gender"])
    bp = st.number_input(
        "Blood Pressure (BP)",
        min_value=float(config["numeric_ranges"]["BP"]["min"]),
        max_value=float(config["numeric_ranges"]["BP"]["max"]),
        value=float(config["numeric_ranges"]["BP"]["min"]),
        step=1.0
    )
    cholesterol = st.number_input(
        "Cholesterol",
        min_value=float(config["numeric_ranges"]["Cholesterol"]["min"]),
        max_value=float(config["numeric_ranges"]["Cholesterol"]["max"]),
        value=float(config["numeric_ranges"]["Cholesterol"]["min"]),
        step=1.0
    )
    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=float(config["numeric_ranges"]["Max HR"]["min"]),
        max_value=float(config["numeric_ranges"]["Max HR"]["max"]),
        value=float(config["numeric_ranges"]["Max HR"]["min"]),
        step=1.0
    )
    st_depression = st.number_input(
        "ST Depression",
        min_value=float(config["numeric_ranges"]["ST depression"]["min"]),
        max_value=float(config["numeric_ranges"]["ST depression"]["max"]),
        value=float(config["numeric_ranges"]["ST depression"]["min"]),
        step=0.1
    )

with c2:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        sorted([int(v) for v in pd.read_csv("train_data.csv")["Chest pain type"].dropna().unique()])
    )
    fbs = st.selectbox("FBS over 120", [0, 1])
    ekg = st.selectbox(
        "EKG Results",
        sorted([int(v) for v in pd.read_csv("train_data.csv")["EKG results"].dropna().unique()])
    )
    exercise_angina = st.selectbox("Exercise Angina", [0, 1])
    slope = st.selectbox(
        "Slope of ST",
        sorted([int(v) for v in pd.read_csv("train_data.csv")["Slope of ST"].dropna().unique()])
    )
    vessels = st.selectbox(
        "Number of Vessels Fluro",
        sorted([int(v) for v in pd.read_csv("train_data.csv")["Number of vessels fluro"].dropna().unique()])
    )
    thallium = st.selectbox(
        "Thallium",
        sorted([int(v) for v in pd.read_csv("train_data.csv")["Thallium"].dropna().unique()])
    )
    work_type = st.selectbox("Work Type", config["categorical_features"]["work_type"])
    smoking_status = st.selectbox("Smoking Status", config["categorical_features"]["smoking_status"])

if st.button("🔍 Predict", use_container_width=True):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Chest pain type": chest_pain,
        "BP": bp,
        "Cholesterol": cholesterol,
        "FBS over 120": fbs,
        "EKG results": ekg,
        "Max HR": max_hr,
        "Exercise angina": exercise_angina,
        "ST depression": st_depression,
        "Slope of ST": slope,
        "Number of vessels fluro": vessels,
        "Thallium": thallium,
        "work_type": work_type,
        "smoking_status": smoking_status
    }])

    # Same preprocessing sequence as the notebook
    processed = imputer.transform(input_df)
    nominal_cols = ["cat__Gender", "cat__work_type", "cat__smoking_status"]

    encoded = ohe.transform(processed[nominal_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(nominal_cols),
        index=processed.index
    )
    processed.drop(columns=nominal_cols, inplace=True)
    processed = pd.concat([processed, encoded_df], axis=1)

    prediction = model.predict(processed)[0]
    probabilities = model.predict_proba(processed)[0]
    label = label_encoder.inverse_transform([prediction])[0]
    confidence = probabilities[prediction] * 100

    st.divider()
    st.subheader("Prediction Result")

    if label == "Yes":
        st.error(f"Prediction: **Heart Disease = Yes**")
    else:
        st.success(f"Prediction: **Heart Disease = No**")

    st.metric("Model confidence", f"{confidence:.1f}%")
