
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Healthcare AI System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Healthcare AI Implementation")
st.subheader("AI-Assisted Patient Risk Analysis System")

st.write(
    "A prototype designed to demonstrate patient data analysis, "
    "risk indicators, and clinical decision support."
)

st.warning(
    "Educational demonstration only. This system uses synthetic data "
    "and must not be used for real medical diagnosis or treatment."
)

@st.cache_resource
def train_model():

    np.random.seed(42)

    n = 1000

    age = np.random.randint(18, 90, n)
    glucose = np.random.randint(70, 250, n)
    blood_pressure = np.random.randint(80, 180, n)
    bmi = np.random.uniform(16, 40, n)

    risk = (
        (age > 60).astype(int)
        + (glucose > 140).astype(int)
        + (blood_pressure > 140).astype(int)
        + (bmi > 30).astype(int)
    )

    risk_label = (risk >= 2).astype(int)

    data = pd.DataFrame({
        "Age": age,
        "Glucose": glucose,
        "Blood_Pressure": blood_pressure,
        "BMI": bmi,
        "Risk": risk_label
    })

    X = data.drop("Risk", axis=1)
    y = data["Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


model, accuracy = train_model()

st.sidebar.header("Patient Information")

patient_id = st.sidebar.text_input(
    "Patient ID",
    value="P001"
)

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

glucose = st.sidebar.slider(
    "Glucose Level (mg/dL)",
    min_value=50,
    max_value=300,
    value=100
)

blood_pressure = st.sidebar.slider(
    "Systolic Blood Pressure (mmHg)",
    min_value=70,
    max_value=200,
    value=120
)

bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=22.0,
    step=0.1
)

patient_data = pd.DataFrame({
    "Age": [age],
    "Glucose": [glucose],
    "Blood_Pressure": [blood_pressure],
    "BMI": [bmi]
})

st.header("Patient Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Patient ID", patient_id)
col2.metric("Age", age)
col3.metric("Glucose", glucose)
col4.metric("BMI", bmi)

st.subheader("Patient Data Summary")

st.dataframe(
    patient_data,
    use_container_width=True
)

st.header("AI Risk Analysis")

if st.button("Analyze Patient Data"):

    prediction = model.predict(patient_data)[0]

    probability = model.predict_proba(
        patient_data
    )[0][1]

    st.subheader("Analysis Result")

    if prediction == 1:

        st.error(
            "Illustrative Risk Flag: Elevated"
        )

    else:

        st.success(
            "Illustrative Risk Flag: Lower"
        )

    st.metric(
        "Model Risk Score",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.info(
        "This output is generated from synthetic training data. "
        "It is not a validated medical risk estimate."
    )

st.header("Model Evaluation")

st.write(
    "The model was evaluated using a held-out portion "
    "of the synthetic dataset."
)

st.metric(
    "Synthetic Test Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.caption(
    "This accuracy measures agreement with artificial labels, "
    "not real-world healthcare performance."
)

st.header("Feature Importance")

importance = pd.DataFrame({
    "Feature": [
        "Age",
        "Glucose",
        "Blood Pressure",
        "BMI"
    ],

    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance.set_index("Feature")
)

st.header("Clinical Review")

st.write(
    "A qualified healthcare professional should review "
    "patient information and independently determine "
    "appropriate clinical action."
)

st.text_area(
    "Clinician Notes",
    placeholder="Enter review notes here..."
)

st.selectbox(
    "Review Status",
    [
        "Pending Review",
        "Reviewed",
        "Follow-up Required"
    ]
)

st.button("Submit Review")

st.divider()

st.caption(
    "Healthcare AI Implementation | "
    "Presented by Aditya Kr. Shukla | "
    "Roll No: 24BCA012"
)
