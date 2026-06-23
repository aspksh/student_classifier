import streamlit as st
import torch
from student_classifier import StudentClassifier

# Model load
model = StudentClassifier()
model.load_state_dict(
    torch.load(
        "student_classifier_pass_fail.pth",
        map_location="cpu"
    )
)
model.eval()

st.title("Student Pass/Fail Predictor")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance %",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

if st.button("Predict"):

    sample = torch.tensor(
        [[study_hours, attendance]],
        dtype=torch.float32
    )

    with torch.no_grad():
        output = model(sample)
        prob = torch.sigmoid(output)

    probability = prob.item() * 100

    prediction = (
        "Pass"
        if probability > 50
        else "Fail"
    )

    st.write(f"Probability: {probability:.2f}%")
    st.success(f"Prediction: {prediction}")
