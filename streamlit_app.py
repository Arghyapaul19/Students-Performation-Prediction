
import streamlit as st
import pandas as pd
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load trained ML model
@st.cache_resource
def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "student_performance_model_simplified.pkl"
    )

    with open(model_path, "rb") as file:
        return pickle.load(file)


model = load_model()

# Application title
st.title("🎓 Student Grade Prediction")
st.write(
    "Predict a student's final academic grade "
    "using previous academic performance."
)

st.divider()

# Student information form
st.subheader("Enter Student Details")

with st.form("prediction_form"):

    student_name = st.text_input(
        "Student Name",
        placeholder="Enter student name"
    )

    previous_grade_average = st.number_input(
    "Previous Grade Average (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=0.5
)

    G1 = st.number_input(
    "Previous Grade",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=0.5
)

    failures = st.number_input(
        "Number of Previous Failures",
        min_value=0,
        step=1,
        value=0
    )

    studytime = st.selectbox(
        "Weekly Study Time",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 - Less than 2 hours",
            2: "2 - 2 to 5 hours",
            3: "3 - 5 to 10 hours",
            4: "4 - More than 10 hours"
        }[x]
    )

    absences = st.number_input(
        "Number of Absences",
        min_value=0,
        step=1,
        value=0
    )

    submitted = st.form_submit_button(
        "Predict Final Grade",
        use_container_width=True
    )



# Prediction
if submitted:

    if not student_name.strip():
        st.error("Please enter the student's name.")

    else:
        try:
            feature_columns = [
                "previous_grade_average",
                "G1",
                "failures",
                "studytime",
                "absences"
            ]

            # Convert percentage inputs to 0-20 scale
            input_data = pd.DataFrame(
                [[
                    previous_grade_average / 5,
                    G1 / 5,
                    failures,
                    studytime,
                    absences
                ]],
                columns=feature_columns
            )

            # Get prediction from trained model
            prediction = float(model.predict(input_data)[0])

            # Convert predicted grade from 0-20 to 0-100
            prediction_percentage = prediction * 5

            # Display prediction result
            st.divider()

            st.subheader("Prediction Result")

            st.success(
                f"Prediction completed for {student_name}!"
            )

            st.write("Predicted Final Grade")

            st.title(f"{prediction_percentage:.2f} / 100")

        except Exception as error:
            st.error(f"Prediction error: {error}")

st.divider()

st.caption(
    "Student Grade Prediction | Machine Learning Project"
)
