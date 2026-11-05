
from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the new simplified ML model
model_path = os.path.join(
    app.root_path,
    "student_performance_model_simplified.pkl"
)

with open(model_path, "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get student name
        student_name = request.form["student_name"]

        # Collect the five academic features
        student_data = {
            "previous_grade_average": float(
                request.form["previous_grade_average"]
            ),
            "G1": float(request.form["G1"]),
            "failures": int(request.form["failures"]),
            "studytime": int(request.form["studytime"]),
            "absences": int(request.form["absences"])
        }

        # Arrange features in training order
        feature_columns = [
            "previous_grade_average",
            "G1",
            "failures",
            "studytime",
            "absences"
        ]

        input_data = pd.DataFrame(
            [[student_data[col] for col in feature_columns]],
            columns=feature_columns
        )

        # Predict final grade
        prediction = model.predict(input_data)[0]

        return render_template(
            "result.html",
            student_name=student_name,
            prediction=round(float(prediction), 2)
        )

    except (ValueError, KeyError) as error:
        return f"Invalid or missing input: {error}", 400


if __name__ == "__main__":
    app.run(debug=True)
