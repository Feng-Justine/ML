from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load("tra_tax_revenue_model.pkl")


# Historical statistics from the training data
LAG1_MEAN = 11.085143
LAG1_STD = 0.642025

LAG2_MEAN = 10.928441
LAG2_STD = 0.681878


def calculate_confidence(lag1, lag2):

    # Calculate how far the inputs are from their historical means
    z1 = abs(lag1 - LAG1_MEAN) / LAG1_STD
    z2 = abs(lag2 - LAG2_MEAN) / LAG2_STD

    average_distance = (z1 + z2) / 2

    # Start from the model's historical reliability
    confidence = 96.2 - (15 * average_distance)

    # Keep the score within a reasonable range
    confidence = max(40, min(98, confidence))

    return round(confidence, 1)


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        lag1 = float(request.form["lag1"])
        lag2 = float(request.form["lag2"])

        data = pd.DataFrame([{
            "Tax Revenue Lag 1": lag1,
            "Tax Revenue Lag 2": lag2
        }])

        prediction = model.predict(data)[0]
        result = round(prediction, 2)

        confidence = calculate_confidence(lag1, lag2)

        return render_template(
            "index.html",
            result=result,
            confidence=confidence
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)