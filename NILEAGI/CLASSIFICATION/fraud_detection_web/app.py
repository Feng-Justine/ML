from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained fraud detection model
model = joblib.load("fraud_detection_random_forest.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Get values from the form
        step = float(request.form["step"])
        transaction_type = request.form["type"]
        amount = float(request.form["amount"])
        oldbalanceOrig = float(request.form["oldbalanceOrig"])
        newbalanceOrig = float(request.form["newbalanceOrig"])
        oldbalanceDest = float(request.form["oldbalanceDest"])
        newbalanceDest = float(request.form["newbalanceDest"])

        # Prepare transaction as a DataFrame
        transaction = pd.DataFrame([{
            "step": step,
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrig": oldbalanceOrig,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }])

        # Make prediction
        prediction = model.predict(transaction)[0]

        # Display result
        if prediction == 1:
            result = "FRAUDULENT TRANSACTION"
        else:
            result = "LEGITIMATE TRANSACTION"

        return render_template("index.html", result=result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)