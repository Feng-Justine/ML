from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained TRA tax revenue model
model = joblib.load("tra_tax_revenue_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        # Get the two values entered by the user
        lag1 = float(request.form["lag1"])
        lag2 = float(request.form["lag2"])

        # Prepare the input for the model
        data = pd.DataFrame([{
            "Tax Revenue Lag 1": lag1,
            "Tax Revenue Lag 2": lag2
        }])

        # Make prediction
        prediction = model.predict(data)[0]

        # Display the prediction
        result = round(prediction, 2)

        return render_template("index.html", result=result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)