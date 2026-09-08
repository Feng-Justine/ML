from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and preprocessing pipeline
model = joblib.load("california_housing_model.pkl")
pipeline = joblib.load("california_housing_pipeline.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    lower_range = None
    upper_range = None

    if request.method == "POST":

        # Get values from the form
        longitude = float(request.form["longitude"])
        latitude = float(request.form["latitude"])
        housing_median_age = float(request.form["housing_median_age"])
        total_rooms = float(request.form["total_rooms"])
        total_bedrooms = float(request.form["total_bedrooms"])
        population = float(request.form["population"])
        households = float(request.form["households"])
        median_income = float(request.form["median_income"])
        ocean_proximity = request.form["ocean_proximity"]

        # Create input DataFrame
        input_data = pd.DataFrame([{
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity,

            # Engineered features used by the final model
            "rooms_per_household": total_rooms / households,
            "bedrooms_per_room": total_bedrooms / total_rooms,
            "population_per_household": population / households
        }])

        # Apply the same preprocessing used during training
        input_prepared = pipeline.transform(input_data)

        # Final prediction
        prediction_value = model.predict(input_prepared)[0]

        # Predictions from each Random Forest tree
        tree_predictions = np.array([
            tree.predict(input_prepared)[0]
            for tree in model.estimators_
        ])

        # Standard deviation between trees
        prediction_std = np.std(tree_predictions)

        # Estimated prediction range
        lower = prediction_value - 1.96 * prediction_std
        upper = prediction_value + 1.96 * prediction_std

        lower = max(0, lower)

        # Estimated model confidence
        if prediction_value != 0:
            uncertainty_ratio = prediction_std / abs(prediction_value)
            confidence_value = 100 * (1 - uncertainty_ratio)
        else:
            confidence_value = 0

        confidence_value = max(0, min(100, confidence_value))

        prediction = f"${prediction_value:,.2f}"
        confidence = f"{confidence_value:.1f}%"
        lower_range = f"${lower:,.2f}"
        upper_range = f"${upper:,.2f}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        lower_range=lower_range,
        upper_range=upper_range
    )


if __name__ == "__main__":
    app.run(debug=True)