from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained KNN model
model = joblib.load("mnist_knn_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the image sent from the browser
        image_file = request.files["image"]

        # Open the image
        image = Image.open(image_file).convert("L")

        # Resize to MNIST size
        image = image.resize((28, 28))

        # Convert image to NumPy array
        image_array = np.array(image)

        # MNIST uses white digits on a black background.
        # The canvas will also use this format.
        image_array = 255 - image_array

        # Flatten 28x28 image into 784 pixels
        image_array = image_array.reshape(1, 784)

        # Make prediction
        prediction = model.predict(image_array)[0]

        # Get class probabilities
        probabilities = model.predict_proba(image_array)[0]

        # Get confidence of predicted digit
        confidence = probabilities.max() * 100

        return jsonify({
            "prediction": str(prediction),
            "confidence": round(float(confidence), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
