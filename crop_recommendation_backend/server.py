import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS 
from crop_recommendation import predict_probabilities

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

# Get the absolute directory where server.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model, scaler, and label encoder using relative paths
try:
    model = joblib.load(os.path.join(BASE_DIR, "crop_recommendation_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "crop_recommendation_scaler.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "crop_recommendation_label_encoder.pkl"))
    print("Model, scaler, and label encoder loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}")
    model = None
    scaler = None
    label_encoder = None

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or label_encoder is None:
        return jsonify({"error": "Model, scaler, or label encoder not loaded."}), 500

    try:
        data = request.get_json()

        # Validate input
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request."}), 400

        # Convert input to a pandas DataFrame with feature names
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_data = pd.DataFrame([data["features"]], columns=feature_names)

        # Scale the input data
        scaled_data = scaler.transform(input_data)

        # Predict probabilities and the most likely crop
        encoded_prediction = model.predict(scaled_data)
        decoded_prediction = label_encoder.inverse_transform(encoded_prediction)

        probabilities = model.predict_proba(scaled_data)[0]
        crop_probabilities = dict(zip(label_encoder.classes_, probabilities * 100))

        # Return the prediction and probabilities as a JSON response
        return jsonify({
            "prediction": decoded_prediction[0],
            "probabilities": crop_probabilities
        })

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return jsonify({"error": f"Value error: {str(ve)}"}), 400
    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    app.run(debug=True)