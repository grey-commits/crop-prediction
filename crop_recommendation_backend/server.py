import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from crop_recommendation import predict_probabilities  # Make sure this exists

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the absolute directory where server.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, scaler, and label encoder
try:
    model = joblib.load(os.path.join(BASE_DIR, "crop_recommendation_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "crop_recommendation_scaler.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "crop_recommendation_label_encoder.pkl"))
    print("✅ Model, scaler, and label encoder loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error loading model files: {e}")
    model = None
    scaler = None
    label_encoder = None

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None or label_encoder is None:
        return jsonify({"error": "Model, scaler, or label encoder not loaded."}), 500

    try:
        data = request.get_json()

        # Input validation
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request."}), 400

        # Expected input feature order
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_data = pd.DataFrame([data["features"]], columns=feature_names)

        # Preprocess
        scaled_data = scaler.transform(input_data)

        # Predict class and probabilities
        encoded_prediction = model.predict(scaled_data)
        decoded_prediction = label_encoder.inverse_transform(encoded_prediction)

        probabilities = model.predict_proba(scaled_data)[0]
        crop_probabilities = dict(zip(label_encoder.classes_, probabilities * 100))

        # Return response
        return jsonify({
            "prediction": decoded_prediction[0],
            "probabilities": crop_probabilities
        })

    except ValueError as ve:
        print(f"❌ ValueError: {ve}")
        return jsonify({"error": f"Value error: {str(ve)}"}), 400
    except Exception as e:
        print(f"❌ Error in /predict endpoint: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    # Use environment variable PORT for deployment, fallback to 5000 locally
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)