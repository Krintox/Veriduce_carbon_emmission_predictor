import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import RobustScaler

tf.config.set_visible_devices([], 'GPU')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://veri-duce.vercel.app"}})

# Load model and scalers
try:
    model = tf.keras.models.load_model("emissions_forecaster_model.keras")
    with open("emissions_forecaster_feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open("emissions_forecaster_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise

FEATURES = ['industrial_output', 'energy_consumption', 'transport_emissions',
            'population_density', 'weather_temp', 'weather_humidity',
            'renewable_energy_share', 'carbon_tax', 'energy_efficiency',
            'traffic_index', 'forest_cover', 'industrial_waste', 'urbanization_rate']
SEQUENCE_LENGTH = 10

@app.route('/')
def home():
    return "Emission Forecasting API is running!"

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "model_loaded": bool(model)})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "sequences" not in data:
            return jsonify({"error": "Missing 'sequences' in request body"}), 400
        
        sequences = np.array(data["sequences"])
        if sequences.shape[1:] != (SEQUENCE_LENGTH, len(FEATURES)):
            return jsonify({"error": f"Expected input shape ({SEQUENCE_LENGTH}, {len(FEATURES)})"}), 400
        
        scaled_sequences = np.array([feature_scaler.transform(seq) for seq in sequences])
        predictions = model.predict(scaled_sequences, verbose=0)
        predicted_emissions = target_scaler.inverse_transform(predictions)
        
        return jsonify({"predictions": predicted_emissions.flatten().tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# REMOVE app.run() here because Gunicorn will run it
