import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import RobustScaler
from waitress import serve  # <-- Added waitress

# Disable GPU and limit memory growth
tf.config.set_visible_devices([], 'GPU')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)
CORS(app)

# Load model and scalers with error handling
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

@app.route('/health', methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": bool(model)}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or "sequences" not in data:
            return jsonify({"error": "Missing 'sequences' in request body"}), 400
        
        sequences = np.array(data["sequences"])

        if sequences.ndim != 3:
            return jsonify({"error": "Input must be a 3D array [batch_size, sequence_length, feature_dim]"}), 400
        
        if sequences.shape[1:] != (SEQUENCE_LENGTH, len(FEATURES)):
            return jsonify({"error": f"Expected input shape ({SEQUENCE_LENGTH}, {len(FEATURES)}), but got {sequences.shape[1:]}"}), 400
        
        # Scale each sequence individually
        scaled_sequences = np.array([feature_scaler.transform(seq) for seq in sequences])
        
        predictions = model.predict(scaled_sequences, verbose=0)
        predicted_emissions = target_scaler.inverse_transform(predictions)
        
        return jsonify({"predictions": predicted_emissions.flatten().tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9515))
    print(f"Starting server on port {port} using Waitress...")
    serve(app, host="0.0.0.0", port=port)  # <-- Serve with Waitress
