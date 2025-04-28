import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import RobustScaler

# ========== MEMORY OPTIMIZATION ==========
tf.config.set_visible_devices([], 'GPU')  # Disable GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ========== APP SETUP ==========
app = Flask(__name__)
CORS(app)

# ========== MODEL LOADING ==========
try:
    # Load with custom objects if needed
    model = tf.keras.models.load_model(
        "emissions_forecaster_model.keras",
        compile=False
    )
    # Set model to inference mode
    model.predict(np.zeros((1, 10, 13)))  # Warmup
    
    # Load scalers
    with open("emissions_forecaster_feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open("emissions_forecaster_target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    raise

# ========== CONSTANTS ==========
FEATURES = [
    'industrial_output', 'energy_consumption', 'transport_emissions',
    'population_density', 'weather_temp', 'weather_humidity',
    'renewable_energy_share', 'carbon_tax', 'energy_efficiency',
    'traffic_index', 'forest_cover', 'industrial_waste', 'urbanization_rate'
]
SEQUENCE_LENGTH = 10

# ========== ROUTES ==========
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": bool(model),
        "memory_usage": f"{os.sys.getsizeof(model)/1024/1024:.2f}MB"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "sequences" not in data:
            return jsonify({"error": "Missing 'sequences' in request body"}), 400
        
        sequences = np.array(data["sequences"])
        if sequences.shape[1:] != (SEQUENCE_LENGTH, len(FEATURES)):
            return jsonify({
                "error": f"Expected shape (n, {SEQUENCE_LENGTH}, {len(FEATURES)})",
                "received": sequences.shape
            }), 400
        
        batch_size = sequences.shape[0]
        sequences_reshaped = sequences.reshape(-1, len(FEATURES))  # (batch_size*10, 13)
        scaled = feature_scaler.transform(sequences_reshaped)
        scaled_sequences = scaled.reshape(batch_size, SEQUENCE_LENGTH, len(FEATURES))
        
        predictions = model.predict(scaled_sequences, verbose=0)
        predicted_emissions = target_scaler.inverse_transform(predictions)
        
        return jsonify({
            "predictions": predicted_emissions.flatten().tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========== SERVER CONFIG ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9515))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True  # Better for I/O bound apps
    )