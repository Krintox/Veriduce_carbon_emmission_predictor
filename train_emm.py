import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from emm_for_main import EmissionsForecaster

# Load dataset
df = pd.read_csv("E:/Bunny/Capstone_2025/machine_learning/emm_for_main/emissions_dataset.csv")

# Define features and target
FEATURES = ['industrial_output', 'energy_consumption', 'transport_emissions',
            'population_density', 'weather_temp', 'weather_humidity',
            'renewable_energy_share', 'carbon_tax', 'energy_efficiency',
            'traffic_index', 'forest_cover', 'industrial_waste', 'urbanization_rate']
TARGET = 'co2_emissions'

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna()  # Alternatively, you can use df.fillna(df.median())

# Convert categorical data (if any) to numeric (not needed here if all are numerical)
# df = pd.get_dummies(df, columns=['categorical_col'])  # Example

# Remove outliers using Z-score
z_scores = np.abs(df[FEATURES].apply(zscore))
df = df[(z_scores < 3).all(axis=1)]  # Keep rows within 3 standard deviations

# Ensure numerical consistency
df = df.astype(float)

# Split into train and test
train_data, test_data = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

# Initialize model
forecaster = EmissionsForecaster(sequence_length=10, forecast_horizon=1)

# Fit scalers
train_features = forecaster.feature_scaler.fit_transform(train_data[FEATURES])
train_target = forecaster.target_scaler.fit_transform(train_data[[TARGET]])

test_features = forecaster.feature_scaler.transform(test_data[FEATURES])
test_target = forecaster.target_scaler.transform(test_data[[TARGET]])

# Prepare sequences
def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_features, train_target, forecaster.sequence_length)
X_test, y_test = create_sequences(test_features, test_target, forecaster.sequence_length)

# Train model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

forecaster.model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)

# Save model
forecaster.save_model('emissions_forecaster')
