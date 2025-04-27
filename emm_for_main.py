import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Bidirectional, Conv1D, MaxPooling1D,
    MultiHeadAttention, LayerNormalization, Flatten
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pickle

class EmissionsForecaster:
    def __init__(
        self,
        sequence_length: int = 10,
        forecast_horizon: int = 1,
        feature_dim: int = 13,
        lstm_units: list = [256, 128, 64],
        dense_units: list = [128, 64, 32],
        attention_heads: int = 4,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        conv_filters: list = [64, 32],
        kernel_size: int = 3
    ):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        
        self.feature_scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        
        self.model = self._build_model()
    
    def _build_model(self) -> Model:
        inputs = Input(shape=(self.sequence_length, self.feature_dim))
        x = self._build_conv_layers(inputs)
        x = self._add_attention_layer(x)
        x = self._build_lstm_stack(x)
        outputs = self._build_dense_layers(x)
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.learning_rate)        
        model.compile(
            optimizer=optimizer,
            loss=Huber(),
            metrics=['mae', 'mse']
        )
        return model

    def _build_conv_layers(self, x):
        for filters in self.conv_filters:
            x = Conv1D(filters=filters, kernel_size=self.kernel_size, activation='relu', padding='same')(x)
            x = MaxPooling1D(pool_size=2)(x)
        return x

    def _add_attention_layer(self, x):
        x = MultiHeadAttention(num_heads=self.attention_heads, key_dim=x.shape[-1] // self.attention_heads)(x, x, x)
        x = LayerNormalization()(x)
        return x

    def _build_lstm_stack(self, x):
        for units in self.lstm_units:
            lstm_out = Bidirectional(LSTM(units, return_sequences=True, dropout=self.dropout_rate))(x)
            if x.shape[-1] != lstm_out.shape[-1]:
                x = Dense(lstm_out.shape[-1])(x)
            x = x + lstm_out
        return x

    def _build_dense_layers(self, x):
        x = Flatten()(x)
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        x = Dense(1)(x)
        return x
    
    def save_model(self, path: str):
        self.model.save(path + '_model.keras')  # Updated to .keras format
        with open(path + '_feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(path + '_target_scaler.pkl', 'wb') as f:
            pickle.dump(self.target_scaler, f)

    def load_model(self, path: str):
        self.model = tf.keras.models.load_model(path + '_model.keras')  # Updated to .keras format
        with open(path + '_feature_scaler.pkl', 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(path + '_target_scaler.pkl', 'rb') as f:
            self.target_scaler = pickle.load(f)


# Load dataset
df = pd.read_csv("E:/Bunny/Capstone_2025/machine_learning/emm_for_main/emissions_dataset.csv")

# Define features and target
FEATURES = ['industrial_output', 'energy_consumption', 'transport_emissions',
            'population_density', 'weather_temp', 'weather_humidity',
            'renewable_energy_share', 'carbon_tax', 'energy_efficiency',
            'traffic_index', 'forest_cover', 'industrial_waste', 'urbanization_rate']
TARGET = 'co2_emissions'

# Data preprocessing
df = df.drop_duplicates()
df = df.dropna()
df = df.astype(float)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Initialize model
forecaster = EmissionsForecaster(sequence_length=10, forecast_horizon=1)

# Fit scalers
train_features = forecaster.feature_scaler.fit_transform(train_data[FEATURES])
train_target = forecaster.target_scaler.fit_transform(train_data[[TARGET]])

test_features = forecaster.feature_scaler.transform(test_data[FEATURES])
test_target = forecaster.target_scaler.transform(test_data[[TARGET]])

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
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)
]

forecaster.model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    callbacks=callbacks
)

# Save model
forecaster.save_model('emissions_forecaster')
