import paho.mqtt.client as mqtt
import json
import numpy as np
from collections import deque
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/sunibian/data"

WINDOW_SIZE = 60
FEATURES = ['timestamp', 'distance', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'intensity']
MODEL_PATH = "tsunami_detection_model.keras"
SCALER_PATH = "scaler.npy"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_buffer = deque(maxlen=WINDOW_SIZE)
scaler = StandardScaler()

plt.style.use('dark_background')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20))
plt.subplots_adjust(hspace=0.4)

class AdaptiveThresholdDetector:
    def __init__(self, window_size=60, z_threshold=3):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.water_levels = deque(maxlen=window_size)
        self.seismic_intensities = deque(maxlen=window_size)

    def update(self, water_level, seismic_intensity):
        self.water_levels.append(water_level)
        self.seismic_intensities.append(seismic_intensity)

    def detect_anomaly(self):
        if len(self.water_levels) < self.window_size:
            return False, 0, 0

        water_mean = np.mean(self.water_levels)
        water_std = np.std(self.water_levels)
        seismic_mean = np.mean(self.seismic_intensities)
        seismic_std = np.std(self.seismic_intensities)

        current_water = self.water_levels[-1]
        current_seismic = self.seismic_intensities[-1]

        water_z_score = (current_water - water_mean) / water_std if water_std > 0 else 0
        seismic_z_score = (current_seismic - seismic_mean) / seismic_std if seismic_std > 0 else 0

        is_anomaly = abs(water_z_score) > self.z_threshold or abs(seismic_z_score) > self.z_threshold
        return is_anomaly, water_z_score, seismic_z_score

class ModelEvaluator:
    def __init__(self):
        self.true_labels = []
        self.predicted_labels = []
        self.accuracy_history = []
        self.precision_history = []
        self.recall_history = []
        self.f1_history = []

    def update(self, true_label, predicted_label):
        self.true_labels.append(true_label)
        self.predicted_labels.append(predicted_label)

    def evaluate(self):
        if len(self.true_labels) < 2:  # Need at least two samples for evaluation
            return

        accuracy = accuracy_score(self.true_labels, self.predicted_labels)
        precision = precision_score(self.true_labels, self.predicted_labels, zero_division=0)
        recall = recall_score(self.true_labels, self.predicted_labels, zero_division=0)
        f1 = f1_score(self.true_labels, self.predicted_labels, zero_division=0)

        self.accuracy_history.append(accuracy)
        self.precision_history.append(precision)
        self.recall_history.append(recall)
        self.f1_history.append(f1)

        logging.info(f"Model Performance:\n"
                     f"  Accuracy: {accuracy:.4f}\n"
                     f"  Precision: {precision:.4f}\n"
                     f"  Recall: {recall:.4f}\n"
                     f"  F1-score: {f1:.4f}")

    def plot_performance(self, ax):
        ax.clear()
        if self.accuracy_history:
            ax.plot(self.accuracy_history, label='Accuracy')
            ax.plot(self.precision_history, label='Precision')
            ax.plot(self.recall_history, label='Recall')
            ax.plot(self.f1_history, label='F1-score')
            ax.set_title('Model Performance Over Time')
            ax.set_xlabel('Evaluation Interval')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for performance metrics', 
                    ha='center', va='center')

detector = AdaptiveThresholdDetector()
evaluator = ModelEvaluator()

def create_model():
    model = Sequential([
        Input(shape=(WINDOW_SIZE, len(FEATURES)-1)),
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=True),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

def load_or_create_model():
    global scaler
    if os.path.exists(MODEL_PATH):
        logging.info("Loading existing model...")
        model = load_model(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            scaler_data = np.load(SCALER_PATH, allow_pickle=True).item()
            scaler.mean_ = scaler_data.get('mean')
            scaler.scale_ = scaler_data.get('scale')
            scaler.n_samples_seen_ = scaler_data.get('n_samples_seen', 0)
        logging.info("Model and scaler loaded successfully.")
    else:
        logging.info("Creating new model...")
        model = create_model()
    return model

model = load_or_create_model()

def save_model_and_scaler():
    model.save(MODEL_PATH)
    np.save(SCALER_PATH, {
        'mean': scaler.mean_ if hasattr(scaler, 'mean_') else None,
        'scale': scaler.scale_ if hasattr(scaler, 'scale_') else None,
        'n_samples_seen': scaler.n_samples_seen_ if hasattr(scaler, 'n_samples_seen_') else 0
    })
    logging.info("Model and scaler saved successfully.")

def on_connect(client, userdata, flags, rc, properties=None):
    logging.info(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        timestamp = datetime.fromtimestamp(data['timestamp'])
        processed_data = [
            timestamp,
            data['distance'],
            data['accel']['x'],
            data['accel']['y'],
            data['accel']['z'],
            data['gyro']['x'],
            data['gyro']['y'],
            data['gyro']['z'],
            data['intensity']
        ]
        data_buffer.append(processed_data)
        
        detector.update(data['distance'], data['intensity'])
        
        logging.info(f"Data received - Timestamp: {timestamp}, Distance: {data['distance']:.2f}, Intensity: {data['intensity']:.2f}")
        
        if len(data_buffer) == WINDOW_SIZE:
            update_model()
            predict_tsunami_risk()
            save_model_and_scaler()
    except Exception as e:
        logging.error(f"Error processing message: {str(e)}")

def update_model():
    global scaler, model
    try:
        X = np.array([d[1:] for d in data_buffer])
        if not hasattr(scaler, 'n_samples_seen_') or scaler.n_samples_seen_ == 0:
            scaler.fit(X)
        else:
            scaler.partial_fit(X)
        X_scaled = scaler.transform(X)
        
        is_anomaly, water_z, seismic_z = detector.detect_anomaly()
        y = np.array([1 if is_anomaly else 0 for _ in range(WINDOW_SIZE)])
        
        X_train = X_scaled.reshape(1, WINDOW_SIZE, -1)
        y_train = y.reshape(1, WINDOW_SIZE, 1)
        
        history = model.fit(X_train, y_train, epochs=1, verbose=0)
        logging.info(f"Model updated - Loss: {history.history['loss'][0]:.4f}")
    except Exception as e:
        logging.error(f"Error updating model: {str(e)}")

def predict_tsunami_risk():
    if len(data_buffer) < WINDOW_SIZE:
        return

    try:
        X = np.array([d[1:] for d in data_buffer])
        if hasattr(scaler, 'n_samples_seen_') and scaler.n_samples_seen_ > 0:
            X_scaled = scaler.transform(X)
            X_pred = X_scaled.reshape(1, WINDOW_SIZE, -1)
            
            model_predictions = model.predict(X_pred)[0]
            model_prediction = np.mean(model_predictions)
        else:
            model_prediction = 0.5  # Default prediction when scaler is not ready
        
        current_distance = data_buffer[-1][1]
        seismic_intensity = data_buffer[-1][-1]
        
        is_anomaly, water_z_score, seismic_z_score = detector.detect_anomaly()
        
        combined_risk = max(model_prediction, abs(water_z_score) / detector.z_threshold, abs(seismic_z_score) / detector.z_threshold)
        
        if combined_risk > 0.8:
            risk_level = "HIGH"
        elif combined_risk > 0.6:
            risk_level = "MODERATE"
        elif combined_risk > 0.3:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"

        logging.info(f"Tsunami Risk Assessment:\n"
                     f"  Risk Level: {risk_level}\n"
                     f"  Combined Risk: {combined_risk:.2f}\n"
                     f"  Model Prediction: {model_prediction:.2f}\n"
                     f"  Water Level Z-Score: {water_z_score:.2f}\n"
                     f"  Seismic Intensity Z-Score: {seismic_z_score:.2f}\n"
                     f"  Current Water Level: {current_distance:.2f}m\n"
                     f"  Seismic Intensity: {seismic_intensity:.2f}")
        
        if risk_level == "HIGH":
            logging.critical("IMMEDIATE ACTION REQUIRED: High probability of tsunami. Initiate evacuation protocols.")
        elif risk_level == "MODERATE":
            logging.warning("ALERT: Moderate tsunami risk detected. Monitor situation closely.")
        elif risk_level == "LOW":
            logging.info("NOTICE: Low tsunami risk detected. Continue monitoring.")
        else:
            logging.info("Status: Minimal risk. Normal operations.")

        true_label = 1 if is_anomaly else 0
        predicted_label = 1 if combined_risk > 0.5 else 0
        
        evaluator.update(true_label, predicted_label)
        
        if len(evaluator.true_labels) % 100 == 0:
            evaluator.evaluate()

    except Exception as e:
        logging.error(f"Error predicting tsunami risk: {str(e)}")

def update_plot(frame):
    if len(data_buffer) > 0:
        try:
            data = np.array([d[1:] for d in data_buffer])
            timestamps = [d[0] for d in data_buffer]

            ax1.clear()
            ax2.clear()
            ax3.clear()

            ax1.plot(timestamps, data[:, 0], label='Water Level', color='cyan')
            ax1.set_title("Water Level")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Distance (m)")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)

            accel_magnitude = np.linalg.norm(data[:, 1:4].astype(float), axis=1)
            gyro_magnitude = np.linalg.norm(data[:, 4:7].astype(float), axis=1)
            ax2.plot(timestamps, accel_magnitude, label='Acceleration Magnitude', color='red')
            ax2.plot(timestamps, gyro_magnitude, label='Gyroscope Magnitude', color='green')
            ax2.set_title("Seismic Activity")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Magnitude")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)

            ax3.plot(timestamps, data[:, -1], label='Seismic Intensity', color='purple')
            if len(data_buffer) == WINDOW_SIZE and hasattr(scaler, 'n_samples_seen_') and scaler.n_samples_seen_ > 0:
                X_scaled = scaler.transform(data)
                X_pred = X_scaled.reshape(1, WINDOW_SIZE, -1)
                predictions = model.predict(X_pred)[0]
                ax3.plot(timestamps, predictions, label='Tsunami Risk', color='yellow', linestyle='--')
            ax3.set_title("Seismic Intensity and Tsunami Risk")
            ax3.set_xlabel("Time")
            ax3.set_ylabel("Intensity / Risk")
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)

            evaluator.plot_performance(ax4)

            plt.tight_layout()
        except Exception as e:
            logging.error(f"Error updating plot: {str(e)}")
            
if __name__ == "__main__":
    client = mqtt.Client(protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    ani = FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False)
    plt.show()