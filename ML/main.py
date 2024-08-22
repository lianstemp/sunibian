#ML.py
import paho.mqtt.client as mqtt
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import time
import os

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/sunibian/data"

SEQUENCE_LENGTH = 50
PREDICTION_LENGTH = 10
FEATURES = ['distance', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'intensity']

DATA_FILE = "sensor_data.csv"
MODEL_FILE = "tsunami_model.keras"
SCALER_FILE = "scaler.pkl"

data_buffer = deque(maxlen=1000)
collected_data = []

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
plt.subplots_adjust(hspace=0.5)

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    processed_data = [
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
    collected_data.append(processed_data)
    save_data(processed_data)

client = mqtt.Client(protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)

scaler = MinMaxScaler()

def save_data(data):
    with open(DATA_FILE, "a") as f:
        f.write(",".join(map(str, data)) + "\n")

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE, header=None, names=FEATURES).values.tolist()
    return []

def preprocess_data(data):
    global scaler
    if os.path.exists(SCALER_FILE):
        scaler = pd.read_pickle(SCALER_FILE)
    else:
        scaler.fit(data)
        pd.to_pickle(scaler, SCALER_FILE)
    
    scaled_data = scaler.transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH - PREDICTION_LENGTH):
        X.append(scaled_data[i:(i + SEQUENCE_LENGTH)])
        y.append(scaled_data[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + PREDICTION_LENGTH, 0])
    return np.array(X), np.array(y)

def create_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(PREDICTION_LENGTH)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    return model

def train_model(X, y, existing_model=None):
    if existing_model:
        model = existing_model
    else:
        model = create_model((X.shape[1], X.shape[2]))
    history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
    return model, history

def predict_tsunami(model, data):
    scaled_data = scaler.transform(data)
    X = scaled_data[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, len(FEATURES))
    prediction = model.predict(X)
    zeros_array = np.zeros((prediction.shape[0], prediction.shape[1], len(FEATURES)-1))
    full_prediction = np.concatenate((prediction.reshape(prediction.shape[0], prediction.shape[1], 1), zeros_array), axis=2)
    return scaler.inverse_transform(full_prediction)[0, :, 0]

def update_plot(frame):
    if len(collected_data) > SEQUENCE_LENGTH:
        df = pd.DataFrame(collected_data, columns=FEATURES)
        
        ax1.clear()
        for feature in FEATURES:
            ax1.plot(df[feature].values[-200:], label=feature)
        ax1.set_title("Real-time Sensor Data")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        if hasattr(update_plot, 'model'):
            try:
                prediction = predict_tsunami(update_plot.model, collected_data[-SEQUENCE_LENGTH:])
                ax2.clear()
                ax2.plot(range(SEQUENCE_LENGTH), df['distance'].values[-SEQUENCE_LENGTH:], label='Actual')
                ax2.plot(range(SEQUENCE_LENGTH, SEQUENCE_LENGTH+PREDICTION_LENGTH), prediction, label='Predicted')
                ax2.set_title("Tsunami Prediction")
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Water Level")
                ax2.legend()
            except Exception as e:
                print(f"Error in prediction: {e}")
        
        if hasattr(update_plot, 'history'):
            ax3.clear()
            ax3.plot(update_plot.history.history['loss'], label='Training Loss')
            ax3.plot(update_plot.history.history['val_loss'], label='Validation Loss')
            ax3.set_title("Model Loss")
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Loss")
            ax3.legend()

def training_thread():
    global collected_data
    while True:
        if len(collected_data) > SEQUENCE_LENGTH + PREDICTION_LENGTH:
            print("Training model...")
            X, y = preprocess_data(collected_data)
            
            try:
                if os.path.exists(MODEL_FILE):
                    existing_model = load_model(MODEL_FILE, custom_objects={'MeanSquaredError': MeanSquaredError})
                    model, history = train_model(X, y, existing_model)
                else:
                    model, history = train_model(X, y)
                
                model.save(MODEL_FILE)
                update_plot.model = model
                update_plot.history = history
                print("Model trained and saved successfully.")
            except Exception as e:
                print(f"Error in training: {e}")
        
        time.sleep(60)

if __name__ == "__main__":
    collected_data = load_data()
    client.loop_start()
    threading.Thread(target=training_thread, daemon=True).start()
    ani = FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False)
    plt.show()