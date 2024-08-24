import paho.mqtt.client as mqtt
import json
import numpy as np
from collections import deque
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/sunibian/data"

WINDOW_SIZE = 60
FEATURES = ['timestamp', 'distance', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'intensity']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_buffer = deque(maxlen=WINDOW_SIZE)

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4)

WATER_RECEDING_THRESHOLD = 20
ACCELERATION_THRESHOLD = 0.5
GYRO_THRESHOLD = 5
INTENSE_SEISMIC_THRESHOLD = 2.0

baseline_water_level = None
water_level_samples = deque(maxlen=30)
receding_water_detected = False
intense_seismic_activity_detected = False

def on_connect(client, userdata, flags, rc):
    logging.info(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    global baseline_water_level
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
        water_level_samples.append(data['distance'])
        
        logging.info(f"Data received - Timestamp: {timestamp}, Distance: {data['distance']:.2f}")
        
        if baseline_water_level is None and len(water_level_samples) == water_level_samples.maxlen:
            baseline_water_level = np.mean(water_level_samples)
            logging.info(f"Baseline water level set: {baseline_water_level:.2f}")
        
        detect_tsunami_precursors()
    except Exception as e:
        logging.error(f"Error processing message: {e}")

def detect_tsunami_precursors():
    global receding_water_detected, intense_seismic_activity_detected, baseline_water_level
    
    if baseline_water_level is None:
        return
    
    current_water_level = np.mean(list(water_level_samples)[-10:])
    water_level_change = baseline_water_level - current_water_level
    if water_level_change > WATER_RECEDING_THRESHOLD and not receding_water_detected:
        logging.warning(f"Water receding detected: {water_level_change:.2f} meters")
        receding_water_detected = True
    
    if len(data_buffer) > 0:
        current_data = np.array([d[1:] for d in data_buffer])
        accel_magnitude = np.linalg.norm(current_data[:, 1:4], axis=1)
        max_accel_magnitude = np.max(accel_magnitude)
        
        if max_accel_magnitude > INTENSE_SEISMIC_THRESHOLD and not intense_seismic_activity_detected:
            logging.warning(f"Intense seismic activity detected: {max_accel_magnitude:.2f} m/s^2")
            intense_seismic_activity_detected = True
    
    if receding_water_detected and intense_seismic_activity_detected:
        logging.critical("TSUNAMI EARLY WARNING: Water receding and intense seismic activity detected!")
    elif receding_water_detected:
        logging.warning("POTENTIAL TSUNAMI THREAT: Water receding detected. Monitor seismic activity.")
    elif intense_seismic_activity_detected:
        logging.warning("POTENTIAL TSUNAMI THREAT: Intense seismic activity detected. Monitor water levels.")

    if water_level_change <= WATER_RECEDING_THRESHOLD / 2:
        receding_water_detected = False
    if len(data_buffer) > 0 and max_accel_magnitude <= INTENSE_SEISMIC_THRESHOLD / 2:
        intense_seismic_activity_detected = False

def update_plot(frame):
    if len(data_buffer) > 0:
        data = np.array([d[1:] for d in data_buffer])
        timestamps = [d[0] for d in data_buffer]

        ax1.clear()
        ax2.clear()

        ax1.plot(timestamps, data[:, 0], label='Water Level', color='cyan')
        ax1.set_title("Real-time Water Level")
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

        if baseline_water_level is not None:
            ax1.axhline(y=baseline_water_level, color='white', linestyle='--', label='Baseline Water Level')
            ax1.axhline(y=baseline_water_level - WATER_RECEDING_THRESHOLD, color='yellow', linestyle='--', label='Water Receding Threshold')
        ax2.axhline(y=INTENSE_SEISMIC_THRESHOLD, color='red', linestyle='--', label='Intense Seismic Threshold')

        plt.tight_layout()

if __name__ == "__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    ani = FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False)
    plt.show()