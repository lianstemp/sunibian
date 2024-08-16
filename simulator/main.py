import paho.mqtt.client as mqtt
import json
import random
import tkinter as tk
from tkinter import ttk
import math
import numpy as np

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/sunibian/data"

root = tk.Tk()
root.title("Advanced Tsunami Simulation with Coastal Area")

box_height = 600
box_width = 1200
water_height = 300
land_width = 300
num_particles = 1000
wave_points = 300
particles = []
houses = []

canvas = tk.Canvas(root, width=box_width, height=box_height, bg="lightblue")
canvas.pack(side=tk.LEFT)

box = canvas.create_rectangle(0, 0, box_width, box_height, outline="black", fill="white")

land = canvas.create_rectangle(0, 0, land_width, box_height, fill="green")

for _ in range(10):
    x = random.uniform(20, land_width - 40)
    y = random.uniform(box_height - water_height - 100, box_height - 60)
    house = canvas.create_rectangle(x, y, x + 40, y + 60, fill="brown")
    houses.append(house)

wave = [canvas.create_line(0, 0, 0, 0, fill="blue") for _ in range(wave_points)]

# Initialize particles
for _ in range(num_particles):
    x = random.uniform(land_width, box_width)
    y = random.uniform(box_height - water_height, box_height)
    particles.append(canvas.create_oval(x-2, y-2, x+2, y+2, fill="blue", outline=""))

def generate_data(water_height, earthquake_intensity):
    distance = water_height
    accel_x = random.gauss(0, 0.2 + earthquake_intensity)
    accel_y = random.gauss(0, 0.2 + earthquake_intensity)
    accel_z = random.gauss(0, 0.2 + earthquake_intensity)
    gyro_x = random.gauss(0, 5 + earthquake_intensity * 50)
    gyro_y = random.gauss(0, 5 + earthquake_intensity * 50)
    gyro_z = random.gauss(0, 5 + earthquake_intensity * 50)
    
    data = {
        "distance": distance,
        "accel": {
            "x": accel_x,
            "y": accel_y,
            "z": accel_z
        },
        "gyro": {
            "x": gyro_x,
            "y": gyro_y,
            "z": gyro_z
        },
        "intensity": earthquake_intensity
    }
    return json.dumps(data)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print("Failed to connect, return code %d\n", rc)

client = mqtt.Client()
client.on_connect = on_connect
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

earthquake_intensity = 0
time = 0
tsunami_phase = 0
tsunami_active = False

def update_simulation():
    global particles, earthquake_intensity, time, water_height, tsunami_phase, tsunami_active

    time += 0.1
    
    if tsunami_active:
        if tsunami_phase < 100:  
            water_height = max(50, 300 - tsunami_phase * 3)
            tsunami_phase += 1
        elif tsunami_phase < 200: 
            water_height = min(550, 50 + (tsunami_phase - 100) * 10)
            tsunami_phase += 1
        else:
            tsunami_active = False
            tsunami_phase = 0
    
    wave_height = water_height
    x = np.linspace(land_width, box_width, wave_points)
    y = box_height - wave_height + np.sin(x/100 + time) * 10 * (1 + earthquake_intensity*5)
    y += np.sin(x/50 - time*1.5) * 5 * (1 + earthquake_intensity*3)
    y += np.sin(x/25 + time*2) * 3 * (1 + earthquake_intensity*2)
    
    y += np.random.normal(0, earthquake_intensity * 10 + (tsunami_active * 20), wave_points)
    
    for i, point in enumerate(wave):
        if i < wave_points - 1:
            canvas.coords(point, x[i], y[i], x[i+1], y[i+1])

    for i, particle in enumerate(particles):
        x1, y1, x2, y2 = canvas.coords(particle)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        wave_height_at_x = np.interp(center_x, x, y)

        dy = (wave_height_at_x - center_y) * 0.1
        dx = np.random.normal(0, earthquake_intensity * 5 + (tsunami_active * 10))

        new_x = center_x + dx
        new_y = center_y + dy

        # Contain within boundaries
        if new_x < land_width:
            new_x = box_width
        elif new_x > box_width:
            new_x = land_width

        if new_y < box_height - water_height - 50:
            new_y = box_height - 5
        elif new_y > box_height:
            new_y = box_height - water_height

        canvas.coords(particle, new_x-2, new_y-2, new_x+2, new_y+2)

    # Update houses
    for house in houses:
        x1, y1, x2, y2 = canvas.coords(house)
        if y2 > box_height - water_height:
            canvas.move(house, random.uniform(-2, 2), random.uniform(-2, 0))
        if y1 < 0:
            canvas.delete(house)
            houses.remove(house)

    root.after(50, update_simulation)

def send_data():
    payload = generate_data(water_height, earthquake_intensity)
    client.publish(MQTT_TOPIC, payload)
    print(f"Message sent: {payload}")
    root.after(1000, send_data)  

def update_intensity(value):
    global earthquake_intensity
    earthquake_intensity = float(value)

def update_water_level(y):
    global water_height
    new_height = box_height - y
    if 0 <= new_height <= box_height:
        water_height = new_height

def on_click(event):
    if event.x > land_width:
        update_water_level(event.y)

def on_drag(event):
    if event.x > land_width:
        update_water_level(event.y)

def trigger_tsunami():
    global tsunami_active, tsunami_phase
    tsunami_active = True
    tsunami_phase = 0

# Bind the mouse click and drag events to the canvas
canvas.bind("<Button-1>", on_click)
canvas.bind("<B1-Motion>", on_drag)

# Create control panel frame
control_panel = tk.Frame(root)
control_panel.pack(side=tk.RIGHT, padx=10, pady=10)

intensity_label = tk.Label(control_panel, text="Earthquake Intensity")
intensity_label.pack()

intensity_slider = ttk.Scale(control_panel, from_=0, to=1, orient=tk.HORIZONTAL, command=update_intensity, length=200)
intensity_slider.set(0)
intensity_slider.pack()

tsunami_button = tk.Button(control_panel, text="Trigger Tsunami", command=trigger_tsunami)
tsunami_button.pack(pady=10)

root.after(1000, send_data)
root.after(50, update_simulation)

root.mainloop()

client.loop_stop()
client.disconnect()