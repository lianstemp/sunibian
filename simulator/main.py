import tkinter as tk
from tkinter import ttk
import math
import random
import numpy as np
import paho.mqtt.client as mqtt
import json

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/sunibian/data"

root = tk.Tk()
root.title("Advanced Tsunami Simulation with Realistic Coastal Area")

box_height = 600
box_width = 1200
water_height = 200
land_width = 300
num_particles = 2000
wave_points = 300
particles = []
houses = []

canvas = tk.Canvas(root, width=box_width, height=box_height, bg="skyblue")
canvas.pack(side=tk.LEFT)

sea_polygon = canvas.create_polygon(0, 0, fill="royalblue", outline="")
wave = canvas.create_line(0, 0, 0, 0, fill="white", width=2)

def create_terrain():
    base_height = box_height - 250
    terrain_points = [(0, box_height)]
    
    for x in range(0, land_width, 10):
        y = base_height + random.randint(-15, 15) + 10 * math.sin(x / 50)
        terrain_points.append((x, y))
    
    slope_start = land_width - 100
    for x in range(slope_start, land_width + 10, 10):
        progress = (x - slope_start) / 100
        y = base_height + (box_height - base_height) * progress
        terrain_points.append((x, y))
    
    terrain_points.append((land_width, box_height))
    return terrain_points

terrain_points = create_terrain()
land = canvas.create_polygon(terrain_points, fill="forestgreen", outline="black")

def get_terrain_height(x):
    for i in range(len(terrain_points) - 1):
        if terrain_points[i][0] <= x < terrain_points[i+1][0]:
            x1, y1 = terrain_points[i]
            x2, y2 = terrain_points[i+1]
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return box_height

class HousePart:
    def __init__(self, canvas, x, y, width, height, color):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.velocity = [0, 0]
        self.shape = canvas.create_rectangle(x, y, x + width, y + height, fill=color, outline="black")

    def update(self, water_level, earthquake_intensity):
        # Apply forces
        buoyancy = max(0, (self.y + self.height - water_level) * 0.05)
        gravity = 0.2
        
        self.velocity[0] += random.uniform(-earthquake_intensity, earthquake_intensity)
        self.velocity[1] += gravity - buoyancy
        
        # Apply drag
        drag = 0.98
        self.velocity[0] *= drag
        self.velocity[1] *= drag
        
        # Update position
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        
        # Boundary checks
        if self.x < 0 or self.x + self.width > land_width:
            self.velocity[0] *= -0.5
        
        ground_level = get_terrain_height(self.x + self.width / 2)
        if self.y + self.height > ground_level:
            self.y = ground_level - self.height
            self.velocity[1] *= -0.3
        
        # Update canvas
        self.canvas.coords(self.shape, self.x, self.y, self.x + self.width, self.y + self.height)

class House:
    def __init__(self, canvas, x, y, width, height):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.parts = []
        
        # Create main structure
        self.parts.append(HousePart(canvas, x, y, width, height * 0.7, "brown"))
        
        # Create roof
        roof_height = height * 0.3
        roof = canvas.create_polygon(
            x, y,
            x + width / 2, y - roof_height,
            x + width, y,
            fill="red", outline="black"
        )
        self.parts.append(HousePart(canvas, x, y - roof_height, width, roof_height, "red"))
        
        # Create windows
        window_size = min(width, height) * 0.2
        for i in range(2):
            for j in range(2):
                wx = x + (i + 0.5) * width / 3 - window_size / 2
                wy = y + (j + 0.5) * height / 3
                self.parts.append(HousePart(canvas, wx, wy, window_size, window_size, "lightblue"))

    def update(self, water_level, earthquake_intensity):
        for part in self.parts:
            part.update(water_level, earthquake_intensity)

num_houses = 5
for _ in range(num_houses):
    x = random.uniform(20, land_width - 100)
    y = get_terrain_height(x) - 80
    house = House(canvas, x, y, 60, 80)
    houses.append(house)

def create_particle(x, y):
    size = random.uniform(1, 3)
    color = random.choice(["royalblue", "deepskyblue", "lightskyblue"])
    return canvas.create_oval(x-size, y-size, x+size, y+size, fill=color, outline="")

particles = [create_particle(random.uniform(0, box_width), random.uniform(box_height - water_height, box_height)) for _ in range(num_particles)]

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
        "accel": {"x": accel_x, "y": accel_y, "z": accel_z},
        "gyro": {"x": gyro_x, "y": gyro_y, "z": gyro_z},
        "intensity": earthquake_intensity
    }
    return json.dumps(data)

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker!" if rc == 0 else f"Failed to connect, return code {rc}")

client = mqtt.Client()
client.on_connect = on_connect
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

earthquake_intensity = 0
time = 0
tsunami_phase = 0
tsunami_active = False
earthquake_active = False
receding_phase = False
earthquake_duration = 200
receding_duration = 500
wave_buildup_duration = 300
wave_crash_duration = 400
default_water_height = 200
target_water_height = default_water_height

def update_simulation():
    global particles, earthquake_intensity, time, water_height, tsunami_phase, tsunami_active, earthquake_active, receding_phase, target_water_height

    time += 0.1
    
    if earthquake_active:
        if tsunami_phase < earthquake_duration:
            earthquake_intensity = min(1, earthquake_intensity + 0.01)
            tsunami_phase += 1
        else:
            earthquake_active = False
            receding_phase = True
            tsunami_phase = 0
    elif tsunami_active:
        if receding_phase:
            earthquake_intensity = max(0, earthquake_intensity - 0.005)
            if tsunami_phase < receding_duration:
                target_water_height = max(50, default_water_height - tsunami_phase * 0.3)
                tsunami_phase += 1
            else:
                receding_phase = False
                tsunami_phase = 0
        elif tsunami_phase < wave_buildup_duration:
            target_water_height = min(550, 50 + (tsunami_phase / wave_buildup_duration) ** 2 * 500)
            tsunami_phase += 1
        elif tsunami_phase < wave_buildup_duration + wave_crash_duration:
            progress = (tsunami_phase - wave_buildup_duration) / wave_crash_duration
            target_water_height = 550 - progress ** 0.5 * (550 - default_water_height)
            tsunami_phase += 1
        else:
            tsunami_active = False
            tsunami_phase = 0
            target_water_height = default_water_height
    else:
        earthquake_intensity = max(0, earthquake_intensity - 0.01)
    
    # Smoothly transition water height
    water_height += (target_water_height - water_height) * 0.05
    
    wave_height = water_height
    x = np.linspace(0, box_width, 300)
    
    y_water = box_height - wave_height + np.sin(x/100 + time) * 10
    y_water += np.sin(x/50 - time*1.5) * 5
    y_water += np.sin(x/25 + time*2) * 3
    
    if tsunami_active:
        if receding_phase:
            tsunami_wave = np.sin(x/200 + tsunami_phase/10) * 30 * (1 - tsunami_phase/receding_duration)
            tsunami_wave *= np.exp(-(x - box_width)**2 / (2 * (box_width/2)**2))
        else:
            tsunami_wave = np.sin(x/400 - tsunami_phase/20) * 100 * (tsunami_phase/wave_buildup_duration)
            tsunami_wave *= np.exp(-(x - box_width)**2 / (2 * (box_width/4)**2))
        y_water += tsunami_wave
    
    sea_points = list(zip(x, y_water))
    sea_points = [(0, box_height)] + sea_points + [(box_width, box_height)]
    canvas.coords(sea_polygon, *[coord for point in sea_points for coord in point])
    
    y_wave = y_water + np.random.normal(0, earthquake_intensity * 5, 300)
    wave_points = list(zip(x, y_wave))
    canvas.coords(wave, *[coord for point in wave_points for coord in point])

    for particle in particles:
        x1, y1, x2, y2 = canvas.coords(particle)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        wave_height_at_x = np.interp(center_x, x, y_water)

        dy = (wave_height_at_x - center_y) * 0.1
        dx = np.random.normal(0, earthquake_intensity * 2 + (tsunami_active * 5))

        if tsunami_active:
            if receding_phase:
                dx += 2  # Move particles right during receding
            else:
                dx -= 3  # Move particles left during tsunami wave

        new_x = center_x + dx
        new_y = center_y + dy

        if new_x < 0:
            new_x = box_width
        elif new_x > box_width:
            new_x = 0

        if new_y < box_height - water_height - 50:
            new_y = box_height - 5
        elif new_y > box_height:
            new_y = box_height - water_height

        canvas.coords(particle, new_x-2, new_y-2, new_x+2, new_y+2)

    water_level = np.mean(y_water)
    for house in houses:
        house.update(water_level, earthquake_intensity)

    canvas.tag_raise(sea_polygon)
    canvas.tag_raise(wave)
    for particle in particles:
        canvas.tag_raise(particle)
    canvas.tag_raise(land)
    for house in houses:
        for part in house.parts:
            canvas.tag_raise(part.shape)

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
    global target_water_height
    new_height = box_height - y
    if 0 <= new_height <= box_height:
        target_water_height = new_height

def on_click(event):
    update_water_level(event.y)

def on_drag(event):
    update_water_level(event.y)

def trigger_tsunami():
    global tsunami_active, tsunami_phase, earthquake_active, receding_phase, earthquake_intensity
    earthquake_active = True
    tsunami_active = True
    tsunami_phase = 0
    receding_phase = False
    earthquake_intensity = 0

canvas.bind("<Button-1>", on_click)
canvas.bind("<B1-Motion>", on_drag)

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