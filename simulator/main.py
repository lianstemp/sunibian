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
root.title("Advanced Tsunami Simulation with Realistic Coastal Area")

box_height = 600
box_width = 1200
water_height = 200
land_width = 300
num_particles = 1000
wave_points = 300
particles = []
houses = []

canvas = tk.Canvas(root, width=box_width, height=box_height, bg="lightblue")
canvas.pack(side=tk.TOP)

spectrum_canvas = tk.Canvas(root, width=box_width, height=200, bg="white")
spectrum_canvas.pack(side=tk.TOP)

sea_polygon = canvas.create_polygon(0, 0, fill="blue", outline="")

wave = canvas.create_line(0, 0, 0, 0, fill="white", width=2)

def create_terrain():
    base_height = box_height - 250
    terrain_points = [(0, box_height)]
    
    for x in range(0, land_width, 10):
        y = base_height + random.randint(-5, 5)
        terrain_points.append((x, y))
    
    slope_start = land_width - 50
    for x in range(slope_start, land_width + 10, 10):
        progress = (x - slope_start) / 50
        y = base_height + (box_height - base_height) * progress
        terrain_points.append((x, y))
    
    terrain_points.append((land_width, box_height))
    return terrain_points

terrain_points = create_terrain()
land = canvas.create_polygon(terrain_points, fill="green", outline="black")

def get_terrain_height(x):
    for i in range(len(terrain_points) - 1):
        if terrain_points[i][0] <= x < terrain_points[i+1][0]:
            x1, y1 = terrain_points[i]
            x2, y2 = terrain_points[i+1]
            return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return box_height

class House:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.velocity = 0
        self.shape = canvas.create_rectangle(x, y, x + width, y + height, fill="brown")
        self.roof = canvas.create_polygon(
            x, y,
            x + width // 2, y - height // 2,
            x + width, y,
            fill="red"
        )

    def update(self, water_level, earthquake_intensity):
        ground_level = get_terrain_height(self.x + self.width / 2)
        
        if self.y + self.height > water_level:
            buoyancy = (self.y + self.height - water_level) * 0.1
            self.velocity -= buoyancy
            self.velocity += 0.2
        else:
            self.velocity += 0.5

        self.velocity += random.uniform(-earthquake_intensity, earthquake_intensity)
        self.y += self.velocity

        if self.y + self.height > ground_level:
            self.y = ground_level - self.height
            self.velocity = -self.velocity * 0.3

        if self.y + self.height > water_level:
            self.x += random.uniform(-1, 1) * (earthquake_intensity + 0.1)
        
        self.x = max(0, min(self.x, land_width - self.width))

        canvas.coords(self.shape, self.x, self.y, self.x + self.width, self.y + self.height)
        canvas.coords(self.roof,
            self.x, self.y,
            self.x + self.width // 2, self.y - self.height // 2,
            self.x + self.width, self.y
        )

num_houses = 5
house_width = 40
house_height = 60
for i in range(num_houses):
    x = random.uniform(20, land_width - 100)
    y = get_terrain_height(x) - house_height
    house = House(x, y, house_width, house_height)
    houses.append(house)

for _ in range(num_particles):
    x = random.uniform(0, box_width)
    y = random.uniform(box_height - water_height, box_height)
    particles.append(canvas.create_oval(x-2, y-2, x+2, y+2, fill="lightblue", outline=""))

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
tsunami_wave_position = box_width
magnitude = 0

def update_simulation():
    global particles, earthquake_intensity, time, water_height, tsunami_phase, tsunami_active, tsunami_wave_position, magnitude

    time += 0.1
    
    if tsunami_active:
        if tsunami_phase < 200:
            water_height = max(50, 200 - tsunami_phase)
            earthquake_intensity = 0.5 + (tsunami_phase / 200) * 1.5
            tsunami_phase += 1
        elif tsunami_phase < 400:
            tsunami_wave_position = max(0, box_width - (tsunami_phase - 200) * 3)
            earthquake_intensity = 2
            tsunami_phase += 1
        elif tsunami_phase < 600:
            tsunami_wave_position = 0
            water_height = min(550, 50 + (tsunami_phase - 400) * 2.5)
            earthquake_intensity = 2 - ((tsunami_phase - 400) / 200) * 1.5
            tsunami_phase += 1
        else:
            tsunami_active = False
            tsunami_phase = 0
            tsunami_wave_position = box_width
            earthquake_intensity = 0
            magnitude = 0
    
    wave_height = water_height
    x = np.linspace(0, box_width, 200)
    
    y_water = box_height - wave_height + np.sin(x/100 + time) * 10
    y_water += np.sin(x/50 - time*1.5) * 5
    y_water += np.sin(x/25 + time*2) * 3
    
    if tsunami_active:
        if tsunami_phase < 200:
            y_water -= (200 - tsunami_phase) * 0.5
        elif tsunami_phase < 400:
            tsunami_wave = np.exp(-(x - tsunami_wave_position)**2 / (2 * 50000)) * 300
            y_water += tsunami_wave
        else:
            y_water += (550 - water_height) * np.exp(-(x - tsunami_wave_position)**2 / (2 * 100000))
    
    sea_points = list(zip(x, y_water))
    sea_points = [(0, box_height)] + sea_points + [(box_width, box_height)]
    canvas.coords(sea_polygon, *[coord for point in sea_points for coord in point])
    
    # Update particle positions
    for particle in particles:
        x1, y1, x2, y2 = canvas.coords(particle)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        wave_height_at_x = np.interp(center_x, x, y_water)

        dy = (wave_height_at_x - center_y) * 0.1
        dx = np.random.normal(0, earthquake_intensity * 5 + (tsunami_active * 10))

        if tsunami_active and tsunami_phase >= 200:
            dx -= 5

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
        canvas.tag_raise(house.shape)
        canvas.tag_raise(house.roof)

    spectrum_canvas.delete("all")
    spectrum_canvas.create_line(0, 100, box_width, 100, fill="black")
    spectrum_data = (np.random.rand(box_width) - 0.5) * earthquake_intensity * 200
    spectrum_points = [(x, 100 + y) for x, y in enumerate(spectrum_data)]
    spectrum_canvas.create_line(spectrum_points, fill="red", width=2)

    magnitude_text = f"Magnitude: {magnitude:.1f}"
    spectrum_canvas.create_text(box_width - 10, 10, text=magnitude_text, anchor=tk.NE, fill="blue", font=("Arial", 12, "bold"))

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
    update_water_level(event.y)

def on_drag(event):
    update_water_level(event.y)

def trigger_tsunami():
    global tsunami_active, tsunami_phase, tsunami_wave_position, magnitude
    if not tsunami_active:
        magnitude = random.uniform(7.5, 9.5)
        tsunami_active = True
        tsunami_phase = 0
        tsunami_wave_position = box_width

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