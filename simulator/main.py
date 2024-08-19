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
MQTT_TOPIC = "sensor/sunibian/gate"

root = tk.Tk()
root.title("Advanced Tsunami Simulation with Realistic Coastal Area and Flooding")

box_height = 600
box_width = 1200
initial_water_height = 50
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
    base_height = box_height - 100
    terrain_points = [(0, box_height)]
    
    for x in range(0, land_width, 10):
        y = base_height + random.randint(-15, 15) + 10 * math.sin(x / 50)
        terrain_points.append((x, y))
    
    terrain_points.append((land_width, base_height))
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
        self.damage = 0

    def update(self, water_level, earthquake_magnitude):
        ground_level = get_terrain_height(self.x + self.width / 2)
        
        water_pressure = max(0, (water_level - self.y) / self.height) * 2
        
        self.damage += water_pressure * 0.01 + earthquake_magnitude * 0.005
        self.damage = min(1, self.damage)
        
        if water_level > self.y + self.height:
            buoyancy = (water_level - (self.y + self.height)) * 0.05 * (1 - self.damage)
            self.velocity += buoyancy
        else:
            self.velocity += 0.2 * (1 - self.damage)
        
        self.velocity += random.uniform(-earthquake_magnitude, earthquake_magnitude) * 0.2
        
        self.y += self.velocity * (1 - self.damage)
        
        if self.y + self.height > ground_level:
            self.y = ground_level - self.height
            self.velocity = 0
        
        if water_level > self.y:
            self.x += random.uniform(-0.5, 0.5) * (earthquake_magnitude + water_pressure) * (1 - self.damage)
        
        self.x = max(0, min(self.x, land_width - self.width))
        
        damage_color = int(255 * (1 - self.damage))
        canvas.itemconfig(self.shape, fill=f'#{damage_color:02x}3300')
        canvas.coords(self.shape, self.x, self.y, self.x + self.width, self.y + self.height)
        canvas.coords(self.roof,
            self.x, self.y,
            self.x + self.width // 2, self.y - self.height // 2,
            self.x + self.width, self.y
        )

num_houses = 5
for _ in range(num_houses):
    x = random.uniform(20, land_width - 100)
    y = get_terrain_height(x) - 80
    house = House(x, y, 60, 80)
    houses.append(house)

def create_particle(x, y):
    size = random.uniform(1, 3)
    color = random.choice(["royalblue", "deepskyblue", "lightskyblue"])
    return canvas.create_oval(x-size, y-size, x+size, y+size, fill=color, outline="")

particles = [create_particle(random.uniform(0, box_width), random.uniform(box_height - initial_water_height, box_height)) for _ in range(num_particles)]

def generate_data(water_height, earthquake_magnitude):
    distance = water_height
    accel_x = random.gauss(0, 0.2 + earthquake_magnitude)
    accel_y = random.gauss(0, 0.2 + earthquake_magnitude)
    accel_z = random.gauss(0, 0.2 + earthquake_magnitude)
    gyro_x = random.gauss(0, 5 + earthquake_magnitude * 50)
    gyro_y = random.gauss(0, 5 + earthquake_magnitude * 50)
    gyro_z = random.gauss(0, 5 + earthquake_magnitude * 50)
    
    data = {
        "distance": distance,
        "accel": {"x": accel_x, "y": accel_y, "z": accel_z},
        "gyro": {"x": gyro_x, "y": gyro_y, "z": gyro_z},
        "magnitude": earthquake_magnitude
    }
    return json.dumps(data)

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker!" if rc == 0 else f"Failed to connect, return code {rc}")

client = mqtt.Client()
client.on_connect = on_connect
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

earthquake_magnitude = 0
time = 0
tsunami_phase = 0
tsunami_active = False
tsunami_wave_position = box_width
water_height = initial_water_height
target_water_height = initial_water_height

tsunami_start_time = 0
earthquake_start_time = 0
earthquake_duration = 60
tsunami_approach_duration = 180
tsunami_impact_duration = 240
tsunami_retreat_duration = 300

max_water_height = 400
min_water_height = 10

def update_simulation():
    global particles, earthquake_magnitude, time, water_height, tsunami_phase, tsunami_active, tsunami_wave_position, target_water_height
    global tsunami_start_time, earthquake_start_time

    time += 0.1
    
    if tsunami_active:
        elapsed_time = time - tsunami_start_time
        
        if elapsed_time < earthquake_duration:
            progress = elapsed_time / earthquake_duration
            earthquake_magnitude = min(9.0, progress * 9.0)
            target_water_height = initial_water_height
        
        elif elapsed_time < tsunami_approach_duration:
            progress = (elapsed_time - earthquake_duration) / (tsunami_approach_duration - earthquake_duration)
            target_water_height = max(min_water_height, initial_water_height - progress * 40)
            earthquake_magnitude = max(0, 9.0 - progress * 4.0)
        
        elif elapsed_time < tsunami_impact_duration:
            progress = (elapsed_time - tsunami_approach_duration) / (tsunami_impact_duration - tsunami_approach_duration)
            target_water_height = min_water_height + progress * (max_water_height - min_water_height)
            tsunami_wave_position = box_width * (1 - progress)
            earthquake_magnitude = 5.0 + progress * 2.0
        
        else:
            progress = min(1, (elapsed_time - tsunami_impact_duration) / tsunami_retreat_duration)
            target_water_height = max(initial_water_height, max_water_height - progress * (max_water_height - initial_water_height))
            tsunami_wave_position = 0
            earthquake_magnitude = max(0, 7.0 - progress * 7.0)
        
        water_height += (target_water_height - water_height) * 0.05
    
    else:
        water_height = initial_water_height
        earthquake_magnitude = max(0, earthquake_magnitude - 0.01)
    
    x = np.linspace(0, box_width, 300)
    y_water = box_height - water_height + np.sin(x/100 + time) * 10
    y_water += np.sin(x/50 - time*1.5) * 5
    y_water += np.sin(x/25 + time*2) * 3
    
    if tsunami_active and tsunami_wave_position < box_width:
        tsunami_wave = np.exp(-(x - tsunami_wave_position)**2 / (2 * 50000)) * 300
        y_water += tsunami_wave
    
    # Allow water to flood the land
    land_mask = x < land_width
    y_water[land_mask] = np.minimum(y_water[land_mask], np.array([get_terrain_height(xi) for xi in x[land_mask]]))
    
    sea_points = list(zip(x, y_water))
    sea_points = [(0, box_height)] + sea_points + [(box_width, box_height)]
    canvas.coords(sea_polygon, *[coord for point in sea_points for coord in point])
    
    y_wave = y_water + np.random.normal(0, earthquake_magnitude * 5, 300)
    wave_points = list(zip(x, y_wave))
    canvas.coords(wave, *[coord for point in wave_points for coord in point])

    for particle in particles:
        x1, y1, x2, y2 = canvas.coords(particle)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

        wave_height_at_x = np.interp(center_x, x, y_water)

        dy = (wave_height_at_x - center_y) * 0.1
        dx = np.random.normal(0, earthquake_magnitude)

        if tsunami_active:
            if elapsed_time < tsunami_approach_duration:
                dx += 2
            else:
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

    water_level = box_height - np.mean(y_water)
    for house in houses:
        house.update(water_level, earthquake_magnitude)

    canvas.tag_raise(sea_polygon)
    canvas.tag_raise(wave)
    for particle in particles:
        canvas.tag_raise(particle)
    canvas.tag_raise(land)
    for house in houses:
        canvas.tag_raise(house.shape)
        canvas.tag_raise(house.roof)

    water_height_label.config(text=f"Water Height: {water_height/10:.2f} m")
    earthquake_magnitude_label.config(text=f"Earthquake Magnitude: {earthquake_magnitude:.2f}")

    root.after(50, update_simulation)

def send_data():
    payload = generate_data(water_height, earthquake_magnitude)
    client.publish(MQTT_TOPIC, payload)
    print(f"Message sent: {payload}")
    root.after(1000, send_data)

def update_magnitude(value):
    global earthquake_magnitude
    earthquake_magnitude = float(value)

def update_water_level(y):
    global water_height, initial_water_height
    new_height = box_height - y
    if 0 <= new_height <= box_height:
        water_height = new_height
        initial_water_height = new_height

def on_click(event):
    update_water_level(event.y)

def on_drag(event):
    update_water_level(event.y)

def trigger_tsunami():
    global tsunami_active, tsunami_phase, tsunami_wave_position, tsunami_start_time
    tsunami_active = True
    tsunami_phase = 0
    tsunami_wave_position = box_width
    tsunami_start_time = time

canvas.bind("<Button-1>", on_click)
canvas.bind("<B1-Motion>", on_drag)

control_panel = tk.Frame(root)
control_panel.pack(side=tk.RIGHT, padx=10, pady=10)

magnitude_label = tk.Label(control_panel, text="Earthquake Magnitude")
magnitude_label.pack()

magnitude_slider = ttk.Scale(control_panel, from_=0, to=9.0, orient=tk.HORIZONTAL, command=update_magnitude, length=200)
magnitude_slider.set(0)
magnitude_slider.pack()

tsunami_button = tk.Button(control_panel, text="Trigger Tsunami", command=trigger_tsunami)
tsunami_button.pack(pady=10)

water_height_label = tk.Label(control_panel, text="Water Height: 0 m")
water_height_label.pack()

earthquake_magnitude_label = tk.Label(control_panel, text="Earthquake Magnitude: 0.0")
earthquake_magnitude_label.pack()

root.after(1000, send_data)
root.after(50, update_simulation)

root.mainloop()

client.loop_stop()
client.disconnect()